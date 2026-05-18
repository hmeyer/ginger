//! Deterministic frame-sequence loader for the offline replay harness.
//!
//! The replay *runner* (detect → match → track) lives in the main crate
//! because it pulls the libcamera-coupled SLAM frontend; the parts that
//! are camera-free — decoding a recorded sequence and its intrinsics —
//! live here so they cross-compile and unit-test without the hardware
//! stack.
//!
//! Frames are binary PGM (`P5`): no image-codec dependency, trivially
//! reproducible, and what the recorder will dump. A sequence is every
//! `*.pgm` in a directory, **lexicographically sorted** so replay order
//! is stable regardless of filesystem enumeration order (zero-pad frame
//! numbers when recording).

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::intrinsics::Intrinsics;

/// A single grayscale frame (8-bit, row-major).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct GrayFrame {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

fn invalid(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

/// Parse a binary PGM (`P5`, maxval ≤ 255). Supports `#` comments and
/// arbitrary ASCII whitespace in the header, per the format.
pub fn parse_pgm(bytes: &[u8]) -> io::Result<GrayFrame> {
    let mut pos = 0;
    let mut token = || -> io::Result<String> {
        // Skip whitespace and #-to-EOL comments.
        loop {
            while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
                pos += 1;
            }
            if pos < bytes.len() && bytes[pos] == b'#' {
                while pos < bytes.len() && bytes[pos] != b'\n' {
                    pos += 1;
                }
            } else {
                break;
            }
        }
        let start = pos;
        while pos < bytes.len() && !bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if start == pos {
            return Err(invalid("PGM: unexpected end of header"));
        }
        Ok(String::from_utf8_lossy(&bytes[start..pos]).into_owned())
    };

    if token()? != "P5" {
        return Err(invalid("PGM: not a binary P5 file"));
    }
    let parse = |s: String| {
        s.parse::<u32>()
            .map_err(|_| invalid("PGM: bad header number"))
    };
    let width = parse(token()?)?;
    let height = parse(token()?)?;
    let maxval = parse(token()?)?;
    if maxval == 0 || maxval > 255 {
        return Err(invalid("PGM: only 8-bit (maxval 1..=255) supported"));
    }
    // Exactly one whitespace byte separates the header from the raster.
    pos += 1;
    let n = (width as usize) * (height as usize);
    let raster = bytes
        .get(pos..pos + n)
        .ok_or_else(|| invalid("PGM: raster shorter than width*height"))?;
    Ok(GrayFrame {
        width,
        height,
        data: raster.to_vec(),
    })
}

/// Encode a [`GrayFrame`] as binary PGM (used by tests and the recorder).
pub fn encode_pgm(f: &GrayFrame) -> Vec<u8> {
    let mut out = format!("P5\n{} {}\n255\n", f.width, f.height).into_bytes();
    out.extend_from_slice(&f.data);
    out
}

/// Load one PGM file.
pub fn load_pgm<P: AsRef<Path>>(path: P) -> io::Result<GrayFrame> {
    parse_pgm(&fs::read(path)?)
}

/// An ordered, deterministic sequence of `*.pgm` frames in a directory,
/// optionally with a sibling `slam.toml` of intrinsics.
pub struct FrameSequence {
    pub frames: Vec<PathBuf>,
    pub intrinsics: Option<Intrinsics>,
}

impl FrameSequence {
    /// Scan `dir` for `*.pgm` (lexicographically sorted) and, if present,
    /// read `dir/slam.toml`.
    pub fn open<P: AsRef<Path>>(dir: P) -> io::Result<Self> {
        let dir = dir.as_ref();
        let mut frames: Vec<PathBuf> = fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "pgm"))
            .collect();
        frames.sort();
        let toml = dir.join("slam.toml");
        let intrinsics = match fs::read_to_string(&toml) {
            Ok(s) => Some(
                Intrinsics::from_toml_str(&s).map_err(|e| invalid(&format!("slam.toml: {e}")))?,
            ),
            Err(e) if e.kind() == io::ErrorKind::NotFound => None,
            Err(e) => return Err(e),
        };
        Ok(Self { frames, intrinsics })
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Decode every frame in order.
    pub fn iter_frames(&self) -> impl Iterator<Item = io::Result<GrayFrame>> + '_ {
        self.frames.iter().map(load_pgm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unique scratch dir under the system temp, cleaned on drop.
    struct TmpDir(PathBuf);
    impl TmpDir {
        fn new(tag: &str) -> Self {
            let t = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let p = std::env::temp_dir().join(format!("slamcore-{tag}-{t}"));
            fs::create_dir_all(&p).unwrap();
            Self(p)
        }
    }
    impl Drop for TmpDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn frame(w: u32, h: u32, fill: u8) -> GrayFrame {
        GrayFrame {
            width: w,
            height: h,
            data: vec![fill; (w * h) as usize],
        }
    }

    #[test]
    fn pgm_roundtrips() {
        let mut f = frame(4, 3, 0);
        for (i, p) in f.data.iter_mut().enumerate() {
            *p = i as u8 * 10;
        }
        assert_eq!(parse_pgm(&encode_pgm(&f)).unwrap(), f);
    }

    #[test]
    fn pgm_header_comments_and_spacing() {
        let raw = b"P5\n# recorded by ginger\n  2 2\n255\n\x01\x02\x03\x04";
        let g = parse_pgm(raw).unwrap();
        assert_eq!((g.width, g.height), (2, 2));
        assert_eq!(g.data, vec![1, 2, 3, 4]);
    }

    #[test]
    fn rejects_truncated_and_non_p5() {
        assert!(parse_pgm(b"P2\n2 2\n255\n1 2 3 4").is_err());
        assert!(parse_pgm(b"P5\n4 4\n255\n\x00\x00").is_err());
    }

    #[test]
    fn sequence_is_lexicographically_ordered_with_intrinsics() {
        let d = TmpDir::new("seq");
        // Write out of order; zero-padded names must replay in order.
        for (name, fill) in [
            ("frame_0002.pgm", 20),
            ("frame_0000.pgm", 0),
            ("frame_0001.pgm", 10),
        ] {
            fs::write(d.0.join(name), encode_pgm(&frame(2, 2, fill))).unwrap();
        }
        fs::write(d.0.join("ignore.txt"), b"not a frame").unwrap();
        fs::write(
            d.0.join("slam.toml"),
            Intrinsics::rev1_3_prior(800, 600).to_toml_string(),
        )
        .unwrap();

        let seq = FrameSequence::open(&d.0).unwrap();
        assert_eq!(seq.len(), 3);
        assert!(!seq.intrinsics.as_ref().unwrap().verified);
        let fills: Vec<u8> = seq.iter_frames().map(|f| f.unwrap().data[0]).collect();
        assert_eq!(fills, vec![0, 10, 20]);
    }

    #[test]
    fn missing_intrinsics_is_ok() {
        let d = TmpDir::new("notoml");
        fs::write(d.0.join("a.pgm"), encode_pgm(&frame(1, 1, 5))).unwrap();
        let seq = FrameSequence::open(&d.0).unwrap();
        assert!(seq.intrinsics.is_none());
        assert_eq!(seq.len(), 1);
    }
}
