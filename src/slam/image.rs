//! Grayscale image + scale pyramid for the SLAM frontend.
//!
//! The camera publishes YUYV (`[Y0 U Y1 V]` per 4 bytes), so the luma
//! plane the detectors want is simply every even byte — extracting it is
//! a cheap strided copy, no colour conversion.

use crate::camera::Frame;

/// A single-channel 8-bit image, row-major, no padding.
#[derive(Clone)]
pub struct GrayImage {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl GrayImage {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0; width * height],
        }
    }

    #[inline]
    pub fn at(&self, x: usize, y: usize) -> u8 {
        self.data[y * self.width + x]
    }

    /// Extract the luma plane from a YUYV [`Frame`] (every even byte).
    pub fn from_yuyv(frame: &Frame) -> Self {
        let (w, h) = (frame.width as usize, frame.height as usize);
        let mut data = vec![0u8; w * h];
        // YUYV is 2 bytes/pixel; byte 2*i is the Y for pixel i.
        for (dst, src) in data.iter_mut().zip(frame.data.chunks_exact(2)) {
            *dst = src[0];
        }
        Self {
            width: w,
            height: h,
            data,
        }
    }

    /// Bilinear downscale to `(w, h)`. Used to build the pyramid; the
    /// source is larger so this is a reduction (mild low-pass + sample).
    pub fn resized(&self, w: usize, h: usize) -> Self {
        let mut out = GrayImage::new(w, h);
        if w == 0 || h == 0 {
            return out;
        }
        // Map output centre back to input space.
        let sx = self.width as f32 / w as f32;
        let sy = self.height as f32 / h as f32;
        for oy in 0..h {
            let fy = ((oy as f32 + 0.5) * sy - 0.5).max(0.0);
            let y0 = fy.floor() as usize;
            let y1 = (y0 + 1).min(self.height - 1);
            let wy = fy - y0 as f32;
            for ox in 0..w {
                let fx = ((ox as f32 + 0.5) * sx - 0.5).max(0.0);
                let x0 = fx.floor() as usize;
                let x1 = (x0 + 1).min(self.width - 1);
                let wx = fx - x0 as f32;
                let p00 = self.at(x0, y0) as f32;
                let p10 = self.at(x1, y0) as f32;
                let p01 = self.at(x0, y1) as f32;
                let p11 = self.at(x1, y1) as f32;
                let top = p00 + (p10 - p00) * wx;
                let bot = p01 + (p11 - p01) * wx;
                out.data[oy * w + ox] = (top + (bot - top) * wy).round() as u8;
            }
        }
        out
    }
}

/// One pyramid level: the image plus the factor mapping a coordinate at
/// this level back to level 0 (`scale_factor ** level`).
pub struct PyramidLevel {
    pub image: GrayImage,
    pub scale: f32,
}

/// ORB-SLAM-style scale pyramid: `n_levels` images, each `scale_factor`×
/// smaller than the previous (level 0 is the original).
pub fn build_pyramid(base: GrayImage, n_levels: usize, scale_factor: f32) -> Vec<PyramidLevel> {
    let mut levels = Vec::with_capacity(n_levels);
    let mut scale = 1.0f32;
    for lvl in 0..n_levels {
        let image = if lvl == 0 {
            base.clone()
        } else {
            let w = (base.width as f32 / scale).round() as usize;
            let h = (base.height as f32 / scale).round() as usize;
            if w < 8 || h < 8 {
                break;
            }
            base.resized(w, h)
        };
        levels.push(PyramidLevel { image, scale });
        scale *= scale_factor;
    }
    levels
}

#[cfg(test)]
mod tests {
    use super::*;

    fn yuyv_frame(w: u32, h: u32, y_fill: u8) -> Frame {
        // [Y0 U Y1 V] groups; set both Y to y_fill, chroma to 128.
        let mut data = vec![128u8; (w * h * 2) as usize];
        for i in 0..(w * h) as usize {
            data[i * 2] = y_fill;
        }
        Frame {
            width: w,
            height: h,
            id: 0,
            data,
        }
    }

    #[test]
    fn yuyv_luma_extraction() {
        let f = yuyv_frame(4, 2, 200);
        let g = GrayImage::from_yuyv(&f);
        assert_eq!((g.width, g.height), (4, 2));
        assert!(g.data.iter().all(|&p| p == 200));
    }

    #[test]
    fn pyramid_shrinks_and_stops() {
        let base = GrayImage::new(640, 480);
        let levels = build_pyramid(base, 8, 1.2);
        assert_eq!(levels.len(), 8);
        assert_eq!((levels[0].image.width, levels[0].image.height), (640, 480));
        assert!((levels[0].scale - 1.0).abs() < 1e-6);
        // Each level is strictly smaller and scale grows by ~1.2.
        for w in levels.windows(2) {
            assert!(w[1].image.width < w[0].image.width);
            assert!((w[1].scale / w[0].scale - 1.2).abs() < 1e-3);
        }
    }

    #[test]
    fn resize_preserves_constant_image() {
        let mut g = GrayImage::new(20, 20);
        g.data.iter_mut().for_each(|p| *p = 123);
        let r = g.resized(7, 9);
        assert_eq!((r.width, r.height), (7, 9));
        assert!(r.data.iter().all(|&p| p == 123));
    }
}
