//! Frame-ID marker codec: a small high-contrast block burned into the
//! luma plane before H.264 encode, so the browser can read back exactly
//! which camera frame a displayed video frame is, and overlay the SLAM
//! features computed from *that* frame. No clock sync needed.
//!
//! # Geometry (per the updated plan)
//!
//! * Data grid: **`MARKER_W`×`MARKER_H` = 48×16 px**, 4×4 px cells →
//!   **12 cols × 4 rows = 48 cells**, row-major (`c = row*12 + col`).
//! * 1-cell (4 px) solid-black alignment border around the grid, plus a
//!   4 px black quiet zone — the whole `ROI` (64×32) is flush to the
//!   bottom-right frame corner.
//! * Cell luma is exactly **16 (0) / 235 (1)** — limited-range BT.709
//!   extremes, never 0/255.
//!
//! # Bit layout (48 cells, 28 used)
//!
//! | cells  | field |
//! |--------|-------|
//! | 0..12  | 12-bit sync prefix `0xA5C` (MSB first) |
//! | 12..20 | 8-bit frame ID (low 8 bits of the frame counter, MSB first) |
//! | 20..28 | CRC-8 (poly 0x07, init 0x00) over the ID byte |
//! | 28..48 | spare, black |
//!
//! Only 8 ID bits travel in the video (robust: bigger relative cells,
//! no Hamming). The full `u64` frame id is sent over SSE; the client
//! disambiguates the low-8 match by proximity to the last-matched id
//! (buffer span ≪ the 256-id wrap interval). CRC-8 + the sync gate keep
//! a corrupted marker from ever yielding a wrong id (it returns `None`).

// ── Geometry ──────────────────────────────────────────────────────────────────

pub const MARKER_W: usize = 48;
pub const MARKER_H: usize = 16;

const CELL: usize = 4;
const COLS: usize = MARKER_W / CELL; // 12
const ROWS: usize = MARKER_H / CELL; // 4
const N_CELLS: usize = COLS * ROWS; // 48
const BORDER: usize = CELL; // 1-cell black ring around the grid
const MARGIN: usize = 4; // black quiet zone, all sides
/// Full reserved/excluded rectangle (grid + border + quiet zone).
pub const ROI_W: usize = MARKER_W + 2 * BORDER + 2 * MARGIN; // 64
pub const ROI_H: usize = MARKER_H + 2 * BORDER + 2 * MARGIN; // 32

const BLACK: u8 = 16;
const WHITE: u8 = 235;
const SYNC: u16 = 0xA5C;
const SYNC_BITS: usize = 12;
const ID_BITS: usize = 8;
const CRC_BITS: usize = 8;
const USED_CELLS: usize = SYNC_BITS + ID_BITS + CRC_BITS; // 28

/// Bottom-right rect of the 48×16 **data grid** within a frame.
/// Both the encoder and the client crop the grid with this.
pub fn data_grid_rect(frame_w: usize, frame_h: usize) -> (usize, usize, usize, usize) {
    let x = frame_w - MARGIN - BORDER - MARKER_W;
    let y = frame_h - MARGIN - BORDER - MARKER_H;
    (x, y, MARKER_W, MARKER_H)
}

/// Bottom-right rect SLAM must exclude (grid + border + quiet zone).
pub fn roi_rect(frame_w: usize, frame_h: usize) -> (usize, usize, usize, usize) {
    (frame_w - ROI_W, frame_h - ROI_H, ROI_W, ROI_H)
}

// ── CRC-8 (poly 0x07, init 0x00, no reflect) ──────────────────────────────────

fn crc8(byte: u8) -> u8 {
    let mut c = byte;
    for _ in 0..8 {
        c = if c & 0x80 != 0 {
            (c << 1) ^ 0x07
        } else {
            c << 1
        };
    }
    c
}

// ── Bit grid ↔ frame ID ───────────────────────────────────────────────────────

/// The 48 cell bits for `frame_id` (white = true). Only the low 8 bits
/// of `frame_id` are encoded.
fn marker_cells(frame_id: u64) -> [bool; N_CELLS] {
    let id = (frame_id & 0xFF) as u8;
    let crc = crc8(id);
    let mut cells = [false; N_CELLS];
    for (i, slot) in cells.iter_mut().enumerate().take(SYNC_BITS) {
        *slot = (SYNC >> (SYNC_BITS - 1 - i)) & 1 != 0;
    }
    for k in 0..ID_BITS {
        cells[SYNC_BITS + k] = (id >> (ID_BITS - 1 - k)) & 1 != 0;
    }
    for k in 0..CRC_BITS {
        cells[SYNC_BITS + ID_BITS + k] = (crc >> (CRC_BITS - 1 - k)) & 1 != 0;
    }
    cells
}

/// Decode 48 cell bits to the 8-bit frame id, or `None` if invalid.
fn cells_to_id(cells: &[bool; N_CELLS]) -> Option<u64> {
    let mut sync = 0u16;
    for &b in &cells[..SYNC_BITS] {
        sync = (sync << 1) | b as u16;
    }
    if sync != SYNC {
        return None; // sync gate — never emit a wrong id
    }
    let mut id = 0u8;
    for &b in &cells[SYNC_BITS..SYNC_BITS + ID_BITS] {
        id = (id << 1) | b as u8;
    }
    let mut crc = 0u8;
    for &b in &cells[SYNC_BITS + ID_BITS..USED_CELLS] {
        crc = (crc << 1) | b as u8;
    }
    if crc8(id) == crc {
        Some(id as u64)
    } else {
        None
    }
}

// ── Public codec ──────────────────────────────────────────────────────────────

/// Paint the marker by calling `put(x, y, luma)` for every pixel in the
/// bottom-right ROI (quiet zone + border black, data cells black/white).
fn paint_marker(
    frame_w: usize,
    frame_h: usize,
    frame_id: u64,
    mut put: impl FnMut(usize, usize, u8),
) {
    if frame_w < ROI_W || frame_h < ROI_H {
        return;
    }
    let (rx, ry, _, _) = roi_rect(frame_w, frame_h);
    for y in ry..ry + ROI_H {
        for x in rx..rx + ROI_W {
            put(x, y, BLACK);
        }
    }
    let (gx, gy, _, _) = data_grid_rect(frame_w, frame_h);
    let cells = marker_cells(frame_id);
    for (c, &on) in cells.iter().enumerate() {
        if !on {
            continue;
        }
        let (col, row) = (c % COLS, c / COLS);
        for dy in 0..CELL {
            for dx in 0..CELL {
                put(gx + col * CELL + dx, gy + row * CELL + dy, WHITE);
            }
        }
    }
}

/// Burn the `frame_id` marker into a **packed** luma plane
/// (`y_plane[y*stride + x]`).
pub fn encode_marker(
    y_plane: &mut [u8],
    stride: usize,
    frame_w: usize,
    frame_h: usize,
    frame_id: u64,
) {
    paint_marker(frame_w, frame_h, frame_id, |x, y, v| {
        y_plane[y * stride + x] = v;
    });
}

/// Burn the `frame_id` marker into a **YUYV** buffer in place (luma is
/// every even byte; row stride = `frame_w * 2`). This is what rides
/// through the H.264/WebRTC path — the same buffer the SLAM thread sees,
/// so no extra copy and a guaranteed-shared id.
pub fn encode_marker_yuyv(yuyv: &mut [u8], frame_w: usize, frame_h: usize, frame_id: u64) {
    let len = yuyv.len();
    paint_marker(frame_w, frame_h, frame_id, |x, y, v| {
        let idx = (y * frame_w + x) * 2;
        if idx < len {
            yuyv[idx] = v;
        }
    });
}

/// Decode the 8-bit frame id from an RGBA buffer that spans **exactly
/// the data grid** (any size ≥ the grid; cell centres are sampled
/// proportionally, so display-scaled crops work too).
pub fn decode_marker(rgba: &[u8], rgba_w: usize, rgba_h: usize) -> Option<u64> {
    if rgba_w < COLS || rgba_h < ROWS || rgba.len() < rgba_w * rgba_h * 4 {
        return None;
    }
    let mut cells = [false; N_CELLS];
    for (c, cell) in cells.iter_mut().enumerate() {
        let (col, row) = (c % COLS, c / COLS);
        let px = (((col as f32 + 0.5) / COLS as f32) * rgba_w as f32) as usize;
        let py = (((row as f32 + 0.5) / ROWS as f32) * rgba_h as f32) as usize;
        let i = (py.min(rgba_h - 1) * rgba_w + px.min(rgba_w - 1)) * 4;
        let luma =
            (rgba[i] as u32 * 299 + rgba[i + 1] as u32 * 587 + rgba[i + 2] as u32 * 114) / 1000;
        *cell = luma > (BLACK as u32 + WHITE as u32) / 2;
    }
    cells_to_id(&cells)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FW: usize = 640;
    const FH: usize = 480;

    /// Render `id` into a frame's luma, then crop the data grid to RGBA.
    fn grid_rgba(id: u64) -> (Vec<u8>, usize, usize) {
        let mut y = vec![100u8; FW * FH];
        encode_marker(&mut y, FW, FW, FH, id);
        let (gx, gy, gw, gh) = data_grid_rect(FW, FH);
        let mut rgba = vec![0u8; gw * gh * 4];
        for r in 0..gh {
            for c in 0..gw {
                let v = y[(gy + r) * FW + gx + c];
                let o = (r * gw + c) * 4;
                rgba[o] = v;
                rgba[o + 1] = v;
                rgba[o + 2] = v;
                rgba[o + 3] = 255;
            }
        }
        (rgba, gw, gh)
    }

    #[test]
    fn roundtrip_all_8bit_ids() {
        // Marker carries the low 8 bits; full ids map onto that.
        for id in [0u64, 1, 7, 0x5A, 0xFF, 0xDEADBE, 0x123456] {
            let (rgba, w, h) = grid_rgba(id);
            assert_eq!(decode_marker(&rgba, w, h), Some(id & 0xFF), "id {id:#x}");
        }
    }

    #[test]
    fn roundtrip_survives_quantization_noise_blur() {
        let mut s = 0x1234_5678u64;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as u32
        };
        let levels = [16u8, 64, 128, 192, 235];
        let mut ok = 0;
        let trials = 1000;
        for _ in 0..trials {
            let id = (rng() & 0xFF) as u64;
            let (mut rgba, w, h) = grid_rgba(id);
            for px in rgba.chunks_exact_mut(4) {
                let q = *levels
                    .iter()
                    .min_by_key(|&&l| (l as i32 - px[0] as i32).abs())
                    .unwrap();
                let n = (rng() % 17) as i32 - 8;
                let v = (q as i32 + n).clamp(0, 255) as u8;
                px[0] = v;
                px[1] = v;
                px[2] = v;
            }
            let src: Vec<u8> = rgba.iter().step_by(4).copied().collect();
            for yy in 0..h {
                for xx in 0..w {
                    let mut sum = 0u32;
                    let mut n = 0u32;
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            let nx = xx as i32 + dx;
                            let ny = yy as i32 + dy;
                            if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                                sum += src[ny as usize * w + nx as usize] as u32;
                                n += 1;
                            }
                        }
                    }
                    let v = (sum / n) as u8;
                    let o = (yy * w + xx) * 4;
                    rgba[o] = v;
                    rgba[o + 1] = v;
                    rgba[o + 2] = v;
                }
            }
            if decode_marker(&rgba, w, h) == Some(id) {
                ok += 1;
            }
        }
        assert!(
            ok as f32 / trials as f32 >= 0.999,
            "decoded {ok}/{trials} after quantization+noise+blur"
        );
    }

    #[test]
    fn single_cell_flip_never_wrong() {
        let (gx, gy, _, _) = data_grid_rect(FW, FH);
        for id in [0u64, 0x0F, 0xFF, 7, 0x5A] {
            for c in 0..N_CELLS {
                let mut y = vec![100u8; FW * FH];
                encode_marker(&mut y, FW, FW, FH, id);
                let (col, row) = (c % COLS, c / COLS);
                for dy in 0..CELL {
                    for dx in 0..CELL {
                        let p = (gy + row * CELL + dy) * FW + gx + col * CELL + dx;
                        y[p] = if y[p] == WHITE { BLACK } else { WHITE };
                    }
                }
                let (gw, gh) = (MARKER_W, MARKER_H);
                let mut rgba = vec![0u8; gw * gh * 4];
                for r in 0..gh {
                    for cc in 0..gw {
                        let v = y[(gy + r) * FW + gx + cc];
                        let o = (r * gw + cc) * 4;
                        rgba[o] = v;
                        rgba[o + 1] = v;
                        rgba[o + 2] = v;
                        rgba[o + 3] = 255;
                    }
                }
                if let Some(d) = decode_marker(&rgba, gw, gh) {
                    assert_eq!(d, id & 0xFF, "cell {c} flip on id {id:#x} → wrong id");
                }
            }
        }
    }

    #[test]
    fn two_cell_flips_none_dominates_wrong() {
        let mut s = 0xC0FFEEu64;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as usize
        };
        let (gx, gy, _, _) = data_grid_rect(FW, FH);
        let (mut wrong, mut none_or_ok) = (0, 0);
        for _ in 0..2000 {
            let id = (rng() & 0xFF) as u64;
            let mut y = vec![100u8; FW * FH];
            encode_marker(&mut y, FW, FW, FH, id);
            let a = rng() % N_CELLS;
            let mut b = rng() % N_CELLS;
            if b == a {
                b = (b + 1) % N_CELLS;
            }
            for cell in [a, b] {
                let (col, row) = (cell % COLS, cell / COLS);
                for dy in 0..CELL {
                    for dx in 0..CELL {
                        let p = (gy + row * CELL + dy) * FW + gx + col * CELL + dx;
                        y[p] = if y[p] == WHITE { BLACK } else { WHITE };
                    }
                }
            }
            let (gw, gh) = (MARKER_W, MARKER_H);
            let mut rgba = vec![0u8; gw * gh * 4];
            for r in 0..gh {
                for cc in 0..gw {
                    let v = y[(gy + r) * FW + gx + cc];
                    let o = (r * gw + cc) * 4;
                    rgba[o] = v;
                    rgba[o + 1] = v;
                    rgba[o + 2] = v;
                    rgba[o + 3] = 255;
                }
            }
            match decode_marker(&rgba, gw, gh) {
                Some(d) if d != id & 0xFF => wrong += 1,
                _ => none_or_ok += 1,
            }
        }
        assert!(
            none_or_ok > wrong,
            "two-cell flips: wrong={wrong} none_or_ok={none_or_ok}"
        );
    }
}
