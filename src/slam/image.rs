//! Grayscale image + scale pyramid for the SLAM frontend.
//!
//! The compute lives in the dependency-light `ginger-fast` crate (so it
//! cross-compiles for aarch64 without the camera stack); this module
//! re-exports it and adds the camera-coupled YUYV→luma adapter.
//!
//! The camera publishes YUYV (`[Y0 U Y1 V]` per 4 bytes), so the luma
//! plane the detectors want is simply every even byte — extracting it is
//! a cheap strided copy, no colour conversion.

pub use ginger_fast::image::{GrayImage, PyramidLevel, build_pyramid};

use crate::camera::Frame;

/// Extract the luma plane from a YUYV [`Frame`] (every even byte).
pub fn gray_from_yuyv(frame: &Frame) -> GrayImage {
    let (w, h) = (frame.width as usize, frame.height as usize);
    let mut img = GrayImage::new(w, h);
    // YUYV is 2 bytes/pixel; byte 2*i is the Y for pixel i.
    for (dst, src) in img.data.iter_mut().zip(frame.data.chunks_exact(2)) {
        *dst = src[0];
    }
    img
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
            data,
        }
    }

    #[test]
    fn yuyv_luma_extraction() {
        let f = yuyv_frame(4, 2, 200);
        let g = gray_from_yuyv(&f);
        assert_eq!((g.width, g.height), (4, 2));
        assert!(g.data.iter().all(|&p| p == 200));
    }
}
