//! 8× WS2812B RGB LED strip via SPI (SPI bus 0, GRB order).
//!
//! Uses MOSI at ~6.4 MHz to encode WS2812B bit timing:
//!   bit=1 → 0xF8, bit=0 → 0x80  (one SPI byte per WS2812B bit)

use rppal::spi::{Bus, Mode, SlaveSelect, Spi};

use crate::Result;

pub const LED_COUNT: usize = 8;
const SPI_HZ: u32 = 6_400_000; // 8 bits / 1.25 µs ≈ 6.4 MHz

pub struct LedStrip {
    spi: Spi,
    buf: [[u8; 3]; LED_COUNT], // stored as [G, R, B]
    brightness: u8,
}

impl LedStrip {
    pub fn new(brightness: u8) -> Result<Self> {
        let spi = Spi::new(Bus::Spi0, SlaveSelect::Ss0, SPI_HZ, Mode::Mode0)?;
        let mut strip = Self {
            spi,
            buf: [[0; 3]; LED_COUNT],
            brightness,
        };
        strip.show()?;
        Ok(strip)
    }

    pub fn set(&mut self, index: usize, r: u8, g: u8, b: u8) {
        let scale = self.brightness as f32 / 255.0;
        self.buf[index] = [
            (g as f32 * scale).round() as u8,
            (r as f32 * scale).round() as u8,
            (b as f32 * scale).round() as u8,
        ];
    }

    pub fn set_all(&mut self, r: u8, g: u8, b: u8) {
        for i in 0..LED_COUNT {
            self.set(i, r, g, b);
        }
    }

    pub fn show(&mut self) -> Result<()> {
        let flat: Vec<u8> = self.buf.iter().flat_map(|p| p.iter().copied()).collect();
        let mut tx = vec![0u8; flat.len() * 8];
        for (byte_idx, &byte) in flat.iter().enumerate() {
            for bit in 0..8u8 {
                let spi_byte = if (byte >> (7 - bit)) & 1 == 1 {
                    0xF8
                } else {
                    0x80
                };
                tx[byte_idx * 8 + bit as usize] = spi_byte;
            }
        }
        self.spi.write(&tx)?;
        Ok(())
    }

    pub fn clear(&mut self) -> Result<()> {
        self.set_all(0, 0, 0);
        self.show()
    }
}
