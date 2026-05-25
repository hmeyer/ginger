//! BMI160 6-DOF IMU (3-axis gyro + 3-axis accel) over I²C.
//!
//! Default I²C address: `0x68` when `SDO/AD0` is tied to GND, `0x69` when
//! tied to VCC. This robot's BMI160 board ties SDO high → `0x69` (verified
//! with `i2cdetect -y 1` and a CHIP_ID read returning `0xD1`).
//!
//! ### Frame convention
//!
//! Gyro and accelerometer triples are returned in the **IMU body frame**
//! `[x_imu, y_imu, z_imu]` exactly as the chip reports them — no rotation,
//! no scaling, no bias correction. The IMU → camera extrinsic (a 90° axis
//! swap, since the camera is mounted Z-forward and the IMU breakout is X-
//! forward) and gyro bias subtraction are the *consumer's* job in the
//! SLAM frontend. Keeping the HAL byte-faithful makes it easy to verify
//! against a `i2cget` capture and avoids burying a sign error inside two
//! abstractions.
//!
//! ### Testability
//!
//! Bus access goes through [`I2cBus`] so the driver — including the
//! power-up sequence and the burst-read parsing — can be exercised in
//! `cargo test --workspace --no-default-features` against a recording
//! mock with zero hardware. [`RppalI2c`] is the production adapter.

use std::thread::sleep;
use std::time::Duration;

use crate::{Error, Result};

// ── Register map (subset; full map in BMI160 datasheet rev 1.2 §2.11) ─────────

pub const REG_CHIP_ID: u8 = 0x00;
/// `SENSORTIME_0..2` — 24-bit chip-internal timer, 39.0625 µs/tick, wraps
/// every ~655 s. Cheapest way to detect dropped I²C reads.
pub const REG_SENSORTIME: u8 = 0x18;
/// `DATA_8` — start of the gyro X/Y/Z block. Accel follows at `DATA_14`
/// (`REG_DATA_GYR + 6`), so a 12-byte burst at this address yields the
/// whole atomic sample.
pub const REG_DATA_GYR: u8 = 0x0C;
pub const REG_ACC_CONF: u8 = 0x40;
pub const REG_ACC_RANGE: u8 = 0x41;
pub const REG_GYR_CONF: u8 = 0x42;
pub const REG_GYR_RANGE: u8 = 0x43;
pub const REG_CMD: u8 = 0x7E;

/// Expected value of `REG_CHIP_ID` for BMI160.
pub const CHIP_ID_BMI160: u8 = 0xD1;

/// `CMD` register opcode: bring accelerometer to normal power mode.
pub const CMD_ACC_NORMAL: u8 = 0x11;
/// `CMD` register opcode: bring gyroscope to normal power mode.
pub const CMD_GYR_NORMAL: u8 = 0x15;

// Datasheet table 8: max wakeup from suspend is 3.8 ms (accel) and
// 80 ms (gyro). Round up — the chip silently ignores writes during
// wakeup, so a too-short delay means default config never lands.
const ACC_WAKEUP: Duration = Duration::from_millis(4);
const GYR_WAKEUP: Duration = Duration::from_millis(81);

// Default configurations (encoded for the touched registers).
//
// `ACC_CONF` bits: `us<<7 | bwp<<4 | acc_odr`. We pick `us=0` (no
// undersampling), `bwp=2` (normal filter), `acc_odr=9` (200 Hz). At
// 200 Hz we average ~6–7 samples per camera frame for pre-integration.
const ACC_CONF_DEFAULT: u8 = (2 << 4) | 0x09;
// `GYR_CONF` bits: `bwp<<4 | gyr_odr`. Same 200 Hz / normal filter.
const GYR_CONF_DEFAULT: u8 = (2 << 4) | 0x09;
// `ACC_RANGE = 0x05` → ±4 g (datasheet table 5). Robot acceleration is
// modest; 4 g leaves headroom for impacts without losing resolution.
const ACC_RANGE_DEFAULT: u8 = 0x05;
// `GYR_RANGE = 0x02` → ±500 dps (datasheet table 4). In-place fast
// spins observed at ~200 dps; 500 dps clears that with margin.
const GYR_RANGE_DEFAULT: u8 = 0x02;

/// Scale factor: raw gyro LSB → degrees per second at the default ±500 dps range.
pub const GYRO_DPS_PER_LSB: f32 = 500.0 / 32768.0;
/// Scale factor: raw accel LSB → m/s² at the default ±4 g range.
pub const ACCEL_MPS2_PER_LSB: f32 = (4.0 * 9.806_65) / 32768.0;
/// Chip-internal time tick (datasheet §2.11.13).
pub const SENSORTIME_TICK_US: f32 = 39.062_5;

// ── Bus abstraction ──────────────────────────────────────────────────────────

/// Minimal I²C surface the driver needs. The blanket impl over `rppal`
/// is in [`RppalI2c`]; tests use a recording mock.
pub trait I2cBus {
    fn write_byte(&mut self, reg: u8, val: u8) -> Result<()>;
    fn read_byte(&mut self, reg: u8) -> Result<u8>;
    /// Plain I²C burst: write `reg`, then read `buf.len()` bytes. **Not**
    /// the SMBus block-read protocol (which prepends a length byte) — the
    /// BMI160 doesn't speak that.
    fn read_block(&mut self, reg: u8, buf: &mut [u8]) -> Result<()>;
}

/// Production [`I2cBus`] over `rppal::i2c::I2c`.
pub struct RppalI2c {
    inner: rppal::i2c::I2c,
}

impl RppalI2c {
    pub fn open(address: u16) -> Result<Self> {
        let mut inner = rppal::i2c::I2c::new()?;
        inner.set_slave_address(address)?;
        Ok(Self { inner })
    }
}

impl I2cBus for RppalI2c {
    fn write_byte(&mut self, reg: u8, val: u8) -> Result<()> {
        self.inner.smbus_write_byte(reg, val)?;
        Ok(())
    }
    fn read_byte(&mut self, reg: u8) -> Result<u8> {
        Ok(self.inner.smbus_read_byte(reg)?)
    }
    fn read_block(&mut self, reg: u8, buf: &mut [u8]) -> Result<()> {
        self.inner.write_read(&[reg], buf)?;
        Ok(())
    }
}

// ── Sample ───────────────────────────────────────────────────────────────────

/// One IMU sample: raw 16-bit signed LSB counts for gyro and accel,
/// straight from the chip in the IMU body frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawSample {
    pub gyro: [i16; 3],
    pub accel: [i16; 3],
}

impl RawSample {
    /// Convert gyro to degrees per second at the default ±500 dps range,
    /// in **chassis frame** (CCW about chassis-vertical = positive).
    ///
    /// `gyro_z` is **negated** vs the raw chip reading. Empirically
    /// established on the live robot (see PR debugging session that
    /// shipped this comment): driving the left wheels only physically
    /// rotates the chassis clockwise (per camera), while the raw
    /// `gyro_z` reads *positive* during that rotation. Either the BMI160
    /// on this breakout has the Z-axis polarity opposite of the
    /// datasheet's right-hand-rule convention, or the chip is mounted
    /// with `+Z_body` axis flipped (the accelerometer still reads
    /// `[~0, ~0, +g]`, so `+Z_body` is up in chassis frame; only the
    /// gyro is reversed). Either way, fixing it once here keeps every
    /// consumer (label worker, pose integrator) in a sane
    /// `CCW = positive` chassis frame without needing to know about the
    /// mounting quirk.
    ///
    /// `gyro_x` / `gyro_y` are passed through unmodified — they aren't
    /// used by Stages 1–4 (only yaw matters), so we haven't validated
    /// their polarity. Revisit if pitch or roll readings ever feed
    /// motion code.
    #[inline]
    pub fn gyro_dps(&self) -> [f32; 3] {
        [
            self.gyro[0] as f32 * GYRO_DPS_PER_LSB,
            self.gyro[1] as f32 * GYRO_DPS_PER_LSB,
            -(self.gyro[2] as f32) * GYRO_DPS_PER_LSB,
        ]
    }
    /// Convert accel to m/s² at the default ±4 g range.
    #[inline]
    pub fn accel_mps2(&self) -> [f32; 3] {
        [
            self.accel[0] as f32 * ACCEL_MPS2_PER_LSB,
            self.accel[1] as f32 * ACCEL_MPS2_PER_LSB,
            self.accel[2] as f32 * ACCEL_MPS2_PER_LSB,
        ]
    }
}

// ── Driver ───────────────────────────────────────────────────────────────────

pub struct Bmi160<B: I2cBus> {
    bus: B,
}

impl<B: I2cBus> Bmi160<B> {
    /// Open the device, verify `CHIP_ID == 0xD1`, bring gyro+accel to
    /// normal power, and configure default range/ODR (±500 dps / ±4 g
    /// @ 200 Hz). Blocks for ~85 ms total during the gyro wakeup.
    pub fn open(mut bus: B) -> Result<Self> {
        let id = bus.read_byte(REG_CHIP_ID)?;
        if id != CHIP_ID_BMI160 {
            return Err(Error::Imu(format!(
                "wrong CHIP_ID: got 0x{id:02x}, expected 0x{CHIP_ID_BMI160:02x}"
            )));
        }
        // Wake accel first (cheap), then gyro (long wakeup). Writes to
        // CONF/RANGE during PMU wakeup are silently dropped — sequence
        // matters here.
        bus.write_byte(REG_CMD, CMD_ACC_NORMAL)?;
        sleep(ACC_WAKEUP);
        bus.write_byte(REG_CMD, CMD_GYR_NORMAL)?;
        sleep(GYR_WAKEUP);
        bus.write_byte(REG_ACC_CONF, ACC_CONF_DEFAULT)?;
        bus.write_byte(REG_ACC_RANGE, ACC_RANGE_DEFAULT)?;
        bus.write_byte(REG_GYR_CONF, GYR_CONF_DEFAULT)?;
        bus.write_byte(REG_GYR_RANGE, GYR_RANGE_DEFAULT)?;
        Ok(Self { bus })
    }

    /// Single 12-byte burst at `DATA_8` → both axes in one transfer.
    ///
    /// This is the *only* read worth using on the hot path: it is
    /// cheaper than two 6-byte reads and atomic on the chip side (gyro
    /// and accel data registers are double-buffered together on a new
    /// sample), so there is no inter-axis jitter.
    pub fn read_both(&mut self) -> Result<RawSample> {
        let mut buf = [0u8; 12];
        self.bus.read_block(REG_DATA_GYR, &mut buf)?;
        Ok(parse_sample(&buf))
    }

    pub fn read_gyro_raw(&mut self) -> Result<[i16; 3]> {
        let mut buf = [0u8; 6];
        self.bus.read_block(REG_DATA_GYR, &mut buf)?;
        Ok(parse_axes(&buf))
    }

    pub fn read_accel_raw(&mut self) -> Result<[i16; 3]> {
        let mut buf = [0u8; 6];
        self.bus.read_block(REG_DATA_GYR + 6, &mut buf)?;
        Ok(parse_axes(&buf))
    }

    /// Chip-internal time at 39.0625 µs/tick (24-bit, wraps ~655 s).
    pub fn read_sensortime(&mut self) -> Result<u32> {
        let mut buf = [0u8; 3];
        self.bus.read_block(REG_SENSORTIME, &mut buf)?;
        Ok(u32::from(buf[0]) | (u32::from(buf[1]) << 8) | (u32::from(buf[2]) << 16))
    }
}

#[inline]
fn parse_axes(buf: &[u8]) -> [i16; 3] {
    [
        i16::from_le_bytes([buf[0], buf[1]]),
        i16::from_le_bytes([buf[2], buf[3]]),
        i16::from_le_bytes([buf[4], buf[5]]),
    ]
}

#[inline]
fn parse_sample(buf: &[u8]) -> RawSample {
    RawSample {
        gyro: parse_axes(&buf[0..6]),
        accel: parse_axes(&buf[6..12]),
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Recording mock: writes go into an ordered log; reads come from a
    /// `regs` map (single-byte) or a `blocks` map keyed by start register
    /// (multi-byte burst). Unknown reads return zero so the open() bring-
    /// up only needs the CHIP_ID stocked.
    #[derive(Default)]
    struct MockBus {
        regs: HashMap<u8, u8>,
        blocks: HashMap<u8, Vec<u8>>,
        writes: Vec<(u8, u8)>,
    }

    impl I2cBus for MockBus {
        fn write_byte(&mut self, reg: u8, val: u8) -> Result<()> {
            self.writes.push((reg, val));
            self.regs.insert(reg, val);
            Ok(())
        }
        fn read_byte(&mut self, reg: u8) -> Result<u8> {
            Ok(self.regs.get(&reg).copied().unwrap_or(0))
        }
        fn read_block(&mut self, reg: u8, buf: &mut [u8]) -> Result<()> {
            let src = self.blocks.get(&reg).cloned().unwrap_or_default();
            for (i, slot) in buf.iter_mut().enumerate() {
                *slot = src.get(i).copied().unwrap_or(0);
            }
            Ok(())
        }
    }

    fn stocked_bus() -> MockBus {
        let mut bus = MockBus::default();
        bus.regs.insert(REG_CHIP_ID, CHIP_ID_BMI160);
        bus
    }

    #[test]
    fn open_verifies_chip_id_and_runs_configured_sequence() {
        let bus = stocked_bus();
        let imu = Bmi160::open(bus).expect("open should succeed with correct CHIP_ID");
        // The exact bring-up: CMDs first (in accel-then-gyro order), then
        // the four CONF/RANGE writes. If somebody reorders writes during
        // wakeup, the chip drops them silently — guard against that.
        assert_eq!(
            imu.bus.writes,
            vec![
                (REG_CMD, CMD_ACC_NORMAL),
                (REG_CMD, CMD_GYR_NORMAL),
                (REG_ACC_CONF, ACC_CONF_DEFAULT),
                (REG_ACC_RANGE, ACC_RANGE_DEFAULT),
                (REG_GYR_CONF, GYR_CONF_DEFAULT),
                (REG_GYR_RANGE, GYR_RANGE_DEFAULT),
            ]
        );
    }

    #[test]
    fn open_rejects_wrong_chip_id() {
        let mut bus = MockBus::default();
        bus.regs.insert(REG_CHIP_ID, 0x42);
        let result = Bmi160::open(bus);
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("should reject non-BMI160 chip id"),
        };
        let msg = err.to_string();
        assert!(msg.contains("0x42"), "error mentions actual id: {msg}");
        assert!(msg.contains("0xd1"), "error mentions expected id: {msg}");
    }

    #[test]
    fn read_both_parses_little_endian_axes() {
        let mut bus = stocked_bus();
        // gyro x=+1, y=-1, z=+256; accel x=-32768, y=0, z=+32767.
        bus.blocks.insert(
            REG_DATA_GYR,
            vec![
                0x01, 0x00, 0xFF, 0xFF, 0x00, 0x01, // gyro
                0x00, 0x80, 0x00, 0x00, 0xFF, 0x7F, // accel
            ],
        );
        let mut imu = Bmi160::open(bus).unwrap();
        let s = imu.read_both().unwrap();
        assert_eq!(s.gyro, [1, -1, 256]);
        assert_eq!(s.accel, [i16::MIN, 0, i16::MAX]);
    }

    #[test]
    fn raw_to_si_uses_default_ranges() {
        let s = RawSample {
            gyro: [32768i32.try_into().unwrap_or(i16::MAX), 0, 0],
            accel: [0, 0, 32768i32.try_into().unwrap_or(i16::MAX)],
        };
        let g = s.gyro_dps();
        let a = s.accel_mps2();
        // Full-scale on the ±500 dps range → 499.98... dps (32767/32768·500).
        assert!(
            (g[0] - 500.0).abs() < 0.05,
            "gyro near full scale: {}",
            g[0]
        );
        // Full-scale on the ±4 g range → 4·9.80665 m/s² ≈ 39.226.
        let expected = 4.0 * 9.806_65;
        assert!(
            (a[2] - expected).abs() < 0.01,
            "accel near full scale: {} vs {}",
            a[2],
            expected
        );
    }

    #[test]
    fn sensortime_assembles_three_byte_little_endian_counter() {
        let mut bus = stocked_bus();
        bus.blocks.insert(REG_SENSORTIME, vec![0x78, 0x56, 0x34]);
        let mut imu = Bmi160::open(bus).unwrap();
        // 0x345678 = 3_430_008 ticks ≈ 134 ms — a realistic value.
        assert_eq!(imu.read_sensortime().unwrap(), 0x0034_5678);
    }
}
