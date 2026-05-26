//! BNO055 9-DOF absolute orientation sensor over I²C.
//!
//! Default I²C address: `0x28` when `COM3` is tied to GND (this robot's
//! wiring — verified with `i2cdetect -y 1`). The alternate address
//! `0x29` is selected by pulling COM3 high.
//!
//! ### Operating mode: IMUPLUS
//!
//! We run the chip in **IMUPLUS** (mode `0x08`): 6-DoF fusion using the
//! accelerometer + gyroscope only. The magnetometer is intentionally
//! left off — the robot drives high PWM currents directly under the IMU
//! board, which will trash any magnetometer-anchored fusion (NDOF /
//! COMPASS). In IMUPLUS the chip outputs an orientation **relative to
//! its boot pose**, which is exactly the indoor-robot use case.
//!
//! Fusion outputs (quaternion + linear acceleration) refresh at the
//! chip's fixed 100 Hz fusion rate; we poll on the same cadence.
//!
//! ### Frame convention
//!
//! The driver returns the chip's outputs *as-the-chip-sees-them*: a
//! unit quaternion in **chip body frame**, and a gravity-removed linear
//! acceleration vector in m/s² in the same chip frame. The chip→chassis
//! rotation (if any) is the consumer's job; see [`crate::imu`].
//!
//! ### Testability
//!
//! Bus access goes through [`I2cBus`] so the driver — including the
//! mode-switching bring-up and the 14-byte burst — can be exercised in
//! `cargo test --workspace --no-default-features` against a recording
//! mock with zero hardware.

use std::thread::sleep;
use std::time::Duration;

use nalgebra::{Quaternion, UnitQuaternion};

use crate::{Error, Result};

// ── Register map (subset; full map in BNO055 datasheet rev 1.7 §4.3) ─────────

pub const REG_CHIP_ID: u8 = 0x00;
/// `QUA_DATA_W_LSB` — start of the quaternion block. Reading 14 bytes
/// from here captures the quaternion (8 B) + linear-acceleration (6 B)
/// in one atomic burst.
pub const REG_QUA_DATA: u8 = 0x20;
/// `CALIB_STAT` — 1 byte. Bits `[7:6]=sys, [5:4]=gyr, [3:2]=acc,
/// [1:0]=mag`, each in `0..=3` (3 = fully calibrated).
pub const REG_CALIB_STAT: u8 = 0x35;
pub const REG_UNIT_SEL: u8 = 0x3B;
pub const REG_OPR_MODE: u8 = 0x3D;
pub const REG_PWR_MODE: u8 = 0x3E;
pub const REG_SYS_TRIGGER: u8 = 0x3F;

/// Expected value of `REG_CHIP_ID` for BNO055.
pub const CHIP_ID_BNO055: u8 = 0xA0;

/// `OPR_MODE` opcode: configuration. Required before changing settings.
pub const OPR_MODE_CONFIG: u8 = 0x00;
/// `OPR_MODE` opcode: 6-DoF accel + gyro fusion, mag off.
pub const OPR_MODE_IMUPLUS: u8 = 0x08;
/// `PWR_MODE` opcode: normal (all sensors at configured ODR).
pub const PWR_MODE_NORMAL: u8 = 0x00;

// Datasheet table 3-6: any mode change is rejected silently while the
// chip is still in the previous mode's wakeup. Config → operating mode
// needs ≥ 7 ms; operating → config needs ≥ 19 ms. Round up generously —
// these are one-shot boot costs, not the hot path.
const MODE_SWITCH_WAIT: Duration = Duration::from_millis(30);

// ── Scale factors ────────────────────────────────────────────────────────────

/// Quaternion component → unit-quaternion value. Datasheet §3.6.5.5:
/// each component is signed 16-bit with `2^14 = 16384` LSB/unit, so a
/// full unit quaternion has components in `[-16384, +16384]`.
const QUAT_SCALE: f32 = 1.0 / 16384.0;
/// Linear-acceleration → m/s². Datasheet §3.6.5.4 with default
/// `UNIT_SEL.ACC_UNIT = 0`: 100 LSB / (m/s²) → 0.01 m/s² per LSB.
const LIN_ACCEL_MPS2_PER_LSB: f32 = 1.0 / 100.0;

// ── Calibration status ───────────────────────────────────────────────────────

/// Per-subsystem calibration status, each in `0..=3`. Decoded from
/// `CALIB_STAT (0x35)`. In IMUPLUS the magnetometer is disabled, so
/// `mag` stays at zero; only `sys / gyr / acc` are meaningful.
///
/// The fusion engine only produces trustworthy orientation once
/// `gyr >= 1` — until then the chip is still doing its stillness-based
/// gyro auto-zero. [`crate::imu::Imu::latest`] gates on this.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct CalibStatus {
    pub sys: u8,
    pub gyr: u8,
    pub acc: u8,
    pub mag: u8,
}

impl CalibStatus {
    pub fn from_byte(b: u8) -> Self {
        Self {
            sys: (b >> 6) & 0x03,
            gyr: (b >> 4) & 0x03,
            acc: (b >> 2) & 0x03,
            mag: b & 0x03,
        }
    }
}

// ── Bus abstraction ──────────────────────────────────────────────────────────

/// Minimal I²C surface the driver needs. The blanket impl over `rppal`
/// is in [`RppalI2c`]; tests use a recording mock.
pub trait I2cBus {
    fn write_byte(&mut self, reg: u8, val: u8) -> Result<()>;
    fn read_byte(&mut self, reg: u8) -> Result<u8>;
    /// Plain I²C burst: write `reg`, then read `buf.len()` bytes. **Not**
    /// the SMBus block-read protocol (which prepends a length byte) — the
    /// BNO055 doesn't speak that.
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

/// One BNO055 fusion sample: chip-frame orientation + linear-accel.
#[derive(Debug, Clone, Copy)]
pub struct FusionSample {
    /// Chip-frame orientation. Identity at the moment the chip booted.
    pub orientation: UnitQuaternion<f32>,
    /// Linear acceleration (gravity removed by the chip) in m/s²,
    /// chip body frame.
    pub linear_accel: [f32; 3],
}

#[inline]
fn s16(buf: &[u8]) -> i16 {
    i16::from_le_bytes([buf[0], buf[1]])
}

/// Parse a 14-byte burst at `REG_QUA_DATA`. Returns `None` if the
/// quaternion is all-zero (chip hasn't produced a fusion frame yet —
/// happens for a few ms after `IMUPLUS` is engaged).
pub fn parse_burst(buf: &[u8; 14]) -> Option<FusionSample> {
    let w = s16(&buf[0..2]);
    let x = s16(&buf[2..4]);
    let y = s16(&buf[4..6]);
    let z = s16(&buf[6..8]);
    if w == 0 && x == 0 && y == 0 && z == 0 {
        return None;
    }
    let q = Quaternion::new(
        w as f32 * QUAT_SCALE,
        x as f32 * QUAT_SCALE,
        y as f32 * QUAT_SCALE,
        z as f32 * QUAT_SCALE,
    );
    // `from_quaternion` normalizes — defensive against the chip's
    // 16-bit truncation that puts |q| slightly off unit.
    let orientation = UnitQuaternion::from_quaternion(q);
    let ax = s16(&buf[8..10]);
    let ay = s16(&buf[10..12]);
    let az = s16(&buf[12..14]);
    Some(FusionSample {
        orientation,
        linear_accel: [
            ax as f32 * LIN_ACCEL_MPS2_PER_LSB,
            ay as f32 * LIN_ACCEL_MPS2_PER_LSB,
            az as f32 * LIN_ACCEL_MPS2_PER_LSB,
        ],
    })
}

// ── Driver ───────────────────────────────────────────────────────────────────

pub struct Bno055<B: I2cBus> {
    bus: B,
}

impl<B: I2cBus> Bno055<B> {
    /// Open the device, verify `CHIP_ID == 0xA0`, and bring the chip up
    /// in IMUPLUS fusion mode. Blocks for ~60 ms total during the two
    /// mode transitions.
    pub fn open(mut bus: B) -> Result<Self> {
        let id = bus.read_byte(REG_CHIP_ID)?;
        if id != CHIP_ID_BNO055 {
            return Err(Error::Imu(format!(
                "wrong CHIP_ID: got 0x{id:02x}, expected 0x{CHIP_ID_BNO055:02x}"
            )));
        }
        // Force CONFIG before changing any settings — register writes
        // are silently rejected in operating modes. The chip starts in
        // CONFIG on power-up, but a soft-reboot of the host without a
        // chip power-cycle leaves us in whatever mode we last selected.
        bus.write_byte(REG_OPR_MODE, OPR_MODE_CONFIG)?;
        sleep(MODE_SWITCH_WAIT);

        // Normal power; clear any sticky reset/calibration triggers;
        // units = m/s² + dps + degrees (matches our SI conversions).
        bus.write_byte(REG_PWR_MODE, PWR_MODE_NORMAL)?;
        bus.write_byte(REG_SYS_TRIGGER, 0x00)?;
        bus.write_byte(REG_UNIT_SEL, 0x00)?;

        // Engage 6-DoF fusion. Wait again — fusion output is garbage
        // for ~7 ms after the transition.
        bus.write_byte(REG_OPR_MODE, OPR_MODE_IMUPLUS)?;
        sleep(MODE_SWITCH_WAIT);

        Ok(Self { bus })
    }

    /// Single 14-byte burst at `REG_QUA_DATA` → quaternion + linear-accel
    /// in one atomic transfer. Returns `None` while the chip hasn't
    /// produced its first fusion frame (all-zero quaternion).
    pub fn read_fusion(&mut self) -> Result<Option<FusionSample>> {
        let mut buf = [0u8; 14];
        self.bus.read_block(REG_QUA_DATA, &mut buf)?;
        Ok(parse_burst(&buf))
    }

    /// Read the 1-byte `CALIB_STAT` register and decode it.
    pub fn read_calib(&mut self) -> Result<CalibStatus> {
        Ok(CalibStatus::from_byte(self.bus.read_byte(REG_CALIB_STAT)?))
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
        bus.regs.insert(REG_CHIP_ID, CHIP_ID_BNO055);
        bus
    }

    #[test]
    fn open_verifies_chip_id_and_runs_configured_sequence() {
        let bus = stocked_bus();
        let imu = Bno055::open(bus).expect("open should succeed with correct CHIP_ID");
        // The exact bring-up: CONFIG first, then power/trigger/units,
        // then IMUPLUS. If somebody reorders, the chip rejects the
        // settings writes — guard against that.
        assert_eq!(
            imu.bus.writes,
            vec![
                (REG_OPR_MODE, OPR_MODE_CONFIG),
                (REG_PWR_MODE, PWR_MODE_NORMAL),
                (REG_SYS_TRIGGER, 0x00),
                (REG_UNIT_SEL, 0x00),
                (REG_OPR_MODE, OPR_MODE_IMUPLUS),
            ]
        );
    }

    #[test]
    fn open_rejects_wrong_chip_id() {
        let mut bus = MockBus::default();
        bus.regs.insert(REG_CHIP_ID, 0x42);
        let result = Bno055::open(bus);
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("should reject non-BNO055 chip id"),
        };
        let msg = err.to_string();
        assert!(msg.contains("0x42"), "error mentions actual id: {msg}");
        assert!(msg.contains("0xa0"), "error mentions expected id: {msg}");
    }

    #[test]
    fn read_fusion_parses_identity_quaternion_and_linear_accel() {
        let mut bus = stocked_bus();
        // w = 16384 → 1.0; x = y = z = 0 → identity rotation.
        // linear-accel x = 200 → 2.0 m/s², y = -50 → -0.5, z = 0.
        let mut burst = vec![0u8; 14];
        burst[0..2].copy_from_slice(&16384i16.to_le_bytes());
        burst[8..10].copy_from_slice(&200i16.to_le_bytes());
        burst[10..12].copy_from_slice(&(-50i16).to_le_bytes());
        bus.blocks.insert(REG_QUA_DATA, burst);
        let mut imu = Bno055::open(bus).unwrap();
        let s = imu.read_fusion().unwrap().expect("non-zero quaternion");
        // Identity quaternion: w near 1, vector part near zero.
        let q = s.orientation.quaternion();
        assert!((q.w - 1.0).abs() < 1e-3, "w = {}", q.w);
        assert!(q.i.abs() < 1e-3 && q.j.abs() < 1e-3 && q.k.abs() < 1e-3);
        // Linear-accel scale: 100 LSB per m/s².
        assert!((s.linear_accel[0] - 2.0).abs() < 1e-3);
        assert!((s.linear_accel[1] + 0.5).abs() < 1e-3);
        assert!(s.linear_accel[2].abs() < 1e-3);
    }

    #[test]
    fn read_fusion_returns_none_when_chip_has_not_produced_a_frame() {
        let mut bus = stocked_bus();
        // All-zero burst — the chip's pre-fusion state. Without this
        // guard, downstream consumers would integrate against a
        // bogus identity-but-uninitialized quaternion.
        bus.blocks.insert(REG_QUA_DATA, vec![0u8; 14]);
        let mut imu = Bno055::open(bus).unwrap();
        assert!(imu.read_fusion().unwrap().is_none());
    }

    #[test]
    fn calib_status_decodes_packed_byte() {
        // sys=3, gyr=2, acc=1, mag=0  →  0b11_10_01_00 = 0xE4.
        let c = CalibStatus::from_byte(0xE4);
        assert_eq!(
            c,
            CalibStatus {
                sys: 3,
                gyr: 2,
                acc: 1,
                mag: 0
            }
        );
        // All zero.
        let c = CalibStatus::from_byte(0x00);
        assert_eq!(c, CalibStatus::default());
        // All max.
        let c = CalibStatus::from_byte(0xFF);
        assert_eq!(
            c,
            CalibStatus {
                sys: 3,
                gyr: 3,
                acc: 3,
                mag: 3
            }
        );
    }

    #[test]
    fn read_calib_round_trips() {
        let mut bus = stocked_bus();
        bus.regs.insert(REG_CALIB_STAT, 0xE4);
        let mut imu = Bno055::open(bus).unwrap();
        assert_eq!(
            imu.read_calib().unwrap(),
            CalibStatus {
                sys: 3,
                gyr: 2,
                acc: 1,
                mag: 0
            }
        );
    }

    /// A 90° rotation about chip-Z is q = (cos45°, 0, 0, sin45°). The
    /// chip emits LSBs as `value · 16384`. We verify the parser round-
    /// trips that into a quaternion whose yaw component matches.
    #[test]
    fn read_fusion_parses_90deg_yaw_rotation() {
        let mut bus = stocked_bus();
        let cos45 = (std::f32::consts::FRAC_PI_4).cos();
        let sin45 = (std::f32::consts::FRAC_PI_4).sin();
        let w = (cos45 * 16384.0) as i16;
        let z = (sin45 * 16384.0) as i16;
        let mut burst = vec![0u8; 14];
        burst[0..2].copy_from_slice(&w.to_le_bytes());
        burst[6..8].copy_from_slice(&z.to_le_bytes());
        bus.blocks.insert(REG_QUA_DATA, burst);
        let mut imu = Bno055::open(bus).unwrap();
        let s = imu.read_fusion().unwrap().unwrap();
        let (_, _, yaw) = s.orientation.euler_angles();
        assert!(
            (yaw - std::f32::consts::FRAC_PI_2).abs() < 1e-2,
            "yaw = {yaw} (expected π/2)"
        );
    }
}
