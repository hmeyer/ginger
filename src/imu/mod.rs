//! IMU sample loop: own a thread that polls the BNO055 at the chip's
//! 100 Hz fusion rate, timestamp each sample on the **same monotonic
//! clock** the camera uses (`std::time::Instant`), and publish the
//! stream into both a latest-sample slot and a short ring buffer.
//!
//! ### Why the two clocks are the same
//!
//! Frame capture times (`camera::Frame::t_capture`) and IMU sample times
//! (`ImuSample::t_read`) both come from `Instant::now()` on the host's
//! `CLOCK_MONOTONIC`. The SLAM frontend computes
//! `frame_now.t_capture - frame_prev.t_capture` to get the camera-frame
//! interval, then pulls the orientation samples whose `t_read` falls
//! inside that interval and computes a single ΔR = `q_curr * q_prev⁻¹`
//! — no clock-domain conversion required.
//!
//! ### Fusion warm-up gate
//!
//! After power-up the BNO055 spends a few seconds doing stillness-based
//! gyro auto-zeroing. While `CALIB_STAT.gyr == 0` the orientation it
//! reports is meaningless. Both [`Imu::latest`] and [`Imu::recent_since`]
//! return empty results during that window, so every consumer's
//! existing "IMU absent" path handles the warm-up automatically. The
//! WebUI surfaces the calibration status via [`Imu::calib_status`].

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use log::{info, warn};
use nalgebra::UnitQuaternion;

use crate::Result;
use crate::hal::bno055::{Bno055, CalibStatus, I2cBus, RppalI2c};

/// Default I²C address for the BNO055 on this robot (`COM3` tied to GND).
/// 0x29 is the alternate when COM3 is pulled high.
pub const DEFAULT_ADDR: u16 = 0x28;

/// Target sample period. 100 Hz → 10 ms — matches the BNO055's fixed
/// fusion-output rate. Polling faster only re-reads stale registers;
/// polling slower drops fusion frames.
const SAMPLE_PERIOD: Duration = Duration::from_millis(10);

/// How often to re-read the 1-byte `CALIB_STAT`. The chip updates this
/// at fusion rate, but consumers only care about it for the warm-up
/// gate and the WebUI badge, so a 1 Hz re-read is plenty.
const CALIB_PERIOD: Duration = Duration::from_secs(1);

/// Ring capacity: 10 s at 100 Hz, which dominates any plausible
/// camera-frame interval. Overflows drop the oldest sample.
const RING_CAPACITY: usize = 1000;

/// Minimum `CALIB_STAT.gyr` value (0..=3) at which the chip's fusion
/// output is trustworthy. `1` is the first non-zero rung — the chip has
/// completed at least one stillness-based gyro-zero pass. `2` would be
/// tighter but takes much longer to reach on a chassis that's never
/// fully still (motor vibration, fans, etc.).
const MIN_GYR_CALIB: u8 = 1;

// ── Sample type ──────────────────────────────────────────────────────────────

/// One IMU sample with the chip's fused orientation + linear-accel.
#[derive(Debug, Clone, Copy)]
pub struct ImuSample {
    /// Chip-frame orientation. Identity at the chip's boot pose; tracks
    /// rotations the chassis has made since then via the BNO's IMUPLUS
    /// fusion. Yaw component is the primary signal for pose / SLAM.
    pub orientation: UnitQuaternion<f32>,
    /// Linear acceleration (gravity removed by the chip) in m/s²,
    /// chip body frame. Drives spike rejection in the label worker.
    pub linear_accel: [f32; 3],
    /// Host monotonic clock at the moment the I²C burst returned.
    /// **Same clock** as `camera::Frame::t_capture` — they can be
    /// subtracted directly.
    pub t_read: Instant,
    /// Monotonically incrementing per-sample counter; a stalled stream
    /// shows as this not advancing across requests.
    pub sample_index: u32,
}

// ── Shared state between the polling thread and HTTP consumers ───────────────

#[derive(Default)]
struct State {
    /// Most-recently-read sample, populated after the first successful
    /// poll that landed *after* the chip's fusion warm-up.
    latest: Option<ImuSample>,
    /// Bounded ring of recent samples in insertion order (oldest at the
    /// front). Only contains samples taken once `calib.gyr >= MIN_GYR_CALIB`.
    ring: VecDeque<ImuSample>,
    /// Rolling EWMA of the achieved sample rate, in Hz.
    rate_hz: f32,
    /// Total I²C read failures since boot.
    read_failures: u64,
    /// Most-recent calibration status. Re-read once per second.
    calib: CalibStatus,
    /// Monotonic counter shared with [`ImuSample::sample_index`].
    next_sample_index: u32,
}

// ── Public handle ────────────────────────────────────────────────────────────

/// Owns the polling thread. Dropping is OK (the thread is detached); the
/// `Arc` keeps the state alive as long as any consumer holds a clone.
pub struct Imu {
    state: Arc<Mutex<State>>,
    _thread: JoinHandle<()>,
}

impl Imu {
    /// Open the chip at `addr`, bring it up in IMUPLUS mode, and start
    /// the polling thread. Returns once the chip is configured but
    /// **before** the first sample lands — callers that need a real
    /// sample should poll [`Self::latest`].
    pub fn open(addr: u16) -> Result<Self> {
        let bus = RppalI2c::open(addr)?;
        let chip = Bno055::open(bus)?;
        Ok(Self::spawn(chip))
    }

    /// Generic constructor used by tests with a mock bus; production
    /// callers want [`Imu::open`].
    fn spawn<B: I2cBus + Send + 'static>(chip: Bno055<B>) -> Self {
        let state = Arc::new(Mutex::new(State::default()));
        let state_thread = state.clone();
        let thread = thread::Builder::new()
            .name("imu".into())
            .spawn(move || sample_loop(chip, state_thread))
            .expect("spawn imu thread");
        Self {
            state,
            _thread: thread,
        }
    }

    /// Most-recent sample, or `None` if the chip hasn't completed its
    /// fusion warm-up yet (`calib.gyr < 1`). Consumers' existing
    /// "IMU absent" paths handle the warm-up correctly without any
    /// per-consumer gating.
    pub fn latest(&self) -> Option<ImuSample> {
        self.state.lock().unwrap().latest
    }

    /// All ring samples whose `t_read >= since`, oldest first. Empty
    /// before the chip's fusion warm-up completes.
    pub fn recent_since(&self, since: Instant) -> Vec<ImuSample> {
        let st = self.state.lock().unwrap();
        st.ring
            .iter()
            .filter(|s| s.t_read >= since)
            .copied()
            .collect()
    }

    /// Most-recent sample whose `t_read <= at` — the chip's best
    /// orientation estimate *at-or-before* the queried instant. `None`
    /// before the chip's fusion warm-up, or if the ring contains no
    /// sample older than `at` (consumer queried with a `t` older than
    /// anything we still have).
    ///
    /// The SLAM rotation hint and the label worker's ω endpoints use
    /// this to pull the two orientation snapshots bracketing a camera
    /// frame or a 200 ms label window — `Δyaw / Δt` from those is the
    /// fusion-engine's drift-compensated answer.
    pub fn latest_before(&self, at: Instant) -> Option<ImuSample> {
        let st = self.state.lock().unwrap();
        st.ring.iter().rev().find(|s| s.t_read <= at).copied()
    }

    /// Rolling EWMA of achieved sample rate (Hz). Should sit near 100 on
    /// the live bus; sagging means I²C contention.
    pub fn rate_hz(&self) -> f32 {
        self.state.lock().unwrap().rate_hz
    }

    /// Latest per-subsystem calibration status. `mag` always zero in
    /// IMUPLUS (magnetometer disabled). Surfaced to the WebUI as a
    /// "fusion ready" badge.
    pub fn calib_status(&self) -> CalibStatus {
        self.state.lock().unwrap().calib
    }
}

// ── Polling thread ───────────────────────────────────────────────────────────

fn sample_loop<B: I2cBus>(mut chip: Bno055<B>, state: Arc<Mutex<State>>) {
    info!(
        "imu: sample loop started at {:?} target period (BNO055 IMUPLUS, 100 Hz fusion)",
        SAMPLE_PERIOD
    );
    let mut last_log = Instant::now();
    let mut last_calib_read = Instant::now() - CALIB_PERIOD; // force first read
    let mut samples_this_sec: u32 = 0;
    let mut last_read = Instant::now();

    loop {
        let next_deadline = last_read + SAMPLE_PERIOD;
        let now = Instant::now();
        if next_deadline > now {
            thread::sleep(next_deadline - now);
        }

        // Re-read calibration status at the slow cadence. Doing it here
        // (in the same thread as the fusion read) avoids any I²C bus
        // races with the fusion polling.
        if last_calib_read.elapsed() >= CALIB_PERIOD {
            match chip.read_calib() {
                Ok(c) => {
                    state.lock().unwrap().calib = c;
                }
                Err(e) => warn!("imu: read_calib failed: {e}"),
            }
            last_calib_read = Instant::now();
        }

        match chip.read_fusion() {
            Ok(Some(fusion)) => {
                let t_read = Instant::now();
                last_read = t_read;
                samples_this_sec += 1;

                let mut st = state.lock().unwrap();
                let idx = st.next_sample_index;
                st.next_sample_index = idx.wrapping_add(1);

                // Gate publication on the chip's fusion warm-up. While
                // `calib.gyr == 0` the quaternion is meaningless; we
                // still increment the counter (so `sample_index` keeps
                // advancing for diagnostics) but don't expose the
                // sample to consumers.
                if st.calib.gyr < MIN_GYR_CALIB {
                    continue;
                }

                let sample = ImuSample {
                    orientation: fusion.orientation,
                    linear_accel: fusion.linear_accel,
                    t_read,
                    sample_index: idx,
                };
                st.latest = Some(sample);
                if st.ring.len() == RING_CAPACITY {
                    st.ring.pop_front();
                }
                st.ring.push_back(sample);
            }
            Ok(None) => {
                // Chip hasn't produced its first fusion frame yet
                // (post-bring-up, ~7 ms window). Skip silently — the
                // next poll will land it.
                last_read = Instant::now();
            }
            Err(e) => {
                let mut st = state.lock().unwrap();
                st.read_failures += 1;
                drop(st);
                warn!("imu: read_fusion failed: {e}");
                thread::sleep(SAMPLE_PERIOD);
                last_read = Instant::now();
            }
        }

        let elapsed = last_log.elapsed();
        if elapsed >= Duration::from_secs(1) {
            let rate = samples_this_sec as f32 / elapsed.as_secs_f32();
            let (latest, failures, calib) = {
                let mut st = state.lock().unwrap();
                st.rate_hz = if st.rate_hz == 0.0 {
                    rate
                } else {
                    0.2 * rate + 0.8 * st.rate_hz
                };
                (st.latest, st.read_failures, st.calib)
            };
            if let Some(s) = latest {
                let (roll, pitch, yaw) = s.orientation.euler_angles();
                let a = s.linear_accel;
                info!(
                    "imu: {rate:.0} Hz | yaw {:+6.1}° pitch {:+5.1}° roll {:+5.1}° | \
                     lin_accel {:+5.2} {:+5.2} {:+5.2} m/s² | \
                     cal sys/gyr/acc {}/{}/{} | failures {failures}",
                    yaw.to_degrees(),
                    pitch.to_degrees(),
                    roll.to_degrees(),
                    a[0],
                    a[1],
                    a[2],
                    calib.sys,
                    calib.gyr,
                    calib.acc,
                );
            } else {
                info!(
                    "imu: {rate:.0} Hz | warming up (cal sys/gyr/acc {}/{}/{}) | failures {failures}",
                    calib.sys, calib.gyr, calib.acc,
                );
            }
            last_log = Instant::now();
            samples_this_sec = 0;
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hal::bno055::{CHIP_ID_BNO055, REG_CALIB_STAT, REG_CHIP_ID, REG_QUA_DATA};
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Counter-driven mock: every burst returns the identity quaternion
    /// plus a small linear-accel that ticks up with the counter, so the
    /// polling thread is observably pumping data. Stocks `CALIB_STAT`
    /// with `gyr = 3` so `latest()` is not gated.
    struct CounterBus {
        i: Arc<AtomicU32>,
        calib_byte: u8,
    }

    impl I2cBus for CounterBus {
        fn write_byte(&mut self, _reg: u8, _val: u8) -> crate::Result<()> {
            Ok(())
        }
        fn read_byte(&mut self, reg: u8) -> crate::Result<u8> {
            match reg {
                REG_CHIP_ID => Ok(CHIP_ID_BNO055),
                REG_CALIB_STAT => Ok(self.calib_byte),
                _ => Ok(0),
            }
        }
        fn read_block(&mut self, reg: u8, buf: &mut [u8]) -> crate::Result<()> {
            for b in buf.iter_mut() {
                *b = 0;
            }
            if reg == REG_QUA_DATA && buf.len() == 14 {
                // Identity quaternion: w = 16384.
                buf[0..2].copy_from_slice(&16384i16.to_le_bytes());
                // linear-accel x ticks up with the counter so consecutive
                // samples are distinguishable.
                let n = self.i.fetch_add(1, Ordering::SeqCst) as i16;
                buf[8..10].copy_from_slice(&n.to_le_bytes());
            }
            Ok(())
        }
    }

    fn build_imu_calibrated() -> (Imu, Arc<AtomicU32>) {
        let i = Arc::new(AtomicU32::new(0));
        // calib_byte = 0xC0 → sys=3, gyr=0, acc=0, mag=0 — gyr is the gate.
        // We want gyr=3 → byte 0x30 (bits [5:4]=11).
        let bus = CounterBus {
            i: i.clone(),
            calib_byte: 0x30,
        };
        let chip = Bno055::open(bus).expect("mock CHIP_ID is stocked");
        (Imu::spawn(chip), i)
    }

    fn build_imu_warming_up() -> (Imu, Arc<AtomicU32>) {
        let i = Arc::new(AtomicU32::new(0));
        let bus = CounterBus {
            i: i.clone(),
            calib_byte: 0x00, // gyr=0 → warm-up gate closed
        };
        let chip = Bno055::open(bus).expect("mock CHIP_ID is stocked");
        (Imu::spawn(chip), i)
    }

    /// Drives the loop for ~120 ms — enough for the 1 Hz calib re-read
    /// to *not* fire (so the calib status the thread learned at startup
    /// stays in effect) and for ~10 fusion polls to happen.
    fn brief_warmup() {
        thread::sleep(Duration::from_millis(120));
    }

    #[test]
    fn polling_thread_produces_a_stream_of_samples() {
        let (imu, counter) = build_imu_calibrated();
        brief_warmup();
        let latest = imu.latest().expect("latest sample after polling");
        // `sample_index` should be advancing. We can't assert an exact
        // number (thread scheduling jitter), but ≥ 2 means we pumped
        // more than one sample.
        assert!(
            latest.sample_index >= 2,
            "sample_index = {}",
            latest.sample_index
        );
        assert!(counter.load(Ordering::SeqCst) >= 2);
    }

    #[test]
    fn latest_is_none_until_calibrated() {
        let (imu, _) = build_imu_warming_up();
        brief_warmup();
        assert!(
            imu.latest().is_none(),
            "latest() should be None while gyr=0"
        );
        assert!(
            imu.recent_since(Instant::now() - Duration::from_secs(1))
                .is_empty()
        );
        // calib_status() still surfaces the raw byte for the WebUI badge.
        let c = imu.calib_status();
        assert_eq!(c.gyr, 0);
    }

    #[test]
    fn imu_timestamps_are_on_the_host_monotonic_clock() {
        let (imu, _) = build_imu_calibrated();
        thread::sleep(Duration::from_millis(40));
        let t_query = Instant::now();
        thread::sleep(Duration::from_millis(40));
        let after = imu.recent_since(t_query);
        assert!(
            !after.is_empty(),
            "expected samples after the query instant"
        );
        for s in &after {
            assert!(s.t_read >= t_query);
        }
    }

    #[test]
    fn recent_since_future_instant_is_empty() {
        let (imu, _) = build_imu_calibrated();
        brief_warmup();
        let future = Instant::now() + Duration::from_secs(60);
        assert!(imu.recent_since(future).is_empty());
    }

    #[test]
    fn calib_status_surfaces_to_consumers() {
        let (imu, _) = build_imu_calibrated();
        brief_warmup();
        let c = imu.calib_status();
        assert_eq!(c.gyr, 3, "calibrated mock should report gyr=3");
        assert_eq!(c.mag, 0, "IMUPLUS leaves mag at 0");
    }
}
