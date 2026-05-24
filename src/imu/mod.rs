//! IMU sample loop: own a thread that polls the BMI160 at the configured
//! ODR, timestamp each sample on the **same monotonic clock** the camera
//! uses (`std::time::Instant`), and publish the stream into both a
//! latest-sample slot (for the `/api/imu/sample` debug endpoint) and a
//! short ring buffer (Stage 4 will consume this from the SLAM frontend
//! for gyro pre-integration between camera frames).
//!
//! ### Why the two clocks are the same
//!
//! Frame capture times (`camera::Frame::t_capture`) and IMU sample times
//! (`ImuSample::t_read`) both come from `Instant::now()` on the host's
//! `CLOCK_MONOTONIC`. That's the load-bearing invariant for stage 4: the
//! SLAM frontend computes `frame_now.t_capture - frame_prev.t_capture`
//! to get the camera-frame interval, then sums gyro samples whose
//! `t_read` falls inside that interval. Both stamps are observation
//! moments on the **host**, so a direct `Duration` subtraction is the
//! gap between events with **no clock-domain conversion required**.
//!
//! The BMI160's chip-internal `sensortime` (39.0625 µs/tick) is *also*
//! captured per sample, but only for sanity / drift detection — it lives
//! on a completely separate clock domain and is not used for sync.
//!
//! ### What's not in this module (yet)
//!
//! * Gyro bias calibration — Stage 3 (`bin/calib-imu.rs` writes
//!   `slam.toml` `[imu]`).
//! * Consumption by the SLAM frontend's tracking-predict — Stage 4
//!   (`src/slam/frontend.rs` will `recent_since(t_prev_frame)` from this
//!   ring and pre-integrate).
//! * IMU-as-BA-constraint factors — Stage 6+ per `PLAN.md`.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use log::{info, warn};

use crate::Result;
use crate::hal::bmi160::{Bmi160, I2cBus, RawSample, RppalI2c};

/// Default I²C address for the BMI160 on this robot (SDO tied to VCC).
/// See `PLAN.md`'s hardware section: 0x68 is the alternative when SDO=GND.
pub const DEFAULT_ADDR: u16 = 0x69;

/// Target sample period. 200 Hz → 5 ms — matches the chip ODR configured
/// in [`crate::hal::bmi160`]. The polling thread sleeps for whatever
/// remains of this period after each I²C transaction; if the bus is busy
/// (PCA9685 motor updates, ADS7830 battery polls) the slip is logged
/// once a second so it's visible in the journal.
const SAMPLE_PERIOD: Duration = Duration::from_millis(5);

/// Ring capacity: 10 s at 200 Hz, which dominates any plausible
/// camera-frame interval (30 fps = 33 ms; even a stalled frontend at 1
/// fps keeps fewer than 200 samples worth of work). Overflows drop the
/// oldest sample — Stage 4 only ever wants the recent tail, and an
/// unbounded ring would mask a stuck consumer.
const RING_CAPACITY: usize = 2000;

// ── Sample type ──────────────────────────────────────────────────────────────

/// One IMU sample with both host and chip timestamps.
#[derive(Debug, Clone, Copy)]
pub struct ImuSample {
    /// Raw gyro/accel in IMU body frame (LSB counts; see
    /// [`crate::hal::bmi160`] for the scale factors at the default
    /// ±500 dps / ±4 g ranges).
    pub raw: RawSample,
    /// Host monotonic clock at the moment the I²C burst returned. **Same
    /// clock** as `camera::Frame::t_capture` — they can be subtracted
    /// directly.
    pub t_read: Instant,
    /// Chip-internal 24-bit counter at 39.0625 µs/tick, wraps every
    /// ~655 s. Compare deltas across consecutive samples to detect a
    /// stalled / dropped I²C read.
    pub sensortime: u32,
}

// ── Shared state between the polling thread and HTTP consumers ───────────────

#[derive(Default)]
struct State {
    /// Most-recently-read sample, populated after the first successful poll.
    latest: Option<ImuSample>,
    /// Bounded ring of recent samples in insertion order (oldest at the
    /// front). Stage 4 will read the tail; for Stage 2 only `latest` is
    /// surfaced, but the ring fills as soon as the thread is up so the
    /// behaviour is testable in isolation.
    ring: VecDeque<ImuSample>,
    /// Rolling EWMA of the achieved sample rate, in Hz. Updated on every
    /// poll so the 1 Hz status log can report it without re-computing.
    rate_hz: f32,
    /// Total I²C read failures since boot — logged at the 1 Hz tick if
    /// non-zero so a flaky bus is visible.
    read_failures: u64,
}

// ── Public handle ────────────────────────────────────────────────────────────

/// Owns the polling thread. Dropping is OK (the thread is detached); the
/// `Arc` keeps the state alive as long as any consumer holds a clone.
pub struct Imu {
    state: Arc<Mutex<State>>,
    _thread: JoinHandle<()>,
}

impl Imu {
    /// Open the chip at `addr`, bring it up, and start the polling
    /// thread. Returns once the chip is configured but **before** the
    /// first sample lands (a few ms later, after the chip's first ODR
    /// tick) — callers wanting to wait for a real sample should poll
    /// `latest()`.
    pub fn open(addr: u16) -> Result<Self> {
        let bus = RppalI2c::open(addr)?;
        let chip = Bmi160::open(bus)?;
        Ok(Self::spawn(chip))
    }

    /// Generic constructor used by tests with a [`MockI2cBus`]; production
    /// callers want [`Imu::open`].
    fn spawn<B: I2cBus + Send + 'static>(chip: Bmi160<B>) -> Self {
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

    /// Most-recent sample, or `None` if the thread hasn't produced one yet.
    pub fn latest(&self) -> Option<ImuSample> {
        self.state.lock().unwrap().latest
    }

    /// All ring samples whose `t_read >= since`, oldest first. Stage 4
    /// will call this with the previous camera frame's `t_capture` to
    /// pull the gyro samples for that inter-frame interval. Returned
    /// vector is owned (cheap: a `Vec<ImuSample>`, each sample is 36 B).
    pub fn recent_since(&self, since: Instant) -> Vec<ImuSample> {
        let st = self.state.lock().unwrap();
        st.ring
            .iter()
            .filter(|s| s.t_read >= since)
            .copied()
            .collect()
    }

    /// Rolling EWMA of achieved sample rate (Hz) — what the 1 Hz log
    /// reports. Useful for tests / debug.
    pub fn rate_hz(&self) -> f32 {
        self.state.lock().unwrap().rate_hz
    }
}

// ── Polling thread ───────────────────────────────────────────────────────────

fn sample_loop<B: I2cBus>(mut chip: Bmi160<B>, state: Arc<Mutex<State>>) {
    info!(
        "imu: sample loop started at {:?} target period",
        SAMPLE_PERIOD
    );
    let mut last_log = Instant::now();
    let mut samples_this_sec: u32 = 0;
    let mut last_read = Instant::now();

    loop {
        let next_deadline = last_read + SAMPLE_PERIOD;
        let now = Instant::now();
        if next_deadline > now {
            thread::sleep(next_deadline - now);
        }
        // Read first, *then* stamp — `t_read` should be as close as
        // possible to the moment the chip's data registers were latched
        // into our buffer, not the start of the I²C transaction. The
        // sensortime read is a separate transaction; it adds ~125 µs of
        // skew on a 400 kHz bus, acceptable for the drift-detection
        // role it serves.
        match chip.read_both() {
            Ok(raw) => {
                let t_read = Instant::now();
                let sensortime = chip.read_sensortime().unwrap_or(0);
                last_read = t_read;
                samples_this_sec += 1;

                let sample = ImuSample {
                    raw,
                    t_read,
                    sensortime,
                };

                let mut st = state.lock().unwrap();
                st.latest = Some(sample);
                if st.ring.len() == RING_CAPACITY {
                    st.ring.pop_front();
                }
                st.ring.push_back(sample);
            }
            Err(e) => {
                let mut st = state.lock().unwrap();
                st.read_failures += 1;
                drop(st);
                warn!("imu: read_both failed: {e}");
                // Don't tight-loop on a busted bus.
                thread::sleep(SAMPLE_PERIOD);
                last_read = Instant::now();
            }
        }

        let elapsed = last_log.elapsed();
        if elapsed >= Duration::from_secs(1) {
            let rate = samples_this_sec as f32 / elapsed.as_secs_f32();
            let (latest, failures) = {
                let mut st = state.lock().unwrap();
                // EWMA so a one-off bus stall doesn't make the readout jumpy.
                st.rate_hz = if st.rate_hz == 0.0 {
                    rate
                } else {
                    0.2 * rate + 0.8 * st.rate_hz
                };
                (st.latest, st.read_failures)
            };
            if let Some(s) = latest {
                let g = s.raw.gyro_dps();
                let a = s.raw.accel_mps2();
                info!(
                    "imu: {rate:.0} Hz | gyro {:+6.1} {:+6.1} {:+6.1} dps | \
                     accel {:+5.2} {:+5.2} {:+5.2} m/s² | failures {failures}",
                    g[0], g[1], g[2], a[0], a[1], a[2],
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
    use crate::hal::bmi160::{CHIP_ID_BMI160, REG_CHIP_ID, REG_DATA_GYR, REG_SENSORTIME};
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Counter-driven mock: every burst read returns a small unique
    /// sample so we can verify the polling thread actually pumps samples
    /// into the ring (not just the same one repeatedly). Sensortime
    /// counts up alongside the sample index.
    struct CounterBus {
        i: Arc<AtomicU32>,
    }

    impl I2cBus for CounterBus {
        fn write_byte(&mut self, _reg: u8, _val: u8) -> crate::Result<()> {
            Ok(())
        }
        fn read_byte(&mut self, reg: u8) -> crate::Result<u8> {
            if reg == REG_CHIP_ID {
                Ok(CHIP_ID_BMI160)
            } else {
                Ok(0)
            }
        }
        fn read_block(&mut self, reg: u8, buf: &mut [u8]) -> crate::Result<()> {
            if reg == REG_DATA_GYR && buf.len() == 12 {
                let n = self.i.fetch_add(1, Ordering::SeqCst) as i16;
                // Gyro x = n, y = z = 0; accel z = n.
                let gx = n.to_le_bytes();
                buf[0..2].copy_from_slice(&gx);
                for b in &mut buf[2..10] {
                    *b = 0;
                }
                let az = n.to_le_bytes();
                buf[10..12].copy_from_slice(&az);
            } else if reg == REG_SENSORTIME && buf.len() == 3 {
                let n = self.i.load(Ordering::SeqCst);
                buf[0] = (n & 0xff) as u8;
                buf[1] = ((n >> 8) & 0xff) as u8;
                buf[2] = ((n >> 16) & 0xff) as u8;
            } else {
                for b in buf.iter_mut() {
                    *b = 0;
                }
            }
            Ok(())
        }
    }

    fn build_imu() -> (Imu, Arc<AtomicU32>) {
        let i = Arc::new(AtomicU32::new(0));
        let bus = CounterBus { i: i.clone() };
        let chip = Bmi160::open(bus).expect("mock CHIP_ID is stocked");
        (Imu::spawn(chip), i)
    }

    /// Run the thread for a short while and assert it produced multiple
    /// distinct samples — i.e. the polling loop actually loops.
    #[test]
    fn polling_thread_produces_a_stream_of_samples() {
        let (imu, _counter) = build_imu();
        // 50 ms ≥ 5 ms target period × ~10 — plenty for several samples.
        thread::sleep(Duration::from_millis(120));
        let latest = imu.latest().expect("latest sample available after polling");
        assert!(
            latest.raw.gyro[0] >= 2,
            "expected several distinct samples; got gyro[0] = {}",
            latest.raw.gyro[0]
        );
    }

    /// The frame-IMU sync invariant: a sample taken just *after* we
    /// snapshot `Instant::now()` has `t_read > t_snap`. This is the
    /// property Stage 4 relies on to bucket gyro samples into the
    /// `[t_prev_frame, t_curr_frame]` interval.
    #[test]
    fn imu_timestamps_are_on_the_host_monotonic_clock() {
        let (imu, _counter) = build_imu();
        thread::sleep(Duration::from_millis(40));
        let t_query = Instant::now();
        thread::sleep(Duration::from_millis(40));
        let after = imu.recent_since(t_query);
        assert!(
            !after.is_empty(),
            "expected at least one sample after the query instant"
        );
        // All returned samples must lie strictly after the query instant.
        for s in &after {
            assert!(
                s.t_read >= t_query,
                "recent_since returned a sample older than the query instant"
            );
        }
    }

    /// `recent_since(now())` returns nothing (no samples after a fresh
    /// query instant), confirming the time filter actually filters.
    #[test]
    fn recent_since_future_instant_is_empty() {
        let (imu, _counter) = build_imu();
        thread::sleep(Duration::from_millis(40));
        let future = Instant::now() + Duration::from_secs(60);
        assert!(imu.recent_since(future).is_empty());
    }
}
