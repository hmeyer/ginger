//! Place-recognition database wiring: the shared BoW index that
//! relocalization and loop detection query.
//!
//! Owns the [`Vocabulary`] + a keyframe [`Database`]. The vocabulary is
//! resolved like the `slam.toml` intrinsics prior: a shipped
//! `slam_vocab.bin` ([`Vocabulary::from_bytes`]) is used if present,
//! otherwise the pipeline **self-trains one deterministically** from the
//! pooled descriptors of the first [`N_VOCAB_KF`] keyframes
//! ([`Vocabulary::build`]) — so headless/CI needs no external asset. The
//! [`LocalMapper`](super::mapper::LocalMapper) calls [`PlaceDb::on_keyframe`]
//! as each keyframe is ingested; entries are added in keyframe-id order
//! and `entry_kf` maps a `Database` entry back to its keyframe id.

use ginger_slam_core::bow::{Database, Vocabulary};
use log::info;

use super::brief::Descriptor;

/// Operator-shipped vocabulary path (optional; absent → self-train).
const VOCAB_PATH: &str = "slam_vocab.bin";
/// Keyframes pooled before self-training the vocabulary.
const N_VOCAB_KF: usize = 6;
/// Self-train vocabulary-tree shape + seed (deterministic).
const VOCAB_K: usize = 10;
const VOCAB_DEPTH: usize = 4;
const VOCAB_SEED: u64 = 0x06B0_57A2_C0DE_1234;

/// Vocabulary + keyframe BoW database. Not `Sync`-shared directly —
/// wrapped in an `Arc<Mutex<…>>` by the frontend so both the tracking
/// thread (relocalize) and the local-mapper thread (loop detect) reach
/// the same index.
pub struct PlaceDb {
    vocab: Option<Vocabulary>,
    db: Database,
    /// `Database` entry index → keyframe id (entries added in kf order).
    entry_kf: Vec<u32>,
    /// `(kf, descriptors)` buffered until the vocabulary exists.
    pending: Vec<(u32, Vec<Descriptor>)>,
}

impl Default for PlaceDb {
    fn default() -> Self {
        Self::new()
    }
}

impl PlaceDb {
    pub fn new() -> Self {
        let vocab = std::fs::read(VOCAB_PATH)
            .ok()
            .and_then(|b| Vocabulary::from_bytes(&b))
            .filter(|v| !v.is_empty());
        if let Some(v) = &vocab {
            info!(
                "slam: loaded BoW vocabulary from {VOCAB_PATH} ({} words)",
                v.len()
            );
        }
        Self {
            vocab,
            db: Database::new(),
            entry_kf: Vec::new(),
            pending: Vec::new(),
        }
    }

    /// A vocabulary exists (loaded or self-trained) so queries work.
    pub fn is_ready(&self) -> bool {
        self.vocab.is_some()
    }

    /// Word count of the trained vocabulary; 0 until one exists.
    pub fn vocab_words(&self) -> usize {
        self.vocab.as_ref().map_or(0, |v| v.len())
    }

    /// Number of keyframes indexed in the database.
    pub fn len(&self) -> usize {
        self.db.len()
    }

    pub fn is_empty(&self) -> bool {
        self.db.is_empty()
    }

    /// Register a keyframe's full descriptor set. Adds its BoW vector
    /// immediately if a vocabulary exists; otherwise buffers it, and
    /// once [`N_VOCAB_KF`] keyframes are buffered, self-trains the
    /// vocabulary once and back-fills the database in keyframe order.
    pub fn on_keyframe(&mut self, kf: u32, descs: &[Descriptor]) {
        if self.vocab.is_some() {
            self.add(kf, descs);
            return;
        }
        self.pending.push((kf, descs.to_vec()));
        if self.pending.len() < N_VOCAB_KF {
            return;
        }
        let imgs: Vec<Vec<Descriptor>> = self.pending.iter().map(|(_, d)| d.clone()).collect();
        let v = Vocabulary::build(&imgs, VOCAB_K, VOCAB_DEPTH, VOCAB_SEED);
        if v.is_empty() {
            // Too little texture yet — keep buffering, retry next kf.
            return;
        }
        info!(
            "slam: self-trained BoW vocabulary ({} words) from {} keyframes",
            v.len(),
            imgs.len()
        );
        self.vocab = Some(v);
        for (k, d) in std::mem::take(&mut self.pending) {
            self.add(k, &d);
        }
    }

    /// Drop every indexed keyframe — the relocalization / loop-closure
    /// candidate set — for a fresh mapping session. The vocabulary is
    /// deliberately *kept*: a trained quantizer stays valid for the new
    /// session (and lets it relocalize without re-training); only the
    /// keyframe database and the pre-vocabulary buffer are cleared.
    pub fn reset(&mut self) {
        self.db = Database::new();
        self.entry_kf.clear();
        self.pending.clear();
    }

    fn add(&mut self, kf: u32, descs: &[Descriptor]) {
        let bow = self.vocab.as_ref().unwrap().transform(descs);
        self.db.add(bow);
        self.entry_kf.push(kf);
    }

    /// Place-recognition query: the best `(keyframe id, score)` matches
    /// for `descs`, best first. `skip(kf)` drops candidates (covisible /
    /// recent keyframes for loop detection; excludes self). Empty until
    /// the vocabulary is ready.
    pub fn query(
        &self,
        descs: &[Descriptor],
        max: usize,
        skip: impl Fn(u32) -> bool,
    ) -> Vec<(u32, f64)> {
        let Some(v) = self.vocab.as_ref() else {
            return Vec::new();
        };
        let q = v.transform(descs);
        if q.is_empty() {
            return Vec::new();
        }
        self.db
            .query(&q, max, |e| skip(self.entry_kf[e as usize]))
            .into_iter()
            .map(|m| (self.entry_kf[m.id as usize], m.score))
            .collect()
    }
}
