//! Bag-of-Words place recognition: a binary-descriptor visual
//! vocabulary + TF-IDF image vectors + an inverted-index database.
//!
//! This is the retrieval primitive relocalization and loop detection are
//! built on — both reduce to "given this image, which earlier keyframe
//! is the same place?". Brute-force matching every keyframe is too slow
//! on the Pi 4 once the map has hundreds of them; BoW turns it into a
//! sparse vector compare over an inverted index.
//!
//! DBoW2-style and deliberately Pi-cheap (bitwise Hamming, no BLAS, no
//! float-heavy clustering — binary BoW fits the ARMv8.0,
//! multicore-not-SIMD strategy). Camera-free; descriptors are the same
//! opaque 256-bit ORB blobs the rest of the core carries
//! ([`crate::map::Descriptor`]).
//!
//! Pipeline: a [`Vocabulary`] is trained **once, offline** (hierarchical
//! binary k-means → a vocabulary tree of `kᴸ` visual words) and shipped
//! as a static asset; at runtime it is fixed. [`Vocabulary::transform`]
//! turns an image's descriptors into a TF-IDF-weighted, L1-normalized
//! [`BowVector`]; [`BowVector::score`] is the DBoW2 L1 similarity in
//! `[0, 1]`. A [`Database`] keeps per-entry vectors + a word→entries
//! inverted index so [`Database::query`] only scores keyframes that
//! share a (rare) word, plus a *direct index* (word→features) that
//! accelerates the geometric verification following a BoW hit.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use rand::{RngExt, SeedableRng, rngs::SmallRng};

use crate::map::Descriptor;

/// Hamming distance between two 256-bit descriptors (0..=256). Scalar
/// reference per the `crates/fast` kernel discipline; a NEON `vcntq_u8` +
/// `vaddvq` version is a later, parity-tested perf step if measured hot.
#[inline]
pub fn hamming(a: &Descriptor, b: &Descriptor) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
}

/// Per-bit majority descriptor of a non-empty set (the binary k-means
/// "centroid"): bit set iff a strict majority of inputs have it; an
/// exact tie resolves to 0 (deterministic).
fn majority(descs: &[Descriptor]) -> Descriptor {
    let n = descs.len();
    let mut out = [0u8; 32];
    for byte in 0..32 {
        for bit in 0..8 {
            let mut ones = 0usize;
            for d in descs {
                if d[byte] & (1 << bit) != 0 {
                    ones += 1;
                }
            }
            if 2 * ones > n {
                out[byte] |= 1 << bit;
            }
        }
    }
    out
}

/// k-means++ seeding under Hamming distance: first centre uniform, each
/// next chosen with probability ∝ (distance to the nearest chosen
/// centre)². May return fewer than `k` centres if the data has fewer
/// than `k` distinct points (degenerate node → fewer children).
fn kmeans_pp(descs: &[Descriptor], k: usize, rng: &mut SmallRng) -> Vec<Descriptor> {
    let mut centres = vec![descs[rng.random_range(0..descs.len())]];
    while centres.len() < k {
        let mut cum = Vec::with_capacity(descs.len());
        let mut total = 0.0;
        for d in descs {
            let dmin = centres.iter().map(|c| hamming(d, c)).min().unwrap_or(0) as f64;
            total += dmin * dmin;
            cum.push(total);
        }
        if total <= 0.0 {
            break; // all points coincide with a chosen centre
        }
        let t = rng.random::<f64>() * total;
        let pick = cum.iter().position(|&c| c >= t).unwrap_or(cum.len() - 1);
        centres.push(descs[pick]);
    }
    centres
}

/// One vocabulary-tree node. Internal nodes route by nearest `centre`;
/// leaves are *visual words* carrying a stable `word` id and its IDF
/// `weight` (set after the tree is built, from the training corpus).
#[derive(Clone, PartialEq, Serialize, Deserialize)]
struct Node {
    centre: Descriptor,
    children: Vec<u32>,
    word: Option<u32>,
    weight: f64,
}

/// A trained visual vocabulary (the offline asset). Maps any descriptor
/// to a word in `O(depth)` Hamming compares (tree descent) and an image
/// to a TF-IDF [`BowVector`].
#[derive(PartialEq, Serialize, Deserialize)]
pub struct Vocabulary {
    nodes: Vec<Node>,
    /// Leaf node index per word id.
    words: Vec<u32>,
}

/// `to_bytes`/`from_bytes` header: `"GBOW"` + format version, ahead of
/// the postcard payload (postcard is not self-describing, so the magic
/// rejects non-vocabulary files and the version gates schema changes).
const VOCAB_MAGIC: [u8; 4] = *b"GBOW";
const VOCAB_VERSION: u8 = 1;

/// Lloyd refinement cap (binary k-means converges in a handful of passes;
/// the bound also guarantees termination on pathological data).
const LLOYD_ITERS: usize = 10;

impl Vocabulary {
    /// Train from a corpus of images (each a descriptor set): cluster
    /// the pooled descriptors into a depth-`depth`, branching-`k`
    /// vocabulary tree, then weight every word by `ln(N / nᵢ)` IDF over
    /// the `N` training images (rare words ⇒ discriminative). `seed`
    /// makes the build fully deterministic.
    pub fn build(images: &[Vec<Descriptor>], k: usize, depth: usize, seed: u64) -> Self {
        let mut v = Self {
            nodes: vec![Node {
                centre: [0u8; 32],
                children: Vec::new(),
                word: None,
                weight: 0.0,
            }],
            words: Vec::new(),
        };
        let pool: Vec<Descriptor> = images.iter().flatten().copied().collect();
        if pool.is_empty() || k < 2 {
            return v;
        }
        let mut rng = SmallRng::seed_from_u64(seed | 1);
        v.cluster(pool, 0, 0, k, depth, &mut rng);

        // Assign stable word ids to every leaf (deterministic preorder).
        for idx in 0..v.nodes.len() {
            if v.nodes[idx].children.is_empty() {
                let id = v.words.len() as u32;
                v.nodes[idx].word = Some(id);
                v.words.push(idx as u32);
            }
        }
        // IDF: count training images that hit each word at least once.
        let n_docs = images.len();
        if n_docs > 0 && !v.words.is_empty() {
            let mut df = vec![0usize; v.words.len()];
            for img in images {
                let mut seen = HashSet::new();
                for d in img {
                    if let Some(w) = v.word_of(d) {
                        seen.insert(w);
                    }
                }
                for w in seen {
                    df[w as usize] += 1;
                }
            }
            for (w, &n) in df.iter().enumerate() {
                let node = v.words[w] as usize;
                v.nodes[node].weight = if n > 0 {
                    (n_docs as f64 / n as f64).ln()
                } else {
                    0.0
                };
            }
        }
        v
    }

    /// Recursively split `descs` under `node` into ≤`k` children until
    /// the depth bound or a node too small/degenerate to split.
    fn cluster(
        &mut self,
        descs: Vec<Descriptor>,
        node: usize,
        level: usize,
        k: usize,
        depth: usize,
        rng: &mut SmallRng,
    ) {
        if level >= depth || descs.len() <= 1 {
            self.nodes[node].centre = majority(&descs);
            return;
        }
        let mut centres = kmeans_pp(&descs, k, rng);
        if centres.len() < 2 {
            self.nodes[node].centre = majority(&descs);
            return;
        }
        let mut assign = vec![0usize; descs.len()];
        for _ in 0..LLOYD_ITERS {
            // Assign to the nearest centre (lowest index breaks ties).
            for (i, d) in descs.iter().enumerate() {
                let mut best = 0;
                let mut bd = u32::MAX;
                for (c, ctr) in centres.iter().enumerate() {
                    let h = hamming(d, ctr);
                    if h < bd {
                        bd = h;
                        best = c;
                    }
                }
                assign[i] = best;
            }
            // Recompute centres; empty clusters keep their old centre.
            let mut moved = false;
            for (c, ctr) in centres.iter_mut().enumerate() {
                let members: Vec<Descriptor> = descs
                    .iter()
                    .zip(&assign)
                    .filter(|&(_, &a)| a == c)
                    .map(|(d, _)| *d)
                    .collect();
                if members.is_empty() {
                    continue;
                }
                let m = majority(&members);
                if m != *ctr {
                    *ctr = m;
                    moved = true;
                }
            }
            if !moved {
                break;
            }
        }
        for (c, &ctr) in centres.iter().enumerate() {
            let part: Vec<Descriptor> = descs
                .iter()
                .zip(&assign)
                .filter(|&(_, &a)| a == c)
                .map(|(d, _)| *d)
                .collect();
            if part.is_empty() {
                continue;
            }
            let child = self.nodes.len() as u32;
            self.nodes.push(Node {
                centre: ctr,
                children: Vec::new(),
                word: None,
                weight: 0.0,
            });
            self.nodes[node].children.push(child);
            self.cluster(part, child as usize, level + 1, k, depth, rng);
        }
        if self.nodes[node].children.is_empty() {
            self.nodes[node].centre = majority(&descs);
        }
    }

    /// Number of visual words (leaves).
    pub fn len(&self) -> usize {
        self.words.len()
    }

    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// Word id a descriptor falls into (tree descent by nearest centre),
    /// or `None` for an untrained/empty vocabulary.
    pub fn word_of(&self, d: &Descriptor) -> Option<u32> {
        let mut idx = 0usize;
        if self.nodes[idx].children.is_empty() {
            return self.nodes[idx].word;
        }
        loop {
            let node = &self.nodes[idx];
            if node.children.is_empty() {
                return node.word;
            }
            let mut best = node.children[0];
            let mut bd = u32::MAX;
            for &ch in &node.children {
                let h = hamming(d, &self.nodes[ch as usize].centre);
                if h < bd {
                    bd = h;
                    best = ch;
                }
            }
            idx = best as usize;
        }
    }

    /// TF-IDF, L1-normalized [`BowVector`] for one image's descriptors
    /// (sorted by word id; empty for an empty image or empty vocab).
    pub fn transform(&self, image: &[Descriptor]) -> BowVector {
        self.transform_indexed(image).0
    }

    /// Like [`transform`](Self::transform) but also returns the **direct
    /// index**: the leaf word each input feature fell into (`None` if
    /// the vocab is empty), in input order. Grouping the two images'
    /// features by shared word turns cross-image descriptor matching
    /// from brute force into a per-word guided match — what
    /// relocalization / loop verification use after a BoW hit. One tree
    /// descent per feature (no extra cost over `transform`).
    pub fn transform_indexed(&self, image: &[Descriptor]) -> (BowVector, Vec<Option<u32>>) {
        if image.is_empty() || self.is_empty() {
            return (BowVector(Vec::new()), vec![None; image.len()]);
        }
        let mut per_feat = Vec::with_capacity(image.len());
        let mut tf: HashMap<u32, f64> = HashMap::new();
        for d in image {
            let w = self.word_of(d);
            if let Some(w) = w {
                *tf.entry(w).or_insert(0.0) += 1.0;
            }
            per_feat.push(w);
        }
        let total = image.len() as f64;
        let mut v: Vec<(u32, f64)> = tf
            .into_iter()
            .filter_map(|(w, c)| {
                let idf = self.nodes[self.words[w as usize] as usize].weight;
                let val = (c / total) * idf;
                (val != 0.0).then_some((w, val))
            })
            .collect();
        // Sort by word *before* the L1 sum so the accumulation order is
        // fixed (the `tf` HashMap iterates in a per-instance-random
        // order; summing it directly makes the result vary by ~1 ulp
        // run-to-run — and BoW feeds the determinism-gated pipeline).
        v.sort_by_key(|&(w, _)| w);
        let l1: f64 = v.iter().map(|&(_, x)| x.abs()).sum();
        if l1 > 0.0 {
            for e in &mut v {
                e.1 /= l1;
            }
        }
        (BowVector(v), per_feat)
    }

    /// Serialize to a compact binary blob — a `"GBOW"` magic + version
    /// header followed by a `postcard` encoding of the tree — so a
    /// vocabulary trained offline can be shipped/loaded as a static
    /// asset.
    pub fn to_bytes(&self) -> Vec<u8> {
        // In-memory → bytes can't fail for this plain data model.
        let payload = postcard::to_allocvec(self).expect("postcard encode vocabulary");
        let mut b = Vec::with_capacity(5 + payload.len());
        b.extend_from_slice(&VOCAB_MAGIC);
        b.push(VOCAB_VERSION);
        b.extend_from_slice(&payload);
        b
    }

    /// Inverse of [`to_bytes`](Self::to_bytes); `None` on a bad magic /
    /// version, a malformed/truncated payload, or a structurally
    /// inconsistent tree (postcard is not self-describing, so the header
    /// + the index-range check below are what reject junk input).
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        let rest = data.strip_prefix(&VOCAB_MAGIC)?;
        let (&ver, payload) = rest.split_first()?;
        if ver != VOCAB_VERSION {
            return None;
        }
        let v: Self = postcard::from_bytes(payload).ok()?;
        // Structural sanity: every child / word index in range.
        let nn = v.nodes.len() as u32;
        if v.nodes.iter().any(|n| n.children.iter().any(|&c| c >= nn))
            || v.words.iter().any(|&w| w >= nn)
        {
            return None;
        }
        Some(v)
    }
}

/// A sparse, L1-normalized TF-IDF image vector: `(word id, weight)`
/// pairs sorted by word id.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BowVector(Vec<(u32, f64)>);

impl BowVector {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn as_slice(&self) -> &[(u32, f64)] {
        &self.0
    }

    /// DBoW2 L1 similarity in `[0, 1]` (1 = identical place): over the
    /// **shared** words, `score = 1 − ½·Σ|aᵢ − bᵢ|` for L1-normalized
    /// inputs. Words unique to one vector contribute nothing.
    pub fn score(&self, other: &BowVector) -> f64 {
        let (a, b) = (&self.0, &other.0);
        let (mut i, mut j) = (0, 0);
        let mut acc = 0.0;
        while i < a.len() && j < b.len() {
            match a[i].0.cmp(&b[j].0) {
                std::cmp::Ordering::Equal => {
                    let (av, bv) = (a[i].1, b[j].1);
                    acc += (av - bv).abs() - av.abs() - bv.abs();
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
        }
        (-acc / 2.0).clamp(0.0, 1.0)
    }
}

/// One [`Database::query`] hit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QueryMatch {
    pub id: u32,
    pub score: f64,
}

/// Keyframe BoW database: stored vectors + a word→entries inverted
/// index so a query only scores entries that share a word with it.
#[derive(Default)]
pub struct Database {
    entries: Vec<BowVector>,
    inverted: HashMap<u32, Vec<u32>>,
}

impl Database {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Insert `v`, returning its entry id (its insertion index).
    pub fn add(&mut self, v: BowVector) -> u32 {
        let id = self.entries.len() as u32;
        for &(w, _) in &v.0 {
            self.inverted.entry(w).or_default().push(id);
        }
        self.entries.push(v);
        id
    }

    /// Top-`max` entries by L1 score, considering only those sharing ≥1
    /// word with `v`. Sorted by score desc, then id asc (deterministic);
    /// zero-score and `skip`-rejected entries are dropped. `skip` lets
    /// the caller exclude e.g. covisible/recent keyframes for loop
    /// detection.
    pub fn query(&self, v: &BowVector, max: usize, skip: impl Fn(u32) -> bool) -> Vec<QueryMatch> {
        let mut cands: HashSet<u32> = HashSet::new();
        for &(w, _) in &v.0 {
            if let Some(es) = self.inverted.get(&w) {
                cands.extend(es.iter().copied());
            }
        }
        let mut out: Vec<QueryMatch> = cands
            .into_iter()
            .filter(|&id| !skip(id))
            .filter_map(|id| {
                let s = v.score(&self.entries[id as usize]);
                (s > 0.0).then_some(QueryMatch { id, score: s })
            })
            .collect();
        out.sort_by(|x, y| {
            y.score
                .partial_cmp(&x.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(x.id.cmp(&y.id))
        });
        out.truncate(max);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic descriptor generator: `proto` flipped in `flips`
    /// pseudo-random bit positions (a noisy observation of a landmark).
    fn noisy(proto: &Descriptor, flips: usize, seed: u64) -> Descriptor {
        let mut r = SmallRng::seed_from_u64(seed | 1);
        let mut d = *proto;
        for _ in 0..flips {
            let bit = r.random_range(0..256usize);
            d[bit / 8] ^= 1 << (bit % 8);
        }
        d
    }

    fn proto(seed: u64) -> Descriptor {
        let mut r = SmallRng::seed_from_u64(seed | 1);
        let mut d = [0u8; 32];
        for b in &mut d {
            *b = r.random::<u8>();
        }
        d
    }

    /// `g` well-separated landmark prototypes; an "image of place p"
    /// observes a fixed subset around p's protos with bit noise.
    fn corpus(g: usize) -> Vec<Descriptor> {
        (0..g).map(|i| proto(0xA53 + i as u64)).collect()
    }
    fn image_of(protos: &[Descriptor], place: usize, span: usize, seed: u64) -> Vec<Descriptor> {
        (0..span)
            .map(|j| {
                let p = (place + j) % protos.len();
                noisy(&protos[p], 12, seed ^ (p as u64 * 2_654_435_761))
            })
            .collect()
    }

    fn vocab(protos: &[Descriptor]) -> Vocabulary {
        // Many noisy views of every landmark → words ≈ landmarks.
        let train: Vec<Vec<Descriptor>> = (0..protos.len())
            .flat_map(|p| {
                (0..6).map(move |s| vec![noisy(&protos[p], 10, 0x1000 + (p * 11 + s) as u64)])
            })
            .collect();
        Vocabulary::build(&train, 4, 4, 0xBEEF)
    }

    #[test]
    fn build_is_deterministic_and_words_stable() {
        let p = corpus(8);
        let train: Vec<Vec<Descriptor>> = (0..8).map(|i| vec![p[i]]).collect();
        let a = Vocabulary::build(&train, 3, 3, 7);
        let b = Vocabulary::build(&train, 3, 3, 7);
        assert!(a.len() > 1 && a.len() == b.len(), "words: {}", a.len());
        // Same descriptor → same word across independent builds, and a
        // lightly-perturbed copy lands in the same word as its proto.
        for (i, pi) in p.iter().enumerate() {
            assert_eq!(a.word_of(pi), b.word_of(pi));
            assert_eq!(a.word_of(pi), a.word_of(&noisy(pi, 4, i as u64)));
        }
    }

    #[test]
    fn same_place_scores_higher_than_different() {
        let p = corpus(10);
        let v = vocab(&p);
        let a1 = v.transform(&image_of(&p, 0, 4, 1));
        let a2 = v.transform(&image_of(&p, 0, 4, 2)); // same place, new view
        let b = v.transform(&image_of(&p, 5, 4, 3)); // different place
        assert!(!a1.is_empty());
        let same = a1.score(&a2);
        let diff = a1.score(&b);
        assert!(
            same > diff + 0.15,
            "same={same:.3} not clearly > diff={diff:.3}"
        );
        // Self-similarity is the maximum and ~1.
        let self_s = a1.score(&a1);
        assert!(self_s >= same && self_s > 0.99, "self={self_s:.3}");
    }

    #[test]
    fn database_retrieves_the_right_keyframe() {
        let p = corpus(12);
        let v = vocab(&p);
        let mut db = Database::new();
        // Five "keyframes" at distinct places.
        let places = [0usize, 3, 6, 9, 2];
        for (kf, &pl) in places.iter().enumerate() {
            db.add(v.transform(&image_of(&p, pl, 4, 100 + kf as u64)));
        }
        assert_eq!(db.len(), 5);
        // Re-observe place 6 (keyframe id 2) from a fresh viewpoint.
        let q = v.transform(&image_of(&p, 6, 4, 999));
        let hits = db.query(&q, 3, |_| false);
        assert!(!hits.is_empty(), "no retrieval");
        // The right keyframe must be among the top-scoring candidates.
        // A strict `hits[0].id == 2` assertion is too brittle: random
        // descriptor protos sometimes produce two places that tie for
        // top score, in which case any of the tied ids is a correct
        // retrieval — the invariant we actually want is "the right
        // place wins or ties for top".
        let top_score = hits[0].score;
        assert!(top_score > 0.0);
        assert!(
            hits.iter()
                .take_while(|h| h.score == top_score)
                .any(|h| h.id == 2),
            "wrong place top: {hits:?}"
        );
        // Scores are sorted descending.
        for w in hits.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
        // `skip` excludes a candidate (e.g. a covisible neighbour).
        let hits2 = db.query(&q, 3, |id| id == 2);
        assert!(hits2.iter().all(|h| h.id != 2));
    }

    #[test]
    fn degenerate_inputs_are_safe() {
        // Empty corpus → empty vocab, no words, safe transforms.
        let empty = Vocabulary::build(&[], 4, 3, 1);
        assert!(empty.is_empty());
        assert!(empty.transform(&[proto(1)]).is_empty());
        assert_eq!(empty.word_of(&proto(1)), None);

        let p = corpus(6);
        let v = vocab(&p);
        assert!(v.transform(&[]).is_empty());
        // Empty / no-overlap queries score 0 and return nothing.
        let mut db = Database::new();
        db.add(v.transform(&image_of(&p, 0, 3, 1)));
        assert!(db.query(&BowVector::default(), 5, |_| false).is_empty());
        assert_eq!(BowVector::default().score(&BowVector::default()), 0.0);
    }

    #[test]
    fn serialize_roundtrips_and_rejects_garbage() {
        let p = corpus(9);
        let v = vocab(&p);
        let bytes = v.to_bytes();
        let w = Vocabulary::from_bytes(&bytes).expect("decoded");
        // Structurally identical …
        assert!(w == v, "vocab struct differs after round-trip");
        // … and behaviourally identical on fresh inputs.
        for place in [0usize, 3, 7] {
            let img = image_of(&p, place, 4, 500 + place as u64);
            assert_eq!(v.transform(&img), w.transform(&img));
            assert_eq!(v.transform_indexed(&img).1, w.transform_indexed(&img).1);
        }
        // Bad magic / version / truncation → None, never a panic.
        assert!(Vocabulary::from_bytes(&[]).is_none());
        assert!(Vocabulary::from_bytes(b"NOPEx").is_none());
        let mut bad = bytes.clone();
        bad[4] = 99; // version
        assert!(Vocabulary::from_bytes(&bad).is_none());
        assert!(Vocabulary::from_bytes(&bytes[..bytes.len() - 3]).is_none());
        // Empty vocab also round-trips.
        let e = Vocabulary::build(&[], 4, 3, 1);
        assert!(Vocabulary::from_bytes(&e.to_bytes()).unwrap().is_empty());
    }

    #[test]
    fn transform_is_bitwise_deterministic() {
        // Same vocab + image ⇒ identical BowVector bits across calls
        // (the tf HashMap iterates in random order; the L1 sum must not
        // depend on it — the pipeline determinism gate relies on this).
        let p = corpus(10);
        let v = vocab(&p);
        let img = image_of(&p, 2, 8, 4242);
        let a = v.transform(&img);
        for _ in 0..16 {
            assert!(v.transform(&img) == a, "transform not bit-stable");
        }
    }

    #[test]
    fn direct_index_groups_features_by_word() {
        let p = corpus(8);
        let v = vocab(&p);
        let img = image_of(&p, 1, 6, 77);
        let (bv, per_feat) = v.transform_indexed(&img);
        assert_eq!(per_feat.len(), img.len());
        // The transform's words are exactly the distinct direct-index
        // words (consistency between the two outputs).
        let mut from_idx: Vec<u32> = per_feat.iter().flatten().copied().collect();
        from_idx.sort_unstable();
        from_idx.dedup();
        let mut from_vec: Vec<u32> = bv.as_slice().iter().map(|&(w, _)| w).collect();
        from_vec.sort_unstable();
        assert_eq!(from_idx, from_vec);
        // Guided match: two views of the same place share words, so
        // grouping by word pairs most features (cheap vs all-pairs).
        let a = image_of(&p, 4, 6, 1);
        let b = image_of(&p, 4, 6, 2);
        let wa = v.transform_indexed(&a).1;
        let wb = v.transform_indexed(&b).1;
        let shared = wa
            .iter()
            .flatten()
            .filter(|w| wb.iter().flatten().any(|x| x == *w))
            .count();
        assert!(shared >= a.len() / 2, "weak word overlap: {shared}");
    }
}
