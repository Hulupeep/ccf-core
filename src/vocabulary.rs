//! Generic sensor vocabulary — the platform-independent context key system.
//!
//! Patent Claims 1 and 8: composite sensor context key as the fundamental unit
//! of situational awareness.
//!
//! # Implementing for a new platform
//!
//! ```rust,ignore
//! use ccf_core::vocabulary::{SensorVocabulary, ContextKey};
//!
//! #[derive(Clone, Debug, PartialEq, Eq, Hash)]
//! pub struct ThreeSensorBot {
//!     pub light: u8,   // 0=dark, 1=dim, 2=bright
//!     pub sound: u8,   // 0=quiet, 1=loud
//!     pub motion: u8,  // 0=still, 1=moving
//! }
//!
//! impl SensorVocabulary<3> for ThreeSensorBot {
//!     fn to_feature_vec(&self) -> [f32; 3] {
//!         [self.light as f32 / 2.0, self.sound as f32, self.motion as f32]
//!     }
//! }
//! // Now ContextKey::<ThreeSensorBot, 3> works with the full CCF stack.
//! ```
//!
//! # Invariants
//! - **I-DIST-001** — no_std compatible; no heap allocation required
//! - **I-DIST-002** — zero platform-specific bounds on the trait
//! - **I-DIST-005** — zero unsafe code

use core::hash::Hash;

// ---------------------------------------------------------------------------
// no_std sqrt via Newton-Raphson (8 iterations, accurate to ~1e-7 for [0, 1])
// ---------------------------------------------------------------------------

/// Compute the square root of a non-negative f32 using Newton-Raphson iteration.
/// This is `no_std` compatible and avoids any platform intrinsics.
fn sqrt_nr(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    // Initial guess using integer bit manipulation (fast inverse sqrt seed)
    let bits = x.to_bits();
    let guess_bits = 0x1fbd_1df5u32.wrapping_add(bits >> 1);
    let mut s = f32::from_bits(guess_bits);
    // Eight Newton-Raphson iterations: s = (s + x/s) / 2
    for _ in 0..8 {
        s = 0.5 * (s + x / s);
    }
    s
}

/// Platform-independent sensor vocabulary trait.
///
/// Implementors define the discrete sensory space the robot operates in.
/// CCF is generic over this trait — the same trust accumulation logic
/// works for any hardware as long as it can produce a discrete, hashable
/// context key and a float feature vector.
///
/// The const generic `N` is the dimensionality of the feature vector.
/// It must match `FEATURE_DIM` on the implementing type for the full CCF stack.
///
/// Patent Claims 1 and 8.
pub trait SensorVocabulary<const N: usize>: Eq + Hash + Clone + core::fmt::Debug {
    /// Dimensionality of the feature vector encoding (equal to the const generic `N`).
    /// Provided as an associated constant for ergonomic access at the type level.
    const FEATURE_DIM: usize = N;

    /// Encode this vocabulary instance as a normalised float feature vector.
    ///
    /// Each element should be in [0.0, 1.0] for cosine similarity to be meaningful.
    /// The order of dimensions must be consistent across calls.
    fn to_feature_vec(&self) -> [f32; N];
}

/// Composite context key — generic over sensor vocabulary.
///
/// Wraps any `SensorVocabulary` implementation and adds:
/// - Deterministic `context_hash_u32()` for HashMap keying
/// - `cosine_similarity()` for graph edge weights
///
/// Patent Claims 1 and 8.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ContextKey<V: SensorVocabulary<N>, const N: usize> {
    /// The sensor vocabulary snapshot for this context.
    pub vocabulary: V,
}

impl<V: SensorVocabulary<N>, const N: usize> ContextKey<V, N> {
    /// Create a new context key from a sensor vocabulary snapshot.
    pub fn new(vocabulary: V) -> Self {
        Self { vocabulary }
    }

    /// Deterministic FNV-1a hash of the feature vector.
    ///
    /// Used to key context entries in fixed-size arrays (no_std compatible).
    /// Deterministic: same vocabulary produces the same hash across restarts.
    pub fn context_hash_u32(&self) -> u32 {
        let vec = self.vocabulary.to_feature_vec();
        let mut h: u32 = 2_166_136_261;
        for &f in vec.iter() {
            // Quantise to u16 for stable hashing of float feature vectors.
            let bits: u16 = (f.clamp(0.0, 1.0) * 65535.0) as u16;
            h ^= bits as u32;
            h = h.wrapping_mul(16_777_619);
        }
        h
    }

    /// Cosine similarity between two context keys via their feature vectors.
    ///
    /// Returns a value in [0.0, 1.0] (assumes non-negative feature vectors).
    /// Used as the raw edge weight in the World Shape graph (Graph A).
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        let a = self.vocabulary.to_feature_vec();
        let b = other.vocabulary.to_feature_vec();

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let sq_a: f32 = a.iter().map(|x| x * x).sum();
        let sq_b: f32 = b.iter().map(|x| x * x).sum();
        let norm_a: f32 = sqrt_nr(sq_a);
        let norm_b: f32 = sqrt_nr(sq_b);

        let epsilon: f32 = 1e-9;
        let tiny_a: bool = norm_a < epsilon;
        let tiny_b: bool = norm_b < epsilon;
        if tiny_a || tiny_b {
            0.0
        } else {
            let raw: f32 = dot / (norm_a * norm_b);
            raw.clamp(0.0, 1.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple two-dimensional test vocabulary used throughout the unit tests.
    // For a production 6-dimensional vocabulary see `ccf_core::mbot::MbotSensors`.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct TwoSensor { light: u8, noise: u8 } // 0=low, 1=mid, 2=high

    impl SensorVocabulary<2> for TwoSensor {
        fn to_feature_vec(&self) -> [f32; 2] {
            [self.light as f32 / 2.0, self.noise as f32 / 2.0]
        }
    }

    fn bright_quiet() -> ContextKey<TwoSensor, 2> {
        ContextKey::new(TwoSensor { light: 2, noise: 0 })
    }

    fn dark_loud() -> ContextKey<TwoSensor, 2> {
        ContextKey::new(TwoSensor { light: 0, noise: 2 })
    }

    #[test]
    fn test_claim_1_context_key_is_deterministic() {
        // Patent Claim 1: discrete context identifier from quantised sensor signals
        let k1 = bright_quiet();
        let k2 = bright_quiet();
        assert_eq!(k1.context_hash_u32(), k2.context_hash_u32());
    }

    #[test]
    fn test_claim_1_different_contexts_have_different_hashes() {
        let k1 = bright_quiet();
        let k2 = dark_loud();
        assert_ne!(k1.context_hash_u32(), k2.context_hash_u32());
    }

    #[test]
    fn test_claim_8_composite_sensor_context_key() {
        // Patent Claim 8: composite sensor vocabulary trait
        let k = bright_quiet();
        let vec = k.vocabulary.to_feature_vec();
        assert_eq!(vec.len(), TwoSensor::FEATURE_DIM);
        // light=2 → 1.0, noise=0 → 0.0
        assert!((vec[0] - 1.0_f32).abs() < 1e-6);
        assert!((vec[1] - 0.0_f32).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_identical_contexts() {
        let k = bright_quiet();
        assert!((k.cosine_similarity(&k) - 1.0_f32).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_dissimilar_contexts() {
        let k1 = bright_quiet();
        let k2 = dark_loud();
        let sim = k1.cosine_similarity(&k2);
        // Bright+Quiet vs Dark+Loud should be low similarity
        assert!(sim < 0.5_f32, "sim={}", sim);
    }

    #[test]
    fn test_custom_vocabulary_works_without_modifying_ccf_core() {
        // Acceptance criterion: custom vocabulary compiles without modifying ccf-core
        #[derive(Clone, Debug, PartialEq, Eq, Hash)]
        struct TwoSensor { a: u8, b: u8 }
        impl SensorVocabulary<2> for TwoSensor {
            fn to_feature_vec(&self) -> [f32; 2] {
                [self.a as f32 / 255.0, self.b as f32 / 255.0]
            }
        }
        let k = ContextKey::new(TwoSensor { a: 100, b: 200 });
        let _hash = k.context_hash_u32(); // just needs to compile and not panic
        let sim = k.cosine_similarity(&k);
        assert!((sim - 1.0_f32).abs() < 1e-5, "self-similarity={}", sim);
    }

    #[test]
    fn test_sqrt_nr_accuracy() {
        // Verify our no_std sqrt helper is accurate enough for cosine similarity
        let cases: &[(f32, f32)] = &[
            (0.0, 0.0),
            (1.0, 1.0),
            (0.25, 0.5),
            (0.5, 0.7071068),
            (4.0, 2.0),
        ];
        for &(input, expected) in cases {
            let got = sqrt_nr(input);
            assert!(
                (got - expected).abs() < 1e-5,
                "sqrt_nr({}) = {}, expected {}",
                input,
                got,
                expected
            );
        }
    }
}
