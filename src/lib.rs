//! # ccf-core
//!
//! Contextual Coherence Fields — earned relational trust for autonomous systems.
//!
//! Patent pending: US Provisional Application 63/988,438 (priority date 23 Feb 2026).
//!
//! ---
//!
//! ## This is not a behaviour system. It is a mathematical architecture.
//!
//! Three primitives combine to produce emergent social behaviour without a single
//! line of behavioural code.
//!
//! **Context-keyed accumulators** — trust is not a single global value. Every distinct
//! sensory environment has its own independent trust history. Bright-and-quiet is not
//! the same as dark-and-loud, and the two histories never share state. A robot that
//! trusts the living room starts at zero in the basement.
//!
//! **The minimum gate** — effective coherence requires agreement between two signals:
//! what the system has *learned* (accumulated context trust) and what it is
//! *experiencing right now* (the instant sensor reading).
//! > "Both must be true, or I stay reserved."
//!
//! Familiar contexts buffer noise — a single bad moment cannot erase earned trust.
//! Unfamiliar contexts demand proof before any expressiveness is permitted.
//!
//! **Graph min-cut boundary** — as contexts accumulate trust histories, the system
//! builds a trust-weighted graph. Stoer-Wagner global min-cut finds the cheapest
//! place to divide it into two clusters.
//! > "This room feels different from that room."
//!
//! The comfort zone *emerges* from the topology — you do not configure a threshold.
//!
//! None of these produces social behaviour alone. Together they produce shyness,
//! warmth, protectiveness, and earned fluency — without programming any of it.
//!
//! ---
//!
//! ## The pipeline
//!
//! ```text
//! Sensors → ContextKey → CoherenceField → SocialPhase → Outputs
//!                ↑              ↑               ↑
//!         SensorVocabulary  Personality    PhaseSpace
//!                               ↓
//!                       MinCutBoundary  (comfort zone)
//!                       SinkhornKnopp  (trust mixing)
//! ```
//!
//! ## Module overview
//!
//! | Module | Key types | What it does |
//! |--------|-----------|--------------|
//! | [`vocabulary`] | [`SensorVocabulary`], [`ContextKey`] | Define your sensor space; hash + cosine similarity |
//! | [`accumulator`] | [`CoherenceAccumulator`], [`CoherenceField`] | Per-context trust with earned floor and minimum gate |
//! | [`phase`] | [`SocialPhase`], [`Personality`], [`PhaseSpace`] | Four-quadrant phase classifier with Schmitt trigger hysteresis |
//! | [`boundary`] | [`MinCutBoundary`] | Stoer-Wagner comfort-zone boundary discovery |
//! | [`sinkhorn`] | [`SinkhornKnopp`] | Doubly stochastic trust mixing |
//! | [`mbot`] | [`mbot::MbotSensors`] | Reference 6-dimensional vocabulary for mBot2 ($50 hardware) |
//! | [`seg`] | [`seg::CcfSegSnapshot`] | Serialisable field snapshot for persistence (requires `serde` feature) |
//!
//! ## Patent claim map
//!
//! | Type | Patent Claims | Description |
//! |------|--------------|-------------|
//! | [`SensorVocabulary`] | 1, 8 | Composite sensor context key trait |
//! | [`ContextKey`] | 1, 8 | Discrete context identifier from quantised sensor signals |
//! | [`CoherenceAccumulator`] | 2–5 | Per-context trust state with earned floor and asymmetric decay |
//! | [`CoherenceField`] | 6–7, 13 | Trust field: context-keyed accumulator map with min-gate |
//! | [`SocialPhase`] | 14–18 | Four-quadrant phase classifier with Schmitt trigger hysteresis |
//! | [`SinkhornKnopp`] | 19–23 | Birkhoff polytope projector — doubly stochastic mixing matrix |
//! | [`MinCutBoundary`] | 9–12 | Stoer-Wagner comfort-zone boundary discovery |
//! | [`Personality`] | 3 (modulators) | Dynamic modulators: curiosity, startle sensitivity, recovery |
//!
//! ## `no_std`
//!
//! This crate is `#![no_std]` by default with no heap required. Enable the `std` feature
//! for persistence helpers. Enable the `serde` feature for serialisation support
//! (required for [`seg::CcfSegSnapshot`] and RVF persistence).
//!
//! ## License
//!
//! Business Source License 1.1. Free for evaluation and non-production use.
//! Change date: 23 February 2032 — Apache License 2.0.
//! Commercial production use requires a license from Flout Labs (cbyrne@floutlabs.com).

#![cfg_attr(not(any(feature = "std", feature = "python-ffi")), no_std)]
#![deny(unsafe_code)]
#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

// Pull in std when the feature is enabled (for persistence helpers, etc.)
#[cfg(any(feature = "std", feature = "python-ffi"))]
extern crate std;

// Placeholder modules — populated by Phase 9 stories #48–#52
pub mod vocabulary;   // #48: SensorVocabulary trait + ContextKey
pub mod accumulator;  // #49: CoherenceAccumulator + CoherenceField
pub mod phase;        // #49: SocialPhase + Personality
pub mod sinkhorn;     // #50: SinkhornKnopp projector
pub mod boundary;     // #51: MinCutBoundary / Stoer-Wagner
pub mod mbot;         // mBot2 reference vocabulary (MbotSensors, 6-dim)
#[cfg(feature = "serde")]
pub mod seg;          // #53: CCF_SEG snapshot format

#[cfg(feature = "python-ffi")]
pub mod ffi;

/// Adaptive coherence mixing — flat or hierarchical.
///
/// Enabled by `features = ["hierarchical"]`.  Compiles to nothing when
/// the feature is absent.
///
/// | Type | Patent Claims |
/// |------|--------------|
/// | [`mixing::HierarchicalMixer`] | Continuation A–D on Claims 19–23 |
/// | [`mixing::MixingStrategy`] | Adaptive mode selection |
/// | [`mixing::CoherenceCluster`] | Per-cluster intra-mixing state |
#[cfg(feature = "hierarchical")]
pub mod mixing;
