//! # ccf-core
//!
//! Contextual Coherence Fields — earned relational trust for autonomous systems.
//!
//! Patent pending: US Provisional Application 63/988,438 (priority date 23 Feb 2026).
//!
//! ## What This Is
//!
//! CCF is a computational architecture for emergent social behaviour in autonomous systems.
//! Instead of a single emotional state, an agent maintains a *field* of trust states —
//! one per sensory context — learned continuously from experience.
//!
//! Trust earned in a bright quiet room does not transfer to a dark noisy room unless the
//! field explicitly learns they are similar. The agent remembers the shape of its own
//! comfort zone.
//!
//! ## Patent Claim Map
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
//! ## no_std
//!
//! This crate is `no_std` by default. Enable the `std` feature for persistence helpers.
//! Enable the `serde` feature for serialisation support (required for CCF_SEG / RVF).
//!
//! ## License
//!
//! Business Source License 1.1. Free for evaluation and non-production use.
//! Change date: 23 February 2032 — Apache License 2.0.
//! Commercial production use requires a license from Flout Labs (cbyrne@floutlabs.com).

#![no_std]
#![deny(unsafe_code)]
#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

// Pull in std when the feature is enabled (for persistence helpers, etc.)
#[cfg(feature = "std")]
extern crate std;

// Placeholder modules — populated by Phase 9 stories #48–#52
pub mod vocabulary;   // #48: SensorVocabulary trait + ContextKey
// pub mod accumulator;  // #49: CoherenceAccumulator + CoherenceField
// pub mod phase;        // #49: SocialPhase + Personality
pub mod sinkhorn;     // #50: SinkhornKnopp projector
pub mod boundary;     // #51: MinCutBoundary / Stoer-Wagner
