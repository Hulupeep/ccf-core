/*
 * Notice of Provisional Patent Filing:
 * The methods and algorithms implemented in this file are the subject of a
 * United States Provisional Patent Application (63/988,438)
 * filed on February 23, 2026.
 *
 * This source code is licensed under the Business Source License 1.1.
 */

//! Adaptive coherence mixing — flat or hierarchical based on active context count.
//!
//! This module provides [`MixingStrategy`], which selects between the flat
//! [`crate::sinkhorn::SinkhornKnopp`] path (for small n) and the hierarchical
//! block-diagonal path (for large n), along with all types needed to operate
//! the hierarchical mixer.
//!
//! Gated behind `#[cfg(feature = "hierarchical")]` — compiles to nothing when
//! the feature is disabled.
//!
//! Implements continuation Claims A–D on Patent Claims 19–23
//! (US Provisional 63/988,438).

pub mod cluster;
pub mod hierarchical;
pub mod transition;

pub use cluster::CoherenceCluster;
pub use hierarchical::{HierarchicalMixer, HierarchicalMixerConfig, MAX_TOTAL_CONTEXTS};
pub use transition::blend_alpha;

// ─── compile-time size constants ─────────────────────────────────────────────

/// Maximum number of clusters the hierarchical mixer supports.
///
/// A larger value increases struct size but allows coarser-grained cluster
/// structures for very large context fields.
pub const MAX_CLUSTERS: usize = 32;

/// Maximum number of member contexts per cluster.
///
/// Also the row/column dimension of each intra-cluster mixing matrix.
pub const MAX_CLUSTER_SIZE: usize = 128;

/// Alias for `MAX_CLUSTER_SIZE` — the capacity of the member-index vector per cluster.
pub const MAX_CONTEXTS_PER_CLUSTER: usize = MAX_CLUSTER_SIZE;

// ─── MixingStrategy ──────────────────────────────────────────────────────────

/// Runtime selector between flat and hierarchical mixing modes.
///
/// The deliberative unit calls [`MixingStrategy::select`] after each
/// consolidation pass to ensure the right mixing path is in use.  When
/// the context count crosses [`HierarchicalMixerConfig::flat_threshold`] in
/// either direction, the strategy switches automatically.
///
/// # Do not modify `CoherenceField`
///
/// Per invariant I-HMX-009, the existing `CoherenceField` struct is not
/// modified.  `MixingStrategy` is used alongside it by the caller.
pub enum MixingStrategy {
    /// Use the flat `SinkhornKnopp` path (n ≤ flat_threshold).
    ///
    /// The caller manages the flat mixing matrix separately via
    /// [`crate::sinkhorn::SinkhornKnopp`].
    Flat,

    /// Use the hierarchical block-diagonal path (n > flat_threshold).
    Hierarchical(HierarchicalMixer),
}

impl MixingStrategy {
    /// Select the appropriate mixing strategy for the given active context count.
    ///
    /// If `n_active > config.flat_threshold`, a new [`HierarchicalMixer`] is
    /// created with the supplied configuration.  Otherwise, returns `Flat`.
    pub fn select(n_active: usize, config: HierarchicalMixerConfig) -> Self {
        if n_active > config.flat_threshold {
            Self::Hierarchical(HierarchicalMixer::new(config))
        } else {
            Self::Flat
        }
    }

    /// Returns `true` if the hierarchical path is currently active.
    pub fn is_hierarchical(&self) -> bool {
        matches!(self, Self::Hierarchical(_))
    }

    /// Returns `true` if the flat path is currently active.
    pub fn is_flat(&self) -> bool {
        matches!(self, Self::Flat)
    }

    /// Return a reference to the inner [`HierarchicalMixer`], if active.
    pub fn hierarchical(&self) -> Option<&HierarchicalMixer> {
        match self {
            Self::Hierarchical(h) => Some(h),
            Self::Flat => None,
        }
    }

    /// Return a mutable reference to the inner [`HierarchicalMixer`], if active.
    pub fn hierarchical_mut(&mut self) -> Option<&mut HierarchicalMixer> {
        match self {
            Self::Hierarchical(h) => Some(h),
            Self::Flat => None,
        }
    }
}
