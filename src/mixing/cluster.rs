/*
 * Notice of Provisional Patent Filing:
 * The methods and algorithms implemented in this file are the subject of a
 * United States Provisional Patent Application (63/988,438)
 * filed on February 23, 2026.
 *
 * This source code is licensed under the Business Source License 1.1.
 */

//! Per-cluster mixing state for the hierarchical coherence mixer.
//!
//! A [`CoherenceCluster`] groups related context accumulators and owns the
//! intra-cluster doubly stochastic mixing matrix for those contexts.
//! Cluster structure is discovered externally by the deliberative unit's
//! min-cut algorithm and delivered via [`crate::mixing::HierarchicalMixer::update_clusters`].
//!
//! Implements the hierarchical extension of Patent Claims 19–23
//! (see continuation Claim A: US Provisional 63/988,438).

use heapless::Vec as HVec;

use super::{MAX_CLUSTER_SIZE, MAX_CONTEXTS_PER_CLUSTER};

// ─── CoherenceCluster ────────────────────────────────────────────────────────

/// A single context cluster within the hierarchical mixing hierarchy.
///
/// Groups a set of context indices (into the parent field's accumulator list)
/// and owns a doubly stochastic intra-cluster mixing matrix of size
/// `size × size`, stored in the top-left corner of a
/// `MAX_CLUSTER_SIZE × MAX_CLUSTER_SIZE` padded array (row stride = `MAX_CLUSTER_SIZE`).
///
/// The projected matrix is kept in sync with the raw parameter matrix by
/// [`crate::mixing::HierarchicalMixer::reproject_all`] and
/// [`crate::mixing::HierarchicalMixer::update_intra_params`].
///
/// Implements the hierarchical extension of Patent Claims 19–23.
#[derive(Clone, Debug)]
pub struct CoherenceCluster {
    /// Numeric identifier matching the assignment index from the deliberative unit.
    pub cluster_id: u16,

    /// Indices into the parent coherence field's ordered context list.
    ///
    /// `member_indices[j]` is the absolute position of the j-th cluster member
    /// in the `coherence_values` slice passed to `HierarchicalMixer::apply`.
    pub member_indices: HVec<usize, MAX_CONTEXTS_PER_CLUSTER>,

    /// Raw (un-projected) intra-cluster mixing parameters.
    ///
    /// Row-major, row stride = `MAX_CLUSTER_SIZE`. Only the top-left `size × size`
    /// sub-block is meaningful; padding entries are ignored during projection.
    pub intra_mix_raw: [f32; MAX_CLUSTER_SIZE * MAX_CLUSTER_SIZE],

    /// Doubly stochastic projection of `intra_mix_raw`.
    ///
    /// Row-major, row stride = `MAX_CLUSTER_SIZE`. Updated by
    /// [`crate::mixing::HierarchicalMixer::reproject_all`].
    /// Padding entries outside the `size × size` sub-block are set to
    /// the identity (diagonal 1, off-diagonal 0).
    pub intra_mix_projected: [f32; MAX_CLUSTER_SIZE * MAX_CLUSTER_SIZE],

    /// Number of active members (= `member_indices.len()`).
    pub size: usize,

    /// True when `intra_mix_raw` has been updated but `intra_mix_projected`
    /// has not yet been re-projected via Sinkhorn-Knopp.
    pub projected_dirty: bool,
}

impl CoherenceCluster {
    /// Create an empty cluster with all-zero mixing matrices.
    ///
    /// After construction `size == 0` and both matrices are all-zero.
    /// Caller is responsible for setting identity or calling
    /// [`crate::mixing::HierarchicalMixer::update_clusters`] which initialises
    /// the matrices to identity for the assigned `size`.
    pub fn new(cluster_id: u16) -> Self {
        Self {
            cluster_id,
            member_indices: HVec::new(),
            intra_mix_raw: [0.0; MAX_CLUSTER_SIZE * MAX_CLUSTER_SIZE],
            intra_mix_projected: [0.0; MAX_CLUSTER_SIZE * MAX_CLUSTER_SIZE],
            size: 0,
            projected_dirty: false,
        }
    }
}
