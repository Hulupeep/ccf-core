/*
 * Notice of Provisional Patent Filing:
 * The methods and algorithms implemented in this file are the subject of a
 * United States Provisional Patent Application (63/988,438)
 * filed on February 23, 2026.
 *
 * This source code is licensed under the Business Source License 1.1.
 */

//! Hierarchical doubly stochastic mixing for large context fields.
//!
//! When the number of active contexts exceeds a configurable threshold (default 50),
//! the flat `SinkhornKnopp` approach becomes expensive (O(n²) per tick).
//! This module provides [`HierarchicalMixer`], a block-diagonal extension that
//! reduces the mixing cost to O(k² + Σnᵢ²) while preserving the full
//! mathematical guarantees of the Birkhoff polytope.
//!
//! Implements the hierarchical extension of Patent Claims 19–23
//! (see continuation Claims A–D: US Provisional 63/988,438).
//!
//! # Mathematical specification
//!
//! Given coherence vector `c ∈ ℝⁿ` partitioned into k clusters:
//!
//! **Step 1 — Intra-cluster mixing:** `c'ᵢ = Hᵢ · cᵢ`
//!
//! **Step 2 — Cluster summary:** `s̄ᵢ = (1/nᵢ) Σⱼ c'ᵢⱼ`
//!
//! **Step 3 — Inter-cluster mixing:** `s̄' = G · s̄`
//!
//! **Step 4 — Inter-cluster correction:**
//! `Δcᵢⱼ = (s̄'ᵢ − s̄ᵢ) × wᵢⱼ`
//! where `wᵢⱼ = interaction_count_j / Σₘ interaction_count_m`
//! (uniform if all counts are zero).
//!
//! **Step 5 — Final coherence:** `c''ᵢⱼ = clamp(c'ᵢⱼ + Δcᵢⱼ, 0.0, 1.0)`

use heapless::Vec as HVec;

use crate::sinkhorn::SinkhornKnopp;
use super::cluster::CoherenceCluster;
use super::transition::blend_alpha;
use super::{MAX_CLUSTERS, MAX_CLUSTER_SIZE, MAX_CONTEXTS_PER_CLUSTER};

/// Maximum total contexts the mixer can handle.
///
/// Equal to `MAX_CLUSTERS × MAX_CONTEXTS_PER_CLUSTER`.  A coherence field with
/// more active contexts than this value cannot be managed by the hierarchical
/// mixer — the deliberative unit is responsible for LRU eviction before this
/// limit is reached.
pub const MAX_TOTAL_CONTEXTS: usize = MAX_CLUSTERS * MAX_CONTEXTS_PER_CLUSTER;

// ─── HierarchicalMixerConfig ─────────────────────────────────────────────────

/// Configuration parameters for [`HierarchicalMixer`].
#[derive(Clone, Debug)]
pub struct HierarchicalMixerConfig {
    /// Context count above which hierarchical mode is engaged.
    ///
    /// When `n_active ≤ flat_threshold`, the caller should use the flat
    /// `SinkhornKnopp` path instead.  Default: 50.
    pub flat_threshold: usize,

    /// Maximum Sinkhorn-Knopp iterations for intra-cluster projection.  Default: 20.
    pub sk_iterations_intra: usize,

    /// Maximum Sinkhorn-Knopp iterations for inter-cluster projection.  Default: 20.
    pub sk_iterations_inter: usize,

    /// Number of ticks over which to blend old→new cluster structure on
    /// restructure.  Default: 100.
    pub transition_blend_ticks: usize,
}

impl Default for HierarchicalMixerConfig {
    fn default() -> Self {
        Self {
            flat_threshold: 50,
            sk_iterations_intra: 20,
            sk_iterations_inter: 20,
            transition_blend_ticks: 100,
        }
    }
}

// ─── HierarchicalMixer ───────────────────────────────────────────────────────

/// Block-diagonal hierarchical coherence mixer for large context fields.
///
/// Replaces the flat `SinkhornKnopp` path when the number of active contexts
/// exceeds [`HierarchicalMixerConfig::flat_threshold`].
///
/// The cluster structure is provided externally by the deliberative unit's
/// min-cut algorithm (see [`crate::boundary::MinCutBoundary`]); this type
/// only performs the mixing and transition smoothing.
///
/// # Hot-path guarantee
///
/// [`HierarchicalMixer::apply`] performs zero heap allocation.  All
/// intermediate buffers are stack-allocated.  Safe at 200 Hz on humanoid
/// hardware with sufficient stack space.
///
/// Implements continuation Claims A–D on Patent Claims 19–23
/// (US Provisional 63/988,438).
///
/// # Memory note
///
/// This struct is large (~8 MiB on 64-bit with the default constants) due to
/// inline fixed-size arrays.  On production humanoid platforms it should be
/// placed in static memory or heap-allocated by the caller.
#[derive(Clone, Debug)]
pub struct HierarchicalMixer {
    /// Active cluster definitions (up to `MAX_CLUSTERS`).
    pub clusters: HVec<CoherenceCluster, MAX_CLUSTERS>,

    /// Raw (un-projected) inter-cluster mixing parameters.
    ///
    /// Row-major, row stride = `MAX_CLUSTERS`.  Only the top-left
    /// `num_clusters × num_clusters` block is active.
    pub inter_mix_raw: [f32; MAX_CLUSTERS * MAX_CLUSTERS],

    /// Doubly stochastic projection of `inter_mix_raw`.
    ///
    /// Updated by [`Self::reproject_all`] and [`Self::update_inter_params`].
    pub inter_mix_projected: [f32; MAX_CLUSTERS * MAX_CLUSTERS],

    /// Number of active clusters (≤ `MAX_CLUSTERS`).
    pub num_clusters: usize,

    /// Runtime configuration.
    pub config: HierarchicalMixerConfig,

    /// True while a cluster restructure transition is in progress.
    pub in_transition: bool,

    /// Ticks elapsed since the current transition started.
    pub transition_tick: usize,

    /// Previous cluster definitions, kept for transition blending.
    ///
    /// Present while `in_transition == true`.
    pub old_clusters: Option<HVec<CoherenceCluster, MAX_CLUSTERS>>,

    /// Previous inter-cluster projected matrix, kept for transition blending.
    ///
    /// Present while `in_transition == true`.
    pub old_inter_mix: Option<[f32; MAX_CLUSTERS * MAX_CLUSTERS]>,
}

impl HierarchicalMixer {
    /// Create a new mixer with the given configuration and no active clusters.
    ///
    /// Both the intra and inter mixing matrices are initialised to identity.
    pub fn new(config: HierarchicalMixerConfig) -> Self {
        let mut inter_raw = [0.0f32; MAX_CLUSTERS * MAX_CLUSTERS];
        let mut inter_proj = [0.0f32; MAX_CLUSTERS * MAX_CLUSTERS];
        // Identity for full matrix — real k×k identity is set on first update_clusters call.
        for i in 0..MAX_CLUSTERS {
            inter_raw[i * MAX_CLUSTERS + i] = 1.0;
            inter_proj[i * MAX_CLUSTERS + i] = 1.0;
        }
        Self {
            clusters: HVec::new(),
            inter_mix_raw: inter_raw,
            inter_mix_projected: inter_proj,
            num_clusters: 0,
            config,
            in_transition: false,
            transition_tick: 0,
            old_clusters: None,
            old_inter_mix: None,
        }
    }

    /// Apply the full five-step hierarchical mixing operation.
    ///
    /// This is the **hot path** — called every reflexive tick.  No allocation.
    ///
    /// `coherence_values[i]` must correspond to `interaction_counts[i]` for
    /// the same context index.  All values in `coherence_values` must be in
    /// `[0.0, 1.0]` on entry; the result is clamped to `[0.0, 1.0]`.
    ///
    /// If a transition is active (see [`Self::update_clusters`]), the output
    /// is blended between the old and new cluster structures using the current
    /// `α(t)` (see [`crate::mixing::transition::blend_alpha`]).
    ///
    /// # Invariant I-HMX-001
    /// No heap allocation — stack buffers only.
    ///
    /// # Invariant I-HMX-002
    /// All output values clamped to `[0.0, 1.0]`.
    pub fn apply(
        &self,
        coherence_values: &mut [f32],
        interaction_counts: &[u32],
    ) {
        if self.in_transition {
            if let (Some(old_clusters), Some(old_inter)) =
                (&self.old_clusters, &self.old_inter_mix)
            {
                let n = coherence_values.len();
                let alpha = blend_alpha(
                    self.transition_tick,
                    self.config.transition_blend_ticks,
                );

                // Buffer to hold old-structure result
                let mut buf_old = [0.0f32; MAX_TOTAL_CONTEXTS];
                buf_old[..n].copy_from_slice(&coherence_values[..n]);

                // Apply old structure to buf_old
                apply_core(
                    old_clusters,
                    old_clusters.len(),
                    old_inter,
                    &mut buf_old[..n],
                    interaction_counts,
                );

                // Apply new structure to coherence_values in-place
                apply_core(
                    &self.clusters,
                    self.num_clusters,
                    &self.inter_mix_projected,
                    coherence_values,
                    interaction_counts,
                );

                // Blend: c_eff = (1-α)·c_old + α·c_new, clamp to [0,1]
                for i in 0..n {
                    coherence_values[i] =
                        ((1.0 - alpha) * buf_old[i] + alpha * coherence_values[i])
                            .clamp(0.0, 1.0);
                }
            } else {
                // Transition state inconsistent — fall through to new structure
                apply_core(
                    &self.clusters,
                    self.num_clusters,
                    &self.inter_mix_projected,
                    coherence_values,
                    interaction_counts,
                );
            }
        } else {
            apply_core(
                &self.clusters,
                self.num_clusters,
                &self.inter_mix_projected,
                coherence_values,
                interaction_counts,
            );
        }
    }

    /// Install a new cluster structure from deliberative min-cut results.
    ///
    /// `assignments[i]` is the `cluster_id` for context index `i`.
    /// `num_clusters` is the total number of distinct cluster IDs.
    ///
    /// If the mixer has an existing cluster structure, the old state is saved
    /// and a transition blend begins (see [`Self::tick_transition`]).
    /// New intra and inter mixing matrices are initialised to identity.
    ///
    /// Called by the deliberative processing unit during consolidation —
    /// **not** on the hot path.
    pub fn update_clusters(&mut self, assignments: &[u16], num_clusters: usize) {
        // Save current state for transition blending if we have clusters already
        if !self.clusters.is_empty() {
            self.in_transition = true;
            self.transition_tick = 0;
            self.old_clusters = Some(self.clusters.clone());
            self.old_inter_mix = Some(self.inter_mix_projected);
        }

        self.num_clusters = num_clusters.min(MAX_CLUSTERS);
        self.clusters.clear();

        // Allocate cluster slots
        for ci in 0..self.num_clusters {
            let _ = self.clusters.push(CoherenceCluster::new(ci as u16));
        }

        // Assign context indices to clusters
        for (context_idx, &cluster_id) in assignments.iter().enumerate() {
            let ci = cluster_id as usize;
            if ci < self.clusters.len() {
                let _ = self.clusters[ci].member_indices.push(context_idx);
                self.clusters[ci].size = self.clusters[ci].member_indices.len();
            }
        }

        // Initialise intra-cluster matrices to identity (n×n block)
        for cluster in self.clusters.iter_mut() {
            let n = cluster.size;
            // Zero the full padded matrix first
            for x in cluster.intra_mix_raw.iter_mut() { *x = 0.0; }
            for x in cluster.intra_mix_projected.iter_mut() { *x = 0.0; }
            // Set n×n identity in the top-left block
            for i in 0..n.min(MAX_CLUSTER_SIZE) {
                cluster.intra_mix_raw[i * MAX_CLUSTER_SIZE + i] = 1.0;
                cluster.intra_mix_projected[i * MAX_CLUSTER_SIZE + i] = 1.0;
            }
            cluster.projected_dirty = false;
        }

        // Initialise inter-cluster matrix to k×k identity
        for x in self.inter_mix_raw.iter_mut() { *x = 0.0; }
        for x in self.inter_mix_projected.iter_mut() { *x = 0.0; }
        for i in 0..self.num_clusters {
            self.inter_mix_raw[i * MAX_CLUSTERS + i] = 1.0;
            self.inter_mix_projected[i * MAX_CLUSTERS + i] = 1.0;
        }
    }

    /// Update raw intra-cluster mixing parameters for a specific cluster.
    ///
    /// `raw_params` is a row-major `size × size` matrix (compact, no padding).
    /// The values are stored into the padded `MAX_CLUSTER_SIZE × MAX_CLUSTER_SIZE`
    /// layout and `projected_dirty` is set to `true`.
    ///
    /// Call [`Self::reproject_all`] to apply Sinkhorn-Knopp after updates.
    ///
    /// Called by the deliberative path during mixing matrix optimisation.
    pub fn update_intra_params(&mut self, cluster_id: u16, raw_params: &[f32]) {
        for cluster in self.clusters.iter_mut() {
            if cluster.cluster_id == cluster_id {
                let n = cluster.size;
                for i in 0..n.min(MAX_CLUSTER_SIZE) {
                    for j in 0..n.min(MAX_CLUSTER_SIZE) {
                        let src = i * n + j;
                        if src < raw_params.len() {
                            cluster.intra_mix_raw[i * MAX_CLUSTER_SIZE + j] = raw_params[src];
                        }
                    }
                }
                cluster.projected_dirty = true;
                break;
            }
        }
    }

    /// Update raw inter-cluster mixing parameters.
    ///
    /// `raw_params` is a row-major `num_clusters × num_clusters` matrix (compact).
    ///
    /// Call [`Self::reproject_all`] to apply Sinkhorn-Knopp after updates.
    ///
    /// Called by the deliberative path during mixing matrix optimisation.
    pub fn update_inter_params(&mut self, raw_params: &[f32]) {
        let k = self.num_clusters;
        for i in 0..k.min(MAX_CLUSTERS) {
            for j in 0..k.min(MAX_CLUSTERS) {
                let src = i * k + j;
                if src < raw_params.len() {
                    self.inter_mix_raw[i * MAX_CLUSTERS + j] = raw_params[src];
                }
            }
        }
    }

    /// Advance transition blending by one tick.
    ///
    /// Returns `true` if the transition is still active after this tick,
    /// `false` once `transition_tick >= transition_blend_ticks`.  When the
    /// transition completes, the old cluster state is dropped.
    ///
    /// Call this every reflexive tick while `in_transition == true`.
    pub fn tick_transition(&mut self) -> bool {
        if !self.in_transition {
            return false;
        }
        self.transition_tick += 1;
        if self.transition_tick >= self.config.transition_blend_ticks {
            self.in_transition = false;
            self.old_clusters = None;
            self.old_inter_mix = None;
            self.transition_tick = 0;
            false
        } else {
            true
        }
    }

    /// Re-project all mixing matrices via Sinkhorn-Knopp.
    ///
    /// Projects each cluster's `intra_mix_raw` and the global `inter_mix_raw`
    /// onto the Birkhoff polytope, updating the corresponding `*_projected`
    /// fields.  Called by the deliberative unit after mixing matrix
    /// optimisation completes.
    ///
    /// This method is **not** on the hot path and may use large stack buffers.
    ///
    /// # Invariant I-HMX-003
    /// Reuses the existing [`SinkhornKnopp`] implementation — no new projector.
    pub fn reproject_all(&mut self) {
        let sk_intra = SinkhornKnopp::new(1e-6, self.config.sk_iterations_intra as u32);
        let sk_inter = SinkhornKnopp::new(1e-6, self.config.sk_iterations_inter as u32);

        for cluster in self.clusters.iter_mut() {
            let n = cluster.size;
            if n == 0 {
                continue;
            }

            // Copy n×n sub-block (padded row-stride MAX_CLUSTER_SIZE) into a
            // compact n×n buffer for the SK projector.
            let mut compact = [0.0f32; MAX_CLUSTER_SIZE * MAX_CLUSTER_SIZE];
            for i in 0..n {
                for j in 0..n {
                    compact[i * n + j] = cluster.intra_mix_raw[i * MAX_CLUSTER_SIZE + j];
                }
            }

            sk_intra.project_flat(&mut compact[..n * n], n);

            // Copy projected compact result back into the padded layout.
            // Padding diagonal is set to 1 (identity), off-diagonal to 0.
            for i in 0..MAX_CLUSTER_SIZE {
                for j in 0..MAX_CLUSTER_SIZE {
                    cluster.intra_mix_projected[i * MAX_CLUSTER_SIZE + j] =
                        if i < n && j < n {
                            compact[i * n + j]
                        } else if i == j {
                            1.0
                        } else {
                            0.0
                        };
                }
            }
            cluster.projected_dirty = false;
        }

        // Project inter-cluster matrix
        let k = self.num_clusters;
        if k > 0 {
            let mut compact_inter = [0.0f32; MAX_CLUSTERS * MAX_CLUSTERS];
            for i in 0..k {
                for j in 0..k {
                    compact_inter[i * k + j] = self.inter_mix_raw[i * MAX_CLUSTERS + j];
                }
            }
            sk_inter.project_flat(&mut compact_inter[..k * k], k);
            for i in 0..MAX_CLUSTERS {
                for j in 0..MAX_CLUSTERS {
                    self.inter_mix_projected[i * MAX_CLUSTERS + j] =
                        if i < k && j < k {
                            compact_inter[i * k + j]
                        } else if i == j {
                            1.0
                        } else {
                            0.0
                        };
                }
            }
        }
    }
}

// ─── apply_core ─────────────────────────────────────────────────────────────

/// Inner five-step hierarchical mixing kernel.
///
/// Separated from [`HierarchicalMixer::apply`] so the same logic can be
/// applied to both current and old cluster structures during transition blending
/// without borrow conflicts.
///
/// All arithmetic is in-place on `coherence_values`.  Stack buffers only.
fn apply_core(
    clusters: &HVec<CoherenceCluster, MAX_CLUSTERS>,
    num_clusters: usize,
    inter_mix: &[f32; MAX_CLUSTERS * MAX_CLUSTERS],
    coherence_values: &mut [f32],
    interaction_counts: &[u32],
) {
    let cv_len = coherence_values.len();
    let ic_len = interaction_counts.len();

    // ── Step 1: intra-cluster mixing ─────────────────────────────────────────
    for cluster in clusters.iter().take(num_clusters) {
        let n = cluster.size;
        if n == 0 {
            continue;
        }

        // c'_i = H_i · c_i  (matrix-vector multiply using top-left n×n block)
        let mut c_out = [0.0f32; MAX_CLUSTER_SIZE];
        for i in 0..n {
            let mut sum = 0.0f32;
            for k in 0..n {
                let global_k = cluster.member_indices[k];
                if global_k < cv_len {
                    sum += cluster.intra_mix_projected[i * MAX_CLUSTER_SIZE + k]
                        * coherence_values[global_k];
                }
            }
            c_out[i] = sum.clamp(0.0, 1.0);
        }
        for i in 0..n {
            let global_i = cluster.member_indices[i];
            if global_i < cv_len {
                coherence_values[global_i] = c_out[i];
            }
        }
    }

    // ── Step 2: cluster summary means ────────────────────────────────────────
    let mut s_bar = [0.0f32; MAX_CLUSTERS];
    for (ci, cluster) in clusters.iter().enumerate().take(num_clusters) {
        let n = cluster.size;
        if n == 0 {
            continue;
        }
        let mut sum = 0.0f32;
        for j in 0..n {
            let idx = cluster.member_indices[j];
            if idx < cv_len {
                sum += coherence_values[idx];
            }
        }
        s_bar[ci] = sum / n as f32;
    }

    // ── Step 3: inter-cluster mixing ─────────────────────────────────────────
    let mut s_bar_prime = [0.0f32; MAX_CLUSTERS];
    for i in 0..num_clusters {
        let mut sum = 0.0f32;
        for k in 0..num_clusters {
            sum += inter_mix[i * MAX_CLUSTERS + k] * s_bar[k];
        }
        s_bar_prime[i] = sum;
    }

    // ── Steps 4 & 5: inter-cluster correction + clamp ────────────────────────
    for (ci, cluster) in clusters.iter().enumerate().take(num_clusters) {
        let n = cluster.size;
        if n == 0 {
            continue;
        }
        let delta_mean = s_bar_prime[ci] - s_bar[ci];

        // Weight denominator: sum of interaction counts within cluster
        let mut total_count: u32 = 0;
        for j in 0..n {
            let idx = cluster.member_indices[j];
            if idx < ic_len {
                total_count = total_count.saturating_add(interaction_counts[idx]);
            }
        }

        for j in 0..n {
            let idx = cluster.member_indices[j];
            if idx >= cv_len {
                continue;
            }
            let w = if total_count > 0 && idx < ic_len {
                interaction_counts[idx] as f32 / total_count as f32
            } else {
                1.0 / n as f32
            };
            coherence_values[idx] = (coherence_values[idx] + delta_mean * w).clamp(0.0, 1.0);
        }
    }
}
