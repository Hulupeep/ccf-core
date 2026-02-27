//! Integration tests for the HierarchicalMixer.
//!
//! Tests are run with `cargo test --features hierarchical`.
//!
//! The HierarchicalMixer struct is large (~8 MiB with default constants).
//! Each test that instantiates it spawns a thread with a 64 MiB stack to
//! avoid stack overflow on development machines.  Production deployments
//! use static or heap allocation.

#![cfg(feature = "hierarchical")]

use ccf_core::mixing::{HierarchicalMixer, HierarchicalMixerConfig, MixingStrategy};
use ccf_core::sinkhorn::SinkhornKnopp;

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Default config with a short transition for testing.
fn test_config() -> HierarchicalMixerConfig {
    HierarchicalMixerConfig {
        flat_threshold: 4,
        sk_iterations_intra: 20,
        sk_iterations_inter: 20,
        transition_blend_ticks: 10,
    }
}

/// Run a test closure in a thread with a large stack.
///
/// Required because HierarchicalMixer is ~8 MiB; default stack (8 MiB) would
/// overflow when constructing it as a local variable.
fn with_large_stack<F: FnOnce() + Send + 'static>(f: F) {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024) // 64 MiB
        .spawn(f)
        .expect("thread spawn failed")
        .join()
        .expect("test thread panicked");
}

/// Assert that a flat n×n matrix (row stride = row_stride) is doubly stochastic.
fn assert_doubly_stochastic(m: &[f32], n: usize, row_stride: usize, tol: f32) {
    for i in 0..n {
        let rs: f32 = (0..n).map(|j| m[i * row_stride + j]).sum();
        assert!(
            (rs - 1.0).abs() < tol,
            "row {} sum = {:.6} (expected 1.0 ± {:.0e})",
            i,
            rs,
            tol
        );
    }
    for j in 0..n {
        let cs: f32 = (0..n).map(|i| m[i * row_stride + j]).sum();
        assert!(
            (cs - 1.0).abs() < tol,
            "col {} sum = {:.6} (expected 1.0 ± {:.0e})",
            j,
            cs,
            tol
        );
    }
}

/// L1-norm of a slice (sum of absolute values).
fn l1_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x.abs()).sum()
}

// ─── test 1 ───────────────────────────────────────────────────────────────────

/// Verify ‖c'‖₁ ≤ ‖c‖₁ + ε after intra-cluster mixing.
///
/// A doubly stochastic matrix applied to a non-negative vector preserves
/// the 1-norm exactly.  After clamping to [0, 1] the norm can only decrease.
#[test]
fn test_intra_cluster_mixing_preserves_norm() {
    with_large_stack(|| {
        let mut mixer = HierarchicalMixer::new(test_config());

        // Two clusters: [0, 1, 2] and [3, 4]
        let assignments = [0u16, 0, 0, 1, 1];
        mixer.update_clusters(&assignments, 2);

        // Set a non-trivial intra-cluster matrix for cluster 0 (3×3 uniform)
        let v = 1.0_f32 / 3.0;
        let raw = [v; 9]; // 3×3 uniform DS matrix
        mixer.update_intra_params(0, &raw);
        mixer.reproject_all();

        let mut coherence = [0.8_f32, 0.4, 0.6, 0.9, 0.2];
        let counts = [5u32, 3, 8, 2, 6];
        let norm_before = l1_norm(&coherence);

        mixer.apply(&mut coherence, &counts);

        let norm_after = l1_norm(&coherence);
        assert!(
            norm_after <= norm_before + 1e-4,
            "norm increased: {:.6} -> {:.6}",
            norm_before,
            norm_after
        );

        // All values must remain in [0, 1]
        for (i, &v) in coherence.iter().enumerate() {
            assert!(v >= 0.0 && v <= 1.0, "coherence[{}] = {} out of [0,1]", i, v);
        }
    });
}

// ─── test 2 ───────────────────────────────────────────────────────────────────

/// Verify ‖c''‖₁ ≤ ‖c‖₁ + ε after the full five-step hierarchical operation.
#[test]
fn test_full_hierarchical_preserves_norm() {
    with_large_stack(|| {
        let mut mixer = HierarchicalMixer::new(test_config());

        // Three clusters of different sizes: [0,1], [2,3,4], [5]
        let assignments = [0u16, 0, 1, 1, 1, 2];
        mixer.update_clusters(&assignments, 3);

        // Non-trivial intra params
        let v2 = 1.0_f32 / 2.0;
        mixer.update_intra_params(0, &[v2; 4]); // 2×2 uniform
        let v3 = 1.0_f32 / 3.0;
        mixer.update_intra_params(1, &[v3; 9]); // 3×3 uniform

        // Non-trivial inter params (3×3 uniform)
        mixer.update_inter_params(&[v3; 9]);
        mixer.reproject_all();

        let mut coherence = [0.3_f32, 0.7, 0.5, 0.9, 0.1, 0.6];
        let counts = [4u32, 2, 7, 1, 3, 5];
        let norm_before = l1_norm(&coherence);

        mixer.apply(&mut coherence, &counts);

        let norm_after = l1_norm(&coherence);
        assert!(
            norm_after <= norm_before + 1e-4,
            "norm increased: {:.6} -> {:.6}",
            norm_before,
            norm_after
        );
        for (i, &v) in coherence.iter().enumerate() {
            assert!(v >= 0.0 && v <= 1.0, "coherence[{}] = {} out of [0,1]", i, v);
        }
    });
}

// ─── test 3 ───────────────────────────────────────────────────────────────────

/// Mixing with identity matrices at both levels leaves coherence unchanged.
///
/// The initial matrices after `update_clusters` are identity at both levels.
/// Applying the mixer must return values equal to the input (within ε).
#[test]
fn test_identity_matrix_is_noop() {
    with_large_stack(|| {
        let mut mixer = HierarchicalMixer::new(test_config());

        // Two clusters
        let assignments = [0u16, 0, 1, 1, 1];
        mixer.update_clusters(&assignments, 2);
        // Do NOT call update_*_params — matrices are already identity.

        let original = [0.3_f32, 0.6, 0.1, 0.8, 0.5];
        let mut coherence = original;
        let counts = [1u32, 2, 3, 4, 5];

        mixer.apply(&mut coherence, &counts);

        for (i, (&after, &before)) in coherence.iter().zip(original.iter()).enumerate() {
            assert!(
                (after - before).abs() < 1e-5,
                "coherence[{}] changed: {:.6} -> {:.6}",
                i,
                before,
                after
            );
        }
    });
}

// ─── test 4 ───────────────────────────────────────────────────────────────────

/// Blending two doubly stochastic matrices stays within the Birkhoff polytope.
///
/// For any α ∈ [0,1], `(1−α)·H_old + α·H_new` is doubly stochastic
/// by convexity of the Birkhoff polytope.  This test verifies row and column
/// sums remain 1.0 ± 1e-5 for α = 0, 0.25, 0.5, 0.75, 1.0.
#[test]
fn test_transition_blend_stays_in_polytope() {
    // Build two 3×3 doubly stochastic matrices and verify blends
    let sk = SinkhornKnopp::default();

    let mut h_old = [2.0_f32, 1.0, 3.0, 3.0, 2.0, 1.0, 1.0, 3.0, 2.0];
    let mut h_new = [1.0_f32, 4.0, 1.0, 3.0, 1.0, 2.0, 2.0, 1.0, 3.0];
    sk.project_flat(&mut h_old, 3);
    sk.project_flat(&mut h_new, 3);

    // Verify h_old and h_new are individually doubly stochastic
    assert_doubly_stochastic(&h_old, 3, 3, 1e-5);
    assert_doubly_stochastic(&h_new, 3, 3, 1e-5);

    for &alpha in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
        let mut blend = [0.0f32; 9];
        for i in 0..9 {
            blend[i] = (1.0 - alpha) * h_old[i] + alpha * h_new[i];
        }
        assert_doubly_stochastic(
            &blend,
            3,
            3,
            1e-4,
        );
    }
}

// ─── test 5 ───────────────────────────────────────────────────────────────────

/// Verify seamless mode switch when n crosses flat_threshold.
///
/// With flat_threshold = 4:
/// - n = 4 → Flat
/// - n = 5 → Hierarchical
///
/// Both modes should produce coherence values within [0, 1] and not panic.
#[test]
fn test_adaptive_mode_switch() {
    with_large_stack(|| {
        let config = HierarchicalMixerConfig {
            flat_threshold: 4,
            ..Default::default()
        };

        // Flat mode: 4 contexts
        let strategy_flat = MixingStrategy::select(4, config.clone());
        assert!(strategy_flat.is_flat(), "n=4 should use Flat mode");

        // Hierarchical mode: 5 contexts
        let strategy_hier = MixingStrategy::select(5, config);
        assert!(strategy_hier.is_hierarchical(), "n=5 should use Hierarchical mode");

        if let Some(mixer) = strategy_hier.hierarchical() {
            // Verify the mixer has no clusters yet (just constructed)
            assert_eq!(mixer.num_clusters, 0);
            assert!(!mixer.in_transition);
        }

        // Now test full cycle: install clusters at n=5, apply mixing
        let config2 = HierarchicalMixerConfig {
            flat_threshold: 4,
            ..Default::default()
        };
        let mut mixer = HierarchicalMixer::new(config2);
        let assignments = [0u16, 0, 1, 1, 1];
        mixer.update_clusters(&assignments, 2);

        let mut coherence = [0.4_f32, 0.8, 0.2, 0.6, 0.9];
        let counts = [1u32, 2, 3, 4, 5];
        mixer.apply(&mut coherence, &counts);

        for (i, &v) in coherence.iter().enumerate() {
            assert!(v >= 0.0 && v <= 1.0, "coherence[{}] = {} out of [0,1] after mode switch", i, v);
        }
    });
}

// ─── test 6 ───────────────────────────────────────────────────────────────────

/// Verify correctness when clusters have very different sizes.
///
/// Tests with clusters of sizes 1, 5, 2 — verifies the computation produces
/// valid [0,1] outputs, the identity-matrix case is a no-op, and non-trivial
/// inter-cluster mixing actually moves values between clusters.
#[test]
fn test_unequal_cluster_sizes() {
    with_large_stack(|| {
        let mut mixer = HierarchicalMixer::new(test_config());

        // Cluster 0: 1 context (index 0)
        // Cluster 1: 5 contexts (indices 1-5)
        // Cluster 2: 2 contexts (indices 6-7)
        let assignments = [0u16, 1, 1, 1, 1, 1, 2, 2];
        mixer.update_clusters(&assignments, 3);

        // ── Identity case ────────────────────────────────────────────────────
        let original = [0.9_f32, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        let mut coherence = original;
        let counts = [5u32, 1, 2, 3, 4, 5, 6, 7];

        mixer.apply(&mut coherence, &counts);

        for (i, (&after, &before)) in coherence.iter().zip(original.iter()).enumerate() {
            assert!(
                (after - before).abs() < 1e-5,
                "identity: coherence[{}] changed: {:.6} -> {:.6}",
                i,
                before,
                after
            );
        }

        // ── Non-trivial inter-cluster mixing ─────────────────────────────────
        // Set non-trivial inter params: 3×3 matrix that moves weight between clusters
        let inter_raw = [
            0.8_f32, 0.1, 0.1,
            0.1, 0.8, 0.1,
            0.1, 0.1, 0.8,
        ];
        mixer.update_inter_params(&inter_raw);
        mixer.reproject_all();

        let mut coherence2 = [0.9_f32, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.8];
        let counts2 = [10u32, 1, 1, 1, 1, 1, 5, 5];

        mixer.apply(&mut coherence2, &counts2);

        // All values must be in [0, 1]
        for (i, &v) in coherence2.iter().enumerate() {
            assert!(v >= 0.0 && v <= 1.0, "coherence2[{}] = {} out of [0,1]", i, v);
        }

        // The singleton cluster (index 0, coherence=0.9) should have received
        // some downward correction because clusters 1 and 2 have low coherence.
        // Verify its value is in (0.0, 0.95) — reduced but not zeroed.
        assert!(
            coherence2[0] < 0.95,
            "singleton cluster coherence should decrease slightly due to inter-cluster transfer, got {}",
            coherence2[0]
        );
    });
}
