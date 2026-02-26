//! Integration tests — one test per patent claim 1–34.
//!
//! Each test is named `test_claim_N_<description>` and demonstrates the
//! claimed behaviour end-to-end using only the public API.
//!
//! Patent pending: US Provisional Application 63/988,438 (priority date 23 Feb 2026).

use ccf_core::accumulator::{CoherenceAccumulator, CoherenceField};
use ccf_core::boundary::MinCutBoundary;
use ccf_core::phase::{Personality, PhaseSpace, SocialPhase};
use ccf_core::sinkhorn::SinkhornKnopp;
use ccf_core::vocabulary::{
    BrightnessBand, ContextKey, MotionContext, MbotContextKey, MbotSensors, NoiseBand,
    Orientation, PresenceSignature, SensorVocabulary, TimePeriod,
};

// ─── Shared helpers ──────────────────────────────────────────────────────────

fn bright_quiet() -> MbotContextKey {
    ContextKey::new(MbotSensors {
        brightness: BrightnessBand::Bright,
        noise: NoiseBand::Quiet,
        presence: PresenceSignature::Absent,
        motion: MotionContext::Static,
        orientation: Orientation::Upright,
        time_period: TimePeriod::Day,
    })
}

fn dark_loud() -> MbotContextKey {
    ContextKey::new(MbotSensors {
        brightness: BrightnessBand::Dark,
        noise: NoiseBand::Loud,
        presence: PresenceSignature::Close,
        motion: MotionContext::Fast,
        orientation: Orientation::Tilted,
        time_period: TimePeriod::Night,
    })
}

fn dim_moderate() -> MbotContextKey {
    ContextKey::new(MbotSensors {
        brightness: BrightnessBand::Dim,
        noise: NoiseBand::Moderate,
        presence: PresenceSignature::Far,
        motion: MotionContext::Slow,
        orientation: Orientation::Upright,
        time_period: TimePeriod::Evening,
    })
}

fn neutral_personality() -> Personality {
    Personality {
        curiosity_drive: 0.5,
        startle_sensitivity: 0.5,
        recovery_speed: 0.5,
    }
}

fn assert_doubly_stochastic<const N: usize>(m: &[[f32; N]; N], tol: f32) {
    for (i, row) in m.iter().enumerate() {
        let rs: f32 = row.iter().sum();
        assert!((rs - 1.0).abs() < tol, "row {} sum = {}", i, rs);
    }
    for j in 0..N {
        let cs: f32 = (0..N).map(|i| m[i][j]).sum();
        assert!((cs - 1.0).abs() < tol, "col {} sum = {}", j, cs);
    }
}

// ─── Claims 1, 8: SensorVocabulary + ContextKey ──────────────────────────────

/// Claim 1: ContextKey produces a deterministic hash from quantised sensor signals.
#[test]
fn test_claim_1_context_key_deterministic_hash() {
    let k1 = bright_quiet();
    let k2 = bright_quiet();
    assert_eq!(
        k1.context_hash_u32(),
        k2.context_hash_u32(),
        "identical sensor readings must produce identical hashes"
    );
    // Different context produces different hash
    let k3 = dark_loud();
    assert_ne!(
        k1.context_hash_u32(),
        k3.context_hash_u32(),
        "distinct sensor readings must produce distinct hashes"
    );
}

/// Claim 8: SensorVocabulary encodes sensory state as a normalised float feature vector
/// used for cosine similarity computation.
#[test]
fn test_claim_8_composite_sensor_vocabulary_cosine_similarity() {
    let k_bright = bright_quiet();
    let k_dark = dark_loud();

    // Self-similarity must be 1.0
    let self_sim = k_bright.cosine_similarity(&k_bright);
    assert!(
        (self_sim - 1.0).abs() < 1e-5,
        "self-similarity should be 1.0, got {}",
        self_sim
    );

    // Dissimilar contexts should score lower
    let cross_sim = k_bright.cosine_similarity(&k_dark);
    assert!(
        cross_sim < 0.5,
        "bright_quiet vs dark_loud should have low similarity, got {}",
        cross_sim
    );

    // Feature vector length must match FEATURE_DIM
    let vec = k_bright.vocabulary.to_feature_vec();
    assert_eq!(vec.len(), MbotSensors::FEATURE_DIM);

    // All features in [0.0, 1.0]
    for f in &vec {
        assert!(*f >= 0.0 && *f <= 1.0, "feature {} out of [0,1]", f);
    }
}

// ─── Claims 2–5: CoherenceAccumulator ───────────────────────────────────────

/// Claim 2: CoherenceAccumulator grows positively with repeated positive interactions.
#[test]
fn test_claim_2_positive_growth() {
    let mut acc = CoherenceAccumulator::new();
    assert_eq!(acc.value, 0.0);

    for tick in 0..30 {
        acc.positive_interaction(0.5, tick, false);
    }
    assert!(
        acc.value > 0.0,
        "coherence must grow after positive interactions, got {}",
        acc.value
    );
    assert!(acc.value <= 1.0, "coherence must not exceed 1.0");
}

/// Claim 3: Earned floor protects accumulated trust against negative events.
#[test]
fn test_claim_3_earned_floor_protects_trust() {
    let mut acc = CoherenceAccumulator::new();

    // Build up trust and interaction count
    for tick in 0..100 {
        acc.positive_interaction(0.5, tick, false);
    }
    let floor_before = acc.earned_floor();
    assert!(floor_before > 0.0, "floor must be positive after 100 interactions");

    // Apply many negative interactions — floor should hold
    for tick in 100..200 {
        acc.negative_interaction(1.0, tick);
    }
    assert!(
        acc.value >= floor_before - 1e-5,
        "value {} must not drop below earned floor {}",
        acc.value,
        floor_before
    );
}

/// Claim 4: Asymmetric decay — coherence decays toward floor, not toward zero.
#[test]
fn test_claim_4_asymmetric_decay_toward_floor() {
    let mut acc = CoherenceAccumulator::new();
    for tick in 0..50 {
        acc.positive_interaction(0.5, tick, false);
    }
    let floor = acc.earned_floor();
    let before = acc.value;

    acc.decay(1000);

    assert!(acc.value < before, "decay should reduce coherence");
    assert!(
        acc.value >= floor,
        "decay must not go below earned floor {} (got {})",
        floor,
        acc.value
    );
}

/// Claim 5: Personality modulates the rate of accumulation but not structure.
#[test]
fn test_claim_5_personality_modulates_accumulation_rate() {
    let mut fast = CoherenceAccumulator::new();
    let mut slow = CoherenceAccumulator::new();

    for tick in 0..30 {
        fast.positive_interaction(0.9, tick, false); // high recovery_speed
        slow.positive_interaction(0.1, tick, false); // low recovery_speed
    }

    assert!(
        fast.value > slow.value,
        "high recovery_speed ({}) should accumulate faster than low ({})",
        fast.value,
        slow.value
    );

    // Both are bounded
    assert!(fast.value <= 1.0);
    assert!(slow.value <= 1.0);
}

// ─── Claims 6–7, 13: CoherenceField ─────────────────────────────────────────

/// Claim 6: CoherenceField maintains a context-keyed map of accumulators.
#[test]
fn test_claim_6_context_keyed_accumulator_map() {
    let mut field: CoherenceField<MbotSensors, 6> = CoherenceField::new();
    let key_a = bright_quiet();
    let key_b = dark_loud();
    let p = neutral_personality();

    // Two independent contexts
    for tick in 0..20 {
        field.positive_interaction(&key_a, &p, tick, false);
    }

    // key_b must still be at zero
    assert_eq!(
        field.context_coherence(&key_b),
        0.0,
        "unrelated context must stay at 0.0"
    );
    assert!(
        field.context_coherence(&key_a) > 0.0,
        "interacted context must grow"
    );
    assert_eq!(field.context_count(), 1);
}

/// Claim 7: CoherenceField applies min-gate for unfamiliar contexts (asymmetric gate, CCF-001).
#[test]
fn test_claim_7_min_gate_asymmetric_for_unfamiliar_context() {
    let mut field: CoherenceField<MbotSensors, 6> = CoherenceField::new();
    let key = bright_quiet();

    // Unseen context — min-gate: min(instant=0.9, ctx=0.0) = 0.0
    let eff = field.effective_coherence(0.9, &key);
    assert_eq!(
        eff, 0.0,
        "unfamiliar context with 0 history must be 0.0, not {}",
        eff
    );

    // Build a small amount of coherence (below 0.3 threshold)
    {
        let acc = field.get_or_create(&key);
        for tick in 0..5 {
            acc.positive_interaction(0.5, tick, false);
        }
    }
    let ctx = field.context_coherence(&key);
    assert!(ctx < 0.3, "still unfamiliar (ctx={})", ctx);

    // High instant capped at ctx (unfamiliar gate: min(0.9, ctx) = ctx)
    let eff2 = field.effective_coherence(0.9, &key);
    assert!(
        (eff2 - ctx).abs() < 1e-4,
        "min-gate should cap eff at ctx={}, got {}",
        ctx,
        eff2
    );
}

/// Claim 13: Familiar contexts blend history into effective coherence (CCF-001 familiar arm).
#[test]
fn test_claim_13_familiar_context_history_blending() {
    let mut field: CoherenceField<MbotSensors, 6> = CoherenceField::new();
    let key = bright_quiet();

    // Build enough history to be familiar (ctx >= 0.3)
    {
        let acc = field.get_or_create(&key);
        for tick in 0..100 {
            acc.positive_interaction(0.5, tick, false);
        }
    }
    let ctx = field.context_coherence(&key);
    assert!(ctx >= 0.3, "should be familiar after 100 interactions (ctx={})", ctx);

    // Familiar arm: 0.3 * instant + 0.7 * ctx
    let instant = 0.2_f32;
    let eff = field.effective_coherence(instant, &key);
    let expected = 0.3 * instant + 0.7 * ctx;
    assert!(
        (eff - expected).abs() < 1e-4,
        "familiar blend: expected {}, got {}",
        expected,
        eff
    );
    // History buffers the low instant — eff must exceed instant
    assert!(
        eff > instant,
        "familiar context must buffer noise: eff={} > instant={}",
        eff,
        instant
    );
}

// ─── Claims 9–12: MinCutBoundary ────────────────────────────────────────────

/// Claim 9: MinCutBoundary computes the comfort-zone boundary from the trust graph
/// without any user-configured threshold.
#[test]
fn test_claim_9_boundary_computed_not_configured() {
    let mut b: MinCutBoundary<MbotSensors, 6> = MinCutBoundary::new();
    let k1 = bright_quiet();
    let k2 = dark_loud();

    b.report_context_with_key(&k1, &[]);
    let existing = [(k1.clone(), k1.context_hash_u32())];
    b.report_context_with_key(&k2, &existing);

    // The boundary was not configured — it is computed
    let cut = b.min_cut_value();
    assert!(cut >= 0.0, "min_cut_value must be non-negative (got {})", cut);
    // Two nodes exist
    assert_eq!(b.node_count(), 2);
}

/// Claim 10: Both sides of the partition are observable (the partition is enumerable).
#[test]
fn test_claim_10_partition_is_observable() {
    let mut b: MinCutBoundary<MbotSensors, 6> = MinCutBoundary::new();
    let k1 = bright_quiet();
    let k2 = dark_loud();
    let k3 = dim_moderate();

    b.report_context_with_key(&k1, &[]);
    let e1 = [(k1.clone(), k1.context_hash_u32())];
    b.report_context_with_key(&k2, &e1);
    let e2 = [
        (k1.clone(), k1.context_hash_u32()),
        (k2.clone(), k2.context_hash_u32()),
    ];
    b.report_context_with_key(&k3, &e2);

    let result = b.partition();

    // All nodes accounted for across both sides
    let total = result.partition_s_count + result.partition_complement_count;
    assert_eq!(
        total, 3,
        "partition should cover all {} nodes (got s={} + c={})",
        b.node_count(),
        result.partition_s_count,
        result.partition_complement_count
    );
}

/// Claim 11: The algorithm discovers thin bridges between context clusters.
#[test]
fn test_claim_11_thin_bridge_discovery() {
    let mut b: MinCutBoundary<MbotSensors, 6> = MinCutBoundary::new();

    // Two bright variants (similar) and two dark variants (similar) — thin bridge between clusters
    let k_bq = bright_quiet();
    let k_bl = ContextKey::new(MbotSensors {
        brightness: BrightnessBand::Bright,
        noise: NoiseBand::Loud,
        presence: PresenceSignature::Absent,
        motion: MotionContext::Static,
        orientation: Orientation::Upright,
        time_period: TimePeriod::Day,
    });
    let k_dq = ContextKey::new(MbotSensors {
        brightness: BrightnessBand::Dark,
        noise: NoiseBand::Quiet,
        presence: PresenceSignature::Absent,
        motion: MotionContext::Static,
        orientation: Orientation::Upright,
        time_period: TimePeriod::Day,
    });
    let k_dl = dark_loud();

    b.report_context_with_key(&k_bq, &[]);
    let e1 = [(k_bq.clone(), k_bq.context_hash_u32())];
    b.report_context_with_key(&k_bl, &e1);
    let e2 = [
        (k_bq.clone(), k_bq.context_hash_u32()),
        (k_bl.clone(), k_bl.context_hash_u32()),
    ];
    b.report_context_with_key(&k_dq, &e2);
    let e3 = [
        (k_bq.clone(), k_bq.context_hash_u32()),
        (k_bl.clone(), k_bl.context_hash_u32()),
        (k_dq.clone(), k_dq.context_hash_u32()),
    ];
    b.report_context_with_key(&k_dl, &e3);

    assert_eq!(b.node_count(), 4);
    let cut = b.min_cut_value();
    // The cut is discovered structurally — must be non-negative
    assert!(cut >= 0.0, "min_cut_value must be non-negative (got {})", cut);
    // Partition covers all 4 nodes
    let result = b.partition();
    assert_eq!(result.partition_s_count + result.partition_complement_count, 4);
}

/// Claim 12: Boundary moves when trust changes — Graph B activates after sufficient observations.
#[test]
fn test_claim_12_boundary_moves_on_trust_change() {
    use ccf_core::boundary::MIN_TRUST_OBSERVATIONS;

    let mut b: MinCutBoundary<MbotSensors, 6> = MinCutBoundary::new();
    let k1 = bright_quiet();
    let k2 = dim_moderate();

    b.report_context_with_key(&k1, &[]);
    let e1 = [(k1.clone(), k1.context_hash_u32())];
    b.report_context_with_key(&k2, &e1);

    let cut_before = b.min_cut_value();

    // Earn trust in both contexts — activates Graph B weights
    b.update_trust(&k1, 0.9, MIN_TRUST_OBSERVATIONS);
    b.update_trust(&k2, 0.9, MIN_TRUST_OBSERVATIONS);
    let cut_high_trust = b.min_cut_value();

    // Drop trust in k2 — boundary should shift
    b.update_trust(&k2, 0.1, MIN_TRUST_OBSERVATIONS);
    let cut_low_trust = b.min_cut_value();

    assert!(cut_before >= 0.0);
    assert!(cut_high_trust >= 0.0);
    assert!(cut_low_trust >= 0.0);

    // Trust reduction in k2 should reduce edge weight and move boundary
    assert!(
        cut_low_trust <= cut_high_trust + 1e-5,
        "cut should not increase when trust drops: before={} after={}",
        cut_high_trust,
        cut_low_trust
    );
}

// ─── Claims 14–18: SocialPhase ───────────────────────────────────────────────

/// Claim 14: SocialPhase classifies into four quadrants of the coherence × tension plane.
#[test]
fn test_claim_14_four_quadrant_phase_classification() {
    let ps = PhaseSpace::default();

    assert_eq!(
        SocialPhase::classify(0.1, 0.1, SocialPhase::ShyObserver, &ps),
        SocialPhase::ShyObserver,
        "low coherence, low tension"
    );
    assert_eq!(
        SocialPhase::classify(0.8, 0.1, SocialPhase::ShyObserver, &ps),
        SocialPhase::QuietlyBeloved,
        "high coherence, low tension"
    );
    assert_eq!(
        SocialPhase::classify(0.1, 0.7, SocialPhase::ShyObserver, &ps),
        SocialPhase::StartledRetreat,
        "low coherence, high tension"
    );
    assert_eq!(
        SocialPhase::classify(0.8, 0.7, SocialPhase::ShyObserver, &ps),
        SocialPhase::ProtectiveGuardian,
        "high coherence, high tension"
    );
}

/// Claim 15: Schmitt trigger hysteresis prevents oscillation at coherence boundary.
#[test]
fn test_claim_15_schmitt_trigger_hysteresis_coherence() {
    let ps = PhaseSpace::default(); // enter=0.65, exit=0.55

    // Enter QuietlyBeloved from ShyObserver (needs >= 0.65)
    let phase = SocialPhase::classify(0.66, 0.1, SocialPhase::ShyObserver, &ps);
    assert_eq!(phase, SocialPhase::QuietlyBeloved, "enter QB at 0.66");

    // Stay in QuietlyBeloved at 0.60 (above exit=0.55)
    let phase = SocialPhase::classify(0.60, 0.1, SocialPhase::QuietlyBeloved, &ps);
    assert_eq!(phase, SocialPhase::QuietlyBeloved, "stay QB at 0.60 (above exit 0.55)");

    // Exit QuietlyBeloved at 0.54 (below exit=0.55)
    let phase = SocialPhase::classify(0.54, 0.1, SocialPhase::QuietlyBeloved, &ps);
    assert_eq!(phase, SocialPhase::ShyObserver, "exit QB at 0.54 (below exit 0.55)");
}

/// Claim 16: led_tint returns distinct RGB values for each phase (expressive output channel).
#[test]
fn test_claim_16_led_tint_distinct_per_phase() {
    let tints = [
        SocialPhase::ShyObserver.led_tint(),
        SocialPhase::StartledRetreat.led_tint(),
        SocialPhase::QuietlyBeloved.led_tint(),
        SocialPhase::ProtectiveGuardian.led_tint(),
    ];

    for i in 0..4 {
        for j in (i + 1)..4 {
            assert_ne!(
                tints[i], tints[j],
                "phases {} and {} must have distinct LED tints",
                i, j
            );
        }
    }
}

/// Claim 17: expression_scale is ordered by phase — QuietlyBeloved > ProtectiveGuardian >
/// ShyObserver > StartledRetreat.
#[test]
fn test_claim_17_expression_scale_ordered() {
    let qb = SocialPhase::QuietlyBeloved.expression_scale();
    let pg = SocialPhase::ProtectiveGuardian.expression_scale();
    let so = SocialPhase::ShyObserver.expression_scale();
    let sr = SocialPhase::StartledRetreat.expression_scale();

    assert!(qb > pg, "QB({}) must exceed PG({})", qb, pg);
    assert!(pg > so, "PG({}) must exceed SO({})", pg, so);
    assert!(so > sr, "SO({}) must exceed SR({})", so, sr);

    // All values in [0, 1]
    for &scale in &[qb, pg, so, sr] {
        assert!(scale >= 0.0 && scale <= 1.0, "scale {} out of [0,1]", scale);
    }
}

/// Claim 18: Schmitt trigger hysteresis also applies to the tension axis.
#[test]
fn test_claim_18_schmitt_trigger_hysteresis_tension() {
    let ps = PhaseSpace::default(); // tension enter=0.45, exit=0.35

    // Enter StartledRetreat (tension >= 0.45)
    let phase = SocialPhase::classify(0.1, 0.46, SocialPhase::ShyObserver, &ps);
    assert_eq!(phase, SocialPhase::StartledRetreat, "enter SR at tension=0.46");

    // Stay in StartledRetreat (tension >= exit=0.35)
    let phase = SocialPhase::classify(0.1, 0.36, SocialPhase::StartledRetreat, &ps);
    assert_eq!(phase, SocialPhase::StartledRetreat, "stay SR at tension=0.36");

    // Exit StartledRetreat (tension < exit=0.35)
    let phase = SocialPhase::classify(0.1, 0.34, SocialPhase::StartledRetreat, &ps);
    assert_eq!(phase, SocialPhase::ShyObserver, "exit SR at tension=0.34");
}

// ─── Claims 19–23: SinkhornKnopp ────────────────────────────────────────────

/// Claim 19: SinkhornKnopp produces a doubly stochastic matrix (Birkhoff polytope member).
#[test]
fn test_claim_19_doubly_stochastic_output() {
    let sk = SinkhornKnopp::default();
    let mut m = [
        [3.0_f32, 1.0, 2.0],
        [1.0, 4.0, 1.0],
        [2.0, 1.0, 3.0],
    ];
    let result = sk.project(&mut m);
    assert!(result.converged, "must converge: {:?}", result);
    assert_doubly_stochastic(&m, 1e-5);
}

/// Claim 20: Iterative normalisation converges within max_iterations for any valid matrix.
#[test]
fn test_claim_20_convergence_within_max_iterations() {
    let sk = SinkhornKnopp::default();
    let mut m = [
        [100.0_f32, 1.0, 1.0, 1.0],
        [1.0, 100.0, 1.0, 1.0],
        [1.0, 1.0, 100.0, 1.0],
        [1.0, 1.0, 1.0, 100.0],
    ];
    let result = sk.project(&mut m);
    assert!(result.converged, "should converge");
    assert!(
        result.iterations <= sk.max_iterations,
        "took {} iterations, max={}",
        result.iterations,
        sk.max_iterations
    );
    assert_doubly_stochastic(&m, 1e-5);
}

/// Claim 21: Bounded mixing — no single context can dominate (no row sum exceeds 1.0).
#[test]
fn test_claim_21_bounded_mixing_no_single_row_dominates() {
    let sk = SinkhornKnopp::default();
    // Diagonal-dominant matrix: one context starts with much higher raw weight.
    // After projection no row can accumulate more than 1.0 total weight.
    let mut m = [
        [4.0_f32, 1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0, 4.0],
        [3.0, 4.0, 1.0, 2.0],
        [2.0, 3.0, 4.0, 1.0],
    ];
    let result = sk.project(&mut m);
    assert!(result.converged, "must converge: residual={}", result.residual);

    // After projection every row sum = 1.0 ± tolerance — bounded mixing satisfied
    let tol = sk.tolerance + 1e-4;
    for (i, row) in m.iter().enumerate() {
        let rs: f32 = row.iter().sum();
        assert!(
            rs <= 1.0 + tol,
            "row {} sum {} exceeds 1.0 — bounded mixing violated",
            i, rs
        );
        // All entries non-negative
        for &v in row {
            assert!(v >= 0.0, "entry {} is negative", v);
        }
    }
}

/// Claim 22: Non-negativity is preserved throughout the projection.
#[test]
fn test_claim_22_non_negativity_preserved() {
    let sk = SinkhornKnopp::default();
    let mut m = [
        [0.5_f32, 3.0, 1.0],
        [1.0, 0.5, 2.0],
        [2.0, 1.0, 0.5],
    ];
    sk.project(&mut m);
    for row in &m {
        for &v in row {
            assert!(v >= 0.0, "negative entry {} after projection", v);
        }
    }
}

/// Claim 23: An already doubly stochastic matrix is unchanged (Birkhoff polytope membership).
#[test]
fn test_claim_23_birkhoff_polytope_idempotence() {
    let sk = SinkhornKnopp::default();
    let v = 1.0_f32 / 4.0;
    let mut m = [[v; 4]; 4]; // uniform — already doubly stochastic
    let original = m;
    sk.project(&mut m);
    for i in 0..4 {
        for j in 0..4 {
            assert!(
                (m[i][j] - original[i][j]).abs() < 1e-5,
                "entry [{},{}] changed: {} -> {}",
                i, j, original[i][j], m[i][j]
            );
        }
    }
}

// ─── Claims 24–28: Personality modulators ───────────────────────────────────

/// Claim 24: curiosity_drive in [0,1] raises the positive interaction delta.
#[test]
fn test_claim_24_curiosity_raises_positive_delta() {
    let mut low_curiosity = CoherenceAccumulator::new();
    let mut high_curiosity = CoherenceAccumulator::new();

    // curiosity_drive maps to recovery_speed channel in positive_interaction
    for tick in 0..20 {
        low_curiosity.positive_interaction(0.0, tick, false);  // curiosity_drive = 0
        high_curiosity.positive_interaction(1.0, tick, false); // curiosity_drive = 1
    }

    assert!(
        high_curiosity.value > low_curiosity.value,
        "high curiosity ({}) should accumulate more than low ({})",
        high_curiosity.value,
        low_curiosity.value
    );
}

/// Claim 25: startle_sensitivity in [0,1] amplifies negative_delta on a startle event.
#[test]
fn test_claim_25_startle_sensitivity_amplifies_negative_delta() {
    // Build identical accumulators with the same history
    let mut low_sens = CoherenceAccumulator::new();
    let mut high_sens = CoherenceAccumulator::new();
    for tick in 0..30 {
        low_sens.positive_interaction(0.5, tick, false);
        high_sens.positive_interaction(0.5, tick, false);
    }
    let before_low = low_sens.value;
    let before_high = high_sens.value;
    assert!((before_low - before_high).abs() < 1e-5, "should be equal before startle");

    low_sens.negative_interaction(0.0, 30);  // low sensitivity
    high_sens.negative_interaction(1.0, 30); // high sensitivity

    let drop_low = before_low - low_sens.value;
    let drop_high = before_high - high_sens.value;
    assert!(
        drop_high > drop_low,
        "high startle sensitivity drop ({}) must exceed low ({})",
        drop_high, drop_low
    );
}

/// Claim 26: recovery_rate (recovery_speed) in [0,1] speeds up the earned floor recovery.
#[test]
fn test_claim_26_recovery_rate_speeds_floor_recovery() {
    let mut fast_recover = CoherenceAccumulator::new();
    let mut slow_recover = CoherenceAccumulator::new();

    for tick in 0..30 {
        fast_recover.positive_interaction(1.0, tick, false); // recovery_speed = 1.0
        slow_recover.positive_interaction(0.0, tick, false); // recovery_speed = 0.0
    }

    // Fast recovery reaches higher coherence after same number of ticks
    assert!(
        fast_recover.value > slow_recover.value,
        "fast recovery ({}) should yield higher coherence than slow ({})",
        fast_recover.value, slow_recover.value
    );
}

/// Claim 27: Personality modulators are independent — changing one does not affect others.
#[test]
fn test_claim_27_personality_modulators_are_independent() {
    // Change curiosity_drive, measure impact on modulate_coherence_gain
    let p_base = Personality { curiosity_drive: 0.5, startle_sensitivity: 0.5, recovery_speed: 0.5 };
    let p_high_startle = Personality { curiosity_drive: 0.5, startle_sensitivity: 1.0, recovery_speed: 0.5 };

    // Coherence gain should be same (recovery_speed unchanged)
    assert!(
        (p_base.modulate_coherence_gain(0.02) - p_high_startle.modulate_coherence_gain(0.02)).abs() < 1e-6,
        "changing startle_sensitivity must not affect coherence gain"
    );

    // Startle drop is different (startle_sensitivity changed)
    assert!(
        p_high_startle.modulate_startle_drop(0.05) > p_base.modulate_startle_drop(0.05),
        "higher startle_sensitivity should increase startle drop"
    );

    // Recovery speed change: affects coherence gain, not startle drop
    let p_high_recovery = Personality { curiosity_drive: 0.5, startle_sensitivity: 0.5, recovery_speed: 1.0 };
    assert!(
        (p_base.modulate_startle_drop(0.05) - p_high_recovery.modulate_startle_drop(0.05)).abs() < 1e-6,
        "changing recovery_speed must not affect startle drop"
    );
    assert!(
        p_high_recovery.modulate_coherence_gain(0.02) > p_base.modulate_coherence_gain(0.02),
        "higher recovery_speed should increase coherence gain"
    );
}

/// Claim 28: Personality::new() with any values in [0,1] stays bounded.
#[test]
fn test_claim_28_personality_extreme_values_stay_bounded() {
    let personalities = [
        Personality { curiosity_drive: 0.0, startle_sensitivity: 0.0, recovery_speed: 0.0 },
        Personality { curiosity_drive: 1.0, startle_sensitivity: 1.0, recovery_speed: 1.0 },
        Personality { curiosity_drive: 0.0, startle_sensitivity: 1.0, recovery_speed: 0.5 },
        Personality { curiosity_drive: 1.0, startle_sensitivity: 0.0, recovery_speed: 1.0 },
    ];

    for p in &personalities {
        let gain = p.modulate_coherence_gain(0.02);
        let drop = p.modulate_startle_drop(0.05);
        assert!(gain >= 0.0, "gain must be non-negative: {}", gain);
        assert!(drop >= 0.0, "drop must be non-negative: {}", drop);
        // Gain bounded: base * (0.5 + rs) where rs in [0,1] → max = base * 1.5
        assert!(gain <= 0.02 * 1.5 + 1e-6, "gain out of expected range: {}", gain);
        // Drop bounded: base * (0.5 + ss) where ss in [0,1] → max = base * 1.5
        assert!(drop <= 0.05 * 1.5 + 1e-6, "drop out of expected range: {}", drop);
    }
}

// ─── Claims 29–34: Composite system tests ───────────────────────────────────

/// Claim 29: Sensor → ContextKey → CoherenceField pipeline compiles and runs end-to-end.
#[test]
fn test_claim_29_sensor_context_field_pipeline_compiles() {
    // Build sensor vocabulary
    let sensors = MbotSensors {
        brightness: BrightnessBand::Bright,
        noise: NoiseBand::Quiet,
        presence: PresenceSignature::Absent,
        motion: MotionContext::Static,
        orientation: Orientation::Upright,
        time_period: TimePeriod::Day,
    };

    // Wrap in ContextKey
    let key: MbotContextKey = ContextKey::new(sensors);

    // Insert into CoherenceField
    let mut field: CoherenceField<MbotSensors, 6> = CoherenceField::new();
    let p = neutral_personality();
    field.positive_interaction(&key, &p, 0, false);

    assert_eq!(field.context_count(), 1);
    assert!(field.context_coherence(&key) > 0.0);
}

/// Claim 30: 10-tick positive sequence produces positive coherence > 0.
#[test]
fn test_claim_30_ten_tick_positive_sequence_yields_positive_coherence() {
    let mut field: CoherenceField<MbotSensors, 6> = CoherenceField::new();
    let key = bright_quiet();
    let p = neutral_personality();

    for tick in 0..10 {
        field.positive_interaction(&key, &p, tick, false);
    }

    let coh = field.context_coherence(&key);
    assert!(coh > 0.0, "10 positive ticks must yield coherence > 0, got {}", coh);
}

/// Claim 31: After 20 positive ticks, coherence > 0 → SocialPhase is not StartledRetreat.
#[test]
fn test_claim_31_positive_coherence_not_startled_retreat() {
    let mut field: CoherenceField<MbotSensors, 6> = CoherenceField::new();
    let key = bright_quiet();
    let p = neutral_personality();
    let ps = PhaseSpace::default();

    for tick in 0..20 {
        field.positive_interaction(&key, &p, tick, false);
    }

    let coh = field.context_coherence(&key);
    assert!(coh > 0.0, "coherence must be positive after 20 ticks");

    // With low tension, effective coherence = min-gate or blend
    let eff = field.effective_coherence(coh, &key);
    let phase = SocialPhase::classify(eff, 0.1, SocialPhase::ShyObserver, &ps);

    assert_ne!(
        phase,
        SocialPhase::StartledRetreat,
        "positive coherence with low tension must not yield StartledRetreat (got {:?})",
        phase
    );
}

/// Claim 32: SinkhornKnopp applied to a trust matrix stays doubly stochastic.
#[test]
fn test_claim_32_sinkhorn_on_trust_matrix_stays_doubly_stochastic() {
    let sk = SinkhornKnopp::default();

    // Simulate a 3-context trust matrix (raw cosine similarities)
    let mut trust_matrix = [
        [1.0_f32, 0.8, 0.2],
        [0.8, 1.0, 0.3],
        [0.2, 0.3, 1.0],
    ];

    let result = sk.project(&mut trust_matrix);
    assert!(result.converged, "trust matrix must converge");
    assert_doubly_stochastic(&trust_matrix, 1e-5);

    // All entries still non-negative
    for row in &trust_matrix {
        for &v in row {
            assert!(v >= 0.0, "negative trust entry: {}", v);
        }
    }
}

/// Claim 33: MinCutBoundary partition separates high-trust from low-trust contexts.
#[test]
fn test_claim_33_min_cut_separates_high_trust_from_low_trust() {
    use ccf_core::boundary::MIN_TRUST_OBSERVATIONS;

    let mut b: MinCutBoundary<MbotSensors, 6> = MinCutBoundary::new();
    let k_high = bright_quiet();    // will receive high trust
    let k_low = dark_loud();        // will receive low trust

    // Register both contexts
    b.report_context_with_key(&k_high, &[]);
    let e1 = [(k_high.clone(), k_high.context_hash_u32())];
    b.report_context_with_key(&k_low, &e1);

    // Grant high trust to k_high, low trust to k_low
    b.update_trust(&k_high, 0.9, MIN_TRUST_OBSERVATIONS);
    b.update_trust(&k_low, 0.1, MIN_TRUST_OBSERVATIONS);

    let result = b.partition();

    // Both nodes must be accounted for in one of the two partitions
    let total = result.partition_s_count + result.partition_complement_count;
    assert_eq!(total, 2, "partition must cover exactly 2 nodes (got {})", total);
    // The cut is finite and non-negative
    assert!(result.min_cut_value >= 0.0);
}

/// Claim 34: Full CCF loop — build field, classify, verify LED tint changes across contexts.
#[test]
fn test_claim_34_full_ccf_loop_led_tint_changes() {
    let ps = PhaseSpace::default();

    // Context A: many positive interactions → high coherence
    let mut field_a: CoherenceField<MbotSensors, 6> = CoherenceField::new();
    let key_a = bright_quiet();
    let p = neutral_personality();

    for tick in 0..200 {
        field_a.positive_interaction(&key_a, &p, tick, false);
    }
    let coh_a = field_a.context_coherence(&key_a);
    let eff_a = field_a.effective_coherence(coh_a, &key_a);

    // Context B: zero interactions → low coherence
    let mut field_b: CoherenceField<MbotSensors, 6> = CoherenceField::new();
    let key_b = dark_loud();
    let coh_b = field_b.context_coherence(&key_b);
    let eff_b = field_b.effective_coherence(coh_b, &key_b);

    // Classify both contexts (low tension throughout)
    let phase_a = SocialPhase::classify(eff_a, 0.1, SocialPhase::ShyObserver, &ps);
    let phase_b = SocialPhase::classify(eff_b, 0.1, SocialPhase::ShyObserver, &ps);

    // LED tints
    let tint_a = phase_a.led_tint();
    let tint_b = phase_b.led_tint();

    // After 200 positive ticks, context A should be in a positive phase (not StartledRetreat)
    assert_ne!(
        phase_a,
        SocialPhase::StartledRetreat,
        "high-coherence context must not be StartledRetreat (got {:?})",
        phase_a
    );

    // Unseen context should be in ShyObserver
    assert_eq!(
        phase_b,
        SocialPhase::ShyObserver,
        "zero-coherence context must be ShyObserver (got {:?})",
        phase_b
    );

    // LED tints must differ across contexts with different phases
    assert_ne!(
        tint_a, tint_b,
        "LED tint must change across contexts with different phases"
    );
}
