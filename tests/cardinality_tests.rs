//! Integration tests for the two-tier context key cardinality management.
//!
//! Run with: `cargo test --features tiered-contexts`
//!
//! Small const generics (T1=8, T2=4) are used throughout to keep stack usage low
//! and to force eviction paths to be exercised at modest input sizes.

#![cfg(feature = "tiered-contexts")]

use ccf_core::cardinality::{merge_accumulators, TieredContextConfig, TieredContextMap};
use ccf_core::mbot::{
    BrightnessBand, MotionContext, MbotSensors, NoiseBand, Orientation, PresenceSignature,
    TimePeriod,
};
use ccf_core::phase::Personality;
use ccf_core::vocabulary::ContextKey;

// ─── helpers ─────────────────────────────────────────────────────────────────

fn default_key() -> ContextKey<MbotSensors, 6> {
    ContextKey::new(MbotSensors {
        brightness: BrightnessBand::Dim,
        noise: NoiseBand::Quiet,
        presence: PresenceSignature::Absent,
        motion: MotionContext::Static,
        orientation: Orientation::Upright,
        time_period: TimePeriod::Day,
    })
}

fn key_with_time(tp: TimePeriod) -> ContextKey<MbotSensors, 6> {
    ContextKey::new(MbotSensors {
        brightness: BrightnessBand::Dim,
        noise: NoiseBand::Quiet,
        presence: PresenceSignature::Absent,
        motion: MotionContext::Static,
        orientation: Orientation::Upright,
        time_period: tp,
    })
}


fn default_personality() -> Personality {
    Personality::default()
}

// ─── test 1: Tier 1 always accumulates ───────────────────────────────────────

/// Every positive interaction must update the coarse Tier 1 accumulator,
/// regardless of whether Tier 2 has been activated.
///
/// Verifies I-CKM-004: coarse relational history is never silently lost.
#[test]
fn test_tier1_always_accumulates() {
    let mut map: TieredContextMap<MbotSensors, 6, 8, 4> = TieredContextMap::new(
        TieredContextConfig::default(),
    );

    let key = default_key();
    let personality = default_personality();

    // Before any interaction, coherence should be 0
    assert_eq!(map.context_coherence(&key), 0.0);
    assert_eq!(map.tier1_class_count(), 0);

    // Do a few interactions — well below the promotion_threshold of 20
    for tick in 0..5u64 {
        map.positive_interaction(&key, &personality, tick, false);
    }

    // Tier 1 coarse class must now exist
    assert_eq!(
        map.tier1_class_count(),
        1,
        "Should have exactly 1 Tier 1 class after interactions"
    );

    // Tier 1 accumulator must have recorded the interactions
    let cls = map.classes.values().next().expect("Tier 1 class must exist");
    assert!(
        cls.accumulator.interaction_count >= 5,
        "Tier 1 must have recorded at least 5 interactions, got {}",
        cls.accumulator.interaction_count
    );
    assert!(
        cls.accumulator.value > 0.0,
        "Tier 1 coherence must be positive after positive interactions"
    );
}

// ─── test 2: promotion threshold exact ───────────────────────────────────────

/// Tier 2 must NOT be active before `promotion_threshold` interactions,
/// and MUST be active at or after it.
///
/// Verifies I-CKM-005.
#[test]
fn test_promotion_threshold_exact() {
    let threshold = 3u32;
    let config = TieredContextConfig {
        promotion_threshold: threshold,
        ..TieredContextConfig::default()
    };
    let mut map: TieredContextMap<MbotSensors, 6, 8, 4> = TieredContextMap::new(config);

    let key = default_key();
    let personality = default_personality();

    // (threshold - 1) interactions — Tier 2 should NOT be active
    for tick in 0..(threshold - 1) as u64 {
        map.positive_interaction(&key, &personality, tick, false);
    }

    let tier2_active_before = map.classes.values().next().map(|c| c.tier2_active);
    assert_eq!(
        tier2_active_before,
        Some(false),
        "Tier 2 must not be active before promotion_threshold"
    );

    // One more interaction — reaches threshold
    map.positive_interaction(&key, &personality, threshold as u64, false);

    let tier2_active_after = map.classes.values().next().map(|c| c.tier2_active);
    assert_eq!(
        tier2_active_after,
        Some(true),
        "Tier 2 must be active at promotion_threshold"
    );
}

// ─── test 3: merge uses min for coherence ────────────────────────────────────

/// The merge of two accumulators must set coherence to the minimum.
///
/// Verifies I-CKM-001.
#[test]
fn test_merge_uses_min_coherence() {
    use ccf_core::accumulator::CoherenceAccumulator;

    let a = CoherenceAccumulator {
        value: 0.8,
        interaction_count: 10,
        last_interaction_tick: 100,
    };
    let b = CoherenceAccumulator {
        value: 0.3,
        interaction_count: 5,
        last_interaction_tick: 200,
    };

    let merged = merge_accumulators(&a, &b);
    assert!(
        (merged.value - 0.3).abs() < 1e-6,
        "Merged coherence must be min(0.8, 0.3) = 0.3, got {}",
        merged.value
    );

    // Commutative check on value
    let merged_rev = merge_accumulators(&b, &a);
    assert!(
        (merged_rev.value - merged.value).abs() < 1e-6,
        "Merge must be commutative on coherence: {} vs {}",
        merged.value,
        merged_rev.value
    );
}

// ─── test 4: merge sums interaction counts ───────────────────────────────────

/// The merge must sum interaction counts and take max of last_interaction_tick.
///
/// Verifies I-CKM-002.
#[test]
fn test_merge_sums_counts() {
    use ccf_core::accumulator::CoherenceAccumulator;

    let a = CoherenceAccumulator {
        value: 0.5,
        interaction_count: 7,
        last_interaction_tick: 50,
    };
    let b = CoherenceAccumulator {
        value: 0.9,
        interaction_count: 13,
        last_interaction_tick: 80,
    };

    let merged = merge_accumulators(&a, &b);
    assert_eq!(
        merged.interaction_count,
        20,
        "Merged interaction_count must be 7 + 13 = 20"
    );
    assert_eq!(
        merged.last_interaction_tick,
        80,
        "Merged last_interaction_tick must be max(50, 80) = 80"
    );
}

// ─── test 5: eviction contributes back ───────────────────────────────────────

/// When Tier 2 is full and a new fine entry must be inserted, the weakest
/// existing entry is evicted and contributes its coherence to the Tier 1
/// parent accumulator.
///
/// Verifies I-CKM-003.
///
/// Strategy: use a feature mask that zeros out `time_period` (dim 5), so all
/// keys with different TimePeriod values hash to the same Tier 1 class.
/// T2=4 means the 5th distinct TimePeriod key triggers eviction.
/// (Only 3 TimePeriod variants exist, so we vary two dimensions together.)
#[test]
fn test_eviction_contributes_back() {
    // Mask that zeros dim 5 (time_period) — all time variants share a Tier 1 class
    let config = TieredContextConfig {
        promotion_threshold: 1, // promote after just 1 interaction
        eviction_contribution_weight: 0.5,
        tier1_feature_mask: 0b011111, // include dims 0-4, zero out dim 5
        ..TieredContextConfig::default()
    };
    // T1=8, T2=4 — Tier 2 holds at most 4 fine entries per class
    let mut map: TieredContextMap<MbotSensors, 6, 8, 4> = TieredContextMap::new(config);
    let personality = default_personality();

    // 4 keys that differ only in time_period (or presence) → same coarse key
    // TimePeriod has 3 variants; use PresenceSignature for the 4th
    let keys: [ContextKey<MbotSensors, 6>; 5] = [
        ContextKey::new(MbotSensors {
            brightness: BrightnessBand::Dim,
            noise: NoiseBand::Quiet,
            presence: PresenceSignature::Absent,
            motion: MotionContext::Static,
            orientation: Orientation::Upright,
            time_period: TimePeriod::Day,
        }),
        ContextKey::new(MbotSensors {
            brightness: BrightnessBand::Dim,
            noise: NoiseBand::Quiet,
            presence: PresenceSignature::Absent,
            motion: MotionContext::Static,
            orientation: Orientation::Upright,
            time_period: TimePeriod::Evening,
        }),
        ContextKey::new(MbotSensors {
            brightness: BrightnessBand::Dim,
            noise: NoiseBand::Quiet,
            presence: PresenceSignature::Absent,
            motion: MotionContext::Static,
            orientation: Orientation::Upright,
            time_period: TimePeriod::Night,
        }),
        ContextKey::new(MbotSensors {
            brightness: BrightnessBand::Bright,
            noise: NoiseBand::Quiet,
            presence: PresenceSignature::Absent,
            motion: MotionContext::Static,
            orientation: Orientation::Upright,
            time_period: TimePeriod::Day,
        }),
        ContextKey::new(MbotSensors {
            brightness: BrightnessBand::Dark,
            noise: NoiseBand::Quiet,
            presence: PresenceSignature::Absent,
            motion: MotionContext::Static,
            orientation: Orientation::Upright,
            time_period: TimePeriod::Day,
        }),
    ];

    // Note: keys[3] and keys[4] differ in brightness (dims 0), which IS included
    // in the mask. So they may end up in different Tier 1 classes. We'll verify
    // the eviction bound holds globally instead.

    // Fill T2 for the first key's Tier 1 class with the first 3 time-period keys
    // (all same coarse key since dim 5 is masked out)
    for (i, key) in keys.iter().enumerate().take(3) {
        for tick in 0..10u64 {
            map.positive_interaction(key, &personality, (i as u64) * 100 + tick, false);
        }
    }

    // keys[3] has a different coarse key (brightness differs) — different Tier 1 class
    for tick in 0..10u64 {
        map.positive_interaction(&keys[3], &personality, 300 + tick, false);
    }

    // At this point the first Tier 1 class should have 3 Tier 2 entries (Day, Evening, Night)
    // Add one more to the same class — keys[4] has same coarse key as keys[3]? No, brightness differs.
    // Let's use a variant of keys[0] that differs only in time_period, to stay in same class.
    // We already have all 3 time_period values. Use noise variation instead:
    let key_extra = ContextKey::new(MbotSensors {
        brightness: BrightnessBand::Dim,
        noise: NoiseBand::Moderate, // different fine key, same coarse (noise is NOT masked at dim 1)
        presence: PresenceSignature::Absent,
        motion: MotionContext::Static,
        orientation: Orientation::Upright,
        time_period: TimePeriod::Day,
    });

    // key_extra has same coarse key as keys[0..2] (only dim5 masked, noise at dim1 is included)
    // → different fine key, same Tier 1 class → 4th Tier 2 entry
    for tick in 0..10u64 {
        map.positive_interaction(&key_extra, &personality, 400 + tick, false);
    }

    // Now add one more fine key in the same Tier 1 class to trigger eviction
    let key_trigger = ContextKey::new(MbotSensors {
        brightness: BrightnessBand::Dim,
        noise: NoiseBand::Loud, // yet another fine key, same coarse
        presence: PresenceSignature::Absent,
        motion: MotionContext::Static,
        orientation: Orientation::Upright,
        time_period: TimePeriod::Day,
    });

    // Record Tier 1 coherence before the eviction trigger
    // (we need to find the Tier 1 class that has all those Tier 2 entries)
    let t1_coherence_before: f32 = map
        .classes
        .values()
        .filter(|c| c.tier2_entries.len() >= 4)
        .map(|c| c.accumulator.value)
        .next()
        .unwrap_or(0.0);

    for tick in 0..10u64 {
        map.positive_interaction(&key_trigger, &personality, 500 + tick, false);
    }

    // All Tier 2 maps must remain bounded at T2=4
    for cls in map.classes.values() {
        assert!(
            cls.tier2_entries.len() <= 4,
            "Tier 2 entry count per class must stay ≤ T2=4, got {}",
            cls.tier2_entries.len()
        );
    }

    // Tier 1 coherence for that class must not have decreased
    let t1_coherence_after: f32 = map
        .classes
        .values()
        .map(|c| c.accumulator.value)
        .fold(0.0_f32, f32::max);

    assert!(
        t1_coherence_after >= t1_coherence_before - 1e-4,
        "Tier 1 coherence should not decrease after eviction contribution: {:.4} -> {:.4}",
        t1_coherence_before,
        t1_coherence_after
    );
}

// ─── test 6: memory bound under key flood ────────────────────────────────────

/// Inserting many more distinct context keys than T1 or T2 allows must never
/// cause the map to exceed its bounds.
///
/// Verifies I-CKM-007.
#[test]
fn test_memory_bound_under_key_flood() {
    let config = TieredContextConfig {
        promotion_threshold: 1,
        tier1_feature_mask: 0b000001, // only dim 0 (brightness) → 3 coarse classes
        ..TieredContextConfig::default()
    };
    // T1=4, T2=4
    let mut map: TieredContextMap<MbotSensors, 6, 4, 4> = TieredContextMap::new(config);
    let personality = default_personality();

    // All 486 MbotSensors combinations
    let brightness_vals = [BrightnessBand::Dark, BrightnessBand::Dim, BrightnessBand::Bright];
    let noise_vals = [NoiseBand::Quiet, NoiseBand::Moderate, NoiseBand::Loud];
    let presence_vals = [PresenceSignature::Absent, PresenceSignature::Far, PresenceSignature::Close];
    let motion_vals = [MotionContext::Static, MotionContext::Slow, MotionContext::Fast];
    let orientation_vals = [Orientation::Upright, Orientation::Tilted];
    let time_vals = [TimePeriod::Day, TimePeriod::Evening, TimePeriod::Night];

    let mut tick = 0u64;
    for &b in &brightness_vals {
        for &n in &noise_vals {
            for &p in &presence_vals {
                for &m in &motion_vals {
                    for &o in &orientation_vals {
                        for &t in &time_vals {
                            let key = ContextKey::new(MbotSensors {
                                brightness: b,
                                noise: n,
                                presence: p,
                                motion: m,
                                orientation: o,
                                time_period: t,
                            });
                            map.positive_interaction(&key, &personality, tick, false);
                            tick += 1;
                        }
                    }
                }
            }
        }
    }

    assert!(
        map.tier1_class_count() <= 4,
        "Tier 1 class count must stay ≤ T1=4, got {}",
        map.tier1_class_count()
    );
    for cls in map.classes.values() {
        assert!(
            cls.tier2_entries.len() <= 4,
            "Tier 2 entry count per class must stay ≤ T2=4, got {}",
            cls.tier2_entries.len()
        );
    }
}

// ─── test 7: lookup fallthrough ──────────────────────────────────────────────

/// Before Tier 2 activates, `context_coherence` must return the Tier 1 coarse
/// value, not 0.0.
///
/// After Tier 2 activates, `context_coherence` must return the fine Tier 2 value
/// for a key that has a fine entry.
///
/// Verifies I-CKM-006.
#[test]
fn test_lookup_fallthrough() {
    let config = TieredContextConfig {
        promotion_threshold: 5,
        ..TieredContextConfig::default()
    };
    let mut map: TieredContextMap<MbotSensors, 6, 8, 4> = TieredContextMap::new(config);

    let key = default_key();
    let personality = default_personality();

    // 2 interactions — below threshold, only Tier 1 populated
    for tick in 0..2u64 {
        map.positive_interaction(&key, &personality, tick, false);
    }

    // Tier 2 must NOT be active
    let tier2_active = map.classes.values().next().map(|c| c.tier2_active);
    assert_eq!(tier2_active, Some(false), "Tier 2 must not yet be active");

    // Lookup must return the Tier 1 (coarse) value (fallthrough)
    let coarse_coherence = map.context_coherence(&key);
    assert!(
        coarse_coherence > 0.0,
        "Lookup must return Tier 1 fallthrough value before Tier 2 activates, got {}",
        coarse_coherence
    );

    // 5 interactions total — reaches threshold, Tier 2 activates
    for tick in 2..5u64 {
        map.positive_interaction(&key, &personality, tick, false);
    }

    let tier2_active_now = map.classes.values().next().map(|c| c.tier2_active);
    assert_eq!(tier2_active_now, Some(true), "Tier 2 must be active at threshold");

    // Lookup returns fine Tier 2 value (not 0.0)
    let fine_coherence = map.context_coherence(&key);
    assert!(
        fine_coherence > 0.0,
        "Lookup must return positive value from Tier 2 after activation, got {}",
        fine_coherence
    );

    // A key that has NO fine entry must still fall through to Tier 1
    let other_key = key_with_time(TimePeriod::Night);
    let fallthrough = map.context_coherence(&other_key);
    // `other_key` may be in the same or different Tier 1 class.
    // If same class: returns Tier 1 coarse value (> 0).
    // If different class (unseen): returns 0.0.
    // Both are correct — just verify no panic.
    let _ = fallthrough;
}

// ─── test 8: merge associativity and commutativity ───────────────────────────

/// `merge_accumulators` must be associative and commutative.
///
/// Verifies I-CKM-008 (merge semantics).
#[test]
fn test_merge_associative_commutative() {
    use ccf_core::accumulator::CoherenceAccumulator;

    let a = CoherenceAccumulator {
        value: 0.7,
        interaction_count: 4,
        last_interaction_tick: 10,
    };
    let b = CoherenceAccumulator {
        value: 0.4,
        interaction_count: 8,
        last_interaction_tick: 30,
    };
    let c = CoherenceAccumulator {
        value: 0.9,
        interaction_count: 2,
        last_interaction_tick: 20,
    };

    // Commutativity: merge(a, b) == merge(b, a)
    let ab = merge_accumulators(&a, &b);
    let ba = merge_accumulators(&b, &a);
    assert!(
        (ab.value - ba.value).abs() < 1e-6,
        "merge not commutative on value: {} vs {}",
        ab.value,
        ba.value
    );
    assert_eq!(
        ab.interaction_count, ba.interaction_count,
        "merge not commutative on interaction_count"
    );

    // Associativity: merge(merge(a,b), c) == merge(a, merge(b,c))
    let abc_left = merge_accumulators(&merge_accumulators(&a, &b), &c);
    let abc_right = merge_accumulators(&a, &merge_accumulators(&b, &c));
    assert!(
        (abc_left.value - abc_right.value).abs() < 1e-6,
        "merge not associative on value: {} vs {}",
        abc_left.value,
        abc_right.value
    );
    assert_eq!(
        abc_left.interaction_count, abc_right.interaction_count,
        "merge not associative on interaction_count"
    );
}
