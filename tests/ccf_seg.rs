//! CCF_SEG round-trip integration tests.
//!
//! Verifies that a live CoherenceField can be captured as a CcfSegSnapshot,
//! serialised to JSON, deserialised back, and that all context values are
//! preserved exactly.

#[cfg(feature = "serde")]
mod tests {
    use ccf_core::accumulator::CoherenceField;
    use ccf_core::mbot::{
        BrightnessBand, MbotSensors, MotionContext, NoiseBand, Orientation, PresenceSignature,
        TimePeriod,
    };
    use ccf_core::phase::Personality;
    use ccf_core::seg::{CcfSegSnapshot, CCF_SEG_VERSION};
    use ccf_core::vocabulary::ContextKey;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn make_key(brightness: BrightnessBand, noise: NoiseBand) -> ContextKey<MbotSensors, 6> {
        ContextKey::new(MbotSensors {
            brightness,
            noise,
            presence: PresenceSignature::Absent,
            motion: MotionContext::Static,
            orientation: Orientation::Upright,
            time_period: TimePeriod::Day,
        })
    }

    fn bright_quiet() -> ContextKey<MbotSensors, 6> {
        make_key(BrightnessBand::Bright, NoiseBand::Quiet)
    }

    fn dark_loud() -> ContextKey<MbotSensors, 6> {
        make_key(BrightnessBand::Dark, NoiseBand::Loud)
    }

    fn dim_moderate() -> ContextKey<MbotSensors, 6> {
        make_key(BrightnessBand::Dim, NoiseBand::Moderate)
    }

    /// Build a CoherenceField with three known contexts and a personality.
    fn make_field() -> (CoherenceField<MbotSensors, 6>, Personality) {
        let mut field: CoherenceField<MbotSensors, 6> = CoherenceField::new();
        let personality = Personality {
            curiosity_drive: 0.8,
            startle_sensitivity: 0.3,
            recovery_speed: 0.7,
        };

        let k1 = bright_quiet();
        let k2 = dark_loud();
        let k3 = dim_moderate();

        // Build context 1: 10 positive interactions
        for tick in 0..10u64 {
            field.positive_interaction(&k1, &personality, tick, false);
        }

        // Build context 2: 5 positive interactions + 1 negative
        for tick in 0..5u64 {
            field.positive_interaction(&k2, &personality, tick, false);
        }
        field.negative_interaction(&k2, &personality, 5);

        // Build context 3: 20 positive interactions (alone = true for faster growth)
        for tick in 0..20u64 {
            field.positive_interaction(&k3, &personality, tick, true);
        }

        (field, personality)
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_snapshot_captures_correct_context_count() {
        let (field, personality) = make_field();
        let snapshot = CcfSegSnapshot::from_field(&field, &personality, 1_000_000, 1_001_000, 36);

        assert_eq!(
            snapshot.context_count(),
            3,
            "snapshot should have 3 contexts, got {}",
            snapshot.context_count()
        );
    }

    #[test]
    fn test_snapshot_version_is_current() {
        let (field, personality) = make_field();
        let snapshot = CcfSegSnapshot::from_field(&field, &personality, 0, 0, 0);
        assert_eq!(
            snapshot.version,
            CCF_SEG_VERSION,
            "snapshot version should be {}",
            CCF_SEG_VERSION
        );
    }

    #[test]
    fn test_snapshot_metadata_preserved() {
        let created_at: i64 = 1_740_000_000;
        let last_active_at: i64 = 1_740_001_000;
        let total_interactions: u64 = 500;

        let (field, personality) = make_field();
        let snapshot = CcfSegSnapshot::from_field(
            &field,
            &personality,
            created_at,
            last_active_at,
            total_interactions,
        );

        assert_eq!(snapshot.created_at, created_at);
        assert_eq!(snapshot.last_active_at, last_active_at);
        assert_eq!(snapshot.total_interactions, total_interactions);
    }

    #[test]
    fn test_snapshot_personality_preserved() {
        let (field, personality) = make_field();
        let snapshot = CcfSegSnapshot::from_field(&field, &personality, 0, 0, 0);

        assert!(
            (snapshot.personality.curiosity_drive - personality.curiosity_drive).abs() < f32::EPSILON,
            "curiosity_drive mismatch: {} vs {}",
            snapshot.personality.curiosity_drive,
            personality.curiosity_drive
        );
        assert!(
            (snapshot.personality.startle_sensitivity - personality.startle_sensitivity).abs()
                < f32::EPSILON,
            "startle_sensitivity mismatch"
        );
        assert!(
            (snapshot.personality.recovery_speed - personality.recovery_speed).abs() < f32::EPSILON,
            "recovery_speed mismatch"
        );
    }

    #[test]
    fn test_snapshot_context_coherence_values_preserved() {
        let (field, personality) = make_field();

        // Capture live values before snapshotting
        let k1 = bright_quiet();
        let k2 = dark_loud();
        let k3 = dim_moderate();

        let coh1 = field.context_coherence(&k1);
        let coh2 = field.context_coherence(&k2);
        let coh3 = field.context_coherence(&k3);
        let count1 = field.context_interaction_count(&k1);
        let count2 = field.context_interaction_count(&k2);
        let count3 = field.context_interaction_count(&k3);

        let snapshot = CcfSegSnapshot::from_field(&field, &personality, 0, 0, 0);

        // Find each context by hash in the snapshot
        let hash1 = k1.context_hash_u32();
        let hash2 = k2.context_hash_u32();
        let hash3 = k3.context_hash_u32();

        let rec1 = snapshot
            .find_context(hash1)
            .expect("context 1 (bright_quiet) not found in snapshot");
        let rec2 = snapshot
            .find_context(hash2)
            .expect("context 2 (dark_loud) not found in snapshot");
        let rec3 = snapshot
            .find_context(hash3)
            .expect("context 3 (dim_moderate) not found in snapshot");

        assert!(
            (rec1.coherence_value - coh1).abs() < 1e-6,
            "context 1 coherence: snapshot={} live={}",
            rec1.coherence_value,
            coh1
        );
        assert!(
            (rec2.coherence_value - coh2).abs() < 1e-6,
            "context 2 coherence: snapshot={} live={}",
            rec2.coherence_value,
            coh2
        );
        assert!(
            (rec3.coherence_value - coh3).abs() < 1e-6,
            "context 3 coherence: snapshot={} live={}",
            rec3.coherence_value,
            coh3
        );

        assert_eq!(
            rec1.interaction_count, count1,
            "context 1 interaction_count mismatch"
        );
        assert_eq!(
            rec2.interaction_count, count2,
            "context 2 interaction_count mismatch"
        );
        assert_eq!(
            rec3.interaction_count, count3,
            "context 3 interaction_count mismatch"
        );
    }

    #[test]
    fn test_ccf_seg_round_trip_json() {
        let (field, personality) = make_field();
        let original =
            CcfSegSnapshot::from_field(&field, &personality, 1_740_000_000, 1_740_001_000, 36);

        // Serialise to JSON
        let json = serde_json::to_string(&original).expect("serialise to JSON");

        // Deserialise back
        let restored: CcfSegSnapshot =
            serde_json::from_str(&json).expect("deserialise from JSON");

        // Structural equality
        assert_eq!(
            original.version, restored.version,
            "version mismatch after round-trip"
        );
        assert_eq!(
            original.created_at, restored.created_at,
            "created_at mismatch after round-trip"
        );
        assert_eq!(
            original.last_active_at, restored.last_active_at,
            "last_active_at mismatch after round-trip"
        );
        assert_eq!(
            original.total_interactions, restored.total_interactions,
            "total_interactions mismatch after round-trip"
        );
        assert_eq!(
            original.context_count(),
            restored.context_count(),
            "context count mismatch after round-trip"
        );

        // Personality round-trip
        assert!(
            (original.personality.curiosity_drive - restored.personality.curiosity_drive).abs()
                < 1e-6,
            "personality.curiosity_drive mismatch after round-trip"
        );
        assert!(
            (original.personality.startle_sensitivity
                - restored.personality.startle_sensitivity)
                .abs()
                < 1e-6,
            "personality.startle_sensitivity mismatch after round-trip"
        );
        assert!(
            (original.personality.recovery_speed - restored.personality.recovery_speed).abs()
                < 1e-6,
            "personality.recovery_speed mismatch after round-trip"
        );

        // Each context record round-trips exactly
        for orig_ctx in &original.contexts {
            let restored_ctx = restored
                .find_context(orig_ctx.context_hash)
                .unwrap_or_else(|| {
                    panic!("context hash 0x{:08x} missing after round-trip", orig_ctx.context_hash)
                });

            assert!(
                (orig_ctx.coherence_value - restored_ctx.coherence_value).abs() < 1e-6,
                "coherence_value mismatch for hash 0x{:08x}: {} vs {}",
                orig_ctx.context_hash,
                orig_ctx.coherence_value,
                restored_ctx.coherence_value
            );
            assert_eq!(
                orig_ctx.interaction_count, restored_ctx.interaction_count,
                "interaction_count mismatch for hash 0x{:08x}",
                orig_ctx.context_hash
            );
            assert_eq!(
                orig_ctx.last_interaction_tick, restored_ctx.last_interaction_tick,
                "last_interaction_tick mismatch for hash 0x{:08x}",
                orig_ctx.context_hash
            );
        }
    }

    #[test]
    fn test_empty_field_snapshot() {
        let field: CoherenceField<MbotSensors, 6> = CoherenceField::new();
        let personality = Personality::new();
        let snapshot = CcfSegSnapshot::from_field(&field, &personality, 0, 0, 0);

        assert_eq!(snapshot.context_count(), 0);
        assert_eq!(snapshot.version, CCF_SEG_VERSION);

        // Empty field round-trips to JSON without error
        let json = serde_json::to_string(&snapshot).expect("serialise empty snapshot");
        let restored: CcfSegSnapshot = serde_json::from_str(&json).expect("deserialise empty snapshot");
        assert_eq!(restored.context_count(), 0);
    }

    #[test]
    fn test_find_context_returns_none_for_unknown_hash() {
        let (field, personality) = make_field();
        let snapshot = CcfSegSnapshot::from_field(&field, &personality, 0, 0, 0);
        // Hash 0 is extremely unlikely to match any real context
        assert!(snapshot.find_context(0xDEAD_BEEF).is_none());
    }
}
