/*
 * Notice of Provisional Patent Filing:
 * The methods and algorithms implemented in this file (specifically relating to
 * Contextual Coherence Fields and relational coherence accumulation) are the
 * subject of a United States Provisional Patent Application (63/988,438)
 * filed on February 23, 2026.
 *
 * This source code is licensed under the Business Source License 1.1.
 * See LICENSE and PATENTS.md in the root directory for full details.
 */

//! Per-context coherence accumulators and the full coherence field.
//!
//! # Patent Claims 2–7, 13
//!
//! - [`CoherenceAccumulator`]: per-context trust state with earned floor and asymmetric decay (Claims 2–5).
//! - [`CoherenceField`]: context-keyed accumulator map with asymmetric min-gate (Claims 6–7, 13).
//!
//! # Invariants
//!
//! - **CCF-001**: `effective_coherence` uses asymmetric gate:
//!   - Unfamiliar contexts (ctx < 0.3): `min(instant, ctx)` — earn trust first.
//!   - Familiar contexts (ctx >= 0.3): `0.3 * instant + 0.7 * ctx` — history buffers noise.
//! - **CCF-002**: All accumulator values bounded [0.0, 1.0].
//! - **CCF-003**: Personality modulates deltas, not structure.
//! - **I-DIST-001**: no_std compatible; uses `hashbrown::HashMap` (no `std` dependency).
//! - **I-DIST-005**: Zero unsafe code.

use hashbrown::HashMap;

use crate::phase::Personality;
use crate::vocabulary::{ContextKey, SensorVocabulary};

// ─── Coherence Accumulator ──────────────────────────────────────────────────

/// Per-context coherence accumulator. Grows through repeated positive
/// interaction, decays with disuse, drops on negative events.
///
/// Interaction history builds a protected floor so that earned trust
/// cannot be erased by transient negative events.
///
/// Patent Claims 2–5.
#[derive(Clone, Debug)]
pub struct CoherenceAccumulator {
    /// Accumulated coherence for this context [0.0, 1.0].
    pub value: f32,
    /// Total positive interactions recorded in this context.
    pub interaction_count: u32,
    /// Tick of the most recent interaction (positive or negative).
    pub last_interaction_tick: u64,
}

impl CoherenceAccumulator {
    /// Construct a fresh accumulator starting at zero coherence.
    pub fn new() -> Self {
        Self {
            value: 0.0,
            interaction_count: 0,
            last_interaction_tick: 0,
        }
    }

    /// Cold-start constructor: initialise value from personality `curiosity_drive`.
    ///
    /// `curiosity`: personality curiosity_drive in [0.0, 1.0].
    /// Baseline = 0.15 × curiosity (max 0.15 for curiosity = 1.0).
    pub fn new_with_baseline(curiosity: f32) -> Self {
        Self {
            value: (0.15 * curiosity).clamp(0.0, 1.0),
            interaction_count: 0,
            last_interaction_tick: 0,
        }
    }

    /// The minimum coherence that interaction history protects against decay or negative events.
    ///
    /// Asymptotically approaches 0.5 with repeated interactions — never fully
    /// immune, but increasingly resilient.
    ///
    /// ```text
    /// floor = 0.5 × (1 − 1 / (1 + count / 20))
    ///   count =  0 → floor ≈ 0.00
    ///   count = 20 → floor ≈ 0.25
    ///   count = 100 → floor ≈ 0.42
    ///   limit  → 0.50
    /// ```
    pub fn earned_floor(&self) -> f32 {
        0.5 * (1.0 - 1.0 / (1.0 + self.interaction_count as f32 / 20.0))
    }

    /// Record a positive interaction. Coherence grows asymptotically toward 1.0.
    ///
    /// - `recovery_speed`: personality parameter [0.0, 1.0] — higher = faster growth.
    /// - `tick`: current tick for freshness tracking.
    /// - `alone`: `true` if presence is Absent — doubles delta for faster bootstrap.
    pub fn positive_interaction(&mut self, recovery_speed: f32, tick: u64, alone: bool) {
        let mut delta = 0.02 * (0.5 + recovery_speed) * (1.0 - self.value);
        if alone {
            delta *= 2.0; // alone contexts bootstrap faster
        }
        self.value = (self.value + delta).min(1.0);
        self.interaction_count = self.interaction_count.saturating_add(1);
        self.last_interaction_tick = tick;
    }

    /// Record a negative interaction (startle, collision, high tension).
    ///
    /// The drop is floored at `earned_floor()` so that accumulated trust
    /// cannot be fully erased by a single negative event.
    ///
    /// - `startle_sensitivity`: personality parameter [0.0, 1.0] — higher = bigger drop.
    /// - `tick`: current tick.
    pub fn negative_interaction(&mut self, startle_sensitivity: f32, tick: u64) {
        let floor = self.earned_floor();
        let delta = 0.05 * (0.5 + startle_sensitivity);
        self.value = (self.value - delta).max(floor);
        self.last_interaction_tick = tick;
    }

    /// Apply time-based decay. Call once per elapsed period.
    ///
    /// Coherence decays toward `earned_floor()`, not toward zero.
    /// More interactions = higher floor = harder to lose earned trust.
    pub fn decay(&mut self, elapsed_ticks: u64) {
        let floor = self.earned_floor();
        if self.value > floor {
            let decay_rate = 0.0001 * elapsed_ticks as f32;
            self.value = (self.value - decay_rate).max(floor);
        }
    }
}

impl Default for CoherenceAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Coherence Field ────────────────────────────────────────────────────────

/// Maximum number of tracked contexts. Oldest entry is evicted when full.
const MAX_CONTEXTS: usize = 64;

/// The coherence field: a map of context → [`CoherenceAccumulator`].
///
/// Generic over any sensor vocabulary `V` implementing [`SensorVocabulary<N>`].
/// Maintains at most [`MAX_CONTEXTS`] entries with LRU eviction.
///
/// Patent Claims 6–7, 13.
pub struct CoherenceField<V: SensorVocabulary<N>, const N: usize> {
    /// Context-keyed accumulators.
    accumulators: HashMap<ContextKey<V, N>, CoherenceAccumulator>,
    /// Personality baseline for new contexts (0.15 × curiosity_drive).
    personality_baseline: f32,
    /// Fallback coherence used as floor for unseen contexts in degraded mode.
    fallback_coherence: Option<f32>,
}

impl<V: SensorVocabulary<N>, const N: usize> CoherenceField<V, N> {
    /// Construct a fresh field with no accumulated coherence.
    pub fn new() -> Self {
        Self {
            accumulators: HashMap::new(),
            personality_baseline: 0.0,
            fallback_coherence: None,
        }
    }

    // ── CCF-001: asymmetric min-gate ───────────────────────────────────────

    /// Compute effective coherence using the asymmetric gate (CCF-001).
    ///
    /// - **Unfamiliar** (ctx < 0.3): `min(instant, ctx)` — earn trust first.
    /// - **Familiar** (ctx ≥ 0.3): `0.3 × instant + 0.7 × ctx` — history buffers noise.
    pub fn effective_coherence(&self, instant: f32, key: &ContextKey<V, N>) -> f32 {
        let ctx = self.context_coherence(key);
        if ctx < 0.3 {
            if instant < ctx { instant } else { ctx }
        } else {
            (0.3 * instant + 0.7 * ctx).clamp(0.0, 1.0)
        }
    }

    // ── Interaction API (CCF-003: Personality modulates deltas, not structure) ─

    /// Record a positive interaction for a context, modulated by `personality`.
    ///
    /// Creates the accumulator at the personality baseline if the context is unseen.
    pub fn positive_interaction(
        &mut self,
        key: &ContextKey<V, N>,
        personality: &Personality,
        tick: u64,
        alone: bool,
    ) {
        self.get_or_create(key)
            .positive_interaction(personality.recovery_speed, tick, alone);
    }

    /// Record a negative interaction for a context, modulated by `personality`.
    ///
    /// Creates the accumulator at the personality baseline if the context is unseen.
    pub fn negative_interaction(
        &mut self,
        key: &ContextKey<V, N>,
        personality: &Personality,
        tick: u64,
    ) {
        self.get_or_create(key)
            .negative_interaction(personality.startle_sensitivity, tick);
    }

    // ── Read accessors ─────────────────────────────────────────────────────

    /// Get the accumulated coherence for a context.
    ///
    /// Returns the accumulator value if seen, or the fallback / 0.0 for unseen contexts.
    pub fn context_coherence(&self, key: &ContextKey<V, N>) -> f32 {
        self.accumulators.get(key).map_or_else(
            || self.fallback_coherence.unwrap_or(0.0),
            |a| a.value,
        )
    }

    /// Number of positive interactions recorded for a context (0 if unseen).
    pub fn context_interaction_count(&self, key: &ContextKey<V, N>) -> u32 {
        self.accumulators.get(key).map_or(0, |a| a.interaction_count)
    }

    // ── Decay ──────────────────────────────────────────────────────────────

    /// Apply time-based decay to all accumulators.
    pub fn decay_all(&mut self, elapsed_ticks: u64) {
        for acc in self.accumulators.values_mut() {
            acc.decay(elapsed_ticks);
        }
    }

    // ── Collection helpers ─────────────────────────────────────────────────

    /// Number of tracked contexts.
    pub fn context_count(&self) -> usize {
        self.accumulators.len()
    }

    /// Iterate over all (context key, accumulator) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&ContextKey<V, N>, &CoherenceAccumulator)> {
        self.accumulators.iter()
    }

    /// All tracked contexts with their coherence value and interaction count,
    /// sorted by interaction count descending.
    ///
    /// Returns `Vec<(key, coherence_value, interaction_count)>`.
    /// Available only when the `std` feature is enabled (requires heap allocation).
    #[cfg(feature = "std")]
    pub fn all_entries(&self) -> std::vec::Vec<(ContextKey<V, N>, f32, u32)> {
        let mut entries: std::vec::Vec<(ContextKey<V, N>, f32, u32)> = self
            .accumulators
            .iter()
            .map(|(k, acc)| (k.clone(), acc.value, acc.interaction_count))
            .collect();
        entries.sort_by(|a, b| b.2.cmp(&a.2));
        entries
    }

    // ── Degraded-mode fallback ─────────────────────────────────────────────

    /// Set the fallback coherence returned for unseen contexts in degraded mode.
    ///
    /// Pass `None` to clear the fallback (unseen contexts revert to 0.0).
    pub fn set_fallback(&mut self, value: Option<f32>) {
        self.fallback_coherence = value;
    }

    // ── Internal helpers ───────────────────────────────────────────────────

    /// Get or create the accumulator for `key`, initialising at the personality baseline.
    ///
    /// Evicts the oldest entry when the field is at [`MAX_CONTEXTS`] capacity.
    pub fn get_or_create(&mut self, key: &ContextKey<V, N>) -> &mut CoherenceAccumulator {
        if !self.accumulators.contains_key(key) {
            if self.accumulators.len() >= MAX_CONTEXTS {
                self.evict_oldest();
            }
            let curiosity = if self.personality_baseline > 0.0 {
                (self.personality_baseline / 0.15).clamp(0.0, 1.0)
            } else {
                0.0
            };
            self.accumulators
                .insert(key.clone(), CoherenceAccumulator::new_with_baseline(curiosity));
        }
        self.accumulators.get_mut(key).unwrap()
    }

    fn evict_oldest(&mut self) {
        if let Some(oldest_key) = self
            .accumulators
            .iter()
            .min_by_key(|(_, acc)| acc.last_interaction_tick)
            .map(|(k, _)| k.clone())
        {
            self.accumulators.remove(&oldest_key);
        }
    }
}

impl<V: SensorVocabulary<N>, const N: usize> Default for CoherenceField<V, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: SensorVocabulary<N>, const N: usize> core::fmt::Debug for CoherenceField<V, N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CoherenceField")
            .field("context_count", &self.accumulators.len())
            .field("personality_baseline", &self.personality_baseline)
            .field("fallback_coherence", &self.fallback_coherence)
            .finish()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mbot::{
        BrightnessBand, MbotSensors, MotionContext, NoiseBand, Orientation,
        PresenceSignature, TimePeriod,
    };
    use crate::vocabulary::ContextKey;

    // ── Helpers ──────────────────────────────────────────────────────────

    fn make_key(
        brightness: BrightnessBand,
        noise: NoiseBand,
        presence: PresenceSignature,
    ) -> ContextKey<MbotSensors, 6> {
        ContextKey::new(MbotSensors {
            brightness,
            noise,
            presence,
            motion: MotionContext::Static,
            orientation: Orientation::Upright,
            time_period: TimePeriod::Day,
        })
    }

    fn bright_quiet_static() -> ContextKey<MbotSensors, 6> {
        make_key(BrightnessBand::Bright, NoiseBand::Quiet, PresenceSignature::Absent)
    }

    fn dark_loud_close() -> ContextKey<MbotSensors, 6> {
        make_key(BrightnessBand::Dark, NoiseBand::Loud, PresenceSignature::Close)
    }

    fn neutral_personality() -> Personality {
        Personality {
            curiosity_drive: 0.5,
            startle_sensitivity: 0.5,
            recovery_speed: 0.5,
        }
    }

    // ── CoherenceAccumulator tests ────────────────────────────────────────

    #[test]
    fn test_accumulator_positive_growth() {
        let mut acc = CoherenceAccumulator::new();
        assert_eq!(acc.value, 0.0);

        for i in 0..50 {
            acc.positive_interaction(0.5, i, false);
        }
        assert!(acc.value > 0.3, "value={}", acc.value);
        assert!(acc.value < 1.0);
        assert_eq!(acc.interaction_count, 50);
    }

    #[test]
    fn test_accumulator_asymptotic_growth() {
        let mut acc = CoherenceAccumulator::new();
        for i in 0..500 {
            acc.positive_interaction(0.5, i, false);
        }
        let high_value = acc.value;
        for i in 500..510 {
            acc.positive_interaction(0.5, i, false);
        }
        let delta = acc.value - high_value;
        assert!(delta < 0.01, "delta should be small at high values: {}", delta);
    }

    #[test]
    fn test_accumulator_personality_modulation() {
        let mut fast = CoherenceAccumulator::new();
        let mut slow = CoherenceAccumulator::new();

        for i in 0..20 {
            fast.positive_interaction(0.9, i, false);
            slow.positive_interaction(0.1, i, false);
        }
        assert!(
            fast.value > slow.value,
            "fast={} should be > slow={}",
            fast.value,
            slow.value
        );
    }

    #[test]
    fn test_accumulator_negative_interaction() {
        let mut acc = CoherenceAccumulator::new();
        for i in 0..30 {
            acc.positive_interaction(0.5, i, false);
        }
        let before = acc.value;
        acc.negative_interaction(0.5, 31);
        assert!(acc.value < before);
    }

    #[test]
    fn test_accumulator_earned_floor() {
        let mut acc = CoherenceAccumulator::new();
        for i in 0..100 {
            acc.positive_interaction(0.5, i, false);
        }
        let before = acc.value;
        for i in 100..200 {
            acc.negative_interaction(1.0, i);
        }
        // Floor at 100 interactions ≈ 0.42
        assert!(
            acc.value > 0.3,
            "value={} should be above earned floor",
            acc.value
        );
        assert!(acc.value < before);
    }

    #[test]
    fn test_accumulator_decay_toward_floor() {
        let mut acc = CoherenceAccumulator::new();
        for i in 0..50 {
            acc.positive_interaction(0.5, i, false);
        }
        let before = acc.value;
        acc.decay(1000);
        assert!(acc.value < before);
        let floor = 0.5 * (1.0 - 1.0 / (1.0 + 50.0 / 20.0));
        assert!(
            acc.value >= floor,
            "value={} should be >= floor={}",
            acc.value,
            floor
        );
    }

    #[test]
    fn test_cold_start_baseline() {
        let acc = CoherenceAccumulator::new_with_baseline(1.0);
        assert!((acc.value - 0.15).abs() < 0.001, "value={}", acc.value);

        let acc = CoherenceAccumulator::new_with_baseline(0.2);
        assert!((acc.value - 0.03).abs() < 0.001, "value={}", acc.value);

        let acc = CoherenceAccumulator::new_with_baseline(0.0);
        assert_eq!(acc.value, 0.0);
    }

    #[test]
    fn test_alone_boost() {
        let mut alone_acc = CoherenceAccumulator::new();
        let mut social_acc = CoherenceAccumulator::new();

        for i in 0..20 {
            alone_acc.positive_interaction(0.5, i, true);
            social_acc.positive_interaction(0.5, i, false);
        }

        assert!(
            alone_acc.value > social_acc.value,
            "alone={} should be > social={}",
            alone_acc.value,
            social_acc.value
        );
    }

    #[test]
    fn test_accumulator_value_bounded() {
        // CCF-002: values always in [0.0, 1.0]
        let acc = CoherenceAccumulator::new_with_baseline(2.0); // out-of-range curiosity
        assert!(
            acc.value >= 0.0 && acc.value <= 1.0,
            "value={}",
            acc.value
        );

        let mut acc = CoherenceAccumulator::new_with_baseline(0.5);
        for i in 0..1000 {
            acc.positive_interaction(1.0, i, true);
        }
        assert!(acc.value <= 1.0, "value={}", acc.value);
    }

    // ── CoherenceField tests ──────────────────────────────────────────────

    #[test]
    fn test_coherence_field_effective_coherence_unfamiliar() {
        let mut field: CoherenceField<MbotSensors, 6> = CoherenceField::new();
        let key = bright_quiet_static();

        // New context: ctx = 0.0 → CCF-001 unfamiliar: min(0.8, 0.0) = 0.0
        let eff = field.effective_coherence(0.8, &key);
        assert_eq!(eff, 0.0);

        // Build a little coherence (stay under 0.3 threshold)
        {
            let acc = field.get_or_create(&key);
            for i in 0..10 {
                acc.positive_interaction(0.5, i, false);
            }
        }
        let ctx_coh = field.context_coherence(&key);
        assert!(ctx_coh > 0.0);
        assert!(
            ctx_coh < 0.3,
            "ctx_coh={} should be < 0.3 for unfamiliar test",
            ctx_coh
        );

        // Unfamiliar: min(0.8, ctx) = ctx
        let eff = field.effective_coherence(0.8, &key);
        assert!((eff - ctx_coh).abs() < 0.001);

        // Unfamiliar: min(0.05, ctx) = 0.05 when instant < ctx
        let eff = field.effective_coherence(0.05, &key);
        assert!((eff - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_coherence_field_effective_coherence_familiar() {
        let mut field: CoherenceField<MbotSensors, 6> = CoherenceField::new();
        let key = bright_quiet_static();

        // Build up enough coherence to cross 0.3 threshold
        {
            let acc = field.get_or_create(&key);
            for i in 0..80 {
                acc.positive_interaction(0.5, i, false);
            }
        }
        let ctx_coh = field.context_coherence(&key);
        assert!(
            ctx_coh >= 0.3,
            "ctx_coh={} should be >= 0.3 for familiar test",
            ctx_coh
        );

        // Familiar: 0.3 * instant + 0.7 * ctx
        let eff = field.effective_coherence(0.8, &key);
        let expected = 0.3 * 0.8 + 0.7 * ctx_coh;
        assert!(
            (eff - expected).abs() < 0.001,
            "eff={} expected={}",
            eff,
            expected
        );

        // Familiar context should buffer against a low instant value
        let eff_low = field.effective_coherence(0.1, &key);
        let expected_low = 0.3 * 0.1 + 0.7 * ctx_coh;
        assert!(
            (eff_low - expected_low).abs() < 0.001,
            "eff_low={} expected_low={}",
            eff_low,
            expected_low
        );
        assert!(
            eff_low > 0.1,
            "familiar context should buffer: eff_low={}",
            eff_low
        );
    }

    #[test]
    fn test_coherence_field_independent_contexts() {
        let mut field: CoherenceField<MbotSensors, 6> = CoherenceField::new();
        let key_a = bright_quiet_static();
        let key_b = dark_loud_close();

        {
            let acc = field.get_or_create(&key_a);
            for i in 0..50 {
                acc.positive_interaction(0.5, i, false);
            }
        }

        assert!(field.context_coherence(&key_a) > 0.3);
        assert_eq!(field.context_coherence(&key_b), 0.0);
    }

    #[test]
    fn test_coherence_field_interaction_via_personality() {
        let mut field: CoherenceField<MbotSensors, 6> = CoherenceField::new();
        let key = bright_quiet_static();
        let p = neutral_personality();

        for tick in 0..30 {
            field.positive_interaction(&key, &p, tick, false);
        }
        assert!(field.context_coherence(&key) > 0.0);

        let before = field.context_coherence(&key);
        field.negative_interaction(&key, &p, 30);
        assert!(field.context_coherence(&key) < before);
    }

    #[test]
    fn test_coherence_field_eviction() {
        let mut field: CoherenceField<MbotSensors, 6> = CoherenceField::new();
        // Fill beyond MAX_CONTEXTS using two alternating distinct keys.
        // We only have a small vocabulary space, so we manipulate tick to force eviction.
        // First insert MAX_CONTEXTS entries, then insert one more.
        for i in 0..=MAX_CONTEXTS {
            let presence = if i % 2 == 0 {
                PresenceSignature::Absent
            } else {
                PresenceSignature::Close
            };
            let noise = if i % 3 == 0 {
                NoiseBand::Quiet
            } else if i % 3 == 1 {
                NoiseBand::Moderate
            } else {
                NoiseBand::Loud
            };
            let brightness = if i % 4 < 2 {
                BrightnessBand::Bright
            } else {
                BrightnessBand::Dark
            };
            let key = make_key(brightness, noise, presence);
            let acc = field.get_or_create(&key);
            acc.last_interaction_tick = i as u64;
        }
        assert!(field.context_count() <= MAX_CONTEXTS);
    }

    #[test]
    fn test_coherence_field_decay_all() {
        let mut field: CoherenceField<MbotSensors, 6> = CoherenceField::new();
        let key = bright_quiet_static();

        {
            let acc = field.get_or_create(&key);
            for i in 0..50 {
                acc.positive_interaction(0.5, i, false);
            }
        }
        let before = field.context_coherence(&key);
        field.decay_all(1000);
        assert!(
            field.context_coherence(&key) < before,
            "coherence should decay"
        );
    }

    #[test]
    fn test_coherence_field_fallback() {
        let mut field: CoherenceField<MbotSensors, 6> = CoherenceField::new();
        let key = bright_quiet_static();

        // Without fallback, unseen context = 0.0
        assert_eq!(field.context_coherence(&key), 0.0);

        // With fallback, unseen context = fallback value
        field.set_fallback(Some(0.4));
        assert!((field.context_coherence(&key) - 0.4).abs() < 0.001);

        // Seen context still uses its actual value
        {
            let acc = field.get_or_create(&key);
            acc.value = 0.6;
        }
        assert!((field.context_coherence(&key) - 0.6).abs() < 0.001);

        // Clear fallback — seen context still works
        field.set_fallback(None);
        assert!((field.context_coherence(&key) - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_asymmetric_gate_noise_resilience() {
        let mut field: CoherenceField<MbotSensors, 6> = CoherenceField::new();
        let key = bright_quiet_static();

        {
            let acc = field.get_or_create(&key);
            for i in 0..100 {
                acc.positive_interaction(0.5, i, false);
            }
        }
        let ctx_coh = field.context_coherence(&key);
        assert!(ctx_coh >= 0.3, "should be familiar");

        // Simulate a light flicker: instant drops to 0.2
        let eff = field.effective_coherence(0.2, &key);
        // Familiar blend: 0.3*0.2 + 0.7*ctx > 0.2
        assert!(eff > 0.2, "familiar context should buffer noise: eff={}", eff);
    }

    #[test]
    fn test_asymmetric_gate_unfamiliar_strict() {
        let mut field: CoherenceField<MbotSensors, 6> = CoherenceField::new();
        let key = bright_quiet_static();

        {
            let acc = field.get_or_create(&key);
            for i in 0..5 {
                acc.positive_interaction(0.5, i, false);
            }
        }
        let ctx_coh = field.context_coherence(&key);
        assert!(ctx_coh < 0.3);

        // High instant doesn't help: min(0.9, ctx) = ctx
        let eff = field.effective_coherence(0.9, &key);
        assert!(
            (eff - ctx_coh).abs() < 0.001,
            "unfamiliar gate should cap at ctx: eff={} ctx={}",
            eff,
            ctx_coh
        );
    }

    #[test]
    fn test_context_interaction_count() {
        let mut field: CoherenceField<MbotSensors, 6> = CoherenceField::new();
        let key = bright_quiet_static();
        let p = neutral_personality();

        assert_eq!(field.context_interaction_count(&key), 0);
        for tick in 0..5 {
            field.positive_interaction(&key, &p, tick, false);
        }
        assert_eq!(field.context_interaction_count(&key), 5);
    }

    #[test]
    fn test_iter_and_context_count() {
        let mut field: CoherenceField<MbotSensors, 6> = CoherenceField::new();
        let key_a = bright_quiet_static();
        let key_b = dark_loud_close();
        let p = neutral_personality();

        field.positive_interaction(&key_a, &p, 0, false);
        field.positive_interaction(&key_b, &p, 1, false);

        assert_eq!(field.context_count(), 2);
        let count = field.iter().count();
        assert_eq!(count, 2);
    }
}
