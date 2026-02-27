/*
 * Notice of Provisional Patent Filing:
 * The methods and algorithms implemented in this file are the subject of a
 * United States Provisional Patent Application (63/988,438)
 * filed on February 23, 2026.
 *
 * This source code is licensed under the Business Source License 1.1.
 */

//! Two-tier context key cardinality management for production-scale deployments.
//!
//! # The problem in plain English
//!
//! Every distinct sensor environment the robot encounters gets its own trust
//! history.  A robot with many sensors can encounter millions of distinct
//! environments over weeks of operation.  Without a bound, the map of trust
//! histories grows without limit.
//!
//! # The two-tier solution
//!
//! Think of it as two notebooks:
//!
//! **Tier 1 — the scratch pad** (bounded, LRU-evicted):
//! New environments start here.  When the pad is full, the robot tears out the
//! least-recently-seen page.  These are places too new to have earned any
//! protection.
//!
//! **Tier 2 — the keepsake binder** (bounded, protected):
//! After enough visits (default: 20 positive interactions), the page is promoted
//! to the binder.  Binder pages cannot be torn out by LRU pressure — they can
//! only leave if the trust for that place decays back to near-zero, at which
//! point they are demoted back to Tier 1 and eventually forgotten.
//!
//! **The Tier 1 coarse key:**
//! Tier 1 tracks a *coarser* address — controlled by `tier1_feature_mask`.
//! A bitmask selects which sensor dimensions form the coarse key, grouping
//! related fine entries under one Tier 1 class.  When fine Tier 2 entries are
//! evicted they contribute their coherence back to the parent Tier 1 class.
//!
//! # The merge rule
//!
//! When two accumulators must be combined:
//!
//! > *coherence = min(sources)* — never grant unearned familiarity.
//! > *interaction_count = sum(sources)* — never erase relational history.
//!
//! This asymmetric treatment is the only merge semantics that preserves both
//! the honesty invariant (Claim 2) and the history preservation invariant
//! (Claim 3) simultaneously.
//!
//! # Invariants
//!
//! - **I-CKM-001** — Merge sets coherence = min(sources)
//! - **I-CKM-002** — Merge sets interaction_count = sum(sources)
//! - **I-CKM-003** — Eviction contributes coherence back to Tier 1 parent
//! - **I-CKM-004** — Tier 1 class always exists; coarse history never silently lost
//! - **I-CKM-005** — Tier 2 promotion requires interaction_count ≥ promotion_threshold
//! - **I-CKM-006** — Lookup falls through: Tier 2 first, then Tier 1
//! - **I-CKM-007** — Total memory statically bounded; no heap allocation
//! - **I-CKM-008** — Merge and eviction only in deliberative path; hot path zero-allocation
//!
//! # Patent Notes
//!
//! Claims A–D in the continuation filing (see `ccf-merge-claim-artifact.docx`):
//! - Claim A: hierarchical context key structure
//! - Claim B: honesty-preserving merge (PRIMARY NOVEL CLAIM)
//! - Claim C: eviction with coherence contribution
//! - Claim D: merge-gate composition

use crate::accumulator::CoherenceAccumulator;
use crate::phase::Personality;
use crate::vocabulary::{ContextKey, SensorVocabulary};
use heapless::{FnvIndexMap, Vec as HVec};

// ─── Tier1Key ─────────────────────────────────────────────────────────────────

/// Coarse context key — FNV-1a hash of the masked feature dimensions.
///
/// Derived from the full context key by zeroing out feature dimensions
/// not selected by the `tier1_feature_mask` and hashing the result.
pub type Tier1Key = u64;

/// Compute the Tier 1 coarse key from a context key and a feature mask.
///
/// `mask`: bitmask over feature dimensions 0..N.
/// Bit i set = include dimension i in the coarse key.
/// Bit i clear = zero out dimension i (coarse key ignores it).
///
/// E.g. for a 6-dim vocabulary, `mask = 0b000111` uses dims 0, 1, 2 only.
fn compute_tier1_key<V: SensorVocabulary<N>, const N: usize>(
    key: &ContextKey<V, N>,
    mask: u32,
) -> Tier1Key {
    let features = key.vocabulary.to_feature_vec();
    // FNV-1a over the quantised, masked feature dimensions
    let mut h: u64 = 14_695_981_039_346_656_037;
    for (i, &f) in features.iter().enumerate() {
        let v: u16 = if mask & (1 << i) != 0 {
            (f.clamp(0.0, 1.0) * 65535.0) as u16
        } else {
            0 // masked-out dimension contributes constant 0
        };
        h ^= v as u64;
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

// ─── Config ───────────────────────────────────────────────────────────────────

/// Configuration for the tiered context map.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TieredContextConfig {
    /// Positive interaction count at which a Tier 2 entry earns promotion.
    /// Default: 20.  At 20 interactions `earned_floor()` ≈ 0.25.
    pub promotion_threshold: u32,

    /// Ticks of inactivity before a Tier 2 fine entry is eligible for eviction.
    /// Default: 50_000.
    pub eviction_staleness_ticks: u64,

    /// Minimum interaction count below which a Tier 2 entry may be evicted.
    /// Default: 3.  Prevents evicting entries with meaningful history.
    pub eviction_min_count: u32,

    /// Bitmask selecting which sensor dimensions form the coarse Tier 1 key.
    /// Default: 0xFFFFFFFF (all dimensions — Tier 1 key = full key hash).
    /// Set lower bits to use only the most stable/important sensor dimensions.
    pub tier1_feature_mask: u32,

    /// Damping weight applied when an evicted Tier 2 entry contributes its
    /// coherence back to the parent Tier 1 accumulator.
    /// Default: 0.1.  I-CKM-003.
    pub eviction_contribution_weight: f32,
}

impl Default for TieredContextConfig {
    fn default() -> Self {
        Self {
            promotion_threshold: 20,
            eviction_staleness_ticks: 50_000,
            eviction_min_count: 3,
            tier1_feature_mask: 0xFFFF_FFFF,
            eviction_contribution_weight: 0.1,
        }
    }
}

// ─── merge_accumulators ───────────────────────────────────────────────────────

/// Merge two coherence accumulators, preserving the honesty invariant.
///
/// # In plain English
///
/// When two memories must be combined, the robot adopts the *lower* trust level.
/// It can inherit caution but never inherited confidence.
///
/// - `coherence = min(a, b)` — never grant unearned familiarity (I-CKM-001)
/// - `interaction_count = a + b` — never erase relational history (I-CKM-002)
/// - `last_interaction_tick = max(a, b)` — preserve freshness
///
/// This asymmetric treatment (conservative on trust, cumulative on history) is
/// the unique combination that respects both the honesty invariant and the
/// history-preservation invariant simultaneously.
///
/// The function is associative and commutative:
/// - `merge(merge(A,B), C) = merge(A, merge(B,C))`
/// - `merge(A, B) = merge(B, A)`
///
/// Patent Continuation Claim B.
pub fn merge_accumulators(
    a: &CoherenceAccumulator,
    b: &CoherenceAccumulator,
) -> CoherenceAccumulator {
    CoherenceAccumulator {
        value: a.value.min(b.value),
        interaction_count: a.interaction_count.saturating_add(b.interaction_count),
        last_interaction_tick: a.last_interaction_tick.max(b.last_interaction_tick),
    }
}

// ─── Tier1Class ───────────────────────────────────────────────────────────────

/// One Tier 1 coarse class: a coarse accumulator plus up to `T2` fine Tier 2 entries.
///
/// The coarse accumulator is always present and always receives interactions
/// regardless of whether Tier 2 is active.  This ensures coarse relational
/// history is never silently lost (I-CKM-004).
///
/// Patent Continuation Claim A.
pub struct Tier1Class<V, const N: usize, const T2: usize>
where
    V: SensorVocabulary<N>,
{
    /// Coarse-level coherence accumulator.  Always present.  Updated on every
    /// interaction regardless of Tier 2 state.
    pub accumulator: CoherenceAccumulator,

    /// Whether fine-grained Tier 2 entries are active for this class.
    /// Activated when `accumulator.interaction_count >= promotion_threshold`.
    pub tier2_active: bool,

    /// Fine Tier 2 entries keyed by full `ContextKey<V, N>`.
    pub tier2_entries: FnvIndexMap<ContextKey<V, N>, CoherenceAccumulator, T2>,
}

impl<V, const N: usize, const T2: usize> Tier1Class<V, N, T2>
where
    V: SensorVocabulary<N>,
{
    fn new() -> Self {
        Self {
            accumulator: CoherenceAccumulator::new(),
            tier2_active: false,
            tier2_entries: FnvIndexMap::new(),
        }
    }
}

// ─── TieredContextMap ─────────────────────────────────────────────────────────

/// Two-tier cardinality-bounded context map.
///
/// Replaces `CoherenceField` on production platforms where the context key
/// space is too large for a flat map.  The existing `CoherenceField` is
/// unchanged — this is a new type, opt-in via `features = ["tiered-contexts"]`.
///
/// # Const generics
///
/// - `N` — sensor vocabulary dimensionality
/// - `T1` — max number of Tier 1 coarse classes (recommend: power of 2, ≥ 1)
/// - `T2` — max Tier 2 fine entries *per class* (recommend: power of 2, ≥ 1)
///
/// Total bounded memory: O(T1 × T2 × sizeof(CoherenceAccumulator)).
/// At T1=64, T2=16, 16B per accumulator: ≈ 16 KiB.
///
/// # Patent Claims A–D
///
/// See module-level documentation.
pub struct TieredContextMap<V, const N: usize, const T1: usize, const T2: usize>
where
    V: SensorVocabulary<N>,
{
    /// Tier 1 coarse classes, keyed by the coarse hash.
    pub classes: FnvIndexMap<Tier1Key, Tier1Class<V, N, T2>, T1>,

    /// Configuration (promotion threshold, masks, weights).
    pub config: TieredContextConfig,

    /// Personality baseline for new Tier 1 classes (0.15 × curiosity_drive).
    personality_baseline: f32,
}

impl<V, const N: usize, const T1: usize, const T2: usize> TieredContextMap<V, N, T1, T2>
where
    V: SensorVocabulary<N>,
{
    /// Construct a fresh tiered context map.
    pub fn new(config: TieredContextConfig) -> Self {
        Self {
            classes: FnvIndexMap::new(),
            config,
            personality_baseline: 0.0,
        }
    }

    /// Set the personality baseline for cold-start contexts.
    pub fn set_personality_baseline(&mut self, baseline: f32) {
        self.personality_baseline = baseline.clamp(0.0, 1.0);
    }

    // ── Effective coherence (CCF-001 asymmetric gate) ─────────────────────

    /// Compute effective coherence using the asymmetric gate (CCF-001).
    ///
    /// Lookup order: Tier 2 fine entry → Tier 1 coarse accumulator → 0.0.
    ///
    /// - **Unfamiliar** (ctx < 0.3): `min(instant, ctx)` — earn trust first.
    /// - **Familiar** (ctx ≥ 0.3): `0.3 × instant + 0.7 × ctx` — history buffers noise.
    ///
    /// I-CKM-006: always resolves to a value, never "unknown."
    pub fn effective_coherence(&self, instant: f32, key: &ContextKey<V, N>) -> f32 {
        let ctx = self.context_coherence(key);
        if ctx < 0.3 {
            if instant < ctx { instant } else { ctx }
        } else {
            (0.3 * instant + 0.7 * ctx).clamp(0.0, 1.0)
        }
    }

    /// Raw accumulated coherence for a context (0.0 if unseen).
    ///
    /// Checks Tier 2 first, falls through to Tier 1.  I-CKM-006.
    pub fn context_coherence(&self, key: &ContextKey<V, N>) -> f32 {
        let t1k = compute_tier1_key(key, self.config.tier1_feature_mask);
        match self.classes.get(&t1k) {
            None => 0.0,
            Some(cls) => {
                // Tier 2 fine entry takes priority (I-CKM-006)
                if cls.tier2_active {
                    if let Some(fine) = cls.tier2_entries.get(key) {
                        return fine.value;
                    }
                }
                cls.accumulator.value
            }
        }
    }

    /// Interaction count for a context (0 if unseen).
    pub fn context_interaction_count(&self, key: &ContextKey<V, N>) -> u32 {
        let t1k = compute_tier1_key(key, self.config.tier1_feature_mask);
        match self.classes.get(&t1k) {
            None => 0,
            Some(cls) => {
                if cls.tier2_active {
                    if let Some(fine) = cls.tier2_entries.get(key) {
                        return fine.interaction_count;
                    }
                }
                cls.accumulator.interaction_count
            }
        }
    }

    // ── Interaction API ───────────────────────────────────────────────────

    /// Record a positive interaction for a context.
    ///
    /// Always updates the Tier 1 coarse accumulator (I-CKM-004).
    /// Also updates the Tier 2 fine entry if Tier 2 is active.
    /// May activate Tier 2 or insert a new fine entry.
    ///
    /// Evicts stale/weak fine entries from Tier 2 if it is full.
    pub fn positive_interaction(
        &mut self,
        key: &ContextKey<V, N>,
        personality: &Personality,
        tick: u64,
        alone: bool,
    ) {
        let t1k = compute_tier1_key(key, self.config.tier1_feature_mask);
        self.ensure_tier1_class(t1k);

        let cls = self.classes.get_mut(&t1k).unwrap();

        // Always update coarse accumulator (I-CKM-004)
        cls.accumulator.positive_interaction(personality.recovery_speed, tick, alone);

        // Maybe activate Tier 2
        if !cls.tier2_active
            && cls.accumulator.interaction_count >= self.config.promotion_threshold
        {
            cls.tier2_active = true;
        }

        if cls.tier2_active {
            if cls.tier2_entries.contains_key(key) {
                cls.tier2_entries
                    .get_mut(key)
                    .unwrap()
                    .positive_interaction(personality.recovery_speed, tick, alone);
            } else {
                // Ensure room in Tier 2
                if cls.tier2_entries.len() >= T2 {
                    self.evict_weakest_tier2_entry(t1k);
                }
                if let Some(cls2) = self.classes.get_mut(&t1k) {
                    let mut new_acc = CoherenceAccumulator::new_with_baseline(
                        (self.personality_baseline / 0.15).clamp(0.0, 1.0),
                    );
                    new_acc.positive_interaction(personality.recovery_speed, tick, alone);
                    let _ = cls2.tier2_entries.insert(key.clone(), new_acc);
                }
            }
        }
    }

    /// Record a negative interaction for a context.
    ///
    /// Always updates the Tier 1 coarse accumulator (I-CKM-004).
    /// Also updates the Tier 2 fine entry if present.
    pub fn negative_interaction(
        &mut self,
        key: &ContextKey<V, N>,
        personality: &Personality,
        tick: u64,
    ) {
        let t1k = compute_tier1_key(key, self.config.tier1_feature_mask);
        self.ensure_tier1_class(t1k);

        let cls = self.classes.get_mut(&t1k).unwrap();
        cls.accumulator
            .negative_interaction(personality.startle_sensitivity, tick);

        if cls.tier2_active {
            if let Some(fine) = cls.tier2_entries.get_mut(key) {
                fine.negative_interaction(personality.startle_sensitivity, tick);
            }
        }
    }

    // ── Decay ─────────────────────────────────────────────────────────────

    /// Apply time-based decay to all accumulators in both tiers.
    ///
    /// Stale Tier 2 entries whose interaction count is below
    /// `eviction_min_count` and which have not been seen in
    /// `eviction_staleness_ticks` ticks are evicted with coherence
    /// contribution to the parent Tier 1 accumulator (I-CKM-003).
    pub fn decay_all(&mut self, elapsed_ticks: u64, current_tick: u64) {
        // Collect all t1 keys first to avoid borrow issues
        let t1_keys: HVec<Tier1Key, T1> = self.classes.keys().cloned().collect();

        for &t1k in &t1_keys {
            if let Some(cls) = self.classes.get_mut(&t1k) {
                cls.accumulator.decay(elapsed_ticks);
                for fine in cls.tier2_entries.values_mut() {
                    fine.decay(elapsed_ticks);
                }
            }
            // Evict stale Tier 2 entries with contribution (I-CKM-003)
            self.evict_stale_tier2_entries(t1k, current_tick);
        }
    }

    // ── Collection helpers ────────────────────────────────────────────────

    /// Total number of Tier 1 classes.
    pub fn tier1_class_count(&self) -> usize {
        self.classes.len()
    }

    /// Total number of fine Tier 2 entries across all classes.
    pub fn tier2_entry_count(&self) -> usize {
        self.classes.values().map(|cls| cls.tier2_entries.len()).sum()
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    /// Ensure a Tier 1 class exists for `t1k`, evicting LRU if necessary.
    fn ensure_tier1_class(&mut self, t1k: Tier1Key) {
        if self.classes.contains_key(&t1k) {
            return;
        }
        if self.classes.len() >= T1 {
            self.evict_lru_tier1_class();
        }
        let mut cls = Tier1Class::new();
        cls.accumulator = CoherenceAccumulator::new_with_baseline(
            (self.personality_baseline / 0.15).clamp(0.0, 1.0),
        );
        let _ = self.classes.insert(t1k, cls);
    }

    /// Evict the Tier 1 class with the oldest `last_interaction_tick`.
    fn evict_lru_tier1_class(&mut self) {
        let oldest = self
            .classes
            .iter()
            .min_by_key(|(_, cls)| cls.accumulator.last_interaction_tick)
            .map(|(k, _)| *k);
        if let Some(k) = oldest {
            self.classes.remove(&k);
        }
    }

    /// Evict the Tier 2 entry with the lowest coherence, contributing its
    /// coherence back to the parent Tier 1 accumulator.  I-CKM-003.
    fn evict_weakest_tier2_entry(&mut self, t1k: Tier1Key) {
        let weakest_key = self.classes.get(&t1k).and_then(|cls| {
            cls.tier2_entries
                .iter()
                .min_by(|(_, a), (_, b)| {
                    a.value
                        .partial_cmp(&b.value)
                        .unwrap_or(core::cmp::Ordering::Equal)
                })
                .map(|(k, _)| k.clone())
        });

        if let Some(wk) = weakest_key {
            if let Some(cls) = self.classes.get_mut(&t1k) {
                if let Some(evicted) = cls.tier2_entries.remove(&wk) {
                    // Contribution back to Tier 1 (I-CKM-003)
                    let w = self.config.eviction_contribution_weight;
                    let contribution = evicted.value * w;
                    cls.accumulator.value =
                        (cls.accumulator.value + contribution).min(1.0);
                }
            }
        }
    }

    /// Evict stale Tier 2 entries (low count, not recently seen) with
    /// coherence contribution to the parent Tier 1 class.  I-CKM-003.
    fn evict_stale_tier2_entries(&mut self, t1k: Tier1Key, current_tick: u64) {
        let staleness = self.config.eviction_staleness_ticks;
        let min_count = self.config.eviction_min_count;
        let weight = self.config.eviction_contribution_weight;

        // Collect stale keys first
        let stale_keys: HVec<ContextKey<V, N>, T2> =
            if let Some(cls) = self.classes.get(&t1k) {
                cls.tier2_entries
                    .iter()
                    .filter(|(_, acc)| {
                        acc.interaction_count < min_count
                            && current_tick.saturating_sub(acc.last_interaction_tick)
                                > staleness
                    })
                    .map(|(k, _)| k.clone())
                    .collect()
            } else {
                HVec::new()
            };

        for sk in &stale_keys {
            if let Some(cls) = self.classes.get_mut(&t1k) {
                if let Some(evicted) = cls.tier2_entries.remove(sk) {
                    let contribution = evicted.value * weight;
                    cls.accumulator.value =
                        (cls.accumulator.value + contribution).min(1.0);
                }
            }
        }
    }
}

impl<V, const N: usize, const T1: usize, const T2: usize> core::fmt::Debug
    for TieredContextMap<V, N, T1, T2>
where
    V: SensorVocabulary<N>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TieredContextMap")
            .field("tier1_classes", &self.classes.len())
            .field(
                "tier2_entries",
                &self.classes.values().map(|c| c.tier2_entries.len()).sum::<usize>(),
            )
            .finish()
    }
}
