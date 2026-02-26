//! CCF_SEG — portable snapshot of a CoherenceField for persistence and transport.
//!
//! # Binary layout (big-endian)
//!
//! ```text
//! Header (32 bytes):
//!   [0..4]   magic:              0x43_43_46_53 ("CCFS")
//!   [4..6]   version:            u16 = 1
//!   [6..8]   context_count:      u16
//!   [8..16]  created_at:         i64 (Unix timestamp, 0 if unknown)
//!   [16..24] last_active_at:     i64 (Unix timestamp of last interaction)
//!   [24..32] total_interactions: u64
//! Personality block (12 bytes): [f32; 3]
//! Context table: context_count × CCFContextRecord
//! ```
//!
//! The snapshot is populated from a live [`CoherenceField`] by iterating its entries.
//! It operates at the hash level — the vocabulary is not stored, only the FNV-1a hash
//! of each context key. The runtime reconstructs the vocabulary from live sensor readings.
//!
//! # no_std
//!
//! This module requires the `serde` feature. It uses `alloc::vec::Vec` via the
//! `serde` feature path and is compatible with no_std + alloc environments.
//!
//! [`CoherenceField`]: crate::accumulator::CoherenceField

extern crate alloc;

use alloc::vec::Vec;

use crate::accumulator::{CoherenceAccumulator, CoherenceField};
use crate::phase::Personality;
use crate::vocabulary::SensorVocabulary;

/// Magic bytes identifying a CCF_SEG binary blob: "CCFS".
pub const CCF_SEG_MAGIC: u32 = 0x43_43_46_53;

/// Current CCF_SEG format version.
pub const CCF_SEG_VERSION: u16 = 1;

/// A serializable snapshot of a [`CoherenceField`] and [`Personality`] state.
///
/// Captures all context accumulators at the hash level so that the snapshot
/// can be persisted and restored without re-running sensor reads. The vocabulary
/// type is erased — only the FNV-1a context hashes are stored.
///
/// # Example
///
/// ```rust,ignore
/// use ccf_core::seg::CcfSegSnapshot;
/// use ccf_core::accumulator::CoherenceField;
/// use ccf_core::phase::Personality;
///
/// let snapshot = CcfSegSnapshot::from_field(&field, &personality, 0, 0, 0);
/// let json = serde_json::to_string(&snapshot).unwrap();
/// let restored: CcfSegSnapshot = serde_json::from_str(&json).unwrap();
/// ```
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug, PartialEq)]
pub struct CcfSegSnapshot {
    /// Format version — always [`CCF_SEG_VERSION`] for newly created snapshots.
    pub version: u16,
    /// Unix timestamp (seconds) when the field was first created. 0 if unknown.
    pub created_at: i64,
    /// Unix timestamp (seconds) of the most recent interaction.
    pub last_active_at: i64,
    /// Total number of interactions recorded across all contexts since creation.
    pub total_interactions: u64,
    /// Personality modulators at snapshot time.
    pub personality: PersonalityRecord,
    /// All tracked context accumulators, in iteration order.
    pub contexts: Vec<ContextRecord>,
}

/// Serializable representation of [`Personality`] modulators.
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug, PartialEq)]
pub struct PersonalityRecord {
    /// Curiosity drive [0.0, 1.0].
    pub curiosity_drive: f32,
    /// Startle sensitivity [0.0, 1.0].
    pub startle_sensitivity: f32,
    /// Recovery speed [0.0, 1.0].
    pub recovery_speed: f32,
}

impl From<&Personality> for PersonalityRecord {
    fn from(p: &Personality) -> Self {
        Self {
            curiosity_drive: p.curiosity_drive,
            startle_sensitivity: p.startle_sensitivity,
            recovery_speed: p.recovery_speed,
        }
    }
}

impl From<&PersonalityRecord> for Personality {
    fn from(r: &PersonalityRecord) -> Self {
        Self {
            curiosity_drive: r.curiosity_drive,
            startle_sensitivity: r.startle_sensitivity,
            recovery_speed: r.recovery_speed,
        }
    }
}

/// Serializable representation of a single context accumulator entry.
///
/// The context is identified by its FNV-1a hash rather than the full vocabulary
/// value. This is sufficient for persistence — the vocabulary is re-built from
/// live sensor readings when the runtime restores the field.
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug, PartialEq)]
pub struct ContextRecord {
    /// FNV-1a hash of the context key (from [`ContextKey::context_hash_u32`]).
    ///
    /// [`ContextKey::context_hash_u32`]: crate::vocabulary::ContextKey::context_hash_u32
    pub context_hash: u32,
    /// Accumulated coherence value [0.0, 1.0].
    pub coherence_value: f32,
    /// Total positive interactions recorded for this context.
    pub interaction_count: u32,
    /// Tick of the most recent interaction.
    pub last_interaction_tick: u64,
}

impl From<(u32, &CoherenceAccumulator)> for ContextRecord {
    fn from((hash, acc): (u32, &CoherenceAccumulator)) -> Self {
        Self {
            context_hash: hash,
            coherence_value: acc.value,
            interaction_count: acc.interaction_count,
            last_interaction_tick: acc.last_interaction_tick,
        }
    }
}

impl CcfSegSnapshot {
    /// Build a snapshot from a live [`CoherenceField`] and [`Personality`].
    ///
    /// - `field`: the coherence field to snapshot.
    /// - `personality`: the personality modulators at snapshot time.
    /// - `created_at`: Unix timestamp when the field was originally created (0 if unknown).
    /// - `last_active_at`: Unix timestamp of the most recent interaction (0 if unknown).
    /// - `total_interactions`: cumulative interaction count since creation.
    pub fn from_field<V, const N: usize>(
        field: &CoherenceField<V, N>,
        personality: &Personality,
        created_at: i64,
        last_active_at: i64,
        total_interactions: u64,
    ) -> Self
    where
        V: SensorVocabulary<N>,
    {
        let contexts: Vec<ContextRecord> = field
            .iter()
            .map(|(key, acc)| ContextRecord::from((key.context_hash_u32(), acc)))
            .collect();

        Self {
            version: CCF_SEG_VERSION,
            created_at,
            last_active_at,
            total_interactions,
            personality: PersonalityRecord::from(personality),
            contexts,
        }
    }

    /// Number of context entries in this snapshot.
    pub fn context_count(&self) -> usize {
        self.contexts.len()
    }

    /// Look up a context record by its FNV-1a hash.
    ///
    /// Returns `None` if the hash is not present in this snapshot.
    pub fn find_context(&self, hash: u32) -> Option<&ContextRecord> {
        self.contexts.iter().find(|r| r.context_hash == hash)
    }
}
