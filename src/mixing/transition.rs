/*
 * Notice of Provisional Patent Filing:
 * The methods and algorithms implemented in this file are the subject of a
 * United States Provisional Patent Application (63/988,438)
 * filed on February 23, 2026.
 *
 * This source code is licensed under the Business Source License 1.1.
 */

//! Transition smoothing helpers for cluster restructure events.
//!
//! When the hierarchical mixer's cluster structure changes (split, merge, or
//! reassignment), the behavioral output is blended between the old and new
//! structures over a configurable number of ticks to prevent discontinuities.
//!
//! The blend is defined as:
//! ```text
//! H_effective(t) = (1 − α(t)) × H_old  +  α(t) × H_new
//! ```
//! where α(t) ramps linearly from 0.0 to 1.0 over `transition_blend_ticks`.
//!
//! This stays within the Birkhoff polytope by convexity of the set of doubly
//! stochastic matrices.  No additional Sinkhorn-Knopp projection is needed
//! during the blend.

// ─── blend_alpha ─────────────────────────────────────────────────────────────

/// Compute the linear blend factor α(t) for transition smoothing.
///
/// Returns a value in `[0.0, 1.0]`:
/// - `0.0` at `transition_tick == 0` (fully old structure)
/// - `1.0` at `transition_tick >= blend_ticks` (fully new structure)
///
/// If `blend_ticks == 0` the result is always `1.0` (instantaneous transition).
///
/// # Arguments
/// - `transition_tick` — number of ticks elapsed since the transition started.
/// - `blend_ticks` — total duration of the blend in ticks (from config).
#[inline]
pub fn blend_alpha(transition_tick: usize, blend_ticks: usize) -> f32 {
    if blend_ticks == 0 {
        return 1.0;
    }
    let alpha = transition_tick as f32 / blend_ticks as f32;
    if alpha > 1.0 { 1.0 } else { alpha }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_alpha_zero_tick_is_zero() {
        assert_eq!(blend_alpha(0, 100), 0.0);
    }

    #[test]
    fn blend_alpha_full_tick_is_one() {
        assert_eq!(blend_alpha(100, 100), 1.0);
    }

    #[test]
    fn blend_alpha_past_end_clamps_to_one() {
        assert_eq!(blend_alpha(150, 100), 1.0);
    }

    #[test]
    fn blend_alpha_zero_blend_ticks_is_one() {
        assert_eq!(blend_alpha(0, 0), 1.0);
    }

    #[test]
    fn blend_alpha_midpoint() {
        let a = blend_alpha(50, 100);
        assert!((a - 0.5).abs() < 1e-6);
    }
}
