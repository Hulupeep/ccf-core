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

//! Social phase classification and personality modulators.
//!
//! # Patent Claims 3, 14–18
//!
//! - [`Personality`]: dynamic modulators — curiosity, startle sensitivity, recovery speed (Claim 3).
//! - [`SocialPhase`]: four-quadrant phase classifier with Schmitt trigger hysteresis (Claims 14–18).
//! - [`PhaseSpace`]: configurable thresholds for quadrant transitions (Claim 14).
//!
//! # Invariants
//!
//! - **CCF-003**: Personality modulates deltas, not structure.
//! - **CCF-004**: Quadrant boundaries use hysteresis (≈0.10 deadband) to prevent oscillation.
//! - **I-DIST-001**: no_std compatible.
//! - **I-DIST-005**: Zero unsafe code.

// ─── Personality ────────────────────────────────────────────────────────────

/// Dynamic personality modulators.
///
/// These three parameters are bounded in [0.0, 1.0] and modulate the *rate* of
/// coherence change, not the structural invariants (CCF-003).
///
/// Patent Claim 3 (modulators).
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Personality {
    /// Drive to explore new contexts. Scales the cold-start baseline and
    /// the rate of positive coherence accumulation.
    ///
    /// Range [0.0, 1.0]. Default 0.5.
    pub curiosity_drive: f32,
    /// Sensitivity to startling or aversive events. Scales the magnitude of
    /// negative-interaction drops.
    ///
    /// Range [0.0, 1.0]. Default 0.5.
    pub startle_sensitivity: f32,
    /// Speed of coherence recovery after disruption. Scales the delta applied
    /// by positive interactions.
    ///
    /// Range [0.0, 1.0]. Default 0.5.
    pub recovery_speed: f32,
}

impl Personality {
    /// Construct the default mid-range personality (all parameters at 0.5).
    pub fn new() -> Self {
        Self {
            curiosity_drive: 0.5,
            startle_sensitivity: 0.5,
            recovery_speed: 0.5,
        }
    }

    /// Scale a base coherence gain delta by this personality's `recovery_speed`.
    ///
    /// Returns `base * (0.5 + recovery_speed)`, clamped to [0.0, 2.0 * base].
    pub fn modulate_coherence_gain(&self, base: f32) -> f32 {
        base * (0.5 + self.recovery_speed)
    }

    /// Scale a base startle drop by this personality's `startle_sensitivity`.
    ///
    /// Returns `base * (0.5 + startle_sensitivity)`, clamped to [0.0, 2.0 * base].
    pub fn modulate_startle_drop(&self, base: f32) -> f32 {
        base * (0.5 + self.startle_sensitivity)
    }
}

impl Default for Personality {
    fn default() -> Self {
        Self::new()
    }
}

// ─── PhaseSpace (configurable Schmitt trigger thresholds) ────────────────────

/// Configurable thresholds for [`SocialPhase`] transitions.
///
/// Uses hysteresis (Schmitt trigger): the *enter* threshold is higher than the
/// *exit* threshold so the robot does not oscillate at phase boundaries (CCF-004).
///
/// Default thresholds:
/// - Coherence high: enter ≥ 0.65, exit ≥ 0.55 (10-point deadband).
/// - Tension high: enter ≥ 0.45, exit ≥ 0.35 (10-point deadband).
///
/// Patent Claim 14.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PhaseSpace {
    /// Coherence threshold to *enter* the high-coherence quadrants (QuietlyBeloved, ProtectiveGuardian).
    pub coherence_high_enter: f32,
    /// Coherence threshold to *stay in* the high-coherence quadrants (exit when below).
    pub coherence_high_exit: f32,
    /// Tension threshold to *enter* the high-tension quadrants (StartledRetreat, ProtectiveGuardian).
    pub tension_high_enter: f32,
    /// Tension threshold to *stay in* the high-tension quadrants (exit when below).
    pub tension_high_exit: f32,
}

impl PhaseSpace {
    /// Construct the standard PhaseSpace with default thresholds.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for PhaseSpace {
    fn default() -> Self {
        Self {
            coherence_high_enter: 0.65,
            coherence_high_exit: 0.55,
            tension_high_enter: 0.45,
            tension_high_exit: 0.35,
        }
    }
}

// ─── SocialPhase ─────────────────────────────────────────────────────────────

/// Behavioral phase from the 2D (coherence × tension) space.
///
/// The four quadrants of the phase plane:
///
/// ```text
///              │ Low tension        │ High tension
/// ─────────────┼────────────────────┼──────────────────────
/// Low coherence│ ShyObserver        │ StartledRetreat
/// High coherence│ QuietlyBeloved    │ ProtectiveGuardian
/// ```
///
/// Transitions use hysteresis (CCF-004) to prevent oscillation at boundaries.
///
/// Patent Claims 14–18.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SocialPhase {
    /// Low coherence, low tension: minimal expression, cautious observation.
    ShyObserver,
    /// Low coherence, high tension: protective reflex with additional withdrawal.
    StartledRetreat,
    /// High coherence, low tension: full expressive range — "small flourishes".
    QuietlyBeloved,
    /// High coherence, high tension: protective but with relational context.
    ProtectiveGuardian,
}

impl SocialPhase {
    /// Determine the current social phase using Schmitt trigger hysteresis (CCF-004).
    ///
    /// - `effective_coherence`: output of `CoherenceField::effective_coherence()` in [0.0, 1.0].
    /// - `tension`: current tension from homeostasis in [0.0, 1.0].
    /// - `prev`: the phase from the previous tick (enables hysteresis).
    /// - `ps`: configurable thresholds for quadrant transitions.
    pub fn classify(
        effective_coherence: f32,
        tension: f32,
        prev: SocialPhase,
        ps: &PhaseSpace,
    ) -> SocialPhase {
        let high_coherence = match prev {
            SocialPhase::QuietlyBeloved | SocialPhase::ProtectiveGuardian => {
                effective_coherence >= ps.coherence_high_exit
            }
            _ => effective_coherence >= ps.coherence_high_enter,
        };

        let high_tension = match prev {
            SocialPhase::StartledRetreat | SocialPhase::ProtectiveGuardian => {
                tension >= ps.tension_high_exit
            }
            _ => tension >= ps.tension_high_enter,
        };

        match (high_coherence, high_tension) {
            (false, false) => SocialPhase::ShyObserver,
            (false, true) => SocialPhase::StartledRetreat,
            (true, false) => SocialPhase::QuietlyBeloved,
            (true, true) => SocialPhase::ProtectiveGuardian,
        }
    }

    /// Scale factor for expressive output in this phase [0.0, 1.0].
    ///
    /// Delegates to [`permeability`] with representative mid-range values
    /// (coherence = 0.5, tension = 0.3) for backward-compatible ordering.
    /// New code should call [`permeability`] directly for full control.
    pub fn expression_scale(&self) -> f32 {
        permeability(0.5, 0.3, *self)
    }

    /// LED color tint for this phase (overlaid on reflex mode color).
    pub fn led_tint(&self) -> [u8; 3] {
        match self {
            SocialPhase::ShyObserver => [40, 40, 80],          // Muted blue-grey
            SocialPhase::StartledRetreat => [80, 20, 20],      // Dark red
            SocialPhase::QuietlyBeloved => [60, 120, 200],     // Warm blue
            SocialPhase::ProtectiveGuardian => [200, 100, 0],  // Amber
        }
    }
}

// ─── Output Permeability ─────────────────────────────────────────────────────

/// Compute output permeability — how much personality expression passes through.
///
/// The quadrant determines qualitative behavior; the position within the quadrant
/// determines intensity. This scalar scales all output channels (motor speed,
/// LED intensity, sound probability, narration depth).
///
/// # Ranges per quadrant
///
/// | Quadrant | Range | Formula |
/// |---|---|---|
/// | ShyObserver | [0.0, 0.3] | `effective_coherence × 0.3` |
/// | StartledRetreat | 0.1 fixed | reflexive, not expressive |
/// | QuietlyBeloved | [0.5, 1.0] | `0.5 + effective_coherence × 0.5` |
/// | ProtectiveGuardian | [0.4, 0.6] | `0.4 + effective_coherence × 0.2` |
pub fn permeability(effective_coherence: f32, _tension: f32, quadrant: SocialPhase) -> f32 {
    match quadrant {
        SocialPhase::ShyObserver => effective_coherence * 0.3,
        SocialPhase::StartledRetreat => 0.1,
        SocialPhase::QuietlyBeloved => 0.5 + effective_coherence * 0.5,
        SocialPhase::ProtectiveGuardian => 0.4 + effective_coherence * 0.2,
    }
}

/// Narration depth levels gated by output permeability.
///
/// Determines how much reflection the robot performs based on the current
/// permeability. Lower permeability means less narration overhead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NarrationDepth {
    /// permeability < 0.2: No reflection.
    None,
    /// permeability 0.2–0.4: Factual observations only.
    Minimal,
    /// permeability 0.4–0.6: Contextual awareness.
    Brief,
    /// permeability 0.6–0.8: Personality-colored narration.
    Full,
    /// permeability > 0.8: Phenomenological reflection.
    Deep,
}

impl NarrationDepth {
    /// Map a permeability scalar to a narration depth level.
    pub fn from_permeability(p: f32) -> Self {
        if p < 0.2 {
            NarrationDepth::None
        } else if p < 0.4 {
            NarrationDepth::Minimal
        } else if p < 0.6 {
            NarrationDepth::Brief
        } else if p < 0.8 {
            NarrationDepth::Full
        } else {
            NarrationDepth::Deep
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Personality tests ─────────────────────────────────────────────────

    #[test]
    fn test_personality_new_mid_range() {
        let p = Personality::new();
        assert!((p.curiosity_drive - 0.5).abs() < f32::EPSILON);
        assert!((p.startle_sensitivity - 0.5).abs() < f32::EPSILON);
        assert!((p.recovery_speed - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_personality_modulate_coherence_gain() {
        let p = Personality { curiosity_drive: 0.5, startle_sensitivity: 0.5, recovery_speed: 0.9 };
        // base * (0.5 + 0.9) = base * 1.4
        let result = p.modulate_coherence_gain(0.02);
        assert!((result - 0.02 * 1.4).abs() < f32::EPSILON, "got {}", result);
    }

    #[test]
    fn test_personality_modulate_startle_drop() {
        let p = Personality { curiosity_drive: 0.5, startle_sensitivity: 0.1, recovery_speed: 0.5 };
        // base * (0.5 + 0.1) = base * 0.6
        let result = p.modulate_startle_drop(0.05);
        assert!((result - 0.05 * 0.6).abs() < f32::EPSILON, "got {}", result);
    }

    // ── PhaseSpace tests ──────────────────────────────────────────────────

    #[test]
    fn test_phase_space_default_values() {
        let ps = PhaseSpace::default();
        assert!((ps.coherence_high_enter - 0.65).abs() < f32::EPSILON);
        assert!((ps.coherence_high_exit - 0.55).abs() < f32::EPSILON);
        assert!((ps.tension_high_enter - 0.45).abs() < f32::EPSILON);
        assert!((ps.tension_high_exit - 0.35).abs() < f32::EPSILON);
    }

    // ── SocialPhase classification tests ──────────────────────────────────

    #[test]
    fn test_social_phase_shy_observer() {
        let ps = PhaseSpace::default();
        let phase = SocialPhase::classify(0.1, 0.1, SocialPhase::ShyObserver, &ps);
        assert_eq!(phase, SocialPhase::ShyObserver);
    }

    #[test]
    fn test_social_phase_quietly_beloved() {
        let ps = PhaseSpace::default();
        let phase = SocialPhase::classify(0.8, 0.1, SocialPhase::ShyObserver, &ps);
        assert_eq!(phase, SocialPhase::QuietlyBeloved);
    }

    #[test]
    fn test_social_phase_startled_retreat() {
        let ps = PhaseSpace::default();
        let phase = SocialPhase::classify(0.1, 0.7, SocialPhase::ShyObserver, &ps);
        assert_eq!(phase, SocialPhase::StartledRetreat);
    }

    #[test]
    fn test_social_phase_protective_guardian() {
        let ps = PhaseSpace::default();
        let phase = SocialPhase::classify(0.8, 0.7, SocialPhase::ShyObserver, &ps);
        assert_eq!(phase, SocialPhase::ProtectiveGuardian);
    }

    #[test]
    fn test_social_phase_hysteresis_coherence() {
        let ps = PhaseSpace::default();

        // Enter QuietlyBeloved above enter threshold
        let phase = SocialPhase::classify(0.66, 0.1, SocialPhase::ShyObserver, &ps);
        assert_eq!(phase, SocialPhase::QuietlyBeloved);

        // Stay in QuietlyBeloved above exit threshold (0.55)
        let phase = SocialPhase::classify(0.56, 0.1, SocialPhase::QuietlyBeloved, &ps);
        assert_eq!(phase, SocialPhase::QuietlyBeloved);

        // Exit QuietlyBeloved below exit threshold
        let phase = SocialPhase::classify(0.54, 0.1, SocialPhase::QuietlyBeloved, &ps);
        assert_eq!(phase, SocialPhase::ShyObserver);
    }

    #[test]
    fn test_social_phase_hysteresis_tension() {
        let ps = PhaseSpace::default();

        // Enter StartledRetreat above enter threshold (0.45)
        let phase = SocialPhase::classify(0.1, 0.46, SocialPhase::ShyObserver, &ps);
        assert_eq!(phase, SocialPhase::StartledRetreat);

        // Stay in StartledRetreat above exit threshold (0.35)
        let phase = SocialPhase::classify(0.1, 0.36, SocialPhase::StartledRetreat, &ps);
        assert_eq!(phase, SocialPhase::StartledRetreat);

        // Exit StartledRetreat below exit threshold
        let phase = SocialPhase::classify(0.1, 0.34, SocialPhase::StartledRetreat, &ps);
        assert_eq!(phase, SocialPhase::ShyObserver);
    }

    #[test]
    fn test_custom_thresholds_stricter() {
        let strict = PhaseSpace {
            coherence_high_enter: 0.80,
            coherence_high_exit: 0.70,
            ..PhaseSpace::default()
        };

        // 0.70 coherence: enough for default QB, but not strict
        let phase = SocialPhase::classify(0.70, 0.1, SocialPhase::ShyObserver, &strict);
        assert_eq!(
            phase,
            SocialPhase::ShyObserver,
            "coherence 0.70 should NOT enter QB with strict threshold 0.80"
        );

        // 0.85 coherence: above strict threshold
        let phase = SocialPhase::classify(0.85, 0.1, SocialPhase::ShyObserver, &strict);
        assert_eq!(
            phase,
            SocialPhase::QuietlyBeloved,
            "coherence 0.85 should enter QB with strict threshold 0.80"
        );

        // Hysteresis: stay in QB at 0.75 (above strict exit 0.70)
        let phase = SocialPhase::classify(0.75, 0.1, SocialPhase::QuietlyBeloved, &strict);
        assert_eq!(
            phase,
            SocialPhase::QuietlyBeloved,
            "coherence 0.75 should stay in QB (above exit 0.70)"
        );

        // Drop below strict exit: leave QB
        let phase = SocialPhase::classify(0.65, 0.1, SocialPhase::QuietlyBeloved, &strict);
        assert_eq!(
            phase,
            SocialPhase::ShyObserver,
            "coherence 0.65 should exit QB (below exit 0.70)"
        );
    }

    #[test]
    fn test_custom_thresholds_looser() {
        let loose = PhaseSpace {
            coherence_high_enter: 0.40,
            coherence_high_exit: 0.30,
            ..PhaseSpace::default()
        };

        // 0.42 coherence: not enough for default (0.65), but enough for loose
        let phase = SocialPhase::classify(0.42, 0.1, SocialPhase::ShyObserver, &loose);
        assert_eq!(
            phase,
            SocialPhase::QuietlyBeloved,
            "coherence 0.42 should enter QB with loose threshold 0.40"
        );

        // With default thresholds, stays ShyObserver
        let ps = PhaseSpace::default();
        let phase = SocialPhase::classify(0.42, 0.1, SocialPhase::ShyObserver, &ps);
        assert_eq!(
            phase,
            SocialPhase::ShyObserver,
            "coherence 0.42 should NOT enter QB with default threshold 0.65"
        );
    }

    #[test]
    fn test_full_quadrant_sweep_with_default_thresholds() {
        let ps = PhaseSpace::default();

        let cases: &[(f32, f32, SocialPhase, SocialPhase)] = &[
            (0.1, 0.1, SocialPhase::ShyObserver, SocialPhase::ShyObserver),
            (0.8, 0.1, SocialPhase::ShyObserver, SocialPhase::QuietlyBeloved),
            (0.1, 0.7, SocialPhase::ShyObserver, SocialPhase::StartledRetreat),
            (0.8, 0.7, SocialPhase::ShyObserver, SocialPhase::ProtectiveGuardian),
            // Hysteresis: stay in QB above exit
            (0.56, 0.1, SocialPhase::QuietlyBeloved, SocialPhase::QuietlyBeloved),
            // Hysteresis: exit QB below exit
            (0.54, 0.1, SocialPhase::QuietlyBeloved, SocialPhase::ShyObserver),
            // Hysteresis: stay in SR above tension exit
            (0.1, 0.36, SocialPhase::StartledRetreat, SocialPhase::StartledRetreat),
            // Hysteresis: exit SR below tension exit
            (0.1, 0.34, SocialPhase::StartledRetreat, SocialPhase::ShyObserver),
        ];

        for &(coh, ten, prev, expected) in cases {
            let result = SocialPhase::classify(coh, ten, prev, &ps);
            assert_eq!(
                result, expected,
                "coh={} ten={} prev={:?}: got {:?}, expected {:?}",
                coh, ten, prev, result, expected
            );
        }
    }

    #[test]
    fn test_expression_scale_ordering() {
        assert!(
            SocialPhase::QuietlyBeloved.expression_scale()
                > SocialPhase::ProtectiveGuardian.expression_scale()
        );
        assert!(
            SocialPhase::ProtectiveGuardian.expression_scale()
                > SocialPhase::ShyObserver.expression_scale()
        );
        assert!(
            SocialPhase::ShyObserver.expression_scale()
                > SocialPhase::StartledRetreat.expression_scale()
        );
    }

    #[test]
    fn test_led_tint_distinct() {
        let so = SocialPhase::ShyObserver.led_tint();
        let sr = SocialPhase::StartledRetreat.led_tint();
        let qb = SocialPhase::QuietlyBeloved.led_tint();
        let pg = SocialPhase::ProtectiveGuardian.led_tint();
        // All four tints must be distinct
        assert_ne!(so, sr);
        assert_ne!(so, qb);
        assert_ne!(so, pg);
        assert_ne!(sr, qb);
        assert_ne!(sr, pg);
        assert_ne!(qb, pg);
    }

    // ── Permeability tests ────────────────────────────────────────────────

    #[test]
    fn test_permeability_shy_observer_range() {
        let p_zero = permeability(0.0, 0.3, SocialPhase::ShyObserver);
        assert!((p_zero - 0.0).abs() < f32::EPSILON, "got {}", p_zero);

        let p_max = permeability(1.0, 0.3, SocialPhase::ShyObserver);
        assert!((p_max - 0.3).abs() < f32::EPSILON, "got {}", p_max);

        let p_mid = permeability(0.5, 0.3, SocialPhase::ShyObserver);
        assert!((p_mid - 0.15).abs() < f32::EPSILON, "got {}", p_mid);
    }

    #[test]
    fn test_permeability_startled_retreat_fixed() {
        for coh in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            for ten in &[0.0_f32, 0.5, 1.0] {
                let p = permeability(*coh, *ten, SocialPhase::StartledRetreat);
                assert!(
                    (p - 0.1).abs() < f32::EPSILON,
                    "SR should always be 0.1, got {} at coh={} ten={}",
                    p,
                    coh,
                    ten
                );
            }
        }
    }

    #[test]
    fn test_permeability_quietly_beloved_range() {
        let p_zero = permeability(0.0, 0.3, SocialPhase::QuietlyBeloved);
        assert!((p_zero - 0.5).abs() < f32::EPSILON, "got {}", p_zero);

        let p_max = permeability(1.0, 0.3, SocialPhase::QuietlyBeloved);
        assert!((p_max - 1.0).abs() < f32::EPSILON, "got {}", p_max);

        let p_mid = permeability(0.5, 0.3, SocialPhase::QuietlyBeloved);
        assert!((p_mid - 0.75).abs() < f32::EPSILON, "got {}", p_mid);
    }

    #[test]
    fn test_permeability_protective_guardian_range() {
        let p_zero = permeability(0.0, 0.3, SocialPhase::ProtectiveGuardian);
        assert!((p_zero - 0.4).abs() < f32::EPSILON, "got {}", p_zero);

        let p_max = permeability(1.0, 0.3, SocialPhase::ProtectiveGuardian);
        assert!((p_max - 0.6).abs() < f32::EPSILON, "got {}", p_max);

        let p_mid = permeability(0.5, 0.3, SocialPhase::ProtectiveGuardian);
        assert!((p_mid - 0.5).abs() < f32::EPSILON, "got {}", p_mid);
    }

    #[test]
    fn test_permeability_ordering() {
        let coh = 0.7;
        let ten = 0.3;
        let qb = permeability(coh, ten, SocialPhase::QuietlyBeloved);
        let pg = permeability(coh, ten, SocialPhase::ProtectiveGuardian);
        let so = permeability(coh, ten, SocialPhase::ShyObserver);
        let sr = permeability(coh, ten, SocialPhase::StartledRetreat);

        assert!(qb > pg, "QB({}) should be > PG({})", qb, pg);
        assert!(pg > so, "PG({}) should be > SO({})", pg, so);
        assert!(so > sr, "SO({}) should be > SR({})", so, sr);
    }

    #[test]
    fn test_expression_scale_matches_permeability() {
        let qb = SocialPhase::QuietlyBeloved.expression_scale();
        let pg = SocialPhase::ProtectiveGuardian.expression_scale();
        let so = SocialPhase::ShyObserver.expression_scale();
        let sr = SocialPhase::StartledRetreat.expression_scale();

        assert!((qb - permeability(0.5, 0.3, SocialPhase::QuietlyBeloved)).abs() < f32::EPSILON);
        assert!(
            (pg - permeability(0.5, 0.3, SocialPhase::ProtectiveGuardian)).abs() < f32::EPSILON
        );
        assert!((so - permeability(0.5, 0.3, SocialPhase::ShyObserver)).abs() < f32::EPSILON);
        assert!((sr - permeability(0.5, 0.3, SocialPhase::StartledRetreat)).abs() < f32::EPSILON);
    }

    // ── NarrationDepth tests ──────────────────────────────────────────────

    #[test]
    fn test_narration_depth_thresholds() {
        assert_eq!(NarrationDepth::from_permeability(0.0), NarrationDepth::None);
        assert_eq!(NarrationDepth::from_permeability(0.19), NarrationDepth::None);
        assert_eq!(NarrationDepth::from_permeability(0.2), NarrationDepth::Minimal);
        assert_eq!(NarrationDepth::from_permeability(0.39), NarrationDepth::Minimal);
        assert_eq!(NarrationDepth::from_permeability(0.4), NarrationDepth::Brief);
        assert_eq!(NarrationDepth::from_permeability(0.59), NarrationDepth::Brief);
        assert_eq!(NarrationDepth::from_permeability(0.6), NarrationDepth::Full);
        assert_eq!(NarrationDepth::from_permeability(0.79), NarrationDepth::Full);
        assert_eq!(NarrationDepth::from_permeability(0.8), NarrationDepth::Deep);
        assert_eq!(NarrationDepth::from_permeability(1.0), NarrationDepth::Deep);
    }

    #[test]
    fn test_narration_depth_matches_quadrants() {
        // ShyObserver at max coherence: p=0.3 -> Minimal
        assert_eq!(
            NarrationDepth::from_permeability(permeability(1.0, 0.3, SocialPhase::ShyObserver)),
            NarrationDepth::Minimal
        );
        // StartledRetreat: p=0.1 -> None
        assert_eq!(
            NarrationDepth::from_permeability(permeability(0.5, 0.5, SocialPhase::StartledRetreat)),
            NarrationDepth::None
        );
        // QuietlyBeloved at max coherence: p=1.0 -> Deep
        assert_eq!(
            NarrationDepth::from_permeability(permeability(1.0, 0.1, SocialPhase::QuietlyBeloved)),
            NarrationDepth::Deep
        );
        // ProtectiveGuardian at mid coherence: p=0.5 -> Brief
        assert_eq!(
            NarrationDepth::from_permeability(permeability(
                0.5,
                0.5,
                SocialPhase::ProtectiveGuardian
            )),
            NarrationDepth::Brief
        );
    }
}
