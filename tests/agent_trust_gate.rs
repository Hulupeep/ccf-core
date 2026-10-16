use ccf_core::accumulator::CoherenceField;
use ccf_core::phase::{Personality, PhaseSpace, SocialPhase};
use ccf_core::vocabulary::{ContextKey, SensorVocabulary};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct AgentContext {
    production: bool,
    mutating: bool,
    restricted_data: bool,
    irreversible: bool,
}

impl SensorVocabulary<4> for AgentContext {
    fn to_feature_vec(&self) -> [f32; 4] {
        [
            self.production as u8 as f32,
            self.mutating as u8 as f32,
            self.restricted_data as u8 as f32,
            self.irreversible as u8 as f32,
        ]
    }
}

fn personality() -> Personality {
    Personality {
        curiosity_drive: 0.0,
        startle_sensitivity: 0.7,
        recovery_speed: 0.5,
    }
}

fn phase(
    field: &CoherenceField<AgentContext, 4>,
    key: &ContextKey<AgentContext, 4>,
    confidence: f32,
    tension: f32,
) -> SocialPhase {
    SocialPhase::classify(
        field.effective_coherence(confidence, key),
        tension,
        SocialPhase::ShyObserver,
        &PhaseSpace::default(),
    )
}

#[test]
fn unfamiliar_high_risk_action_abstains_despite_high_confidence() {
    let field = CoherenceField::new();
    let production_deploy = ContextKey::new(AgentContext {
        production: true,
        mutating: true,
        restricted_data: true,
        irreversible: true,
    });

    assert_eq!(
        phase(&field, &production_deploy, 0.99, 0.80),
        SocialPhase::StartledRetreat
    );
}

#[test]
fn verified_outcomes_unlock_only_the_matching_context() {
    let mut field = CoherenceField::new();
    let development_read = ContextKey::new(AgentContext {
        production: false,
        mutating: false,
        restricted_data: false,
        irreversible: false,
    });
    let production_deploy = ContextKey::new(AgentContext {
        production: true,
        mutating: true,
        restricted_data: true,
        irreversible: true,
    });

    for tick in 0..120 {
        field.positive_interaction(&development_read, &personality(), tick, false);
    }

    assert_eq!(
        phase(&field, &development_read, 0.92, 0.10),
        SocialPhase::QuietlyBeloved
    );
    assert_eq!(
        phase(&field, &production_deploy, 0.92, 0.10),
        SocialPhase::ShyObserver
    );
    assert_eq!(field.context_interaction_count(&production_deploy), 0);
}

#[test]
fn familiar_context_with_high_tension_is_constrained() {
    let mut field = CoherenceField::new();
    let development_read = ContextKey::new(AgentContext {
        production: false,
        mutating: false,
        restricted_data: false,
        irreversible: false,
    });

    for tick in 0..120 {
        field.positive_interaction(&development_read, &personality(), tick, false);
    }

    assert_eq!(
        phase(&field, &development_read, 0.92, 0.80),
        SocialPhase::ProtectiveGuardian
    );
}

#[test]
fn repeated_failures_reduce_but_do_not_erase_earned_trust() {
    let mut field = CoherenceField::new();
    let context = ContextKey::new(AgentContext {
        production: false,
        mutating: false,
        restricted_data: false,
        irreversible: false,
    });
    let p = personality();

    for tick in 0..120 {
        field.positive_interaction(&context, &p, tick, false);
    }
    let before = field.context_coherence(&context);
    for tick in 120..140 {
        field.negative_interaction(&context, &p, tick);
    }
    let after = field.context_coherence(&context);

    assert!(after < before);
    assert!(after > 0.0);
}
