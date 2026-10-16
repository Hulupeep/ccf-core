//! # Context-aware trust gate for an autonomous agent
//!
//! This example treats an agent action as a contextual interaction. An external
//! decision system supplies `instant_confidence`; CCF contributes trust earned
//! from verified outcomes in the exact operational context.
//!
//! Run with:
//!
//! ```text
//! cargo run --example agent_trust_gate
//! ```

use ccf_core::accumulator::CoherenceField;
use ccf_core::phase::{Personality, PhaseSpace, SocialPhase};
use ccf_core::vocabulary::{ContextKey, SensorVocabulary};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Environment {
    Development,
    Production,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Action {
    Read,
    Deploy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum DataSensitivity {
    Internal,
    Restricted,
}

/// The context intentionally keeps environment and action class separate.
/// Success in a development read context must not grant production deploy trust.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct AgentContext {
    environment: Environment,
    action: Action,
    sensitivity: DataSensitivity,
    reversible: bool,
}

impl SensorVocabulary<4> for AgentContext {
    fn to_feature_vec(&self) -> [f32; 4] {
        [
            match self.environment {
                Environment::Development => 0.0,
                Environment::Production => 1.0,
            },
            match self.action {
                Action::Read => 0.0,
                Action::Deploy => 1.0,
            },
            match self.sensitivity {
                DataSensitivity::Internal => 0.25,
                DataSensitivity::Restricted => 1.0,
            },
            if self.reversible { 0.0 } else { 1.0 },
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AgentPolicy {
    /// Context is familiar and operational tension is low.
    Allow,
    /// Context is familiar, but current risk calls for restricted execution.
    Constrain,
    /// Context is unfamiliar; keep the agent read-only and ask a human.
    Confirm,
    /// Context is unfamiliar and current risk is high; do not execute.
    Abstain,
}

fn policy_for(phase: SocialPhase) -> AgentPolicy {
    match phase {
        SocialPhase::QuietlyBeloved => AgentPolicy::Allow,
        SocialPhase::ProtectiveGuardian => AgentPolicy::Constrain,
        SocialPhase::ShyObserver => AgentPolicy::Confirm,
        SocialPhase::StartledRetreat => AgentPolicy::Abstain,
    }
}

fn evaluate(
    field: &CoherenceField<AgentContext, 4>,
    key: &ContextKey<AgentContext, 4>,
    instant_confidence: f32,
    operational_tension: f32,
) -> (f32, SocialPhase, AgentPolicy) {
    let effective = field.effective_coherence(instant_confidence, key);
    let phase = SocialPhase::classify(
        effective,
        operational_tension,
        SocialPhase::ShyObserver,
        &PhaseSpace::default(),
    );
    (effective, phase, policy_for(phase))
}

fn main() {
    let personality = Personality {
        curiosity_drive: 0.0,
        startle_sensitivity: 0.7,
        recovery_speed: 0.5,
    };
    let mut field: CoherenceField<AgentContext, 4> = CoherenceField::new();

    let development_read = ContextKey::new(AgentContext {
        environment: Environment::Development,
        action: Action::Read,
        sensitivity: DataSensitivity::Internal,
        reversible: true,
    });
    let production_deploy = ContextKey::new(AgentContext {
        environment: Environment::Production,
        action: Action::Deploy,
        sensitivity: DataSensitivity::Restricted,
        reversible: false,
    });

    // Verified outcomes build trust only for this exact context.
    for tick in 0..120 {
        field.positive_interaction(&development_read, &personality, tick, false);
    }

    let development = evaluate(&field, &development_read, 0.92, 0.10);
    let production = evaluate(&field, &production_deploy, 0.92, 0.80);
    let tense_development = evaluate(&field, &development_read, 0.92, 0.80);

    println!("Context-aware agent trust gate");
    println!("==============================");
    println!(
        "development read  | effective {:.2} | {:?} | {:?}",
        development.0, development.1, development.2
    );
    println!(
        "production deploy | effective {:.2} | {:?} | {:?}",
        production.0, production.1, production.2
    );
    println!(
        "dev read + alert  | effective {:.2} | {:?} | {:?}",
        tense_development.0, tense_development.1, tense_development.2
    );

    assert_eq!(development.2, AgentPolicy::Allow);
    assert_eq!(production.2, AgentPolicy::Abstain);
    assert_eq!(tense_development.2, AgentPolicy::Constrain);
    assert_eq!(field.context_interaction_count(&production_deploy), 0);
}
