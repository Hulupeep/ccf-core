//! # mBot2 — Emergent Social Behaviour on a $50 Robot
//!
//! The [mBot2](https://www.makeblock.com/mbot2) by Makeblock is a programmable robot
//! that retails for around $50–$80 USD. It runs a CyberPi microcontroller with onboard
//! light, sound, proximity, and motion sensors.
//!
//! This example simulates a full CCF behavioural loop as it would run on the mBot2.
//! No hardware required — the sensor readings are generated in code to illustrate the
//! emergent dynamics.
//!
//! ## What "emergent" means here
//!
//! Most robotic behavioural systems are programmed: "if loud noise, retreat". The robot
//! does what it was told, every time, regardless of context or history.
//!
//! CCF works differently. The robot *accumulates experience*. A kitchen that has been
//! noisy every morning for two weeks is not threatening — it is familiar. The same noise
//! level in an unknown basement is. The behaviour that emerges is a product of what the
//! robot has actually experienced, not what a programmer anticipated.
//!
//! This is what earned trust looks like. It is the same mechanism that governs how a
//! person feels comfortable in their own home during a storm but uneasy in an unfamiliar
//! building in the same storm. The environment carries history. CCF gives your robot
//! the same capacity.
//!
//! ## The four phases
//!
//! As the robot accumulates trust in a context, it moves through a 2D phase space
//! (coherence × tension) with four behavioural quadrants:
//!
//! ```text
//!                   │ Low tension          │ High tension
//! ──────────────────┼──────────────────────┼───────────────────────
//! Low coherence     │ ShyObserver          │ StartledRetreat
//! High coherence    │ QuietlyBeloved       │ ProtectiveGuardian
//! ```
//!
//! - **ShyObserver** — New or unfamiliar context. Minimal expression. Watching carefully.
//! - **StartledRetreat** — Unfamiliar context with threat signal. Withdrawal. LED red.
//! - **QuietlyBeloved** — Deeply familiar, relaxed. Full expressiveness. LED warm white.
//! - **ProtectiveGuardian** — Trusted but alert. Grounded response to threat. LED amber.
//!
//! Transitions use Schmitt trigger hysteresis — the robot does not flicker between phases
//! at the boundary. It commits and holds until the signal moves well past the threshold.
//!
//! ## Running this example
//!
//! ```
//! cargo run --example mbot2
//! ```

use ccf_core::accumulator::CoherenceField;
use ccf_core::mbot::{
    BrightnessBand, MbotSensors, MotionContext, NoiseBand, Orientation, PresenceSignature,
    TimePeriod,
};
use ccf_core::phase::{Personality, PhaseSpace, SocialPhase};
use ccf_core::vocabulary::ContextKey;

fn main() {
    println!("CCF mBot2 — Emergent Behaviour Simulation");
    println!("==========================================\n");

    // ── Personality ──────────────────────────────────────────────────────────
    //
    // These three parameters shape how the robot accumulates and loses trust.
    // They are analogous to temperament — the same events produce different
    // magnitudes of response depending on personality.
    //
    // This configuration: eager to engage (high curiosity), not easily startled
    // (low sensitivity), recovers quickly from disruption (high recovery).
    let personality = Personality {
        curiosity_drive:     0.8,
        startle_sensitivity: 0.3,
        recovery_speed:      0.7,
    };

    // ── Trust field ──────────────────────────────────────────────────────────
    //
    // The field maintains an independent trust state per sensory context.
    // It starts empty — the robot has no experience yet.
    let mut field: CoherenceField<MbotSensors, 6> = CoherenceField::new();

    // ── Phase space ──────────────────────────────────────────────────────────
    //
    // Configurable thresholds for the four-quadrant classifier.
    // The default values (enter 0.65 / exit 0.55 on coherence) are a good
    // starting point. Adjust to tune how quickly the robot "warms up".
    let ps = PhaseSpace::new();
    let mut phase = SocialPhase::ShyObserver;

    // ── Two sensor contexts ──────────────────────────────────────────────────
    //
    // The living room: bright, quiet, a familiar person nearby.
    // The basement:    dark, moderate noise, no one present.
    //
    // These are entirely independent trust contexts. Experience in one
    // does not transfer to the other.
    let living_room = ContextKey::new(MbotSensors {
        brightness:  BrightnessBand::Bright,
        noise:       NoiseBand::Quiet,
        presence:    PresenceSignature::Close,
        motion:      MotionContext::Static,
        orientation: Orientation::Upright,
        time_period: TimePeriod::Day,
    });

    let basement = ContextKey::new(MbotSensors {
        brightness:  BrightnessBand::Dark,
        noise:       NoiseBand::Moderate,
        presence:    PresenceSignature::Absent,
        motion:      MotionContext::Static,
        orientation: Orientation::Upright,
        time_period: TimePeriod::Evening,
    });

    // ── Phase 1: Build trust in the living room ───────────────────────────────
    //
    // Simulate 80 ticks of positive interaction in the living room.
    // The person is present, the environment is calm, interactions go well.
    println!("Phase 1: 80 positive interactions in the living room");
    println!("------------------------------------------------------");

    for tick in 0u64..80 {
        // alone=false: the person is present; the interaction is social
        field.positive_interaction(&living_room, &personality, tick, false);

        // instant: the current sensor-level signal (1.0 = fully coherent reading)
        // In a real system this comes from your homeostasis / sensor fusion layer.
        let instant: f32 = 0.9;
        let coherence = field.effective_coherence(instant, &living_room);
        let tension: f32 = 0.1; // calm environment
        phase = SocialPhase::classify(coherence, tension, phase, &ps);

        if tick % 20 == 19 {
            let led = phase.led_tint();
            println!(
                "  tick {:>3} | coherence {:.2} | phase {:?} | LED #{:02X}{:02X}{:02X}",
                tick + 1,
                coherence,
                phase,
                led[0], led[1], led[2]
            );
        }
    }

    // ── Phase 2: The basement — zero trust, same robot ───────────────────────
    //
    // The robot is moved to the basement. It has never been here before.
    // All the trust it built in the living room is irrelevant in this context.
    // It starts from zero — exactly as it should.
    println!("\nPhase 2: Robot moved to the basement (zero prior experience)");
    println!("--------------------------------------------------------------");

    let basement_coherence = field.effective_coherence(0.9, &basement);
    let basement_tension: f32 = 0.5; // unfamiliar, slightly unnerving
    phase = SocialPhase::classify(basement_coherence, basement_tension, phase, &ps);

    let led = phase.led_tint();
    println!(
        "  coherence in basement: {:.2} | phase {:?} | LED #{:02X}{:02X}{:02X}",
        basement_coherence,
        phase,
        led[0], led[1], led[2]
    );
    println!("  (Trust does not transfer. The robot earned nothing here yet.)");

    // ── Phase 3: A startle event back in the living room ─────────────────────
    //
    // The robot returns to the familiar living room. Someone slams a door.
    // Because the robot has deep trust in this context (earned floor > 0),
    // the negative event causes a drop — but cannot erase the accumulated history.
    println!("\nPhase 3: Return to living room — startle event (door slam)");
    println!("------------------------------------------------------------");

    let before = field.effective_coherence(0.9, &living_room);
    field.negative_interaction(&living_room, &personality, 81);
    let after = field.effective_coherence(0.9, &living_room);

    println!("  coherence before startle: {:.2}", before);
    println!("  coherence after startle:  {:.2}", after);
    println!(
        "  drop: {:.2} (earned floor protects against erasure)",
        before - after
    );

    let tension: f32 = 0.55; // brief spike from the startle
    phase = SocialPhase::classify(after, tension, phase, &ps);
    let led = phase.led_tint();
    println!(
        "  phase: {:?} | LED #{:02X}{:02X}{:02X}",
        phase, led[0], led[1], led[2]
    );

    // ── Phase 4: Recovery ─────────────────────────────────────────────────────
    //
    // A few more positive interactions and the robot recovers. The familiar
    // environment reasserts itself. This is not scripted — it emerges from the
    // accumulated trust floor protecting the relationship.
    println!("\nPhase 4: Recovery — 5 positive interactions");
    println!("--------------------------------------------");

    for tick in 82u64..87 {
        field.positive_interaction(&living_room, &personality, tick, false);
    }
    let recovered = field.effective_coherence(0.9, &living_room);
    let tension: f32 = 0.1;
    phase = SocialPhase::classify(recovered, tension, phase, &ps);
    let led = phase.led_tint();
    println!(
        "  coherence after recovery: {:.2} | phase {:?} | LED #{:02X}{:02X}{:02X}",
        recovered, phase, led[0], led[1], led[2]
    );

    // ── Summary ───────────────────────────────────────────────────────────────
    println!("\nSummary");
    println!("-------");
    println!(
        "  Living room trust:  {:.2} ({} interactions)",
        field.effective_coherence(0.9, &living_room),
        field.context_interaction_count(&living_room),
    );
    println!(
        "  Basement trust:     {:.2} ({} interactions)",
        field.effective_coherence(0.9, &basement),
        field.context_interaction_count(&basement),
    );
    println!("\nThe robot knows where it belongs.");
    println!("No rules. No scripts. Earned.");
}
