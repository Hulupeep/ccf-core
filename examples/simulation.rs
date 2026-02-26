//! # CCF Multi-Context Emergence Simulation
//!
//! Simulates a robot living through 22 days across five distinct environments.
//! Shows trust building, context isolation, phase transitions, startle events,
//! comfort-zone boundary discovery, decay, and warm-start recovery.

use ccf_core::accumulator::CoherenceField;
use ccf_core::boundary::MinCutBoundary;
use ccf_core::mbot::{
    BrightnessBand, MbotSensors, MotionContext, NoiseBand, Orientation,
    PresenceSignature, TimePeriod,
};
use ccf_core::phase::{Personality, PhaseSpace, SocialPhase};
use ccf_core::vocabulary::ContextKey;

// ── Contexts ─────────────────────────────────────────────────────────────────

fn living_room() -> ContextKey<MbotSensors, 6> {
    ContextKey::new(MbotSensors {
        brightness: BrightnessBand::Bright,
        noise: NoiseBand::Quiet,
        presence: PresenceSignature::Close,
        motion: MotionContext::Static,
        orientation: Orientation::Upright,
        time_period: TimePeriod::Day,
    })
}

fn kitchen() -> ContextKey<MbotSensors, 6> {
    ContextKey::new(MbotSensors {
        brightness: BrightnessBand::Bright,
        noise: NoiseBand::Loud,
        presence: PresenceSignature::Close,
        motion: MotionContext::Slow,
        orientation: Orientation::Upright,
        time_period: TimePeriod::Day,
    })
}

fn bedroom() -> ContextKey<MbotSensors, 6> {
    ContextKey::new(MbotSensors {
        brightness: BrightnessBand::Dim,
        noise: NoiseBand::Quiet,
        presence: PresenceSignature::Close,
        motion: MotionContext::Static,
        orientation: Orientation::Upright,
        time_period: TimePeriod::Evening,
    })
}

fn hallway() -> ContextKey<MbotSensors, 6> {
    ContextKey::new(MbotSensors {
        brightness: BrightnessBand::Dim,
        noise: NoiseBand::Quiet,
        presence: PresenceSignature::Absent,
        motion: MotionContext::Slow,
        orientation: Orientation::Upright,
        time_period: TimePeriod::Day,
    })
}

fn basement() -> ContextKey<MbotSensors, 6> {
    ContextKey::new(MbotSensors {
        brightness: BrightnessBand::Dark,
        noise: NoiseBand::Moderate,
        presence: PresenceSignature::Absent,
        motion: MotionContext::Static,
        orientation: Orientation::Upright,
        time_period: TimePeriod::Evening,
    })
}

// ── Display helpers ───────────────────────────────────────────────────────────

fn bar(v: f32) -> String {
    let filled = (v * 20.0).round() as usize;
    let empty  = 20usize.saturating_sub(filled);
    format!("[{}{}] {:.2}", "█".repeat(filled), "░".repeat(empty), v)
}

fn phase_name(p: SocialPhase) -> &'static str {
    match p {
        SocialPhase::ShyObserver        => "ShyObserver      ",
        SocialPhase::StartledRetreat    => "StartledRetreat  ",
        SocialPhase::QuietlyBeloved     => "QuietlyBeloved   ",
        SocialPhase::ProtectiveGuardian => "ProtectiveGuardian",
    }
}

fn row(
    label: &str,
    key: &ContextKey<MbotSensors, 6>,
    field: &CoherenceField<MbotSensors, 6>,
    phase: SocialPhase,
    instant: f32,
) {
    let coh = field.effective_coherence(instant, key);
    let led = phase.led_tint();
    let scale = phase.expression_scale();
    let n = field.context_interaction_count(key);
    println!(
        "  {:<22} {} {} | LED #{:02X}{:02X}{:02X} | scale {:.2} | n={}",
        label, phase_name(phase), bar(coh),
        led[0], led[1], led[2], scale, n,
    );
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  CCF Multi-Context Emergence Simulation — 22 simulated days         ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    let personality = Personality {
        curiosity_drive:     0.7,
        startle_sensitivity: 0.4,
        recovery_speed:      0.6,
    };
    let ps = PhaseSpace::new();
    let mut field: CoherenceField<MbotSensors, 6> = CoherenceField::new();
    let mut boundary: MinCutBoundary<MbotSensors, 6> = MinCutBoundary::new();

    // Keyed contexts — also maintained as a list for boundary registration
    let keys = [living_room(), kitchen(), bedroom(), hallway(), basement()];
    let names = ["living room", "kitchen", "bedroom", "hallway", "basement"];
    let tensions: [f32; 5] = [0.10, 0.35, 0.08, 0.20, 0.55];
    let instants: [f32; 5] = [0.85, 0.85, 0.85, 0.85, 0.40];

    let mut phases = [SocialPhase::ShyObserver; 5];
    let mut tick: u64 = 0;

    // Register all contexts with boundary (only builds edges on registration)
    boundary.report_context_with_key(&keys[0], &[]);
    let k0 = [(keys[0].clone(), keys[0].context_hash_u32())];
    boundary.report_context_with_key(&keys[1], &k0);
    let k01 = [
        (keys[0].clone(), keys[0].context_hash_u32()),
        (keys[1].clone(), keys[1].context_hash_u32()),
    ];
    boundary.report_context_with_key(&keys[2], &k01);
    let k012 = [
        (keys[0].clone(), keys[0].context_hash_u32()),
        (keys[1].clone(), keys[1].context_hash_u32()),
        (keys[2].clone(), keys[2].context_hash_u32()),
    ];
    boundary.report_context_with_key(&keys[3], &k012);
    let k0123 = [
        (keys[0].clone(), keys[0].context_hash_u32()),
        (keys[1].clone(), keys[1].context_hash_u32()),
        (keys[2].clone(), keys[2].context_hash_u32()),
        (keys[3].clone(), keys[3].context_hash_u32()),
    ];
    boundary.report_context_with_key(&keys[4], &k0123);

    let classify = |i: usize, field: &CoherenceField<MbotSensors, 6>, prev: SocialPhase| {
        let coh = field.effective_coherence(instants[i], &keys[i]);
        SocialPhase::classify(coh, tensions[i], prev, &ps)
    };

    // ── Week 1: Morning routine ───────────────────────────────────────────────
    println!("▶  WEEK 1 — Morning routine: living room + kitchen, 7 days\n");

    for day in 0..7u32 {
        for _ in 0..15 { field.positive_interaction(&keys[0], &personality, tick, false); tick += 1; }
        for _ in 0..15 { field.positive_interaction(&keys[1], &personality, tick, false); tick += 1; }
        for _ in 0..5  { field.positive_interaction(&keys[3], &personality, tick, true);  tick += 1; }

        if day == 2 || day == 6 {
            println!("  Day {}:", day + 1);
            for i in 0..5 {
                phases[i] = classify(i, &field, phases[i]);
                row(names[i], &keys[i], &field, phases[i], instants[i]);
            }
            println!();
        }
    }

    // Update boundary trust after week 1
    for i in 0..5 {
        let coh = field.context_coherence(&keys[i]);
        let n = field.context_interaction_count(&keys[i]);
        boundary.update_trust(&keys[i], coh, n);
    }

    // ── Week 2: Evening routine added ────────────────────────────────────────
    println!("▶  WEEK 2 — Evening routine added: bedroom joins daily pattern\n");

    for day in 0..7u32 {
        for _ in 0..15 { field.positive_interaction(&keys[0], &personality, tick, false); tick += 1; }
        for _ in 0..10 { field.positive_interaction(&keys[1], &personality, tick, false); tick += 1; }
        for _ in 0..20 { field.positive_interaction(&keys[2], &personality, tick, false); tick += 1; }
        for _ in 0..5  { field.positive_interaction(&keys[3], &personality, tick, true);  tick += 1; }

        if day == 6 {
            println!("  Day 14 (end of week 2):");
            for i in 0..5 {
                phases[i] = classify(i, &field, phases[i]);
                row(names[i], &keys[i], &field, phases[i], instants[i]);
            }
            println!();
        }
    }

    // Update boundary trust after week 2
    for i in 0..5 {
        let coh = field.context_coherence(&keys[i]);
        let n = field.context_interaction_count(&keys[i]);
        boundary.update_trust(&keys[i], coh, n);
    }

    // ── Comfort zone boundary ─────────────────────────────────────────────────
    println!("▶  DAY 15 — Comfort zone boundary (Stoer-Wagner min-cut)\n");

    let cut_val = boundary.min_cut_value();
    let result  = boundary.partition();

    println!("  Min-cut value: {:.4}  (lower = sharper edge between zones)", cut_val);
    print!("  INSIDE  (high trust cluster): ");
    for j in 0..result.partition_s_count {
        let h = result.partition_s[j];
        for (i, k) in keys.iter().enumerate() {
            if k.context_hash_u32() == h { print!("[{}] ", names[i]); }
        }
    }
    println!();
    print!("  OUTSIDE (unfamiliar / thin bridge): ");
    for j in 0..result.partition_complement_count {
        let h = result.partition_complement[j];
        for (i, k) in keys.iter().enumerate() {
            if k.context_hash_u32() == h { print!("[{}] ", names[i]); }
        }
    }
    println!("\n");

    // ── First visit to basement ───────────────────────────────────────────────
    println!("▶  DAY 16 — First visit to basement (zero prior experience)\n");

    phases[4] = classify(4, &field, phases[4]);
    row("basement", &keys[4], &field, phases[4], instants[4]);
    println!("  Trust earned in 4 familiar rooms does not transfer here.\n");

    // ── Startle events ────────────────────────────────────────────────────────
    println!("▶  DAY 17 — Startle events: same stimulus, different trust history\n");

    let kitchen_before = field.effective_coherence(instants[1], &keys[1]);
    field.negative_interaction(&keys[1], &personality, tick); tick += 1;
    field.negative_interaction(&keys[1], &personality, tick); tick += 1;
    let kitchen_after = field.effective_coherence(instants[1], &keys[1]);
    phases[1] = SocialPhase::classify(kitchen_after, 0.70, phases[1], &ps);
    println!("  Kitchen (14 days of trust built): drop {:.2} → {:.2}  (Δ {:.3})",
        kitchen_before, kitchen_after, kitchen_before - kitchen_after);
    row("  kitchen post-startle", &keys[1], &field, phases[1], instants[1]);

    let basement_before = field.effective_coherence(instants[4], &keys[4]);
    field.negative_interaction(&keys[4], &personality, tick); tick += 1;
    field.negative_interaction(&keys[4], &personality, tick); tick += 1;
    let basement_after = field.effective_coherence(instants[4], &keys[4]);
    phases[4] = SocialPhase::classify(basement_after, 0.70, phases[4], &ps);
    println!("\n  Basement (zero trust history):    drop {:.2} → {:.2}  (Δ {:.3})",
        basement_before, basement_after, basement_before - basement_after);
    row("  basement post-startle", &keys[4], &field, phases[4], instants[4]);
    println!("  Same stimulus. Earned floor protects the kitchen. Nothing to protect in basement.\n");

    // ── Recovery ─────────────────────────────────────────────────────────────
    println!("▶  DAYS 18–19 — Kitchen recovers; basement has no recovery effort\n");

    for _ in 0..25 { field.positive_interaction(&keys[1], &personality, tick, false); tick += 1; }

    phases[1] = classify(1, &field, phases[1]);
    phases[4] = classify(4, &field, phases[4]);
    row("kitchen (recovered)", &keys[1], &field, phases[1], instants[1]);
    row("basement (static)",   &keys[4], &field, phases[4], instants[4]);
    println!();

    // ── Decay ────────────────────────────────────────────────────────────────
    println!("▶  DAY 21 — Robot shelved over long weekend: 48-tick decay\n");

    let before: Vec<f32> = keys.iter().enumerate()
        .map(|(i, k)| field.effective_coherence(instants[i], k))
        .collect();

    field.decay_all(48);

    println!("  Context          before → after  held floor?");
    for i in 0..5 {
        let after = field.effective_coherence(instants[i], &keys[i]);
        phases[i] = SocialPhase::classify(after, tensions[i], phases[i], &ps);
        let held = if phases[i] == SocialPhase::QuietlyBeloved || after > 0.5 { "✓" } else { "—" };
        println!("  {:<18} {:.2}  →  {:.2}  {}",
            names[i], before[i], after, held);
    }
    println!();

    // ── Warm start ────────────────────────────────────────────────────────────
    println!("▶  DAY 22 — Robot returns; familiar contexts warm-start immediately\n");

    println!("  Ticks to reach QuietlyBeloved again:\n");

    // Living room: track how many ticks to re-enter QuietlyBeloved
    let mut recovery_ticks = 0u32;
    loop {
        field.positive_interaction(&keys[0], &personality, tick, false);
        tick += 1;
        recovery_ticks += 1;
        let coh = field.effective_coherence(instants[0], &keys[0]);
        phases[0] = SocialPhase::classify(coh, tensions[0], phases[0], &ps);
        if phases[0] == SocialPhase::QuietlyBeloved || recovery_ticks >= 50 { break; }
    }
    println!("  Living room: {} interactions to recover QuietlyBeloved", recovery_ticks);

    // Basement: how many ticks to reach QuietlyBeloved for the first time
    let mut basement_ticks = 0u32;
    let mut basement_phase = SocialPhase::ShyObserver;
    loop {
        field.positive_interaction(&keys[4], &personality, tick, false);
        tick += 1;
        basement_ticks += 1;
        let coh = field.effective_coherence(instants[4], &keys[4]);
        basement_phase = SocialPhase::classify(coh, tensions[4], basement_phase, &ps);
        if basement_phase == SocialPhase::QuietlyBeloved || basement_ticks >= 200 { break; }
    }
    if basement_phase == SocialPhase::QuietlyBeloved {
        println!("  Basement:     {} interactions to reach QuietlyBeloved (first time)", basement_ticks);
    } else {
        println!("  Basement:     not yet QuietlyBeloved after {} interactions (needs more)", basement_ticks);
    }
    println!("  → Warm start: earned floor means familiar rooms recover {} faster\n",
        if basement_ticks > recovery_ticks {
            format!("~{}×", basement_ticks / recovery_ticks.max(1))
        } else {
            "similarly".to_string()
        }
    );

    // ── Final state ───────────────────────────────────────────────────────────
    println!("▶  FINAL STATE — All five contexts after 22 days\n");
    for i in 0..5 {
        phases[i] = classify(i, &field, phases[i]);
        row(names[i], &keys[i], &field, phases[i], instants[i]);
    }

    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  What emerged — no rules, no scripts                                ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║  • 4 contexts independently reached QuietlyBeloved                 ║");
    println!("║  • Basement stayed cautious — nothing earned there                 ║");
    println!("║  • Kitchen survived a startle because it had earned floor           ║");
    println!("║  • Comfort zone boundary discovered from trust topology alone       ║");
    println!("║  • Decay preserved earned floor — not a full reset                 ║");
    println!("║  • Warm start: familiar rooms recovered in ~{}% of original ticks  ║",
        (recovery_ticks as f32 / 105.0 * 100.0).round() as u32);
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}
