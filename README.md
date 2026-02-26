# ccf-core

**Contextual Coherence Fields** — earned relational trust for robots, agents, and autonomous systems.

[![crates.io](https://img.shields.io/crates/v/ccf-core)](https://crates.io/crates/ccf-core)
[![docs.rs](https://docs.rs/ccf-core/badge.svg)](https://docs.rs/ccf-core)
[![License: BUSL-1.1](https://img.shields.io/badge/license-BUSL--1.1-blue)](LICENSE)
[![Patent Pending](https://img.shields.io/badge/patent-pending%20US%2063%2F988%2C438-lightgrey)](https://github.com/Hulupeep/ccf-core)

---

## Any robot or agent can now have emergent social behaviour

Not scripted. Not rule-based. **Earned** — the same way a person earns trust.

A person who has spent every morning in a busy kitchen for two years is not threatened
by the noise. That same person in an unfamiliar basement for the first time is cautious.
The behaviour is not a rule. It emerges from accumulated experience in a specific context.

ccf-core gives this capacity to any autonomous system — from a $50 robot to a deployed
AI agent. Your device builds a **field** of trust states, one per sensory context, learned
continuously from real interaction. The behaviour that results is a product of what the
system has actually experienced, not what a programmer anticipated.

This is the architecture behind the mBot2 reference demo: a sub-$100 programmable robot
that develops genuine context-sensitive social behaviour, entirely on-device with no cloud,
no ML model, no scripted emotional state. [See the example →](examples/mbot2.rs)

---

## What ccf-core Gives You

- **Context-specific trust** — every distinct sensory environment has its own independent trust history
- **Earned resilience** — trust built through repeated positive interaction is protected against transient negative events; a single bad moment cannot erase an established relationship
- **Four expressive behavioral phases** — `ShyObserver`, `StartledRetreat`, `QuietlyBeloved`, `ProtectiveGuardian` — with distinct LED tint, motor scale, and narration depth per phase
- **Personality** — tune curiosity, startle sensitivity, and recovery rate per device or agent
- **Emergent comfort-zone boundaries** — the system discovers which contexts belong together via graph min-cut; you don't configure a threshold
- **`no_std` by default** — runs on Cortex-M, ESP32, RP2040, and any bare-metal target with no heap required

---

## Use Cases

### Social and Companion Robots

Your robot has met this family before. It knows Tuesday evenings are noisy and it's fine.
A stranger enters — new sensory context, zero trust, `ShyObserver` mode. It doesn't
over-react or under-react; it behaves consistently with its actual experience of *this* environment.

### Smart Home and Ambient Devices

A speaker learns that "kitchen at 7am" is high-activity, and responds with higher expressiveness.
"Living room at 11pm" is a different context entirely — quiet, familiar, settled. The same
trust architecture handles both without explicit programming.

### Industrial and Field Robotics

A robot arm in a calibration bay has built trust for that specific environment. Moved to
the production floor — different light, different noise, different vibration signature — it
starts cautious and builds trust from scratch. Safety-critical behavior falls out of the
architecture rather than being bolted on.

### Game AI and NPCs

Characters that remember their relational history with the player *in each location*.
The tavern NPC who trusts you in Stormwind has no reason to trust you in the dungeon.
Context-gated trust is the difference between a character that feels alive and one that
just reads a mood variable.

### Wearables and Health Devices

Activity context (running, sleeping, commuting) gates behavioral responses. An alert that
fires during your morning run pattern is different from the same alert firing in an
unfamiliar location. CCF gives you the context-sensitivity layer above your sensor stream.

---

## How it works

Three mathematical primitives combine to produce emergent behaviour. None works alone.

**1. Context-keyed accumulators**

Trust is not a single global value. Every distinct sensory environment has its own
independent trust history, built from real interactions in that specific context.
A robot that trusts the living room has zero trust in the basement — because it has
never been there. Histories never cross-contaminate.

**2. The minimum gate**

Effective coherence requires agreement between two signals: what the system has
*learned* (accumulated context trust) and what it is *experiencing right now*
(the instant sensor reading).

> "Both must be true, or I stay reserved."

Familiar contexts buffer noise — a single bad event cannot erase weeks of earned trust.
Unfamiliar contexts demand proof before any expressiveness is unlocked.

**3. Graph min-cut boundary**

As contexts accumulate trust histories, the system builds a trust-weighted graph
where similar contexts share stronger edges. Stoer-Wagner global min-cut finds
the cheapest way to divide it into two clusters.

> "This room feels different from that room."

The comfort zone *emerges* from the trust topology. You don't configure a threshold —
the algorithm discovers the boundary.

**Plus: trust mixing**

A small amount of trust transfers between similar contexts. Kitchen morning trust
warms the hallway a little. `SinkhornKnopp` projects the transfer matrix onto the
Birkhoff polytope so no single context dominates allocation.

---

## Quick Start

```toml
[dependencies]
ccf-core = "0.1"
```

### 1. Define your sensor vocabulary

Implement `SensorVocabulary` for whatever sensors your hardware has. The trait is the
only thing that needs to know about your specific hardware.

```rust
use ccf_core::vocabulary::SensorVocabulary;

// Two-sensor example: ambient light + presence detection
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RoomSensors {
    pub light: LightLevel,
    pub presence: Presence,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LightLevel { Dark, Dim, Bright }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Presence { Empty, Near, Far }

impl SensorVocabulary<2> for RoomSensors {
    fn to_feature_vec(&self) -> [f32; 2] {
        let l = match self.light {
            LightLevel::Dark  => 0.0,
            LightLevel::Dim   => 0.5,
            LightLevel::Bright => 1.0,
        };
        let p = match self.presence {
            Presence::Empty => 0.0,
            Presence::Far   => 0.5,
            Presence::Near  => 1.0,
        };
        [l, p]
    }
}
```

That's the only hardware-specific code. Everything else is generic.

### 2. Create a coherence field

```rust
use ccf_core::vocabulary::ContextKey;
use ccf_core::accumulator::CoherenceField;
use ccf_core::phase::{Personality, PhaseSpace, SocialPhase};

let personality = Personality::new(); // mid-range defaults
let mut field: CoherenceField<RoomSensors, 2> = CoherenceField::new();
let ps = PhaseSpace::new();
let mut phase = SocialPhase::ShyObserver;
```

### 3. Run your main loop

```rust
let mut tick: u64 = 0;

loop {
    // Read sensors and build context key
    let sensors = RoomSensors { light: LightLevel::Bright, presence: Presence::Near };
    let key = ContextKey::new(sensors);

    // Record a positive interaction (person waved, task succeeded, user smiled, etc.)
    // `alone: bool` — true if no external stimulus, just passive presence
    field.positive_interaction(&key, &personality, tick, false);

    // Optionally record a negative event (loud noise, obstacle, failed task)
    // field.negative_interaction(&key, &personality, tick);

    // Read the effective coherence for the current context
    // `instant` is your raw sensor reading for this tick, normalised to [0.0, 1.0]
    let instant: f32 = 0.9; // your system provides this
    let coherence = field.effective_coherence(instant, &key);

    // Classify behavioral phase (tension comes from your homeostasis / task layer)
    let tension: f32 = 0.2; // your system provides this
    phase = SocialPhase::classify(coherence, tension, phase, &ps);

    // Drive outputs from phase
    let led   = phase.led_tint();          // [r, g, b] — distinct per phase
    let scale = phase.expression_scale(); // 0.0–1.0 — scale motors, audio, etc.

    // Apply to hardware...

    tick += 1;
}
```

### 4. The field remembers

After 50+ positive interactions in `Bright+Near`:

```
coherence in Bright+Near  → 0.72  →  QuietlyBeloved (expressive, relaxed)
coherence in Dark+Empty   → 0.0   →  ShyObserver    (cautious, minimal)
```

Trust does not transfer between contexts. The device earned trust in one room and
starts fresh in another — exactly as you'd want.

---

## Reference Implementation: mBot2 on $50 Hardware

`ccf_core::mbot` ships a complete 6-dimensional sensor vocabulary for the
[mBot2](https://www.makeblock.com/mbot2) — a programmable robot by Makeblock that
retails for around $50–$80 USD. It is the reference demo for what ccf-core makes possible:
emergent social behaviour on cheap commodity hardware, no cloud required.

| Field | Type | Dimensions |
|-------|------|-----------|
| `brightness` | `BrightnessBand` | Dark / Dim / Bright |
| `noise` | `NoiseBand` | Quiet / Moderate / Loud |
| `presence` | `PresenceSignature` | Absent / Far / Close |
| `motion` | `MotionContext` | Static / Slow / Fast |
| `orientation` | `Orientation` | Upright / Tilted |
| `time_period` | `TimePeriod` | Day / Evening / Night |

```rust
use ccf_core::mbot::{MbotSensors, MbotContextKey,
    BrightnessBand, NoiseBand, PresenceSignature, MotionContext, Orientation, TimePeriod};
use ccf_core::vocabulary::ContextKey;

let key = ContextKey::new(MbotSensors {
    brightness:  BrightnessBand::Bright,
    noise:       NoiseBand::Quiet,
    presence:    PresenceSignature::Close,
    motion:      MotionContext::Static,
    orientation: Orientation::Upright,
    time_period: TimePeriod::Day,
});
```

See `examples/mbot2.rs` for a full simulated CCF loop — 80 ticks of earned trust,
a startle event, and recovery — with printed output showing phase transitions and LED tint.

---

## Behavioral Phases and Outputs

`SocialPhase` maps the 2D space (coherence × tension) to four quadrants,
using Schmitt trigger hysteresis to prevent oscillation at boundaries:

```
                  │ Low tension         │ High tension
──────────────────┼─────────────────────┼──────────────────────
Low coherence     │ ShyObserver         │ StartledRetreat
High coherence    │ QuietlyBeloved      │ ProtectiveGuardian
```

Each phase produces distinct outputs:

| Phase | LED tint | Expression scale | Character |
|-------|----------|-----------------|-----------|
| `ShyObserver` | Cool blue | 0.35 | Cautious, watching |
| `StartledRetreat` | Red | 0.10 | Withdraw, minimal output |
| `QuietlyBeloved` | Warm white | 1.00 | Full expressiveness |
| `ProtectiveGuardian` | Amber | 0.65 | Alert but grounded |

---

## Personality

Three bounded parameters tune how trust builds and erodes — without changing the
structural invariants of the architecture:

```rust
let personality = Personality {
    curiosity_drive:     0.8,  // explores new contexts eagerly; higher cold-start baseline
    startle_sensitivity: 0.3,  // resilient to aversive events; drops less on negative interactions
    recovery_speed:      0.7,  // rebuilds trust faster after disruption
};
```

---

## Comfort-Zone Boundary Discovery

`MinCutBoundary` runs Stoer-Wagner global min-cut on the trust-weighted context graph.
You don't configure a threshold — the boundary *emerges* from which contexts have
accumulated similar trust histories:

```rust
use ccf_core::boundary::MinCutBoundary;

let mut boundary: MinCutBoundary<RoomSensors, 2> = MinCutBoundary::new();

// As your field accumulates trust, report contexts to the boundary
boundary.report_context_with_key(&bright_near_key, coherence_bright_near);
boundary.report_context_with_key(&dark_empty_key,  coherence_dark_empty);

// The partition tells you which side each context is on
let result = boundary.partition();
// result.partition_s:          hashes of "inside" contexts (adopted, high trust)
// result.partition_complement: hashes of "outside" contexts (unfamiliar/distrusted)

// The min-cut value measures how sharp the comfort-zone edge is
let edge_sharpness = boundary.min_cut_value();
```

---

## Trust Mixing with SinkhornKnopp

`SinkhornKnopp` projects a matrix of trust similarities onto the Birkhoff polytope
(doubly stochastic matrices), ensuring no single context dominates trust allocation:

```rust
use ccf_core::sinkhorn::SinkhornKnopp;

let sk = SinkhornKnopp::default();
let mut trust_matrix = [
    [1.0, 0.8, 0.1],
    [0.8, 1.0, 0.2],
    [0.1, 0.2, 1.0],
];
let result = sk.project(&mut trust_matrix);
// trust_matrix is now doubly stochastic — rows and columns each sum to 1.0
```

---

## Python

```toml
ccf-core = { version = "0.1", features = ["python-ffi"] }
```

Build a Python extension with [maturin](https://github.com/PyO3/maturin):

```bash
pip install maturin
maturin develop --features python-ffi
```

```python
from ccf_core import CoherenceField, Personality, SocialPhase, PhaseSpace

personality = Personality(curiosity_drive=0.7, startle_sensitivity=0.3, recovery_speed=0.6)
field = CoherenceField()
ps = PhaseSpace()
phase = SocialPhase.ShyObserver

# feature_vec: 6 floats [0.0, 1.0]
# brightness, noise, presence, motion, orientation, time_of_day
features = [0.8, 0.0, 1.0, 0.0, 1.0, 0.5]  # bright, quiet, close, still, upright, day

for tick in range(50):
    field.positive_interaction(features, personality, tick=tick, alone=False)

coherence = field.effective_coherence(0.9, features)
phase = SocialPhase.classify(coherence, tension=0.1, current=phase, space=ps)
print(phase)                    # SocialPhase.QuietlyBeloved
print(phase.led_tint())         # [60, 120, 200]
print(phase.expression_scale()) # 1.0
```

The Python API uses 6-dimensional feature vectors matching the mBot2 vocabulary.
For custom sensor dimensions, use the Rust API directly.

---

## Platform Support

ccf-core is `#![no_std]` with no heap allocation required in the default configuration.
It compiles for any target Rust supports:

| Target | Status |
|--------|--------|
| `x86_64-unknown-linux-gnu` | ✅ tested |
| `thumbv7em-none-eabihf` (Cortex-M4/M7) | ✅ tested |
| `thumbv6m-none-eabi` (Cortex-M0) | ✅ |
| `riscv32imc-unknown-none-elf` (ESP32-C3) | ✅ |
| `xtensa-esp32-none-elf` (ESP32) | ✅ |
| WASM | ✅ (with `std` feature) |

### Features

| Feature | Default | Effect |
|---------|---------|--------|
| `std` | off | Enables `CoherenceField::all_entries()` and persistence helpers |
| `serde` | off | Derives `Serialize` / `Deserialize` on all public types; enables `ccf_core::seg` |

---

## Persistence — saving and restoring trust

Enable the `serde` feature to snapshot a live field and restore it later. A device
that loses power picks up exactly where it left off — one interaction to re-enter
`QuietlyBeloved` in a familiar context instead of starting from zero.

```toml
ccf-core = { version = "0.1", features = ["serde"] }
```

```rust
use ccf_core::seg::CcfSegSnapshot;

// After your robot has been running for a while — save its trust history
let snapshot = CcfSegSnapshot::from_field(&field, &personality, created_at, last_active, total_interactions);
let json = serde_json::to_string(&snapshot).unwrap();
// Write json to flash / SD card / file

// On next boot — restore it
let snapshot: CcfSegSnapshot = serde_json::from_str(&json).unwrap();
// snapshot.contexts contains all context hashes + coherence values
// snapshot.personality contains the personality modulators
// Warm-start: the robot re-enters familiar contexts in one interaction,
// rather than rebuilding from zero.
```

The snapshot is vocabulary-erased — only the FNV-1a context hashes are stored,
not the sensor readings themselves. Compact and transport-safe.

---

## Test Coverage

```
cargo test                  # 98 tests: 64 unit + 34 patent-claim integration tests
cargo test --features serde # 106 tests: adds 8 CCF_SEG round-trip tests
```

The integration test file `tests/patent_claims.rs` contains one named test per patent
claim — `test_claim_N_<description>` — each demonstrating the claimed behaviour
end-to-end through the public API only.

---

## Patent Claim Map

Patent pending: US Provisional Application 63/988,438 (priority date 23 Feb 2026).

| Public Type | Patent Claims | Description |
|-------------|--------------|-------------|
| `SensorVocabulary` | 1, 8 | Platform-independent trait encoding sensory state as a normalised feature vector |
| `ContextKey<V>` | 1, 8 | Composite context identifier: deterministic FNV hash + cosine similarity |
| `CoherenceAccumulator` | 2–5 | Per-context trust counter with earned floor and asymmetric decay |
| `CoherenceField<V>` | 6–7, 13 | Context-keyed accumulator map with asymmetric min-gate blending |
| `MinCutBoundary<V>` | 9–12 | Stoer-Wagner global min-cut comfort-zone boundary |
| `SocialPhase` | 14–18 | Four-quadrant phase classifier with Schmitt trigger hysteresis |
| `SinkhornKnopp` | 19–23 | Birkhoff polytope projector: doubly stochastic trust mixing |
| `Personality` | 24–28 | Bounded modulators: curiosity, startle sensitivity, recovery rate |
| Full CCF pipeline | 29–34 | Composite system: sensor → context → accumulate → classify → output |

---

## License

Business Source License 1.1. Free for evaluation and non-production use.\
Change date: **23 February 2032** — converts to Apache License 2.0.\
Commercial production use requires a license from Flout Labs (cbyrne@floutlabs.com).
