# ccf-core

**Contextual Coherence Fields** — earned relational trust for autonomous systems.

Patent pending: US Provisional Application 63/988,438 (priority date 23 Feb 2026).

## What This Is

CCF is a computational architecture for emergent social behaviour in autonomous systems.
An agent maintains a *field* of trust states — one per sensory context — learned
continuously from experience. Trust earned in a bright quiet room does not transfer to
a dark noisy room unless the field explicitly learns they are similar.

## Patent Claim Map

| Public Type | Patent Claims | One-line Description |
|-------------|--------------|----------------------|
| `SensorVocabulary` | 1, 8 | Platform-independent trait encoding sensory state as a normalised feature vector |
| `ContextKey<V>` | 1, 8 | Composite context identifier: deterministic FNV hash + cosine similarity over `SensorVocabulary` |
| `CoherenceAccumulator` | 2–5 | Per-context trust counter with asymptotic positive growth and an earned interaction floor |
| `CoherenceField<V>` | 6–7, 13 | Context-keyed map of accumulators with LRU eviction and asymmetric min-gate blending |
| `MinCutBoundary<V>` | 9–12 | Stoer-Wagner global min-cut on the trust-weighted context graph — comfort-zone boundary |
| `SocialPhase` | 14–18 | Four-quadrant (coherence × tension) phase classifier with Schmitt trigger hysteresis |
| `SinkhornKnopp` | 19–23 | Birkhoff polytope projector: iterative row/column normalisation to doubly stochastic matrix |
| `Personality` | 24–28 | Bounded modulators (curiosity, startle sensitivity, recovery rate) that scale deltas, not structure |
| Full CCF pipeline | 29–34 | Composite system: sensor → context key → field accumulation → phase classification → output |

### Claim-by-Claim Summary

| Claim | Type | Description |
|-------|------|-------------|
| 1 | `SensorVocabulary` + `ContextKey<V>` | Discrete context identifier from quantised sensor signals with deterministic hash |
| 2 | `CoherenceAccumulator` | Positive growth: coherence increases asymptotically with repeated positive interactions |
| 3 | `CoherenceAccumulator` | Earned floor: interaction history protects against transient negative events |
| 4 | `CoherenceAccumulator` | Asymmetric decay: coherence decays toward earned floor, not toward zero |
| 5 | `CoherenceAccumulator` | Personality modulation: recovery_speed scales growth rate without changing structure |
| 6 | `CoherenceField<V>` | Context-keyed accumulator map: independent trust per sensory context |
| 7 | `CoherenceField<V>` | Min-gate (unfamiliar arm): min(instant, ctx) — must earn trust before it counts |
| 8 | `SensorVocabulary` + `ContextKey<V>` | Composite sensor vocabulary trait with cosine similarity over feature vectors |
| 9 | `MinCutBoundary<V>` | Comfort-zone boundary computed from graph topology, not configured as a threshold |
| 10 | `MinCutBoundary<V>` | Both sides of the min-cut partition are observable and enumerable |
| 11 | `MinCutBoundary<V>` | Thin bridges between context clusters are automatically discovered |
| 12 | `MinCutBoundary<V>` | Boundary is dynamic: trust changes shift the cut (Graph B activation) |
| 13 | `CoherenceField<V>` | Familiar context blending: 0.3×instant + 0.7×ctx buffers transient noise |
| 14 | `SocialPhase` | Four-quadrant phase plane: ShyObserver, StartledRetreat, QuietlyBeloved, ProtectiveGuardian |
| 15 | `SocialPhase` | Schmitt trigger hysteresis on coherence axis prevents oscillation at phase boundary |
| 16 | `SocialPhase` | `led_tint()` returns distinct RGB values per phase for expressive LED output |
| 17 | `SocialPhase` | `expression_scale()` ordered QB > PG > SO > SR — scales all output channels |
| 18 | `SocialPhase` | Schmitt trigger hysteresis also applied to the tension axis |
| 19 | `SinkhornKnopp` | Output is a doubly stochastic matrix (element of the Birkhoff polytope) |
| 20 | `SinkhornKnopp` | Iterative row+column normalisation converges within `max_iterations` |
| 21 | `SinkhornKnopp` | Bounded mixing: no single context can accumulate more than 1.0 total weight |
| 22 | `SinkhornKnopp` | Non-negativity preserved throughout Sinkhorn-Knopp iteration |
| 23 | `SinkhornKnopp` | Idempotence: already doubly stochastic input is unchanged (Birkhoff membership) |
| 24 | `Personality` | `curiosity_drive` in [0,1] raises the positive interaction delta |
| 25 | `Personality` | `startle_sensitivity` in [0,1] amplifies the negative delta on startle events |
| 26 | `Personality` | `recovery_rate` (recovery_speed) in [0,1] speeds up coherence floor recovery |
| 27 | `Personality` | Modulators are independent: changing one parameter does not affect others |
| 28 | `Personality` | Extreme parameter values remain bounded; outputs stay within expected ranges |
| 29 | Full pipeline | Sensor → `ContextKey` → `CoherenceField` pipeline compiles and runs end-to-end |
| 30 | Full pipeline | 10-tick positive sequence produces strictly positive coherence |
| 31 | Full pipeline | Positive coherence with low tension → `SocialPhase` is not `StartledRetreat` |
| 32 | Full pipeline | `SinkhornKnopp` applied to a trust similarity matrix stays doubly stochastic |
| 33 | Full pipeline | `MinCutBoundary` partition separates high-trust from low-trust contexts |
| 34 | Full pipeline | Full CCF loop: build field, classify phase, verify LED tint changes across contexts |

## Usage

```toml
[dependencies]
ccf-core = "0.1"
```

## Features

- `std` — enables persistence helpers (off by default; crate is `no_std`)
- `serde` — enables `Serialize`/`Deserialize` on all public types

## Test Coverage

```
cargo test    # runs 98 tests: 64 unit tests + 34 patent-claim integration tests
```

The integration test suite (`tests/patent_claims.rs`) contains exactly one named test
per patent claim (`test_claim_N_<description>`), demonstrating each claimed behaviour
end-to-end using only the public API.

## License

Business Source License 1.1. Free for evaluation and non-production use.
Change date: 23 February 2032 → Apache License 2.0.
Commercial use requires a license from Flout Labs (cbyrne@floutlabs.com).
