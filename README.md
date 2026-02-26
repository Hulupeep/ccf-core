# ccf-core

**Contextual Coherence Fields** — earned relational trust for autonomous systems.

Patent pending: US Provisional Application 63/988,438 (priority date 23 Feb 2026).

## What This Is

CCF is a computational architecture for emergent social behaviour in autonomous systems.
An agent maintains a *field* of trust states — one per sensory context — learned
continuously from experience. Trust earned in a bright quiet room does not transfer to
a dark noisy room unless the field explicitly learns they are similar.

## Patent Claim Map

| Type | Patent Claims | Description |
|------|--------------|-------------|
| `SensorVocabulary` | 1, 8 | Composite sensor context key — the situational fingerprint |
| `ContextKey<V>` | 1, 8 | Discrete context identifier from quantised sensor signals |
| `CoherenceAccumulator` | 2–5 | Per-context trust state with earned floor and asymmetric decay |
| `CoherenceField<V>` | 6–7, 13 | Trust field: context-keyed accumulator map with min-gate |
| `SocialPhase` | 14–18 | Four-quadrant phase classifier with Schmitt trigger hysteresis |
| `SinkhornKnopp` | 19–23 | Birkhoff polytope projector — doubly stochastic mixing matrix |
| `MinCutBoundary<V>` | 9–12 | Stoer-Wagner comfort-zone boundary discovery |
| `Personality` | 3 (modulators) | Dynamic modulators: curiosity, startle sensitivity, recovery speed |

## Usage

```toml
[dependencies]
ccf-core = "0.1"
```

## Features

- `std` — enables persistence helpers (off by default; crate is `no_std`)
- `serde` — enables `Serialize`/`Deserialize` on all public types

## License

Business Source License 1.1. Free for evaluation and non-production use.
Change date: 23 February 2032 → Apache License 2.0.
Commercial use requires a license from Flout Labs (cbyrne@floutlabs.com).
