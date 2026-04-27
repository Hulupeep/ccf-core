# ccf-core v1.0 — Canonical QAC Implementation

**Product Requirements Document**

| | |
|---|---|
| **Owner** | Colm Byrne, Flout Labs |
| **Status** | DRAFT — awaiting approval |
| **Date** | 2026-04-27 |
| **Trigger** | ccf-core v0.1.4 audit (2026-04-27) found Claims 1, 2, and 3 of the QAC representation theorem FAIL. Crate ships scalar arithmetic where the patent specifies matrix-form QAC, weighted average where it specifies hard min, and no Sinkhorn-Knopp projection at all. |
| **Patent backing** | US Provisionals 1, 2, 3, 4, 5 (filed); 6 (drafting). Non-provisional deadline: 2027-02-23. |
| **License** | BSL 1.1 with 2032-04-15 Apache 2.0 change date (matches v0.1.x) |

---

## 1. Why this exists

### 1.1 The audit finding, in one sentence

The published ccf-core Rust crate (v0.1.4, on crates.io, referenced by theshyrobot.com and the patent filings) implements a scalar approximation of the canonical Contextual Coherence Fields mathematics. It is not the math the patents describe.

### 1.2 What v1.0 is

`ccf-core v1.0.0` is the canonical, patent-aligned, production Rust implementation of CCF. Each of the seven QAC representation theorem claims has a corresponding implementation that matches the canonical form within numerical tolerance, with tests that assert canonical-form fidelity, plus a runtime certificate (κ_t) that streams a measurable deviation quantity at every update.

### 1.3 What v1.0 is NOT

- A patch of v0.1.x. The architecture is wrong; patching is not feasible.
- A research artifact. v1.0 is the production substrate.
- A re-derivation of the patent math. The math is fixed by the filed claims; this PRD specifies the *implementation* of that fixed math.
- A general-purpose ML library. v1.0 implements CCF, nothing else.

### 1.4 What happens to v0.1.x

- v0.1.x stays on crates.io for backwards compatibility with anyone who has already integrated it.
- v0.1.x is reframed in its README as "research preview, scalar approximation. See v1.0+ for canonical implementation."
- v0.1.x is no longer represented as the patent reduction-to-practice in any external materials.
- The mBot2 hardware work that ran on v0.1.x is reframed as "empirical exploration on a scalar approximation" rather than reduction-to-practice.

### 1.5 Deployment artifact — what "v1.0 ships" actually means

v1.0.0 is **not just a library**. It ships as a Rust workspace containing two crates, both at the same version:

| Crate | Type | Purpose |
|---|---|---|
| `ccf-core` | lib | The canonical math. The library that implementers integrate against. |
| `ccf-agent` | bin | A reference deployment binary. Reads config, connects to a Cognitum Seed (or alternative substrate), runs the canonical CCF loop, exposes runtime state via a small HTTP server. **The agent is the deployable unit for the Seed demo.** |

Plus `deploy/seed/` containing:
- `ccf-agent.service` — systemd unit for the Pi Zero 2 W
- `install.sh` — SSH-to-Seed install script (copy binary, install service, start)
- `config.example.toml` — example configuration with bearer token slot, Seed endpoint, taxonomy parameters

**The success state:** the user SSHs to a Cognitum Seed, runs `./install.sh`, the agent starts as a service, the user `curl`s a known endpoint and observes a κ_t stream + current trust matrix. The agent survives reboot. CCF is running on the Seed, end-to-end, observably.

This is the gate that closes v1.0. Without it, v1.0 is a library with no deployment story, and the headline demo cannot run.

---

## 2. Acceptance criteria — what "done" means

v1.0.0 ships when, and only when, all of the following hold:

| | Criterion |
|---|---|
| 1 | Canonical QAC update implemented as `A_{t+1} = L_t · (A_t^(1−α) ⊙ R^α) · C_t` with positive diagonal L_t and C_t, scalar α, Hadamard product, reference matrix R. Test asserts each component of the formula matches against a hand-computed reference. |
| 2 | Min-gate implemented as hard `g_t = min(C_inst, C_ctx)`. No smoothing, no weighted average. Test asserts `g_t = min(...)` exactly for a battery of `(C_inst, C_ctx)` pairs across the unit interval. |
| 3 | Sinkhorn-Knopp projection to Birkhoff polytope implemented in pure Rust, no_std-compatible. Test asserts row sums and column sums equal 1 within tolerance 1e-9 after projection, asserts non-negativity, asserts convergence within bounded iterations. |
| 4 | κ_t computed at every update step. Test asserts κ_t = 0 (within 1e-9) across all canonical-form unit tests. Documented streaming format. |
| 5 | ε_t pre-check fires before update; degenerate inputs (singular matrices, NaN, out-of-range) are detected and refused. Test asserts pre-check catches each documented failure mode. |
| 6 | Per-context independent accumulators with wall-clock anchoring. Test asserts that updating context K1 cannot influence context K2's accumulator state or last-update timestamp. (This claim already passes in v0.1; preserve it.) |
| 7 | Pinned-zero category mechanism implemented as structural. Test asserts pinned entries are bit-identical zero before and after update attempts; asserts κ_t excursion fires when an update is attempted on a pinned entry. |
| 8 | `cargo test --all-features` passes with zero failures, zero ignored. |
| 9 | `cargo build --no-default-features --target thumbv7em-none-eabihf` succeeds for `ccf-core` (or alternative documented embedded target). |
| 10 | `ccf-core` cross-compiles for Cognitum Seed target (ARM Linux glibc, Pi Zero 2 W) and runs the full test suite on-device. |
| 11 | `ccf-agent` binary cross-compiles for the Seed target, installs via `deploy/seed/install.sh` over SSH, runs as a systemd service, survives reboot. |
| 12 | A query to the running agent (`curl http://<seed>:<port>/state`) returns a structured response containing: current κ_t value, current C_eff per active context, current SocialPhase, last-update timestamp, partition state, and ε_t status. The endpoint produces fresh data within 1 second of a sensor event. |
| 13 | Documentation updated: README, /docs, theshyrobot.com primitives section, audit document included in /docs, deploy guide for Seed installation. |
| 14 | Release post drafted, dated, signed by Colm Byrne, published. |

A claim fails any criterion → v1.0.0 does not ship. No partial releases.

---

## 3. Scope

### 3.1 In scope

- Pure Rust implementation, no FFI dependencies for core math
- `std` and `no_std` feature flags (preserve from v0.1.x)
- The seven QAC representation theorem claims, all implemented faithfully
- **Stoer-Wagner min-cut partitioning, in core, no_std-compatible** (per §6.1)
- Tests asserting canonical-form fidelity for each claim
- κ_t and ε_t for Provisional 6 backing
- Public library API for: per-context update, gate evaluation, current state query, certificate streaming
- **`ccf-agent` reference binary** (per §1.5) — runtime host that uses the library, connects to a Cognitum Seed, exposes state via HTTP
- **Seed deployment artifacts** — systemd unit, install script, example configuration (per §1.5)
- Cross-compile and validate `ccf-core` on Cognitum Seed
- End-to-end install + run + observe validation on Cognitum Seed hardware
- Migration documentation for v0.1.x users (likely zero, but document the intent)

### 3.2 Explicitly out of scope for v1.0

- LLM integration. Provisional 5's LLM-bound action gate is a separate crate or feature in a future release.
- Sponsored Bridging (Prov 4 EXT 8). Future release.
- Merge-Split (Prov 4 EXT 6). Future release.
- Causation trail signing. The Cognitum Seed handles this externally; ccf-core does not need to embed crypto.
- Self-Model (Prov 4 EXT 3). Future release.
- Causation Trail (Prov 4 EXT 7). Future release.
- Performance optimisation beyond the minimum required to run on the Seed at 1 Hz tick rate. Optimisation is a v1.1 concern.
- A Python binding. ccf-py can be re-bound after v1.0 ships; not gating.

### 3.3 Deferred to v1.1+

- Higher-tick-rate operation (10 Hz+) requires SIMD or other optimisation
- Multi-fleet identity merging
- LLM-bound action space gating (Prov 5 reduction-to-practice)
- Web Assembly target (the Cognitum stack uses RuVector WASM; CCF-on-WASM may be useful but is not gating)

---

## 4. The canonical mathematics — implementation specification

Each subsection below specifies one of the seven claims. The implementer must produce code that matches the specification, plus a test that asserts the match.

### 4.1 The QAC update step (Claim 1)

**Canonical form:**

```
A_{t+1} = L_t · (A_t^(1−α) ⊙ R^α) · C_t
```

Where:
- `A_t` is an n×n trust accumulator matrix at time t (entries in [0, 1], rows and columns summing to ≤1)
- `L_t` is an n×n positive diagonal matrix (the left gauge)
- `C_t` is an n×n positive diagonal matrix (the right gauge)
- `R` is the n×n reference matrix (entries in [0, 1])
- `α ∈ (0, 1)` is the contraction exponent (scalar)
- `^` denotes element-wise power
- `⊙` denotes Hadamard (element-wise) product
- `·` denotes standard matrix multiplication

**Implementation requirements:**
- Use a small linear algebra crate (`nalgebra` for `std`, or hand-coded for `no_std`). No `ndarray` (heavy).
- L_t and C_t MUST be enforced as diagonal positive matrices structurally — represent them as `Vec<f64>` of length n, never as full matrices.
- α MUST be a runtime scalar parameter, not learned, not per-element, not time-varying within a single update. (α may be reset between deployments; it does not vary mid-update.)
- The Hadamard product and matrix multiplication MUST be in the order `L_t · (A_t^(1−α) ⊙ R^α) · C_t` exactly. No re-ordering.
- After the update, project back to the Birkhoff polytope via §4.3. The update step itself does not enforce row/column sum constraints; the projection does.

**Test requirements:**
- Hand-compute `A_{t+1}` for a 3×3 example by hand. Assert the implementation matches to tolerance 1e-12.
- Test with α at boundary values (0.01, 0.99) — must not blow up, must not collapse.
- Test that swapping L_t and C_t changes the result (catches accidental commutation).
- Test that replacing the Hadamard with arithmetic interpolation produces a different, detectable κ_t excursion.

### 4.2 The min-gate (Claim 2)

**Canonical form:**

```
g_t = min(C_inst, C_ctx)
```

**Implementation requirements:**
- Use `f64::min` or equivalent hard-min function. Not `(a + b) / 2`. Not `0.3 * a + 0.7 * b`. Not `(a^p + b^p)^(1/p)` for any p.
- The gate is not parameterised. There is no soft-min option in v1.0.
- Phase computation (Withdrawn / Cautious / Engaged / Familiar / Quietly Beloved) is *downstream* of the gate. Phase is a function of `C_eff = g_t` after the gate, not a replacement for it.
- The action permitted set is determined by `g_t` crossing thresholds per category. Thresholds are configurable; the gate computation is not.

**Test requirements:**
- Test `g_t == min(C_inst, C_ctx)` exactly (no tolerance) for 1000 random `(C_inst, C_ctx)` pairs in [0,1]².
- Test boundary cases: `(0, 0)`, `(0, 1)`, `(1, 0)`, `(1, 1)`, `(0.5, 0.5)`.
- Negative test: assert that a weighted-average implementation `0.3 * C_inst + 0.7 * C_ctx` would *fail* the test on a specific input pair (e.g., `(0.1, 0.9)`).

### 4.3 Sinkhorn-Knopp projection (Claim 3)

**Canonical form:**

Iteratively normalise rows and columns until convergence:

```
while not converged:
    A = A / row_sums(A)
    A = A / col_sums(A)
```

**Implementation requirements:**
- Operates on the trust accumulator matrix `A` after each QAC update (per §4.1).
- Convergence criterion: `max(|row_sum - 1|) < 1e-9` and `max(|col_sum - 1|) < 1e-9`.
- Iteration cap: 100 iterations. If not converged, return error (this is an ε_t condition, see §4.5).
- Non-negativity preserved: any negative entry → ε_t failure.
- Singular handling: if a row or column sums to zero, the matrix is degenerate; ε_t fails before projection runs.

**Test requirements:**
- After projection, assert all row sums are 1.0 ± 1e-9.
- After projection, assert all column sums are 1.0 ± 1e-9.
- After projection, assert all entries ≥ 0.
- Assert convergence within 100 iterations for a battery of well-conditioned inputs.
- Assert the ε_t pre-check rejects singular inputs without invoking the iteration.
- Compare convergence behaviour against the `demos/gavalas/sinkhorn.py` reference for matching inputs (sanity check, not formal verification).

### 4.4 κ_t runtime certificate (Claim 4 / Provisional 6)

**Canonical form:**

`κ_t` is a non-negative scalar quantity computed at every update step that measures the deviation between the executed update and the canonical QAC form. `κ_t = 0` (within numerical tolerance) means the implementation is faithful at this step.

**Specification source:** Provisional 6 draft. Implementation must match the Prov 6 formula exactly.

**Implementation requirements:**
- Computed at every update step. Not periodically. Not on demand.
- Streamed via a callback or channel mechanism so external observers (e.g., the MCP HUD overlay) can subscribe.
- An excursion (`κ_t > tolerance`) triggers gate fail-closed behaviour: the next gate evaluation returns "no actions permitted" until the excursion is acknowledged or resolved.

**Test requirements:**
- Assert `κ_t < 1e-9` across all canonical-form unit tests in §4.1.
- Assert `κ_t > tolerance` when a deliberately incorrect update is performed (e.g., replacing min-gate with weighted average mid-test).
- Assert that an excursion triggers gate fail-closed within one tick.

### 4.5 ε_t pre-check (Claim 5 / Provisional 6)

**Canonical form:**

`ε_t` runs *before* the QAC update step and detects conditions under which the update would be invalid. If ε_t fires, the update is skipped, the gate fails closed for the affected context, and the failure is logged.

**Documented failure modes ε_t must catch:**
- NaN or Inf in any input (A_t, L_t, C_t, R, α)
- α ∉ (0, 1)
- L_t or C_t with non-positive diagonal entries
- A_t with any negative entries
- A_t row or column sums equal to zero (singular projection target)
- ContextKey not in the registered taxonomy (unknown context)
- Update attempt on a pinned-zero category (see §4.7)

**Implementation requirements:**
- ε_t runs before §4.1, never after.
- Failure of ε_t is fail-closed: skip update, log, return.
- ε_t failures are visible to the κ_t certificate stream as a separate event class.

**Test requirements:**
- One test per documented failure mode. Each must catch the bad input and prevent the update from running.
- Test that a valid input passes ε_t cleanly (no false positives).

### 4.6 Per-context accumulators with wall-clock anchoring (Claim 6)

**Canonical form:**

Each ContextKey K_i has its own independent accumulator A_t^{K_i} with its own last-update timestamp τ_t^{K_i}. The state of K_i evolves independently of K_j for i ≠ j.

**Implementation requirements:**
- Storage: `HashMap<ContextKey, Accumulator>` (std) or equivalent for no_std (e.g., `heapless::IndexMap`).
- Each Accumulator carries: A_t matrix, last-update timestamp, update count.
- The QAC update step touches only the accumulator at the active context. No global normalisation across contexts.
- Wall-clock timestamps from a runtime-provided clock source (not a global tick counter).

**Test requirements:**
- Assert that updating K1 does not change K2's accumulator state or timestamp. (Existing v0.1.x test passes; port and preserve.)
- Assert that wall-clock timestamps are per-context, not shared.
- Assert that adding a new context does not invalidate existing context state.

### 4.7 Permanently restricted entries / pinned-zero categories (Claim 7)

**Canonical form:**

Some action categories are *structurally* zero and cannot accumulate trust. Updates targeting these entries are detected and refused. The mechanism is structural (a flag or separate matrix), not threshold-based.

**Implementation requirements:**
- Each Accumulator has a `pinned_categories: BitSet` (or equivalent for no_std).
- The QAC update step skips columns corresponding to pinned categories. Pinned columns remain bit-identical zero through any number of updates.
- An attempted update to a pinned category triggers κ_t excursion and ε_t failure (this is the canonical falsifiability test from PiCar-X spec §4.2).

**Test requirements:**
- Pin a category at construction. Run 10,000 updates. Assert the pinned column is bit-identical zero (not 1e-15, zero).
- Attempt to update a pinned category via debug/test API. Assert κ_t excursion, gate fail-closed.
- Assert pinned status persists across update calls.

---

## 5. Public API

### 5.1 ccf-core library — core types

```rust
pub struct ContextKey(/* opaque, hashable */);

pub struct CcfConfig {
    pub n_categories: usize,
    pub alpha: f64,
    pub kappa_tolerance: f64,
    pub max_sinkhorn_iters: usize,
    pub pinned_categories: Vec<usize>,
}

pub struct CcfEngine { /* ... */ }

pub struct UpdateResult {
    pub kappa_t: f64,
    pub epsilon_failed: Option<EpsilonReason>,
    pub gate_state: GateState,
    pub phase: SocialPhase,
}

pub enum EpsilonReason {
    NaNInput, AlphaOutOfRange, NegativeEntry, SingularInput,
    UnknownContext, PinnedCategoryWrite, /* ... */
}

pub enum SocialPhase { Withdrawn, Cautious, Engaged, Familiar, QuietlyBeloved }

pub struct GateState {
    pub permitted: BitSet,
    pub c_eff: f64,
}
```

### 5.2 ccf-core library — core methods

```rust
impl CcfEngine {
    pub fn new(config: CcfConfig) -> Self;
    pub fn update(&mut self, ctx: ContextKey, sensor_vec: &[f64; 8], now: Instant) -> UpdateResult;
    pub fn gate(&self, ctx: ContextKey) -> GateState;
    pub fn phase(&self, ctx: ContextKey) -> SocialPhase;
    pub fn kappa_stream(&self) -> impl Stream<Item = f64>;
    pub fn snapshot(&self) -> EngineSnapshot;
}
```

### 5.3 ccf-agent HTTP API surface

The reference agent exposes the following read-only HTTP endpoints (no auth in v1.0; the agent is intended to be local-network-only and protected by the Seed's network boundary):

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | `200 OK` with `{"status":"ok","uptime_secs":N}` |
| GET | `/state` | Current C_eff, current phase, current ContextKey, last-update timestamp, ε_t status |
| GET | `/trust-matrix` | Full active trust matrix as a 2D array, with row/column labels |
| GET | `/kappa-stream` | Server-Sent Events stream: one `data:` line per update tick, JSON payload `{"t":timestamp,"kappa":value,"class":"in_class"}` |
| GET | `/partition` | Stoer-Wagner min-cut partition; if Seed partition is being verified, includes agreement status |
| GET | `/certificate` | Last falsifiability class, last excursion details, total ticks, total excursions |
| GET | `/config` | Current configuration (with bearer token redacted) |

All endpoints return JSON. Errors return appropriate HTTP status codes with a JSON error body.

### 5.4 Public stability

- `ccf-core` lib API: stable for the v1.x line. Breaking changes ship as v2.0.
- `ccf-agent` HTTP API: stable for the v1.x line.
- `ccf-agent` config TOML schema: stable for the v1.x line; new fields default to optional.

---

## 6. Architectural decisions

### 6.1 Min-cut partition: implemented in core, optionally verified against Cognitum

v1.0 ships a Stoer-Wagner min-cut implementation in pure Rust, no_std-compatible, as a first-class component of the canonical math.

**Rationale:**
- Patent reduction-to-practice requires CCF to compute its own canonical answer. Outsourcing the partition would mean CCF cannot *certify* the partition is canonical — it can only certify agreement with whatever Cognitum produces.
- κ_t certificate must be able to detect partition deviations. A certificate that doesn't observe the partition computation has a hole.
- Non-Seed deployments (laptop testing, mBot2 alone, future hardware, any non-Cognitum substrate) have no underlying min-cut to consume. CCF must function correctly on its own.
- Stoer-Wagner is a fixed, well-specified algorithm — implementing it in Rust adds engineering work but does not introduce mathematical risk.

**Cognitum integration policy:** When deployed on a Cognitum Seed, v1.0 may consume the Seed's existing partition via its API as a *performance hint*. The hint is verified against CCF's own canonical computation at every tick. If they agree (within tolerance), the system uses the cached result for speed. If they disagree, the disagreement is logged as a κ_t excursion classified `partition_disagreement` (a new falsifiability class added in this PRD), and CCF's own computation is the authoritative answer.

**Implication:** v1.0's min-cut is portable. The Cognitum integration is an optimization, not a dependency.

**Implementation note:** Stoer-Wagner on a 16-node graph (the PiCar-X taxonomy) is well within performance bounds for a 1 Hz tick rate on Pi Zero 2 W hardware. Larger graphs (n > 64) are not a v1.0 target.

### 6.2 Linear algebra: nalgebra for std, hand-coded for no_std

- `std` feature flag: depends on `nalgebra` for matrix operations
- `no_std` feature flag: hand-coded n×n operations using stack-allocated arrays. Maximum supported `n` is 16 (16-context taxonomy from PiCar-X spec). Larger n requires `std`.

### 6.3 Numerical precision: f64 throughout

- f64 for all computation
- Tolerances (κ_t, Sinkhorn convergence, ε_t boundary) all defined relative to f64 epsilon
- f32 not supported in v1.0

### 6.4 Determinism

- The QAC update is deterministic given inputs and α.
- No randomness in core paths.
- Wall-clock timestamps are observed but do not affect computation (they label state, they do not perturb it).

### 6.5 Concurrency

- v1.0 is single-threaded.
- CcfEngine is `!Send` and `!Sync` in v1.0. (Multi-threading is a v1.1 concern.)
- The κ_t stream may be read from a different thread provided proper synchronisation; the engine itself is not.

### 6.6 No-std target

- Primary embedded target: `thumbv7em-none-eabihf` (Cortex-M7, used by mBot2 CyberPi and similar)
- Secondary: ARM Linux glibc (Cognitum Seed, Raspberry Pi)
- Both must build clean; no_std must pass the canonical-form tests using the same test code.
- **`no_std` applies to `ccf-core` only.** `ccf-agent` is `std` and Linux-targeted.

### 6.7 ccf-agent architecture

The agent is a Rust binary that:

1. **Loads a TOML config** specifying the Seed endpoint, bearer token (from environment or file), the active taxonomy parameters, log paths, and the HTTP listener port.
2. **Establishes a session with the Seed** by hitting `/api/v1/identity` and `/api/v1/store/status` to confirm reachability and matching expected device fingerprint.
3. **Runs the canonical CCF loop:**
   - Polls `/api/v1/store/sync` (or subscribes if the Seed exposes a streaming endpoint) for new sensor vectors
   - For each vector, derives the active ContextKey from the configured taxonomy
   - Runs `ccf-core::CcfEngine::update(ctx, vector, now)` — full canonical pipeline (ε_t pre-check, QAC update, Sinkhorn projection, Stoer-Wagner partition, κ_t computation, gate evaluation, phase resolution)
   - Writes a signed `gate_decision` event back to the Seed's `/api/v1/store/ingest` (so the audit chain captures the decision, not just the input)
   - Updates internal state for the HTTP query endpoints
4. **Exposes a small HTTP server** (default `:8080`) with read-only endpoints:
   - `GET /health` — liveness probe (no auth)
   - `GET /state` — current C_eff, current phase, current context, last-update timestamp, ε_t status
   - `GET /trust-matrix` — full active trust matrix (16×5 in the PiCar-X taxonomy)
   - `GET /kappa-stream` — Server-Sent Events stream of κ_t per tick
   - `GET /partition` — current Stoer-Wagner partition + agreement status with Seed-reported partition (if applicable)
   - `GET /certificate` — current falsifiability class, last excursion details
5. **Logs** structured JSON to stdout (captured by systemd journal) and a rotating log file.
6. **Handles graceful shutdown** on SIGTERM (drains the queue, flushes logs, releases the Seed session).

The agent is a reference implementation. Integrators may write their own agents using `ccf-core` directly. The reference agent exists so the Seed demo runs out of the box.

**Authentication:** the agent uses the Seed's bearer-token auth model. Token is read from environment variable `COGNITUM_TOKEN` or from a sidecar file referenced in the config. Never logged.

**Failure handling:** if the Seed becomes unreachable, the agent enters a degraded mode — local CCF state continues to evolve based on cached data, but new ingest is queued. Recovery is automatic on Seed return.

---

## 7. Test discipline

Every claim in §4 has tests that assert canonical-form fidelity. The test suite is the audit mechanism for v1.0; if v1.0's tests pass, v1.0 is canonical.

### 7.1 Test categories

1. **Canonical-form tests.** Hand-computed reference values, asserted to f64 tolerance.
2. **Negative tests.** Demonstrate that a deliberately wrong implementation (weighted average, soft-min, scalar update) would *fail* the test.
3. **ε_t coverage tests.** One per documented failure mode in §4.5.
4. **Pinned-category persistence tests.** 10,000 updates, asserted bit-identical zero.
5. **Cross-context independence tests.** Updating K1 does not perturb K2.
6. **No_std tests.** All canonical-form tests pass under `--no-default-features`.
7. **Cognitum-on-device tests.** Test suite runs on Pi Zero 2 W, all pass.

### 7.2 Coverage requirement

Every line of canonical math in §4.1, §4.2, §4.3 covered by at least one canonical-form test and at least one negative test.

### 7.3 No silent skips

- `cargo test --all-features` reports zero failures and zero ignored tests.
- Any test that cannot run on a given target fails fast with a clear message.

---

## 8. Performance requirements

v1.0 must run on the Cognitum Seed (Pi Zero 2 W, 512MB, single-core slice) at the canonical tick rate.

| Constraint | Bound |
|---|---|
| Single update step (n=16 contexts, 6 categories) | < 10 ms |
| Sinkhorn projection (16×16 matrix) | < 5 ms |
| κ_t computation per update | < 1 ms |
| Resident memory | < 50 MB (leaves headroom for Cognitum firmware + RVF store) |
| Tick rate sustained | 1 Hz minimum, 2 Hz target |

Performance optimisation beyond these bounds is v1.1 scope. Anything below these bounds blocks v1.0.

---

## 9. Migration from v0.1.x

### 9.1 Breaking changes (acknowledged)

- Public API completely replaced (see §5)
- Scalar accumulator state semantics replaced with matrix-form
- Configuration structure replaced

### 9.2 Migration path for v0.1.x users

There are no known external v0.1.x integrators beyond Flout Labs's own work. The v0.1.x crate stays on crates.io with a README reframing as research preview. v1.0 is published as a separate version; users opt in.

Internal migration:
- mBot2 firmware stays on v0.1.x for the existing tabletop demos until a hardware refresh
- New mBot2 + Seed work uses v1.0
- PiCar-X demo uses v1.0 from day 1

### 9.3 Public README updates

- v0.1.x README: prepend "**Research preview.** This crate uses a scalar approximation. The canonical patent-aligned implementation is v1.0+."
- v1.0 README: lead with the canonical math, link to the audit document, link to the patent filings.

---

## 10. Documentation deliverables

Shipped with v1.0.0:

1. `README.md` (workspace root) — what the workspace is, what each crate does, link to docs
2. `crates/ccf-core/README.md` — library-focused: how to integrate `ccf-core` into your own binary
3. `crates/ccf-agent/README.md` — binary-focused: what the agent does, how to configure it
4. `docs/canonical-spec.md` — the seven claims and their implementations, indexed to this PRD
5. `docs/audit-2026-04-27.md` — the original audit report that motivated v1.0 (preserves causation trail)
6. `docs/migration-from-v0.1.md` — for the (probably zero) external integrators
7. `docs/embedded-targets.md` — building `ccf-core` for Pi Zero 2 W and Cortex-M7
8. `docs/deploy-seed.md` — step-by-step Seed installation guide using `deploy/seed/install.sh`
9. `examples/single-update.rs` — minimal example: one context, one update, observe result (uses `ccf-core` directly, no agent)
10. `examples/cognitum-bridge.rs` — sketch of the Cognitum Seed integration pattern from a custom binary
11. Public API docs via `cargo doc` — every public type and method in both crates documented

---

## 11. Timeline — 8-week sprint

| Week | Deliverable |
|---|---|
| 1 (Apr 28 – May 4) | PRD approved, repo set up (workspace with `ccf-core` and `ccf-agent` crates), §4.1 (QAC update) implemented + tests |
| 2 (May 5 – May 11) | §4.2 (min-gate) and §4.3 (Sinkhorn) implemented + tests |
| 3 (May 12 – May 18) | Stoer-Wagner min-cut implemented + tests; partition disagreement detection |
| 4 (May 19 – May 25) | §4.4 (κ_t) and §4.5 (ε_t) implemented + tests |
| 5 (May 26 – Jun 1) | §4.6 (per-context) ported and §4.7 (pinned-zero) implemented + tests; ccf-core acceptance criteria 1–10 met |
| 6 (Jun 2 – Jun 8) | `ccf-agent` binary: config loading, Seed session, CCF loop, HTTP server endpoints |
| 7 (Jun 9 – Jun 15) | Deployment artifacts (systemd unit, install.sh, deploy docs); cross-compile both crates; end-to-end install + run + observe on Cognitum Seed |
| 8 (Jun 16 – Jun 22) | Documentation completion, performance validation against §8 bounds, release post, v1.0.0 published to crates.io |

**Slip allowance:** 1 week. If by 2026-06-29 v1.0.0 is not published, escalate to a patent attorney conversation about whether the non-provisional should narrow scope to claims that *are* implemented vs. claims that are specified-but-not-implemented.

---

## 12. Risks

### 12.1 Mathematical risk

Implementing canonical math is fundamentally specified work, but two risks remain:

- **Sinkhorn convergence on near-degenerate matrices.** May require additional ε_t conditions during implementation. Acceptable; document and add tests.
- **κ_t formula precision on f64.** The Provisional 6 formula must produce 0 within 1e-9 for canonical updates. If it doesn't, either the tolerance widens or the formula gets refined. Either way, document.

### 12.2 Schedule risk

- Six weeks is tight. If the implementer is Colm-only, this is realistic but consumes hyperfocus completely. Worth budgeting an hour a week for the rest of the project (causation trail updates, Notion roadmap maintenance, mathematician handoff prep).
- If a contract Rust developer is engaged, NDA + patent context briefing required first.

### 12.3 Patent risk

- If v1.0 ships with κ_t but Provisional 6 has not yet been filed, the implementation pre-dates the filing. This is fine — Prov 6 is in draft and should be filed before or alongside v1.0. **Recommend: file Prov 6 in May 2026 alongside Week 3 of the implementation.**
- Public crate updates create timestamped prior art. v1.0 release post is patent evidence. Date it carefully.

### 12.4 Commercial risk

- theshyrobot.com currently overstates what the v0.1 crate does. Until v1.0 ships, the site needs a temporary tone-down. **Recommend: 30-minute edit this week to add "v1.0 in development" framing.**
- The Ruv conversation should not happen before v1.0 is at least 50% implemented. The interim demo footage on v0.1 (if filmed at all) cannot be the artifact sent to Ruv.

---

## 13. Approval

Pending: Colm Byrne sign-off on this PRD.

Once approved, this document becomes the contract for v1.0.0 development. Any deviation from the canonical math in §4 requires either a PRD amendment or a documented falsifiability-class admission in the release notes.

---

## Appendix A — falsifiability classes (for κ_t excursion classification)

For the κ_t certificate, when an excursion is detected, the falsifiability table classifies the cause:

| Class | Description |
|---|---|
| `in_class` | κ_t is within tolerance; canonical |
| `quantization_artifact` | Discrete approximation introduces bounded error |
| `non_gauge_approximation` | Gauge transformation not preserved |
| `soft_min_substitution` | Min replaced with smooth approximation |
| `weighted_average_substitution` | Min replaced with linear combination (the v0.1.x failure mode) |
| `approximate_log_geodesic` | Log/exp map approximation issues |
| `arithmetic_interpolation_regression` | Linear interpolation where geodesic was required (the v0.1.x QAC failure mode) |
| `partition_disagreement` | CCF's canonical min-cut and an external partition source (e.g. Cognitum Seed) disagree beyond tolerance — CCF's answer is authoritative |
| `other` | Document explicitly |

This table is exposed via the κ_t stream as part of the diagnostic payload.

## Appendix B — Audit findings, for posterity

Findings from the 2026-04-27 audit that motivated v1.0:

- **Claim 1 FAIL** (`arithmetic_interpolation_regression`): scalar additive update, no matrix form, no L_t/C_t, no Hadamard, no R. Evidence: `crates/mbot-core/src/coherence/mod.rs:459`.
- **Claim 2 FAIL** (`weighted_average_substitution`): `0.3 * instant + 0.7 * ctx` for familiar contexts. Evidence: `crates/mbot-core/src/coherence/mod.rs:607`.
- **Claim 3 FAIL** (no implementation): no Rust Sinkhorn-Knopp; only a Python demo at `demos/gavalas/sinkhorn.py`.
- **Claims 4, 5, 7 not implemented** (Provisional 6 in draft).
- **Claim 6 PASS** (narrow): per-context HashMap with per-accumulator timestamps. Evidence: `crates/mbot-core/src/coherence/mod.rs:425, 555`.
- **Tests:** 1165 passed, 0 failed, 1 ignored. None assert canonical QAC, Sinkhorn, κ_t, ε_t, or pinned-zero.

The audit document is preserved in full at `/docs/audit-2026-04-27.md` in the v1.0 repository.
