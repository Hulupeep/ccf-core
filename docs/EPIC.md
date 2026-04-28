# ccf-core v1.0 Epic Snapshot

This is a frozen snapshot. The active artifact is the GitHub epic issue and Project board.

Generated: 2026-04-27T17:41:13Z

## Live Artifacts

- Project board: [https://github.com/users/Hulupeep/projects/27](https://github.com/users/Hulupeep/projects/27)
- Epic issue: [#1](https://github.com/Hulupeep/ccf-core/issues/1)
- PRD snapshot: [docs/ccf-core-v1-prd.md](ccf-core-v1-prd.md)
- Audit anchor: [docs/audit-2026-04-27.md](audit-2026-04-27.md)
- Coverage matrix: [docs/COVERAGE.md](COVERAGE.md)

## Goal

Deliver the PRD success state: a user SSHs to a Cognitum Seed, runs `./install.sh` from `deploy/seed/`, `ccf-agent` starts as a systemd service, `curl http://<seed>:<port>/state` returns live κ_t/trust/phase/partition/ε_t state, and the service survives reboot.

## Strategic Context

The 2026-04-27 audit found that v0.1.x did not implement the canonical QAC representation theorem: Claim 1 used scalar arithmetic rather than matrix-form QAC, Claim 2 used a weighted average rather than hard min, Claim 3 had no Rust Sinkhorn-Knopp projection, and Claims 4, 5, and 7 were not implemented. v1.0 is the canonical implementation contract that remediates those findings.

As of 2026-04-28, Provisional 6 is not filed. Stories #5, #10, #11, #13, and #14 are therefore spec-lock blockers before implementation: the QAC theorem source, κ_t formula, ε_t behavior, pinned-zero mechanism, and falsifiability classes must be frozen in an unfiled Prov 6 draft, filed Prov 6, or equivalent signed mathematical specification before those stories start.

## Dependency Graph

- Phase 0 scaffold: #2, #3, #4.
- Canonical math: #5, #6, #7.
- Min-cut and certificate substrate: #8, #9, #10, #11, #14, #15.
- State management: #12, #13.
- no_std and Seed core validation: #15, #16.
- Agent runtime and API: #17, #18, #19, #20.
- Deployment and operational proof: #21, #22.
- Release closure: #23, #24, #25.

## Stories

| Slice | Issue | Title | Phase | Inputs | PRD Coverage |
|---|---:|---|---|---|---|
| 00 | [#2](https://github.com/Hulupeep/ccf-core/issues/2) | Repository scaffolding | `phase:0-scaffold` | None | criterion 8, criterion 9 |
| 01 | [#3](https://github.com/Hulupeep/ccf-core/issues/3) | Audit document committed to /docs | `phase:0-scaffold` | #2 [00] | criterion 13 |
| 02 | [#4](https://github.com/Hulupeep/ccf-core/issues/4) | Public API skeleton | `phase:0-scaffold` | #2 [00] | criterion 8 |
| 03 | [#5](https://github.com/Hulupeep/ccf-core/issues/5) | QAC update step | `phase:1-canonical-math` | #4 [02] | criterion 1, criterion 8 |
| 04 | [#6](https://github.com/Hulupeep/ccf-core/issues/6) | Min-gate | `phase:1-canonical-math` | #4 [02] | criterion 2, criterion 8 |
| 05 | [#7](https://github.com/Hulupeep/ccf-core/issues/7) | Sinkhorn-Knopp projection | `phase:2-min-cut` | #5 [03] | criterion 3, criterion 8 |
| 06 | [#8](https://github.com/Hulupeep/ccf-core/issues/8) | Stoer-Wagner min-cut | `phase:2-min-cut` | #4 [02] | criterion 8, criterion 10 |
| 07 | [#9](https://github.com/Hulupeep/ccf-core/issues/9) | Partition disagreement detection | `phase:2-min-cut` | #8 [06] | criterion 4, criterion 10, criterion 12 |
| 08 | [#10](https://github.com/Hulupeep/ccf-core/issues/10) | κ_t runtime certificate | `phase:3-certificate` | #5 [03], #6 [04], #7 [05], #9 [07] | criterion 4, criterion 8 |
| 09 | [#11](https://github.com/Hulupeep/ccf-core/issues/11) | ε_t pre-check | `phase:3-certificate` | #5 [03], #7 [05], #10 [08] | criterion 5, criterion 8 |
| 10 | [#12](https://github.com/Hulupeep/ccf-core/issues/12) | Per-context accumulators | `phase:4-state-mgmt` | #5 [03], #11 [09] | criterion 6, criterion 8 |
| 11 | [#13](https://github.com/Hulupeep/ccf-core/issues/13) | Pinned-zero categories | `phase:4-state-mgmt` | #11 [09], #12 [10] | criterion 7, criterion 8 |
| 12 | [#14](https://github.com/Hulupeep/ccf-core/issues/14) | Falsifiability class instrumentation | `phase:3-certificate` | #9 [07], #10 [08], #11 [09], #13 [11] | criterion 4, criterion 5, criterion 7, criterion 12 |
| 13 | [#15](https://github.com/Hulupeep/ccf-core/issues/15) | no_std target validation for ccf-core | `phase:1-canonical-math` | #5 [03], #6 [04], #7 [05], #8 [06], #10 [08], #11 [09], #12 [10], #13 [11], #14 [12] | criterion 8, criterion 9 |
| 14 | [#16](https://github.com/Hulupeep/ccf-core/issues/16) | Cognitum Seed cross-compile of ccf-core | `phase:6-deploy` | #15 [13] | criterion 10 |
| 15 | [#17](https://github.com/Hulupeep/ccf-core/issues/17) | ccf-agent configuration loading and Seed session establishment | `phase:5-agent` | #4 [02], #16 [14] | criterion 11, criterion 12 |
| 16 | [#18](https://github.com/Hulupeep/ccf-core/issues/18) | ccf-agent canonical CCF runtime loop | `phase:5-agent` | #5 [03], #6 [04], #7 [05], #8 [06], #10 [08], #11 [09], #12 [10], #13 [11], #14 [12], #17 [15] | criterion 1, criterion 2, criterion 3, criterion 4, criterion 5, criterion 6, criterion 7, criterion 12 |
| 17 | [#19](https://github.com/Hulupeep/ccf-core/issues/19) | ccf-agent HTTP server with all endpoints | `phase:5-agent` | #18 [16] | criterion 4, criterion 12 |
| 18 | [#20](https://github.com/Hulupeep/ccf-core/issues/20) | ccf-agent structured logging, graceful shutdown, degraded-mode handling | `phase:5-agent` | #17 [15], #18 [16] | criterion 11, criterion 12 |
| 19 | [#21](https://github.com/Hulupeep/ccf-core/issues/21) | Deployment artifacts: systemd unit, install.sh, config.example.toml | `phase:6-deploy` | #17 [15], #19 [17], #20 [18] | criterion 11 |
| 20 | [#22](https://github.com/Hulupeep/ccf-core/issues/22) | End-to-end deployment validation on Cognitum Seed | `phase:6-deploy` | #16 [14], #21 [19] | criterion 11, criterion 12 |
| 21 | [#23](https://github.com/Hulupeep/ccf-core/issues/23) | Performance validation against PRD §8 bounds | `phase:7-release` | #22 [20] | criterion 10, criterion 11, criterion 12 |
| 22 | [#24](https://github.com/Hulupeep/ccf-core/issues/24) | Documentation suite | `phase:7-release` | #3 [01], #15 [13], #19 [17], #21 [19], #22 [20] | criterion 13 |
| 23 | [#25](https://github.com/Hulupeep/ccf-core/issues/25) | Release post draft and v1.0.0 publish to crates.io | `phase:7-release` | #22 [20], #23 [21], #24 [22] | criterion 8, criterion 13, criterion 14 |

## Spec-Lock Blockers

- [ ] #5 [03] QAC update step: QAC representation theorem source frozen/filed.
- [ ] #10 [08] κ_t runtime certificate: Prov 6 formula frozen/filed.
- [ ] #11 [09] ε_t pre-check: Prov 6 ε_t behavior frozen/filed.
- [ ] #13 [11] Pinned-zero categories: Prov 6 and/or PiCar-X pinned-zero specification frozen/filed.
- [ ] #14 [12] Falsifiability class instrumentation: Prov 6 excursion/classification semantics frozen/filed.

## Story Checklist

- [ ] [#2](https://github.com/Hulupeep/ccf-core/issues/2) [00] Repository scaffolding
- [ ] [#3](https://github.com/Hulupeep/ccf-core/issues/3) [01] Audit document committed to /docs
- [ ] [#4](https://github.com/Hulupeep/ccf-core/issues/4) [02] Public API skeleton
- [ ] [#5](https://github.com/Hulupeep/ccf-core/issues/5) [03] QAC update step
- [ ] [#6](https://github.com/Hulupeep/ccf-core/issues/6) [04] Min-gate
- [ ] [#7](https://github.com/Hulupeep/ccf-core/issues/7) [05] Sinkhorn-Knopp projection
- [ ] [#8](https://github.com/Hulupeep/ccf-core/issues/8) [06] Stoer-Wagner min-cut
- [ ] [#9](https://github.com/Hulupeep/ccf-core/issues/9) [07] Partition disagreement detection
- [ ] [#10](https://github.com/Hulupeep/ccf-core/issues/10) [08] κ_t runtime certificate
- [ ] [#11](https://github.com/Hulupeep/ccf-core/issues/11) [09] ε_t pre-check
- [ ] [#12](https://github.com/Hulupeep/ccf-core/issues/12) [10] Per-context accumulators
- [ ] [#13](https://github.com/Hulupeep/ccf-core/issues/13) [11] Pinned-zero categories
- [ ] [#14](https://github.com/Hulupeep/ccf-core/issues/14) [12] Falsifiability class instrumentation
- [ ] [#15](https://github.com/Hulupeep/ccf-core/issues/15) [13] no_std target validation for ccf-core
- [ ] [#16](https://github.com/Hulupeep/ccf-core/issues/16) [14] Cognitum Seed cross-compile of ccf-core
- [ ] [#17](https://github.com/Hulupeep/ccf-core/issues/17) [15] ccf-agent configuration loading and Seed session establishment
- [ ] [#18](https://github.com/Hulupeep/ccf-core/issues/18) [16] ccf-agent canonical CCF runtime loop
- [ ] [#19](https://github.com/Hulupeep/ccf-core/issues/19) [17] ccf-agent HTTP server with all endpoints
- [ ] [#20](https://github.com/Hulupeep/ccf-core/issues/20) [18] ccf-agent structured logging, graceful shutdown, degraded-mode handling
- [ ] [#21](https://github.com/Hulupeep/ccf-core/issues/21) [19] Deployment artifacts: systemd unit, install.sh, config.example.toml
- [ ] [#22](https://github.com/Hulupeep/ccf-core/issues/22) [20] End-to-end deployment validation on Cognitum Seed
- [ ] [#23](https://github.com/Hulupeep/ccf-core/issues/23) [21] Performance validation against PRD §8 bounds
- [ ] [#24](https://github.com/Hulupeep/ccf-core/issues/24) [22] Documentation suite
- [ ] [#25](https://github.com/Hulupeep/ccf-core/issues/25) [23] Release post draft and v1.0.0 publish to crates.io

## SpecFlow Shape Used

Each story issue contains: scope, gate/invariant, inputs, outputs, named failure modes, verification command, invariants referenced, acceptance criteria, Gherkin scenario, journey contract note, PRD/audit references, causation note, persona simulation, definition of done, and pre-flight findings.
