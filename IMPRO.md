# IMPRO.md – Implementation Plan for Improvement Opportunities

## Objective
Create a systematic, DRY‑compliant, and modular roadmap to address the identified **Improvement Opportunities** in the `llama-rs` codebase. The plan will guide developers through refactoring, feature extraction, and reusable component creation while ensuring test coverage and CI compliance.

---

## 1. Scope Definition
| Area | Current Pain Point | Desired Outcome |
|------|-------------------|-----------------|
| **Core Logic** | Repeated parsing and token‑generation code across `crates/llama` and `crates/ggml` | Centralised parsing module with a single public API |
| **Configuration** | Multiple ad‑hoc config structs duplicated in `llama-cli` and `llama-server` | Shared `config` crate exposing a unified `Config` type |
| **Error Handling** | Inconsistent `Result` patterns, many `unwrap()` calls | Adopt a project‑wide `Error` enum and propagate errors via `Result<T, Error>` |
| **Testing** | Overlap between unit tests and integration tests, missing coverage for new modules | Consolidated test suite with `#[cfg(test)]` modules and integration tests under `tests/` |
| **Build Scripts** | Cargo features scattered, duplicate `build.rs` logic | Single `build.rs` in workspace root, feature flags defined centrally |

---

## 2. High‑Level Phases
| Phase | Description | Deliverables |
|-------|-------------|--------------|
| **0️⃣ Preparation** | Verify CI passes, generate baseline metrics (test coverage, clippy warnings). | CI status badge, coverage report (`target/coverage/`) |
| **1️⃣ Architecture Review** | Map existing duplicated code using CodeGraph, identify common abstractions. | `ARCHITECTURE.md` update, impact diagram (Mermaid) |
| **2️⃣ Core Refactorings** | Extract shared utilities, create new crates (`common`, `config`, `error`). | New crates with `Cargo.toml` entries, updated `Cargo.lock` |
| **3️⃣ DRY Enforcement** | Replace duplicated snippets with calls to the new shared modules. | No duplicate code warnings, updated modules across crates |
| **4️⃣ Modular API Surface** | Publish public APIs, add documentation comments (`///`). | Updated `README.md` usage examples, `cargo doc` passes without warnings |
| **5️⃣ Test Consolidation** | Write missing tests, migrate duplicated tests to shared test harness. | 100 % of new code covered, `cargo test --workspace` green |
| **6️⃣ CI / Lint Integration** | Ensure `cargo fmt` and `cargo clippy` succeed with new code. | CI pipeline passes, badge updates |
| **7️⃣ Documentation & Release** | Draft `IMPRO.md`, update `CHANGELOG.md`, tag release. | New version tag, release notes, merged PR |

---

## 3. Detailed Task Breakdown (per Phase)
### Phase 0 – Preparation (Completed)
1. Run `cargo fmt --all -- --check` and `cargo clippy --workspace -- -D warnings`.
2. Capture current test coverage (`cargo tarpaulin -o Html` or `cargo llvm-cov`).
3. Record baseline metrics in `docs/metrics/README.md`.

### Phase 1 – Architecture Review
1. **CodeGraph Query** – `codegraph_context` for symbols related to parsing, config, and error handling.
2. Generate a Mermaid diagram of module dependencies.
3. Identify duplicated functions (e.g., `parse_input`, `load_model`).
4. Document findings in `ARCHITECTURE.md` under a new *Improvement Opportunities* section.

### Phase 2 – Core Refactorings
| Sub‑task | Owner | Files Affected |
|----------|-------|----------------|
| Create `crates/common` with utilities (`log`, `timer`, `bytes_util`). | Refactor Lead | `crates/*/src/**/*.rs` |
| Create `crates/config` exposing `Config` struct and `load_from_env`. | Refactor Lead | `crates/llama-cli/src/main.rs`, `crates/llama-server/src/main.rs` |
| Create `crates/error` with a unified `Error` enum and `Result<T>` alias. | Refactor Lead | All crates using `anyhow` or `unwrap!` |
| Update workspace `Cargo.toml` to include new crates and set appropriate feature flags. | Build Engineer | `Cargo.toml` |

### Phase 3 – DRY Enforcement
1. Replace each duplicated parsing block with `common::parser::parse_input`.
2. Swap direct `std::env::var` usage for `config::Config::from_env`.
3. Convert `unwrap()` calls to `?` with proper error mapping.
4. Run `cargo test` after each substitution to ensure no regressions.

### Phase 4 – Modular API Surface
1. Add `pub` visibility where needed; keep internal modules private.
2. Write `///` documentation for each public function.
3. Generate docs locally (`cargo doc --open`) and fix warnings.
4. Update `README.md` with usage examples for the new API.

### Phase 5 – Test Consolidation
1. Create a `tests/common` harness that can be imported by integration tests.
2. Move duplicated test cases into the harness.
3. Add property‑based tests for parsing and config loading (use `proptest`).
4. Ensure `cargo test --workspace --doc` passes (doctests).

### Phase 6 – CI / Lint Integration
1. Add the new crates to CI matrix.
2. Update GitHub Actions workflow to run `cargo fmt`, `cargo clippy`, and coverage.
3. Add a badge for `codecov` or `tarpaulin`.

### Phase 7 – Documentation & Release
1. Finalise `IMPRO.md` (this file) with the full plan.
2. Update `CHANGELOG.md` with a *Improvement Opportunities* entry.
3. Create a PR titled `feat: DRY & modularization improvements`.
4. After approval, merge and tag `vX.Y.Z`.
5. Publish release notes via the `release` skill.

---

## 4. Timeline (Estimated)
| Week | Milestones |
|------|------------|
| **1** | Phase 0 & Phase 1 completed, metrics recorded |
| **2** | Phase 2 crates created, CI updated |
| **3** | Phase 3 DRY replacements, compile‑time verification |
| **4** | Phase 4 documentation, API surface stabilised |
| **5** | Phase 5 test suite consolidated, coverage > 90 % |
| **6** | Phase 6 CI passes, badges added |
| **7** | Phase 7 final docs, PR merged, release tag created |

---

## 5. Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking public API | Medium | Keep backward‑compatible wrappers for one release cycle |
| CI timeouts due to added crates | Low | Cache Cargo registry, run tests in parallel (`cargo test --jobs N`) |
| Insufficient test coverage for new modules | High | Enforce coverage gate in CI (e.g., `tarpaulin --fail-under 90`) |
| Merge conflicts across crates | Medium | Use `git worktree` to isolate work per crate, rebase frequently |

---

## 6. Acceptance Criteria
- No duplicate code blocks remain (verified by `git grep` after refactor).
- All crates compile with `cargo build --workspace`.
- Test coverage for new/modified code ≥ 90 %.
- `cargo clippy` reports zero warnings.
- Documentation builds without broken links.
- CI pipeline passes on every push to `main`.

---

*Prepared by the Orchestrator – see `IMPRO.md` for the full implementation roadmap.*