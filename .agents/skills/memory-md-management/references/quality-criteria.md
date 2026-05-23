# Quality Criteria Rubric

Each memory file is scored across 6 dimensions (100 points total). This document defines the detailed criteria for each dimension, including scoring guidelines, common deductions, and examples.

---

## 1. Commands & Workflows (20 pts)

Evaluates whether build, test, run, deploy, and common workflow commands are documented accurately.

### Score Breakdown

| Score | Condition |
|-------|-----------|
| 18-20 | All essential commands documented (build, test, dev/run, lint/format, deploy if applicable). Commands are verified working. Copy-paste ready. |
| 12-17 | Most commands present, but missing one essential command or some commands are incomplete (e.g., missing flags). |
| 6-11 | Only basic commands (e.g., `npm install`) present. No test, build, or deploy commands. |
| 0-5 | No commands documented, or commands are clearly wrong/stale. |

### What to Check

- **Build command** — Does it compile/package the project? (e.g., `cargo build`, `npm run build`, `make`)
- **Test command** — How to run tests? Single test? All tests? (e.g., `cargo test`, `npm test`)
- **Dev command** — How to start a development server or watch mode?
- **Lint/format** — How to check code quality? (e.g., `cargo clippy`, `npm run lint`)
- **Deploy** — How to deploy? (if applicable to the project)
- **Database** — Any migration, seed, or reset commands?
- Are commands copy-paste ready? (no placeholders user must guess)

### Common Deductions

| Issue | Deduction |
|-------|-----------|
| Missing test command | -5 pts |
| Missing build command | -4 pts |
| Missing dev/run command | -3 pts |
| Commands use placeholders (e.g., `<your-key>`) without explanation | -2 pts |
| Commands reference wrong package manager | -3 pts |
| Commands don't work (if verifiable) | -10 pts |

---

## 2. Architecture Clarity (20 pts)

Evaluates whether the agent can understand the codebase structure, module relationships, and key architectural decisions.

### Score Breakdown

| Score | Condition |
|-------|-----------|
| 18-20 | Clear directory tree or module map with descriptions. Key architectural decisions documented (why this framework? why this pattern?). Entry points identified. |
| 12-17 | Directory tree present but module responsibilities vague. Some architectural context but gaps remain. |
| 6-11 | High-level description only. No directory tree. Agent would need to explore to understand structure. |
| 0-5 | No architecture documentation at all. |

### What to Check

- **Directory tree** — Is there a visual map of src/ or equivalent?
- **Module responsibilities** — Does each directory/module have a one-line purpose?
- **Entry points** — Where does execution start? (e.g., `src/main.rs`, `src/index.ts`, `app.py`)
- **Key technologies** — Framework, database, major libraries documented?
- **Architectural patterns** — Is there MVC, layered architecture, event-driven, etc.? Is it documented?
- **Data flow** — How does data move through the system? (useful for complex apps)

### Common Deductions

| Issue | Deduction |
|-------|-----------|
| No directory tree | -5 pts |
| Missing module responsibilities | -3 pts |
| No entry point identified | -3 pts |
| Missing framework/library stack | -2 pts |
| Tree is flat (just files, no grouping) | -1 pt |

---

## 3. Non-Obvious Patterns (15 pts)

Evaluates whether gotchas, quirks, unusual patterns, and hard-won knowledge are documented.

### Score Breakdown

| Score | Condition |
|-------|-----------|
| 13-15 | Multiple gotchas documented. Known pitfalls and their workarounds present. Decision rationale for non-obvious choices included. |
| 9-12 | Some gotchas documented but gaps remain. A new developer would hit at least one known issue. |
| 4-8 | Only obvious or well-known patterns. Missing project-specific quirks. |
| 0-3 | No gotchas, quirks, or non-obvious information documented. |

### What to Check

- **Gotchas** — Is there something unusual about how this project works that would surprise someone?
- **Known issues** — Workarounds for known bugs or limitations
- **Decision rationale** — Why was a particular library/approach chosen over alternatives?
- **Configuration quirks** — Unusual settings, environment variables, or runtime requirements
- **Testing quirks** — Flaky tests, required test fixtures, special setup
- **Edge cases** — Common edge cases developers should handle

### Examples

| Good Gotcha | Bad Gotcha |
|-------------|------------|
| "Auth tokens expire every 15 min in dev; run `dev-refresh` to re-authenticate" | "Be careful with authentication" |
| "CSS modules don't support @apply; use compose() instead" | "Follow CSS best practices" |

### Common Deductions

| Issue | Deduction |
|-------|-----------|
| No gotchas at all | -10 pts |
| Gotchas are generic/obvious | -5 pts |
| Gotchas lack remediation steps | -3 pts |

---

## 4. Conciseness (15 pts)

Evaluates whether the content is dense, useful, and free of filler.

### Score Breakdown

| Score | Condition |
|-------|-----------|
| 13-15 | Every line carries signal. No filler, no generic advice, no obvious information. Dense and readable. |
| 9-12 | Mostly dense but has some filler or generic statements. Could be tightened. |
| 4-8 | Significant filler. Long paragraphs where bullets would suffice. Generic advice mixed in. |
| 0-3 | Mostly filler. Very little project-specific content. Difficult to extract useful information. |

### What to Check

- **Filler** — Is there content that doesn't help the agent? (e.g., "Write good code", "Follow best practices")
- **Density** — Is each line specific and informative?
- **Formatting** — Are lists/ tables used where appropriate instead of prose?
- **Redundancy** — Is the same information repeated in multiple sections?

### Common Deductions

| Issue | Deduction |
|-------|-----------|
| Generic advice present (e.g., "Write clean code") | -3 pts per instance |
| Long paragraphs where 3 bullets would work | -2 pts |
| Obvious information documented (e.g., "This project uses JavaScript") | -2 pts |
| >50% of content is generic/not project-specific | -8 pts |

---

## 5. Currency (15 pts)

Evaluates whether the documentation reflects the current state of the codebase.

### Score Breakdown

| Score | Condition |
|-------|-----------|
| 13-15 | All paths, commands, and references are current. No mention of removed files or old patterns. |
| 9-12 | Mostly current but 1-2 minor outdated references (e.g., old file path, deprecated command). |
| 4-8 | Multiple outdated sections. References to deleted files, old architectures, or removed features. |
| 0-3 | Completely stale. References code that no longer exists. Commands that fail. |

### What to Check

- **File paths** — Do referenced files/directories still exist?
- **Commands** — Do they still work with current tooling?
- **Dependencies** — Are mentioned libraries still in use?
- **Architecture** — Does the described structure match reality?
- **Version references** — Are version numbers current?

### Common Deductions

| Issue | Deduction |
|-------|-----------|
| Referenced file/dir no longer exists | -3 pts per instance |
| Command no longer works | -3 pts per instance |
| Outdated architecture description | -5 pts |
| References to removed features | -4 pts per instance |

---

## 6. Actionability (15 pts)

Evaluates whether the instructions are executable and useful without additional clarification.

### Score Breakdown

| Score | Condition |
|-------|-----------|
| 13-15 | Every instruction is concrete, specific, and immediately actionable. Copy-paste code snippets, commands, and configs. |
| 9-12 | Mostly actionable but some vague instructions remain. Some context needed to use. |
| 4-8 | Several vague or ambiguous instructions. Agent would need to ask clarifying questions. |
| 0-3 | Instructions are too vague to follow. Heavy interpretation required. |

### What to Check

- **Specificity** — Are instructions concrete? ("Run `npm run test:integration`" vs "Run integration tests")
- **Copy-paste ready** — Can the user (or agent) copy commands directly?
- **Context included** — Are prerequisites documented? (e.g., "Requires Docker running")
- **Examples** — Do complex operations include examples?
- **Error handling** — Are common failure modes documented with solutions?

### Common Deductions

| Issue | Deduction |
|-------|-----------|
| Vague instructions ("Run the tests") without specifics | -3 pts |
| Missing prerequisites | -3 pts |
| No examples for complex operations | -2 pts |
| Instructions assume knowledge without stating it | -2 pts |

---

## Scoring Summary Table

| Grade | Score | Meaning |
|-------|-------|---------|
| A | 90-100 | Excellent. Agent can work effectively with minimal exploration. |
| B | 70-89 | Good. Some gaps but agent can function with minor exploration. |
| C | 50-69 | Acceptable. Agent will need moderate exploration to fill gaps. |
| D | 30-49 | Poor. Agent will struggle. Significant updates needed. |
| F | 0-29 | Failing. Agent cannot rely on this documentation at all. |

## Automated Scoring Aid

For each dimension, follow this quick triage:

1. **Does it exist?** (0 pts if not)
2. **Is it minimally present?** (50% of weight)
3. **Is it good?** (80% of weight)
4. **Is it excellent?** (90-100% of weight)

Calibrate by asking: *"If I were a new agent dropped into this codebase, would this documentation help me be productive immediately?"*
