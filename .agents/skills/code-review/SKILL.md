---
name: code-review
description: "Review code changes for correctness, security, performance, and code quality. Use when the user asks to review code, review a diff, review commits, or perform a code review. Supports three modes: (1) Basic review - single-pass analysis, (2) Cross review - use a specific model, (3) Comprehensive review - parallel specialized subagents. Input can be: a text diff, git commit hashes, a git range, or a PR URL."
metadata:
  version: 2.0.0
---

# Code Review

Expert code reviewer combining rigorous analysis with deep expertise in clarity, consistency, and maintainability. Prioritize readable, explicit code over overly compact solutions while ensuring correctness and security.

## Operating Modes

This skill supports three modes:

| Mode | When to Use |
|------|-------------|
| **Basic** | Default. Single-pass review of changes |
| **Cross** | User specifies a model (e.g., "review with opus") |
| **Comprehensive** | User explicitly requests parallel specialized review |

### Mode Detection

Parse the user request to determine the mode:

1. **Cross mode** — If user specifies a model: "review with X", "use X to review", "review using X"
2. **Comprehensive mode** — If user explicitly requests: "comprehensive", "full review", "parallel review"
3. **Basic mode** — All other requests

## Inputs

Accept any combination of:
1. **Text diff** — pasted directly by the user
2. **Git commit hashes** — one or more SHAs; extract the diff with git
3. **GitHub PR URL** — for comprehensive reviews: `https://github.com/<OWNER>/<REPO>/pull/<NUMBER>`
4. **Task description / requirements** — context for what the change is supposed to accomplish

## Review Workflow

### Step 1: Determine Mode and Obtain Diff

- **Cross mode**: Extract the model from user's request, gather changes from conversation history
- **Comprehensive mode**: Use parallel subagents (see Comprehensive section below)
- **Basic mode**: Single-pass review of the diff

If user provides a text diff, use it directly.
If user provides commit hashes, extract with git:
```bash
git diff "<commit>^..<commit>"     # single commit
git diff "<commit1>..<commit2>"  # two commits
git diff "<range>"                # range syntax
```
If neither diff nor commits provided, ask the user for input.

### Step 2: Gather Context

- Read the changed files fully (not just diff hunks) to understand surrounding code
- Search for code that depends on or is affected by the changed code — callers, importers, subclasses
- Keep user's stated requirements in mind — flag deviations where implementation doesn't match intent

### Step 3: Analyze Changes

Review against the checklist below.

#### Priority Levels

| Level | Meaning | Action |
|-------|---------|--------|
| P0 | Critical — security vulnerability, data loss risk, crash | Must fix |
| P1 | Major — significant bug, performance regression, broken feature | Must fix |
| P2 | Minor — code smell, clarity issue, inconsistency | Nice to fix |
| P3 | Suggestion — improvement idea, optional refactor | Optional |

#### Critical Issues (P0–P1)

**Correctness:**
- Logic errors and off-by-one mistakes
- Unhandled edge cases (null, empty, boundary values)
- Broken control flow (early returns, missing breaks)
- Incorrect type conversions or comparisons
- State mutation side effects

**Security:**
- Injection vulnerabilities (SQL, command, XSS)
- Exposed secrets, tokens, or credentials
- Unsafe deserialization
- Missing input validation at system boundaries
- Improper access control or authorization checks

**Performance:**
- Inefficient algorithms (quadratic where linear is possible)
- N+1 queries or unbounded database calls
- Memory leaks or unbounded growth
- Missing pagination on large datasets
- Blocking operations in async contexts

**Data Integrity:**
- Race conditions in concurrent code
- Missing transactions for multi-step writes
- Data loss on error paths
- Inconsistent state after partial failures

#### Code Quality (P2–P3)

**Clarity:**
- Unnecessary complexity or deep nesting
- Poor naming (vague, misleading, or inconsistent)
- Confusing logic flow or convoluted conditionals
- Nested ternary operators (prefer switch/if-else)
- Magic numbers or unexplained constants

**Consistency:**
- Violations of project conventions
- Inconsistent naming conventions
- Mixed patterns for the same concern
- Import style inconsistencies

**Maintainability:**
- Missing abstractions for duplicated logic
- Tight coupling between unrelated modules
- Over-engineering simple problems
- Dead code or unreachable branches

**Simplification:**
- Redundant null checks or type guards
- Overly verbose constructs with simpler alternatives
- Unnecessary intermediate variables
- Code that reimplements standard library functions

### Step 4: Produce the Review

Output this format:

```
## Code Review

**Verdict**: [APPROVE | REQUEST CHANGES | NEEDS DISCUSSION]
**Confidence**: [HIGH | MEDIUM | LOW]
**Mode**: [Basic | Cross | Comprehensive]
**Model** (Cross mode only): <model-id used>

### Summary
[1-2 sentences: what the change does and overall assessment]

### Findings

| Priority | Issue | Location |
|----------|-------|----------|
| P0 | Description | file:line |
| P1 | Description | file:line |
| P2 | Description | file:line |

### Details

#### [P0/P1] Issue title
**File:** `path/to/file.ext:line`

Description of the issue and why it matters.

**Suggested fix:**
```
code suggestion
```

### Recommendation
[Concise actionable recommendation for the author]
```

**Rules:**
- Use `APPROVE` only when there are no P0 or P1 findings.
- Use `REQUEST CHANGES` when P0 or P1 findings exist.
- Use `NEEDS DISCUSSION` when findings are ambiguous or require author's context.
- Include detailed write-ups with suggested fixes for every P0 and P1 finding.
- P2/P3 findings go in the table; add detail sections only when a code suggestion adds clarity.
- Keep it concise — don't pad with praise or filler.

## Cross Review Mode

When user explicitly specifies a model:

### Step 1: Parse the Request

Extract from the user's prompt:
- **Model**: The model ID to use for the subagent
- **Review instructions**: Any additional instructions after "Review instructions:"
- **Change scope**: What should be reviewed

### Step 2: Gather Context

You already know what changed — reconstruct from your conversation history:
1. Compose a unified diff summary of all changes made (by Edit, Write, Bash tools)
2. Read the final state of changed files
3. Check for related context (tests, config, type definitions)

If no changes were made, inform the user and stop.

### Step 3: Spawn Review Subagent

Use spawn_subagent with:
- **skill**: "code-review" (this skill, for basic mode)
- **model**: The model specified by user
- **prompt**:
```
Review the changes below using "code-review" skill.

## Review Instructions
{user's instructions verbatim}

## Changes
{reconstructed diff or before/after summary, grouped by file}

## Additional Context
{links to related files, tests, type definitions}
```

### Step 4: Relay Results

Relay the subagent's review output exactly as received. Do NOT summarize, rephrase, or add commentary. Do NOT act on findings — let the user decide.

## Comprehensive Review Mode

When user explicitly requests comprehensive review.

### Workflow

1. **Determine review mode**:
   - PR mode: GitHub PR URL provided
   - Local mode: No PR URL, uses local branch diff

2. **Fetch diff** (use subagent):
   - Read `<SKILL_DIRECTORY>/../comprehensive-review/fetch-diff.md` for full instructions
   - Return: diff file path, line count, title, description, complexity

3. **Run specialized reviews** based on complexity:

   | Complexity | Strategy |
   |------------|----------|
   | Simple | Self-review against all criteria |
   | Medium | 6 parallel subagents (1 per criterion) |
   | Hard | 12 parallel subagents (2 per criterion, different models) |

   **Criteria** (each needs instruction file):
   - architecture
   - security
   - performance
   - code-quality
   - requirements-compliance
   - bugs

4. **Merge, filter, prioritize**:
   - Deduplicate findings by file/line
   - Filter false positives
   - Assign priorities (P0–P3)
   - Cross-criteria signals get higher priority

5. **Output format**:
```
## Comprehensive Code Review

| # | Priority | Issue | File:Line | Review type |
|---|----------|-------|-----------|------------|
| 1 | P0 | Description | file:line | architecture(model-a), security(model-b) |

### Details
[P0 findings with suggested fixes]
```

**Note**: Comprehensive mode requires instruction files from `comprehensive-review/` subdirectory. If not available, fall back to basic mode.
