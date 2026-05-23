---
name: memory-md-management
description: "Use when explicitly asked to check, audit, update, improve, fix, or maintain AGENTS.md / memory.md files, when assessing documentation quality, when starting work on a new codebase and needing to understand existing documentation, or when a user presses # during a session to incorporate learnings into a memory file - provides 5-phase workflow (discovery, quality assessment, reporting, targeted updates, application) with scoring across 6 dimensions to ensure project memory files are current, actionable, and concise."
---

# Memory.md Management

## Overview

Provides comprehensive project memory file management including discovery, quality assessment, scoring, reporting, and targeted improvements. Ensures the coding agent has optimal project context by maintaining high-quality documentation files such as AGENTS.md or MEMORY.md.

Memory files are the primary mechanism for providing project-specific context to coding agent sessions. This skill manages their complete lifecycle across 5 phases, evaluating against 6 quality dimensions to produce scores (0-100), letter grades (A-F), and specific improvement recommendations.

## When to Use

Use this skill when:

- User explicitly asks to "check", "audit", "update", "improve", "fix", or "maintain" AGENTS.md
- User mentions "AGENTS.md quality", "documentation review", or "project memory optimization"
- A project memory file needs to be created from scratch for a new project
- User asks about improving agent understanding of the codebase
- Documentation has become stale or outdated
- Starting work on a new codebase and need to understand existing documentation
- User presses **#** during a session to incorporate learnings into a memory file

**Trigger phrases:** "audit AGENTS.md", "check documentation quality", "improve project context", "review AGENTS.md", "validate documentation", "update memory", "fix AGENTS.md"

## Core Workflow

### Phase 1: Discovery

Find all memory files in the repository:

```bash
find . -name "AGENTS.md" -o -name ".agents.md" -o -name ".agents.local.md" -o -name "MEMORY.md" -o -name "memory.md" 2>/dev/null | head -50
```

**File Types & Locations:**

| Type | Location | Purpose |
|------|----------|---------|
| Project root | `./AGENTS.md` | Primary project context (checked into git, shared) |
| Local overrides | `./.agents.local.md` | Personal/local settings (gitignored) |
| Global defaults | `~/.agents/AGENTS.md` | User-wide defaults across all projects |
| Memory | `./MEMORY.md` or `./memory.md` | Alternative naming for project memory |
| Subdirectory | Any nested location | Feature/domain-specific context |

### Phase 2: Quality Assessment

For each file found, evaluate against **6 criteria**. Consult `references/quality-criteria.md` for the detailed rubric.

| Criterion | Weight | What to Check |
|-----------|--------|---------------|
| Commands/workflows | 20 pts | Are build/test/deploy commands present and working? |
| Architecture clarity | 20 pts | Can the agent understand the codebase structure? |
| Non-obvious patterns | 15 pts | Are gotchas and quirks documented? |
| Conciseness | 15 pts | Is content dense without filler? |
| Currency | 15 pts | Does it reflect current codebase state? |
| Actionability | 15 pts | Are instructions executable and copy-paste ready? |

**Quality Scores:** A (90-100), B (70-89), C (50-69), D (30-49), F (0-29)

### Phase 3: Quality Report Output

**CRITICAL: Always output the quality report BEFORE making any updates.**

```markdown
## Memory.md Quality Report

### Summary
- Files found: X
- Average score: X/100
- Files needing update: X

### File-by-File Assessment

#### 1. ./AGENTS.md (Project Root)
**Score: XX/100 (Grade: X)**

| Criterion | Score | Notes |
|-----------|-------|-------|
| Commands/workflows | X/20 | ... |
| Architecture clarity | X/20 | ... |
| Non-obvious patterns | X/15 | ... |
| Conciseness | X/15 | ... |
| Currency | X/15 | ... |
| Actionability | X/15 | ... |

**Issues:**
- [List specific problems]

**Recommended additions:**
- [List what should be added]
```

### Phase 4: Targeted Updates

After report output, ask user for confirmation before updating. Consult `references/update-guidelines.md` for detailed principles.

**Key principles:**
- Propose targeted additions only — commands, gotchas, patterns, configs
- Keep it minimal — avoid obvious info, generic advice, one-off fixes
- Show diffs with explanation

**Diff format:**

```markdown
### Update: ./AGENTS.md
**Why:** [Brief explanation of why this helps]

```diff
+ ## Quick Start
+
+ ```bash
+ npm install
+ npm run dev
+ ```
```
```

### Phase 5: Apply Updates

After user approval, use the Edit tool to apply changes. Preserve existing content structure. Do not reformat or reorganize unless explicitly requested.

## Examples

### Example 1: Quality Assessment

**User:** "Check if my AGENTS.md is good"

**Response:**

```markdown
## AGENTS.md Quality Report

### Summary
- Files found: 1
- Average score: 65/100
- Files needing update: 1

### File-by-File Assessment

#### 1. ./AGENTS.md
**Score: 65/100 (Grade: C)**

| Criterion | Score | Notes |
|-----------|-------|-------|
| Commands/workflows | 10/20 | Only npm install documented |
| Architecture clarity | 15/20 | Basic tree, missing module relationships |
| Non-obvious patterns | 5/15 | No gotchas documented |
| Conciseness | 15/15 | Concise and dense |
| Currency | 10/15 | Some outdated paths |
| Actionability | 10/15 | Some vague instructions |

**Issues:**
- Missing test command
- Missing build command
- No documentation of auth flow complexity
- References to old src/structure

**Recommended additions:**
- Add Commands section with test/build
- Document auth module gotcha
- Update architecture paths
```

### Example 2: Creating from Scratch

**User:** "Create a AGENTS.md for this React project"

**Response:**

```markdown
I'll create a AGENTS.md for your React project. Let me first analyze the codebase structure.

[Scans package.json, src/, and key config files]

## AGENTS.md Quality Report

### Summary
- Files found: 0
- Creating new AGENTS.md at project root

### Proposed Content

# Project Name

## Commands
| Command | Description |
|---------|-------------|
| `npm install` | Install dependencies |
| `npm run dev` | Start dev server (Vite on port 5173) |
| `npm run build` | Production build |
| `npm run test` | Run Vitest tests |
| `npm run lint` | ESLint check |

## Architecture
src/
├── components/   # Reusable UI components
├── pages/        # Route pages
├── hooks/        # Custom React hooks
├── lib/          # Utilities and API clients
└── main.tsx      # Entry point

## Gotchas
- Vite requires files in `public/` for static assets
- State updates in `useEffect` need proper dependency arrays

Shall I create this file?
```

### Example 3: Update Proposal

**User:** "Update my AGENTS.md with what you learned"

**Response:**

```markdown
### Update: ./AGENTS.md
**Why:** Discovered that tests require database setup which wasn't documented.

```diff
 ## Testing

+ **Prerequisites:** Start test database first with `docker-compose up -d test-db`
+ `npm run test` - Run all tests
+ `npm run test:unit` - Unit tests only (no DB required)

 ## Environment
 - `API_BASE_URL` - API endpoint (default: localhost:3000)
+ `API_BASE_URL` - API endpoint (required for production, defaults to http://localhost:3000 in dev)
```

Apply these changes?
```

## Best Practices

1. **Read references before acting**: Quality criteria and update guidelines contain essential rubric details
2. **Report before editing**: Never modify memory files without first presenting a quality report
3. **Preserve existing structure**: When updating, maintain the existing organization and style
4. **Be project-specific**: Only add information specific to this codebase, not generic advice
5. **Verify commands work**: Before suggesting commands, verify they execute correctly
6. **Use progressive disclosure**: Keep SKILL.md lean, reference detailed rubrics in separate files
7. **Score consistently**: Apply the same scoring standards across all files

## Constraints

1. **Never modify without approval**: Always get user confirmation before editing memory files
2. **Don't remove content without asking**: If suggesting deletions, explicitly mark them and get approval
3. **Respect `.agents.local.md`**: These are personal settings; never suggest modifying them in shared docs
4. **Avoid generic advice**: Do not add "write good code" type content
5. **Keep diffs concise**: Show only the actual changes, not entire file contents
6. **Verify file paths**: Ensure all referenced files exist before documenting them
7. **Score objectively**: Use the rubric consistently; don't inflate scores

## References

- `references/quality-criteria.md` — Detailed scoring rubric for each of the 6 dimensions
- `references/update-guidelines.md` — Principles and patterns for proposing updates
