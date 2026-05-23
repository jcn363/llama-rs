# Update Guidelines

Principles and patterns for proposing and applying updates to memory files.

---

## Core Principles

### 1. Propose Targeted Additions Only

Do not rewrite or reformat entire files. Propose discrete, targeted additions:

- ✅ Add a missing command section
- ✅ Add a single gotcha
- ✅ Update an outdated path
- ❌ Reorganize all sections
- ❌ Rewrite in a different style
- ❌ Add content unrelated to what the user asked about

### 2. Keep It Minimal

Every addition must earn its place. Before proposing, ask:

- **Is this specific to this codebase?** — If not, don't add it
- **Is this non-obvious?** — If everyone already knows, don't add it
- **Is this stable?** — If it changes weekly, don't add it (or mark as volatile)
- **Will the agent use this?** — If it only helps humans, reconsider

### 3. Show Diffs, Not Full Files

Always show changes as diffs with explanations:

```markdown
### Update: ./AGENTS.md
**Why:** [1-2 sentence explanation of why this helps]

```diff
+ ## Commands
+
+ | Command | Description |
+ |---------|-------------|
+ | `cargo test` | Run all tests |
+ | `cargo test -- --nocapture` | Run tests with stdout visible |
+ | `cargo clippy` | Lint check |
```
```

### 4. Preserve Existing Structure

When applying updates:

- Insert new sections in the logical location (not always at the end)
- Use the same heading style, formatting, and conventions already in the file
- Don't reformat existing content unless it contains errors
- Don't change the order of existing sections

### 5. Validate Before Proposing

Before suggesting content:

- Verify file paths by checking they exist
- Verify commands by checking they work
- Verify architecture descriptions match actual code
- Verify version numbers and dependency names

---

## Common Update Patterns

### Pattern A: Adding Missing Commands

```markdown
**Why:** No test or build commands were documented.

```diff
+
+ ## Commands
+
+ | Command | Description |
+ |---------|-------------|
+ | `npm run dev` | Start dev server (port 5173) |
+ | `npm run build` | Production build to dist/ |
+ | `npm test` | Run Vitest test suite |
+ | `npm run lint` | ESLint check |
```
```

### Pattern B: Adding a Gotcha

```markdown
**Why:** Discovered that the auth token expires every 15 minutes in dev, causing flaky API tests.

```diff
+ ## Gotchas
+
+ ### Auth Token Expiry
+ Auth tokens expire every 15 minutes in dev mode. If API tests start failing with
+ 401 errors, run `npm run dev:refresh` to re-authenticate.
```
```

### Pattern C: Updating an Outdated Path

```markdown
**Why:** The config module moved from `src/config/` to `src/settings/`.

```diff
- `src/config/` - Application configuration
+ `src/settings/` - Application configuration
```
```

### Pattern D: Adding Architecture Context

```markdown
**Why:** The database layer uses Prisma with a connection pooling pattern that isn't obvious.

```diff
+ ### Database
+
+ Prisma ORM with PgBouncer connection pooling.
+ **Important:** All raw SQL queries must use the pooler-compatible connection string
+ from `DATABASE_POOL_URL` (not `DATABASE_URL`).
```
```

---

## What NOT to Add

| ❌ Don't Add | Why |
|-------------|-----|
| "Write clean code" | Generic advice, zero signal |
| "Follow best practices" | Vague, not actionable |
| "This project uses React" | Obvious from package.json |
| "Be careful with state management" | No specific insight |
| "Test your code" | Assumed knowledge |
| Full-file rewrites | Destroys existing context and user trust |
| Duplicate content already in the file | Redundant, wastes attention |

---

## Approval Flow

```
Propose Update (diff + explanation)
        ↓
User Reviews
        ↓
User Approves ──→ Apply changes with Edit tool
        ↓
User Rejects ───→ Move on, no changes made
        ↓
User Requests Changes ──→ Revise proposal and re-present
```

Never bypass this flow. Memory files are the agent's primary source of project context — changes must be deliberate and reviewed.

---

## When to Skip Update Patterns

Some situations don't warrant updates:

- **One-off issue**: A single bug doesn't warrant a permanent memory entry — unless it's likely to recur
- **Obvious from code**: If the code itself is self-documenting, don't duplicate in memory
- **Ephemeral state**: Current task progress, temporary workarounds, debug settings
- **Personal preferences**: Code style preferences that aren't project conventions
