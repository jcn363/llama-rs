---
name: code-review-jcn
description: "Review code changes for correctness, security, performance, and code quality. Use when the user asks to review code, review a diff, review commits, or perform a code review. Supports three modes: (1) Basic review - single-pass analysis, (2) Cross review - use a specific model, (3) Comprehensive review - parallel specialized subagents. Covers structured 8-step methodology, SOLID principles, security audit, prioritized checklists, anti-pattern detection, and vulnerability corrections."
metadata:
  version: 2.0.0
---

# Code Review

Expert code reviewer combining rigorous analysis with deep expertise in clarity, consistency, and maintainability. Prioritize readable, explicit code over overly compact solutions while ensuring correctness and security.

## When to Use

- Reviewing pull requests
- Checking code quality
- Providing feedback on implementations
- Identifying potential bugs
- Suggesting improvements
- Security audits
- Performance analysis

## Operating Modes

| Mode | When to Use |
|------|-------------|
| **Basic** | Default. Single-pass review of changes |
| **Cross** | User specifies a model (e.g., "review with opus") |
| **Comprehensive** | User explicitly requests parallel specialized review |

### Mode Detection

1. **Cross mode** — User specifies a model: "review with X", "use X to review", "review using X"
2. **Comprehensive mode** — User explicitly requests: "comprehensive", "full review", "parallel review"
3. **Basic mode** — All other requests

## Inputs

Accept any combination of:
1. **Text diff** — pasted directly by the user
2. **Git commit hashes** — one or more SHAs; extract the diff with git
3. **GitHub PR URL** — for comprehensive reviews: `https://github.com/<OWNER>/<REPO>/pull/<NUMBER>`
4. **Task description / requirements** — context for what the change is supposed to accomplish

## Review Workflow

### Step 1: Determine Mode and Obtain Diff

- **Basic mode**: Single-pass review of the diff
- **Cross mode**: Extract the model from user's request, gather changes from conversation history
- **Comprehensive mode**: Use parallel subagents (see Comprehensive section below)

If user provides a text diff, use it directly.
If user provides commit hashes, extract with git:
```bash
git diff "<commit>^..<commit>"     # single commit
git diff "<commit1>..<commit2>"    # two commits
git diff "<range>"                 # range syntax
```
If neither diff nor commits provided, ask the user for input.

### Step 2: Gather Context

- Read the PR description — what is the goal? Which issues does it address?
- Read the changed files fully (not just diff hunks) to understand surrounding code
- Search for code that depends on or is affected by the changed code — callers, importers, subclasses
- Keep user's stated requirements in mind — flag deviations where implementation doesn't match intent
- Check scope: how many files changed? What type of changes? (feature, bugfix, refactor) Are tests included?

### Step 3: High-Level Review

**Architecture and design:**
- Does the approach make sense?
- Is it consistent with existing patterns?
- Are there simpler alternatives?
- Is the code in the right place?

**Code organization:**
- Clear separation of concerns?
- Appropriate abstraction levels?
- Logical file/folder structure?

### Step 4: Detailed Code Analysis

**Naming:**
- Variables: descriptive, meaningful names
- Functions: verb-based, clear purpose
- Classes: noun-based, single responsibility
- Constants: `UPPER_CASE` for true constants
- Avoid abbreviations unless widely known

**Functions:**
- Single responsibility
- Reasonable length (<50 lines ideally)
- Clear inputs and outputs
- Minimal side effects
- Proper error handling

**SOLID:**
- Single Responsibility — one reason to change
- Open/Closed — open for extension, closed for modification
- Liskov Substitution — subtypes must be substitutable for base types
- Interface Segregation — small, focused interfaces
- Dependency Inversion — depend on abstractions, not concretions

**Error handling:**
- All errors caught and handled
- Meaningful error messages
- Proper logging
- No silent failures
- User-friendly errors for UI

**Code quality:**
- No code duplication (DRY)
- No dead code
- No commented-out code
- No magic numbers
- Consistent formatting

### Step 5: Security Review

- Injection vulnerabilities (SQL, command, XSS) — use parameterized queries, no raw `innerHTML`
- Exposed secrets, tokens, or credentials — no hardcoded keys, use environment variables
- Unsafe deserialization
- Missing input validation at system boundaries — type, range, and format checks
- Improper access control or authorization checks
- Authentication: proper checks, session management, password hashing/salting
- No hardcoded secrets
- CSRF protection
- Dependencies: no vulnerable packages, up-to-date, minimal usage

### Step 6: Performance Review

- Inefficient algorithms (quadratic where linear is possible)
- N+1 queries or unbounded database calls
- Memory leaks or unbounded growth
- Missing pagination on large datasets
- Blocking operations in async contexts
- Proper database indexing
- Connection pooling
- Appropriate caching strategy with invalidation
- Resource management (files closed, connections released)

### Step 7: Testing & Documentation Review

**Testing:**
- Unit tests for new code, integration tests if needed
- Edge cases and error cases covered
- Tests are readable, maintainable, deterministic
- No test interdependencies
- Proper test data setup/teardown
- Descriptive test names: `test_user_creation_with_valid_data_succeeds` not `test1`

**Documentation:**
- Complex logic has comments explaining *why*
- No obvious comments
- TODOs have linked tickets
- Docstrings follow Args/Returns/Raises format
- README, API docs, migration guide updated

### Step 8: Provide Constructive Feedback

**Be constructive:**
```
✅ Good: "Consider extracting this logic into a separate function for better
testability and reusability."

❌ Bad: "This is wrong. Rewrite it."
```

**Be specific — reference exact line numbers and symbols:**
```
✅ Good: "On line 45, this query could cause an N+1 problem. Consider using
.select_related('author') to fetch related objects in a single query."

❌ Bad: "Performance issues here."
```

**Prioritize issues:**
| Level | Meaning | Action |
|-------|---------|--------|
| P0 | Critical — security vulnerability, data loss risk, crash | Must fix |
| P1 | Major — significant bug, performance regression, broken feature | Must fix |
| P2 | Minor — code smell, clarity issue, inconsistency | Nice to fix |
| P3 | Suggestion — improvement idea, optional refactor | Optional |

**Acknowledge good work:**
> "Nice use of the strategy pattern here! This makes it easy to add new payment methods in the future."

### Produce the Review

Output this format:

```
## Code Review

**Verdict**: [APPROVE | REQUEST CHANGES | NEEDS DISCUSSION]
**Confidence**: [HIGH | MEDIUM | LOW]
**Mode**: [Basic | Cross | Comprehensive]

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
- Use `APPROVE` only when there are no P0 or P1 findings
- Use `REQUEST CHANGES` when P0 or P1 findings exist
- Use `NEEDS DISCUSSION` when findings are ambiguous or require author's context
- Include detailed write-ups with suggested fixes for every P0 and P1 finding
- P2/P3 findings go in the table; add detail sections only when a code suggestion adds clarity
- Keep it concise — don't pad with praise or filler

## Review Checklist

### Functionality
- [ ] Code does what it's supposed to do
- [ ] Edge cases handled (null, empty, boundary values)
- [ ] Error cases handled
- [ ] No obvious bugs or logic errors
- [ ] State mutation side effects considered

### Code Quality
- [ ] Clear, descriptive naming
- [ ] Functions are small and focused (SRP)
- [ ] No code duplication (DRY)
- [ ] Consistent with codebase style and conventions
- [ ] No code smells (god class, deep nesting, magic numbers)
- [ ] No dead code or commented-out code

### Security
- [ ] Input validation on all user-facing endpoints
- [ ] No hardcoded secrets (use env vars)
- [ ] Authentication/authorization enforced
- [ ] Parameterized queries (no SQL injection)
- [ ] XSS prevention (no raw innerHTML)
- [ ] CSRF protection on state-changing endpoints

### Performance
- [ ] No obvious bottlenecks
- [ ] Efficient algorithms (not quadratic where linear works)
- [ ] Proper database queries (no N+1, indexed FKs)
- [ ] Resource management (files closed, connections released)
- [ ] No blocking ops in async contexts

### Testing
- [ ] Tests included
- [ ] Good test coverage (edge cases, errors)
- [ ] Tests are maintainable and deterministic
- [ ] No test interdependencies

### Documentation
- [ ] Code is self-documenting (good names, clear flow)
- [ ] Comments where complexity demands it
- [ ] Docs updated (README, API, migration guide)

## Common Anti-Patterns

**God class:**
```python
# ❌ Bad: One class does everything
class UserManager:
    def create_user(self): pass
    def send_email(self): pass
    def process_payment(self): pass
    def generate_report(self): pass
```

**Magic numbers:**
```python
# ❌ Bad
if user.age > 18: pass

# ✅ Good
MINIMUM_AGE = 18
if user.age > MINIMUM_AGE: pass
```

**Deep nesting / missing early returns:**
```python
# ❌ Bad
if condition1:
    if condition2:
        if condition3:
            if condition4:
                ...  # deeply nested

# ✅ Good
if not condition1: return
if not condition2: return
if not condition3: return
if not condition4: return
```

**Nested ternaries:**
```javascript
// ❌ Bad
const result = a ? b : c ? d : e ? f : g;

// ✅ Good — use if/else or switch
```

## Security Vulnerabilities

**SQL Injection:**
```python
# ❌ Bad
query = f"SELECT * FROM users WHERE id = {user_id}"

# ✅ Good
query = "SELECT * FROM users WHERE id = %s"
cursor.execute(query, (user_id,))
```

**XSS:**
```javascript
// ❌ Bad
element.innerHTML = userInput;

// ✅ Good
element.textContent = userInput;
```

**Hardcoded secrets:**
```python
# ❌ Bad
API_KEY = "sk-1234567890abcdef"

# ✅ Good
API_KEY = os.environ.get("API_KEY")
```

## Cross Review Mode

When user explicitly specifies a model (e.g., "review with opus").

### Workflow

1. **Parse the request**: Extract model ID and any additional review instructions
2. **Gather context**: Compose a unified diff summary of all changes made; read final state of changed files; check for related context (tests, config, types)
3. **Spawn review subagent** with `skill: "code-review"`, the specified model, and a prompt containing the review instructions plus reconstructed diff
4. **Relay results**: Return the subagent's output exactly as received. Do NOT summarize, rephrase, or add commentary. Do NOT act on findings — let the user decide.

## Comprehensive Review Mode

When user explicitly requests comprehensive/full/parallel review.

### Workflow

1. **Determine review variant**:
   - PR mode: GitHub PR URL provided
   - Local mode: No PR URL, uses local branch diff

2. **Fetch diff** using a subagent; return diff file path, line count, title, description, complexity

3. **Run specialized reviews** based on complexity:
   - **Simple** (≤small change): Self-review against all criteria
   - **Medium**: 6 parallel subagents (1 per criterion)
   - **Hard**: 12 parallel subagents (2 per criterion, different models)

   **Criteria**: architecture, security, performance, code-quality, requirements-compliance, bugs

4. **Merge, filter, prioritize**:
   - Deduplicate findings by file/line
   - Filter false positives
   - Assign priorities (P0–P3)
   - Findings flagged by multiple criteria get higher priority

5. **Output format**:
```
## Comprehensive Code Review

| # | Priority | Issue | File:Line | Review type |
|---|----------|-------|-----------|-------------|
| 1 | P0 | Description | file:line | architecture, security |

### Details
[P0 findings with suggested fixes]
```

## Constraints

### Required (MUST)
- **Understand before commenting**: Read the full PR context before leaving feedback
- **Be specific**: Reference exact line numbers and symbols
- **Prioritize**: Tag issues as P0–P3 with clear reasoning
- **Explain why**: Don't just say what's wrong — explain the reasoning and suggest a fix
- **Write tests first for refactoring**: When reviewing refactors, ensure tests exist pre-change

### Prohibited (MUST NOT)
- Personal criticism — review the code, not the person
- Vague feedback like "fix this" without explanation
- Nitpicking style without automated tools handling formatting
- Approving without understanding the change
- Premature optimization suggestions without profiling data

## Best Practices

- **Review promptly**: Don't make authors wait
- **Be respectful**: Focus on code, not the person
- **Suggest alternatives**: Show better approaches with code examples
- **Pick your battles**: Focus on P0 and P1 issues
- **Acknowledge good work**: Positive feedback matters
- **Review your own code first**: Catch obvious issues before requesting review
- **Use automated tools**: Let linters handle style — focus on logic and design
- **Be consistent**: Apply the same standards to all code
- **Read beyond the diff**: Load full files for context and check callers/dependents

## Tools

| Category | Tools |
|----------|-------|
| **Linters** | Python: pylint, flake8, black · JS/TS: eslint, prettier · Go: golint, gofmt · Rust: clippy, rustfmt |
| **Security** | Bandit (Python), npm audit (Node.js), OWASP Dependency-Check |
| **Code quality** | SonarQube, CodeClimate, Codacy |

## Related Skills

- **code-refactoring**: Follow-up refactoring for issues found during review
- **security-best-practices**: Deep-dive on security vulnerabilities
- **performance-optimization**: Performance bottleneck diagnosis

## References

- [Google Code Review Guidelines](https://google.github.io/eng-practices/review/)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- *Clean Code* (Robert C. Martin)
