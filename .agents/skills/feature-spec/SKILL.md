---
name: feature-spec
description: "Use when writing product requirements documents (PRDs), defining feature specifications, creating user stories with acceptance criteria, prioritizing requirements with MoSCoW, setting success metrics, or managing scope - provides structured PRD templates spanning problem statements, goals/non-goals, user stories, P0/P1/P2 requirements, leading/lagging indicators, and scope creep prevention."
---

# Feature Spec

## Overview

Write structured product requirements documents with problem statements, user stories, success metrics, and scope management — translating user needs into actionable specifications.

## When to Use

- Defining a new feature or product area
- Writing a PRD for cross-team alignment
- Prioritizing requirements and making scope decisions
- Setting measurable success criteria before building
- Preventing scope creep during implementation

## PRD Structure

A well-structured PRD follows these eight sections:

### 1. Problem Statement

Describe the user problem in 2-3 sentences:
- Who experiences this problem and how often?
- What is the cost of not solving it? (user pain, business impact, competitive risk)
- Ground in evidence: user research, support data, metrics, customer feedback

### 2. Goals

3-5 specific, measurable outcomes this feature should achieve:
- Each goal answers: "How will we know this succeeded?"
- Distinguish user goals (what users get) from business goals (what the company gets)
- Goals are outcomes, not outputs: "Reduce time to first value by 50%" not "Build onboarding wizard"

### 3. Non-Goals

3-5 things this feature will **not** do:
- Adjacent capabilities explicitly out of scope for this version
- For each, explain why: insufficient impact, too complex, separate initiative, premature
- Non-goals prevent scope creep and set stakeholder expectations

### 4. User Stories

Standard format: **As a [user type], I want [capability] so that [benefit]**

**Guidelines:**
- User type is specific enough to be meaningful ("enterprise admin" not "user")
- Capability describes *what* they want to accomplish, not how
- Benefit explains the *why* — what value is delivered
- Include edge cases: error states, empty states, boundary conditions
- Cover different user personas if the feature serves multiple types
- Order by priority — most important stories first

**Examples:**
- "As a team admin, I want to configure SSO for my organization so that my team members can log in with their corporate credentials"
- "As a team member, I want to be automatically redirected to my company's SSO login so that I do not need to remember a separate password"
- "As a team admin, I want to see which members have logged in via SSO so that I can verify the rollout is working"

### 5. Requirements (P0 / P1 / P2)

**Must-Have (P0):** Feature cannot ship without these. The minimum viable version. Ask: "If we cut this, does the feature still solve the core problem?" If no, it's P0.

**Nice-to-Have (P1):** Significantly improves the experience but the core use case works without them. Often become fast follow-ups after launch.

**Future Considerations (P2):** Explicitly out of scope for v1 but design should support them later. Prevents accidental architectural decisions that make them hard to add.

For each requirement, include:
- Clear, unambiguous description of expected behavior
- Acceptance criteria (Given/When/Then or checklist)
- Technical considerations or constraints
- Dependencies on other teams or systems

### 6. Success Metrics

See **Success Metrics** section below for detailed guidance.

### 7. Open Questions

- Questions needing answers before or during implementation
- Tag each with who should answer (engineering, design, legal, data, stakeholder)
- Distinguish blocking vs non-blocking questions

### 8. Timeline Considerations

- Hard deadlines (contractual commitments, events, compliance dates)
- Dependencies on other teams' work or releases
- Suggested phasing if the feature is too large for one release

## User Story Writing (INVEST)

Good user stories are:

| Principle | Meaning |
|-----------|---------|
| **I**ndependent | Can be developed and delivered separately |
| **N**egotiable | Details can be discussed; not a contract |
| **V**aluable | Delivers value to the user |
| **E**stimable | Team can roughly estimate effort |
| **S**mall | Can be completed in one sprint/iteration |
| **T**estable | Clear way to verify it works |

### Common Mistakes

| Mistake | Correction |
|---------|-----------|
| Too vague: "As a user, I want the product to be faster" | What specifically should be faster? |
| Solution-prescriptive: "As a user, I want a dropdown menu" | Describe the need, not the widget |
| No benefit: "As a user, I want to click a button" | Why? What does it accomplish? |
| Too large: "As a user, I want to manage my team" | Break into specific capabilities |
| Internal focus: "Engineering wants to refactor the DB" | This is a task, not a user story |

## Requirements Categorization (MoSCoW)

| Category | Meaning | Action |
|----------|---------|--------|
| **Must Have** | Feature is not viable without these. Non-negotiable. | P0 — ship-blocking |
| **Should Have** | Important but not critical for launch. High-priority fast follow. | P1 — next |
| **Could Have** | Desirable if time permits. Will not delay delivery if cut. | P2 — nice to have |
| **Won't Have** | Explicitly out of scope. May revisit in future versions. | Not now |

### Tips for Categorization

- Be ruthless about P0s — the tighter the must-have list, the faster you ship and learn
- If everything is P0, nothing is P0. Challenge each: "Would we really not ship without this?"
- P1s should be things you are confident you'll build soon, not a wish list
- P2s are architectural insurance — they guide design even though you're not building them now

## Success Metrics

### Leading Indicators (change quickly — days to weeks)

| Metric | What It Measures |
|--------|-----------------|
| Adoption rate | % of eligible users who try the feature |
| Activation rate | % of users who complete the core action |
| Task completion rate | % of users who successfully accomplish their goal |
| Time to complete | How long the core workflow takes |
| Error rate | How often users encounter errors or dead ends |
| Feature usage frequency | How often users return to use the feature |

### Lagging Indicators (weeks to months)

| Metric | What It Measures |
|--------|-----------------|
| Retention impact | Does this feature improve user retention? |
| Revenue impact | Does this drive upgrades, expansion, or new revenue? |
| NPS / satisfaction change | Does this improve how users feel about the product? |
| Support ticket reduction | Does this reduce support load? |
| Competitive win rate | Does this help win more deals? |

### Setting Targets

- Targets must be specific: "50% adoption within 30 days" not "high adoption"
- Base targets on comparable features, industry benchmarks, or explicit hypotheses
- Set both a "success" threshold and a "stretch" target
- Define the measurement method: what tool, what query, what time window
- Specify when you will evaluate: 1 week, 1 month, 1 quarter post-launch

## Acceptance Criteria

Write in **Given/When/Then** format or **checklist** format:

**Given/When/Then:**
```
Given [precondition or context]
When [action the user takes]
Then [expected outcome]
```

**Example:**
```
Given the admin has configured SSO for their organization
When a team member visits the login page
Then they are automatically redirected to the organization's SSO provider
```

**Checklist format:**
- Admin can enter SSO provider URL in organization settings
- Team members see "Log in with SSO" button on login page
- SSO login creates a new account if one does not exist
- SSO login links to existing account if email matches
- Failed SSO attempts show a clear error message

### Tips

- Cover the happy path, error cases, and edge cases
- Be specific about expected behavior, not implementation
- Include what should NOT happen (negative test cases)
- Each criterion should be independently testable
- Avoid ambiguous words: "fast", "user-friendly", "intuitive" — define them concretely

## Scope Management

### Recognizing Scope Creep

- Requirements keep getting added after the spec is approved
- "Small" additions accumulate into a significantly larger project
- The team is building features no user asked for ("while we're at it...")
- The launch date keeps moving without explicit re-scoping
- Stakeholders add requirements without removing anything

### Prevention

- Write explicit non-goals in every spec
- Require any scope addition to come with a scope removal or timeline extension
- Separate "v1" from "v2" clearly in the spec
- Review the spec against the original problem statement — does everything serve it?
- Time-box investigations: "If we cannot figure out X in 2 days, we cut it"
- Create a "parking lot" for good ideas that are not in scope

## Constraints

### Required (MUST)
- **Problem anchored**: Every requirement must trace back to the stated problem
- **Measurable goals**: Every goal must have a specific success metric and target
- **Explicit non-goals**: Every spec must state what is intentionally out of scope
- **Prioritized requirements**: Every requirement must have a P0/P1/P2 priority

### Prohibited (MUST NOT)
- **Solution-prescriptive stories**: User stories describe the need, not the UI widget
- **Unscoped additions**: No "while we're at it" without explicit re-scoping
- **Ambiguous success criteria**: No "high adoption" or "better experience" without definitions

## Related Skills

- **task-estimation**: Story point estimation for requirements
- **task-planning**: Sprint planning and backlog organization from specs

## References

- *Inspired* (Marty Cagan) — Product discovery and PRD best practices
- [MoSCoW Prioritization](https://www.productplan.com/glossary/moscow-prioritization/)
- [INVEST Principles](https://xp123.com/articles/invest-in-good-stories-and-smart-tasks/)
