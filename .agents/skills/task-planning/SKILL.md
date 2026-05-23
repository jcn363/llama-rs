---
name: task-planning
description: "Use when breaking down features into user stories, organizing sprints, prioritizing backlogs, sprint planning, backlog grooming, or writing acceptance criteria - provides INVEST-compliant story templates, epic decomposition, MoSCoW prioritization, and sprint planning with capacity planning and Definition of Done."
---

# Task Planning

## Overview

Structured task planning with INVEST-compliant user stories, epic decomposition, MoSCoW prioritization, and sprint organization for agile teams.

## When to Use

- **Feature development**: Break down a new feature into small, estimable tasks
- **Sprint Planning**: Select work to include in the sprint with capacity planning
- **Backlog Grooming**: Clean up the backlog and set priorities using MoSCoW

## Core Workflow

### Step 1: Write User Stories (INVEST)

Write stories that satisfy INVEST:

| Principle | Meaning |
|-----------|---------|
| **I**ndependent | Can be developed and delivered separately |
| **N**egotiable | Details can be refined through discussion |
| **V**aluable | Delivers clear value to the user/stakeholder |
| **E**stimable | Team can size it with reasonable confidence |
| **S**mall | Fits within a single sprint (≤13 points) |
| **T**estable | Clear acceptance criteria can be verified |

#### Template

```markdown
## User Story: [title]

**As a** [user type]
**I want** [feature]
**So that** [value/reason]

### Acceptance Criteria
- [ ] Given [context] When [action] Then [outcome]
- [ ] Given [context] When [action] Then [outcome]

### Technical Notes
- API endpoint: [endpoint]
- Database: [table/collection]
- Frontend: [component]

### Estimation
- Story Points: [1/2/3/5/8/13]
- T-Shirt: [XS/S/M/L/XL]

### Dependencies
- [prerequisite work or issue #]

### Priority
- MoSCoW: [Must Have / Should Have / Could Have / Won't Have]
- Business Value: [High / Medium / Low]
```

#### Example

```markdown
## User Story: User Registration

**As a** new visitor
**I want** to create an account
**So that** I can access personalized features

### Acceptance Criteria
- [ ] Given valid email and password When user submits form Then account is created
- [ ] Given duplicate email When user submits Then error message is shown
- [ ] Given weak password When user submits Then validation error is shown
- [ ] Given successful registration When account created Then welcome email is sent

### Technical Notes
- Hash password with bcrypt
- Validate email format
- Send welcome email via SendGrid
- Store user in PostgreSQL

### Estimation
- Story Points: 5

### Dependencies
- Email service integration (#123)

### Priority
- MoSCoW: Must Have
```

### Step 2: Decompose Epic → Story → Task

Break epics into stories, then stories into time-estimated tasks.

```markdown
## Epic: User Management System

### Story 1: User Registration
- **Points**: 5
- Tasks:
  - [ ] Design registration form UI (2h)
  - [ ] Create POST /api/users endpoint (3h)
  - [ ] Implement email validation (1h)
  - [ ] Add password strength checker (2h)
  - [ ] Write unit tests (2h)
  - [ ] Integration testing (2h)

### Story 2: User Login
- **Points**: 3
- Tasks:
  - [ ] Design login form (2h)
  - [ ] Create POST /api/auth/login endpoint (2h)
  - [ ] Implement JWT token generation (2h)
  - [ ] Add "Remember Me" functionality (1h)
  - [ ] Write tests (2h)

### Story 3: Password Reset
- **Points**: 5
- Tasks:
  - [ ] "Forgot Password" UI (2h)
  - [ ] Generate reset token (2h)
  - [ ] Send reset email (1h)
  - [ ] Reset password form (2h)
  - [ ] Update password API (2h)
  - [ ] Tests (2h)
```

### Step 3: MoSCoW Prioritization

Categorize every item into one of four buckets:

| Category | Meaning | Sprint Placement |
|----------|---------|-----------------|
| **Must Have** | Non-negotiable for release | Sprint 1 |
| **Should Have** | Important but not critical | Sprint 2 |
| **Could Have** | Nice to have | Sprint 3 (or later) |
| **Won't Have** | Explicitly out of scope this release | Future |

#### Example

```markdown
## Feature Prioritization (MoSCoW)

### Must Have (Sprint 1)
- User Registration
- User Login
- Basic Profile Page

### Should Have (Sprint 2)
- Password Reset
- Email Verification
- Profile Picture Upload

### Could Have (Sprint 3)
- Two-Factor Authentication
- Social Login (Google, GitHub)
- Account Deletion

### Won't Have (This Release)
- Biometric Authentication
- Multiple Sessions Management
```

### Step 4: Sprint Planning

Plan a sprint with capacity, velocity, and Definition of Done.

```markdown
## Sprint 10 Planning

**Sprint Goal**: Complete user authentication system

**Duration**: 2 weeks
**Team Capacity**: 40 hours × 4 people = 160 hours
**Estimated Velocity**: 30 story points

### Selected Stories
1. User Registration (5 points) - Must Have
2. User Login (3 points) - Must Have
3. Password Reset (5 points) - Must Have
4. Email Verification (3 points) - Should Have
5. Profile Edit (5 points) - Should Have
6. JWT Refresh Token (3 points) - Should Have
7. Rate Limiting (2 points) - Should Have
8. Security Audit (4 points) - Must Have

**Total**: 30 points

### Sprint Backlog
- [ ] User Registration (#101)
- [ ] User Login (#102)
- [ ] Password Reset (#103)
- [ ] Email Verification (#104)
- [ ] Profile Edit (#105)
- [ ] JWT Refresh Token (#106)
- [ ] Rate Limiting (#107)
- [ ] Security Audit (#108)

### Definition of Done
- [ ] Code written and reviewed
- [ ] Unit tests passing (80%+ coverage)
- [ ] Integration tests passing
- [ ] Documentation updated
- [ ] Deployed to staging
- [ ] QA approved
```

## Output Format: Task Board Structure

```
Backlog → To Do → In Progress → Review → Done
```

| Column | Rules |
|--------|-------|
| **Backlog** | Sorted by priority, groomed stories |
| **To Do** | Work selected for the sprint, owner assigned |
| **In Progress** | WIP Limit: 2 per person |
| **Review** | Waiting for code review / In QA testing |
| **Done** | Meets DoD, deployed |

## Constraints

### Required (MUST)
- **Clear AC**: Every story must have acceptance criteria
- **Estimation done**: Assign story points to every story
- **Dependencies identified**: Specify prerequisite work

### Prohibited (MUST NOT)
- **Stories too large**: Split anything 13+ points
- **Vague requirements**: Avoid "improve" and "optimize" — be specific

## Best Practices

- **INVEST**: Write good user stories — test them against each principle
- **Definition of Ready**: Story must be ready (estimated, AC written, deps known) before sprint starts
- **Definition of Done**: Clear, verifiable completion criteria for every story
- **Vertical slicing**: Cut stories across the full stack (UI + API + DB), not by layer
- **Limit WIP**: Max 2 items in progress per person to reduce context switching

## Common Mistakes

| Mistake | Correction |
|---------|-----------|
| Stories written as tasks ("Add button") | Write as user value: "As a... I want... So that..." |
| Gigantic stories (21+ points) | Split into smaller vertical slices |
| No acceptance criteria | Write Given/When/Then for every story |
| Mixing priority and order | MoSCoW for priority; sprint backlog for order |
| Skipping Definition of Done | Define DoD before sprint starts and enforce it |
| Capacity without velocity data | Track 2-3 sprints of historical data before planning |

## Related Skills

- **task-estimation**: Story point estimation, planning poker, and t-shirt sizing
- **prioritize-features**: Impact/effort scoring frameworks for backlog prioritization

## References

- [User Story Guide (Atlassian)](https://www.atlassian.com/agile/project-management/user-stories)
- [MoSCoW Prioritization](https://www.productplan.com/glossary/moscow-prioritization/)
- [INVEST Principles](https://xp123.com/articles/invest-in-good-stories-and-smart-tasks/)
