---
name: task-estimation
description: "Use when estimating story points, planning poker, sprint planning, backlog grooming, roadmap creation, resource planning, or t-shirt sizing tasks - provides frameworks for relative estimation with risk and uncertainty adjustment."
---

# Task Estimation

## Overview

Relative estimation framework using story points, planning poker, and t-shirt sizing to forecast effort, complexity, and risk. Enables team consensus and data-driven planning.

## When to Use

- **Sprint Planning**: Decide what work to include in the sprint
- **Roadmap creation**: Build long-term plans
- **Resource planning**: Estimate team size and schedule
- **Backlog grooming**: Initial sizing of backlog items
- **Quick prioritization**: Rough estimates for decision-making

## Core Pattern

### Step 1: Story Points (Relative Estimation)

Use the Fibonacci sequence: **1, 2, 3, 5, 8, 13, 21**

| Points | Size | Example | Time | Complexity | Risk |
|--------|------|---------|------|-----------|------|
| 1 | Very Small | Text change, constant update | 1-2 hours | Very low | None |
| 2 | Small | Simple bug fix, add logging | 2-4 hours | Low | Low |
| 3 | Medium | Simple CRUD API endpoint | 4-8 hours | Medium | Low |
| 5 | Medium-Large | Complex form, auth middleware | 1-2 days | Medium | Medium |
| 8 | Large | New feature (frontend + backend) | 2-3 days | High | Medium |
| 13 | Very Large | Payment system integration | ~1 week | Very high | High |
| 21+ | Epic | Must be split into smaller stories | — | — | — |

### Step 2: Planning Poker

1. Product Owner explains the story
2. Team asks clarifying questions
3. Everyone picks a card (1, 2, 3, 5, 8, 13)
4. Reveal simultaneously
5. Highest/lowest scores explain reasoning
6. Re-vote
7. Reach consensus

**Example**: "Users can upload a profile photo"
- Member A: 3 points (simple frontend)
- Member B: 5 points (image resizing needed)
- Member C: 8 points (S3 upload, security considerations)
- Discussion → image processing library exists, S3 is set up, file size validation needed
- Re-vote → consensus on **5 points**

### Step 3: T-Shirt Sizing (Quick Estimation)

| Size | Story Points | Timeline |
|------|-------------|----------|
| XS | 1-2 | Within 1 hour |
| S | 2-3 | Half day |
| M | 5 | 1-2 days |
| L | 8 | ~1 week |
| XL | 13+ | Needs splitting |

**Use t-shirt sizing for**: initial backlog grooming, rough roadmap planning, quick prioritization.

### Step 4: Adjust for Risk and Uncertainty

Apply buffers based on risk level and uncertainty factor:

```typescript
interface TaskEstimate {
  baseEstimate: number;  // base estimate in story points
  risk: 'low' | 'medium' | 'high';
  uncertainty: number;   // 0-1 (0% to 100%)
  finalEstimate: number; // adjusted estimate
}

function adjustEstimate(estimate: TaskEstimate): number {
  let buffer = 1.0;

  // Risk buffer
  if (estimate.risk === 'medium') buffer *= 1.3;
  if (estimate.risk === 'high') buffer *= 1.5;

  // Uncertainty buffer
  buffer *= (1 + estimate.uncertainty);

  return Math.ceil(estimate.baseEstimate * buffer);
}

// Example: 5-point task, medium risk, 20% uncertainty
const task = {
  baseEstimate: 5,
  risk: 'medium',
  uncertainty: 0.2
}; // 5 * 1.3 * 1.2 = 7.8 → 8 points
```

## Output Format

Present estimates using this template:

```markdown
## Task: [Task Name]

### Description
[Work description]

### Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

### Estimation
- **Story Points**: 5
- **T-Shirt Size**: M
- **Estimated Time**: 1-2 days

### Breakdown
- Frontend UI: 2 points
- API Endpoint: 2 points
- Testing: 1 point

### Risks
- Uncertain API response time (medium risk)
- External library dependency (low risk)

### Dependencies
- User authentication must be completed first

### Notes
- Need to discuss design with UX team
```

## Constraints

### Required (MUST)
- **Relative estimation**: Compare complexity, not absolute time
- **Team consensus**: Agreement from the whole team, not individuals
- **Use historical data**: Plan based on past velocity

### Prohibited (MUST NOT)
- Pressuring individuals — estimates are not promises
- Overly granular estimation — split anything 13+ points
- Turning estimates into deadlines — estimate ≠ commitment

## Best Practices

- **Break Down**: Split big work into smaller pieces (anything 13+ points)
- **Reference Stories**: Compare against similar past work
- **Include buffer**: Prepare for the unexpected using the risk adjustment formula
- **Re-estimate when context changes**: New information warrants re-evaluation
- **Track velocity**: Use actual completion rates to calibrate future estimates

## Common Mistakes

| Mistake | Correction |
|---------|-----------|
| Estimating in hours | Use relative story points instead |
| One person estimates for everyone | Use planning poker for team consensus |
| Ignoring uncertainty | Apply risk/uncertainty buffers |
| Treating estimates as commitments | Communicate: estimates are forecasts, not promises |
| Gold-plating estimates | Reference similar past work to ground estimates |
| Not splitting epics | Anything 13+ points must be broken down |

## References

- [Scrum Guide](https://scrumguides.org/)
- [Planning Poker](https://en.wikipedia.org/wiki/Planning_poker)
- [Story Points](https://www.atlassian.com/agile/project-management/estimation)
