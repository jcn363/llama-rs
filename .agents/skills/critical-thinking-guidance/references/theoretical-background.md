# Theoretical Background

The thinking guidance mechanism is grounded in three key theoretical concerns.

## 1. Knowledge Equality

The agent and the user hold different knowledge — the agent has broad technical knowledge, while the user has unique domain context, goals, and constraints. Neither is strictly superior; they are complementary. The guidance mechanism exists to bridge this gap efficiently, not to assume the user hasn't thought enough.

**Implication**: Guidance is information exchange, not deficit correction. The goal is to align the agent's answer with the user's unstated context.

## 2. Risk of Cognitive Degradation

When AI consistently fills in missing details without asking, users may:
- Stop providing context (learned laziness)
- Lose the habit of framing problems clearly
- Rely on the agent to make judgment calls that should be theirs

**Implication**: The mechanism should *preserve* the user's thinking muscle, not atrophy it. Lightweight guidance maintains engagement; Forced Thinking draws a boundary.

## 3. Tool Positioning

The agent is a tool for extending human capability, not a replacement for human judgment. Critical decisions (architecture, priorities, trade-offs) benefit from human context. The guidance mechanism ensures the tool asks for that context when it matters.

**Implication**: The threshold for Forced Thinking should align with decision impact — high-stakes calls warrant more clarification than low-risk factual lookups.
