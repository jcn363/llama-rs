# Producer-Critic Orchestrator

Pattern for orchestrating Generate-Critique-Refine cycles in agent workflows.

## Architecture

```
                     ┌──────────────────┐
                     │  Master Agent    │
                     │  (Orchestrator)  │
                     └────────┬─────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
      ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
      │  Producer    │ │   Critic    │ │  Memory      │
      │  Agent       │ │  Agent      │ │  Curator     │
      │  (generate)  │ │  (evaluate) │ │  (store)     │
      └──────────────┘ └──────────────┘ └──────────────┘
              │               │
              └───────┬───────┘
                      │
                      ▼
              ┌──────────────┐
              │   Refine     │
              │   Loop       │
              │  (≤ N iters) │
              └──────────────┘
```

## Cycle

1. **Master** sends task to Producer
2. **Producer** generates candidate output
3. **Critic** evaluates against quality criteria
4. If criteria met → output delivered
5. If not → refinement instructions sent to Producer
6. Loop until criteria met or budget exhausted

## Budget Constraints

```yaml
refinement:
  max_iterations: 3
  max_tokens_per_iter: 2000
  timeout_seconds: 30

criteria:
  min_confidence: 0.7
  required_checks: ["coherence", "completeness", "correctness"]
```

## Implementation Guidance

- Producer and Critic can share the same model with different prompts, or use different models for diversity
- The Critic should receive the task requirements verbatim to evaluate against stated goals
- Budget constraints prevent infinite refinement loops
- Each iteration should produce measurably different output (not superficial rephrasing)
