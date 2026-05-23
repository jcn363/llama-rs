# Reasoning Router

Problem complexity detection and routing to optimal reasoning method.

## Routing Logic

```
Input Problem
    │
    ├─ Complexity Analysis
    │   ├─ Number of viable paths
    │   ├─ Interconnection density
    │   ├─ Verification requirement
    │   └─ Cost budget
    │
    └─ Method Selection
        ├─ CoT:  Simple, single path, low cost
        ├─ ToT:  Complex, branching, high accuracy needed
        ├─ GoT:  Interconnected sub-problems
        └─ SC:   Critical decisions, needs voting
```

## Method Profiles

### Chain-of-Thought (CoT)
- **When**: Single solution path, linear reasoning
- **Cost**: Low (1-2 API calls)
- **Example**: "Calculate the total cost of 3 items with tax"
- **Prompt pattern**: "Let's think step by step..."

### Tree-of-Thoughts (ToT)
- **When**: Multiple viable paths, planning, search
- **Modes**: BFS (breadth) for exploration, DFS (depth) for deep reasoning
- **Cost**: High (N branches × depth API calls)
- **Example**: "Plan a 3-day itinerary balancing budget and interests"
- **Prompt pattern**: "Consider these N approaches..." then evaluate each

### Graph-of-Thoughts (GoT)
- **When**: Interconnected sub-problems with dependencies
- **Cost**: Very high
- **Example**: "Design a distributed system architecture"
- **Prompt pattern**: Decompose into nodes, resolve dependencies

### Self-Consistency (SC)
- **When**: Critical decisions needing verification
- **Cost**: Very high (N full reasoning chains)
- **Example**: "Is this medical diagnosis correct?"
- **Prompt pattern**: Generate N chains, majority vote

## Implementation

```python
class ReasoningRouter:
    def select(self, problem):
        complexity = self.assess_complexity(problem)
        if complexity.task_type == "simple_linear":
            return ChainOfThought()
        elif complexity.task_type == "complex_planning":
            return TreeOfThoughts(mode=complexity.recommended_mode)
        elif complexity.task_type == "interconnected":
            return GraphOfThoughts()
        elif complexity.task_type == "critical_decision":
            return SelfConsistency(num_chains=5)
```

## Complexity Assessment Factors

- **Path count**: How many valid approaches exist?
- **Depth**: How many reasoning steps needed?
- **Verification need**: Is a single correct answer expected?
- **Cost tolerance**: What's the API call budget?
