# Fallback Handler

Graceful degradation strategies when primary reasoning methods fail or exceed budgets.

## Degradation Hierarchy

```
Primary Method (ToT/GoT)
    │
    ├─ Budget exhausted?
    │   └─ Downgrade to CoT with key insights preserved
    │
    ├─ Confidence too low?
    │   └─ Request human verification on uncertain steps
    │
    ├─ Repeated failure?
    │   └─ Fallback to simpler heuristic / default response
    │
    └─ System error?
        └─ Return cached result or graceful error message
```

## Fallback Strategies

| Failure Mode | Fallback | User Impact |
|-------------|----------|-------------|
| Method budget exhausted | CoT with partial reasoning | May miss some branches |
| Low confidence across all paths | Request clarification | User interaction needed |
| API/model unavailable | Cached response for known patterns | Slightly stale results |
| Context overflow | Summarize → continue | Loss of detail |
| Timeout | Return best-so-far | Possibly incomplete |

## Implementation

```python
class FallbackHandler:
    def handle(self, error: FailureMode, context: AgentContext) -> Response:
        if error == FailureMode.BUDGET_EXCEEDED:
            return self.downgrade_method(context)
        elif error == FailureMode.LOW_CONFIDENCE:
            return self.request_verification(context)
        elif error == FailureMode.API_UNAVAILABLE:
            return self.serve_cached(context)
        elif error == FailureMode.TIMEOUT:
            return self.best_effort_response(context)

    def downgrade_method(self, context) -> Response:
        # Extract key insights from partial reasoning
        # Run simplified CoT on remaining problem
        # Merge partial + simplified results
        return merged_response

    def best_effort_response(self, context) -> Response:
        # Return whatever reasoning was completed
        # Clearly mark incomplete sections
        return Response(
            content=context.partial_output,
            warning="Response may be incomplete due to timeout"
        )
```
