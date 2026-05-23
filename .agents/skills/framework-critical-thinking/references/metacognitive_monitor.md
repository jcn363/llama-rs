# Metacognitive Monitor

Self-assessment and error detection in the reasoning process using the Producer-Critic pattern.

## Core Concepts

The monitor continuously evaluates each reasoning step for quality, coherence, and confidence, triggering corrections when metrics fall below thresholds.

## Confidence Scoring

```python
class ConfidenceScore:
    step_coherence: float     # 0-1, logical flow between steps
    evidence_support: float   # 0-1, how well supported
    contradiction_free: float # 0-1, no internal conflicts
    overall: float            # weighted combination

def assess_confidence(step: Thought) -> ConfidenceScore:
    return ConfidenceScore(
        step_coherence=measure_coherence(step),
        evidence_support=measure_evidence(step),
        contradiction_free=check_contradictions(step),
        overall=weighted_average([...])
    )
```

## Anomaly Detection

| Pattern | Indicator | Action |
|---------|-----------|--------|
| Sudden confidence drop | Score delta > 0.3 | Re-route to ToT |
| Repeated same action | Action count > N | Escalate to reflection |
| Circular reasoning | Self-reference loop | Break chain, restart |
| Overconfidence on weak evidence | High score + low support | Force verification |

## Human Handoff Protocol

Trigger handoff when:
1. Confidence stays below threshold after 3 refine attempts
2. Detected anomaly pattern unknown
3. Task requires external authorization
4. Budget exhausted without resolution

## Threshold Configuration

```yaml
thresholds:
  confidence_minimum: 0.7
  coherence_minimum: 0.6
  refine_attempts_max: 3
  anomaly_retry_max: 2
  handoff_on_pattern: ["circular", "escalating", "unknown"]
```
