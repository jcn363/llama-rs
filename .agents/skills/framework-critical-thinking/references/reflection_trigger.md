# Reflection Trigger

Rule-based triggers to activate self-correction loops based on specific conditions.

## Trigger Conditions

### Confidence Threshold Violations
```yaml
triggers:
  - condition: "step_confidence < 0.5"
    action: "re-route to ToT with broader search"
  - condition: "avg_confidence < 0.6 over 3 steps"
    action: "activate self-verification on all steps"
  - condition: "confidence_drop > 0.3 between consecutive steps"
    action: "flag anomaly, request explanation"
```

### Repeated Action Patterns
```yaml
triggers:
  - condition: "same action repeated > 3 times"
    action: "reflect: is this productive? try alternative approach"
  - condition: "backtrack detected > 2 times in 5 steps"
    action: "evaluate: is the current approach correct?"
```

### Latency Spikes
```yaml
triggers:
  - condition: "step_time > 2x average"
    action: "check for complexity explosion, consider simplification"
  - condition: "total_time > budget * 0.8"
    action: "force decision or handoff"
```

### Complexity Indicators
```yaml
triggers:
  - condition: "context exceeds 70% of limit"
    action: "summarize and consolidate before continuing"
  - condition: "branching factor > 5 in ToT"
    action: "prune low-confidence branches"
```

## Trigger Engine

```python
class ReflectionTrigger:
    def __init__(self):
        self.rules = self.load_rules()

    def evaluate(self, state: AgentState) -> list[Action]:
        triggered = []
        for rule in self.rules:
            if rule.condition.evaluate(state):
                triggered.append(rule.action)
        return triggered

    def load_rules(self) -> list[Rule]:
        # Load from YAML configuration
        return [Rule.from_yaml(r) for r in TRIGGER_CONFIG]
```
