# Reasoning Validator

Logical consistency checker and structural validation for reasoning chains.

## Validation Types

### Logical Consistency
- Check for contradictions between consecutive steps
- Verify transitive relationships (if A→B and B→C, then A→C)
- Detect circular reasoning (A→B→C→A)
- Flag non-sequiturs (conclusion doesn't follow from premises)

### Structural Completeness
- Every premise has a conclusion
- No dangling premises (introduced but never used)
- All required reasoning steps are present
- Chain has clear start (premises) and end (conclusion)

### Assumption Validation
- All assumptions are explicitly stated
- Each assumption is justified or flagged
- Hidden assumptions are surfaced
- Assumption sensitivity is assessed

### Contradiction Detection

```python
class ReasoningValidator:
    def check_contradictions(self, chain: list[Statement]) -> list[Contradiction]:
        contradictions = []
        for i, a in enumerate(chain):
            for b in chain[i+1:]:
                if self.are_contradictory(a, b):
                    contradictions.append(
                        Contradiction(a, b, severity=self.assess_severity(a, b))
                    )
        return contradictions

    def check_completeness(self, chain: list[Statement]) -> list[Gap]:
        # Identify missing intermediate steps
        gaps = []
        for i in range(len(chain) - 1):
            if not self.logically_follows(chain[i], chain[i+1]):
                gaps.append(Gap(chain[i], chain[i+1]))
        return gaps
```

## Output

```json
{
  "is_valid": true,
  "contradictions": [],
  "gaps": [],
  "unstated_assumptions": ["Assuming X implies Y"],
  "confidence": 0.92
}
```
