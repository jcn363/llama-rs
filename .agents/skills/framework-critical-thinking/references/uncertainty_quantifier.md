# Uncertainty Quantifier

Confidence calibration and uncertainty measurement for reasoning steps.

## Calibration Methods

### Token-level Uncertainty
- Analyze log probabilities of generated tokens
- Low probability tokens indicate uncertainty
- Aggregate across output for overall confidence

### Consistency-based Uncertainty
- Generate multiple samples → measure agreement
- High variance = high uncertainty
- Use entropy of answer distribution

### Decomposition-based Uncertainty
- Break problem into sub-questions
- Measure uncertainty per sub-question
- Propagate uncertainties through reasoning graph

## Confidence Calibration

```python
class UncertaintyQuantifier:
    def calibrate(self, raw_confidence: float, sample_variance: float,
                  historical_accuracy: float) -> float:
        # Adjust raw confidence based on:
        # - Sample variance (high variance = lower confidence)
        # - Historical accuracy of similar tasks
        adjusted = raw_confidence * (1 - sample_variance)
        adjusted *= historical_accuracy
        return min(max(adjusted, 0.0), 1.0)

    def quantify(self, output: str, samples: list[str]) -> Uncertainty:
        return Uncertainty(
            token_confidence=self.token_based(output),
            consistency=self.consistency_based(samples),
            calibrated=self.calibrate(...)
        )
```

## Output

```json
{
  "uncertainty": {
    "overall": 0.15,
    "per_step": [0.1, 0.2, 0.05, 0.3],
    "high_uncertainty_regions": ["step_3: boundary condition handling"],
    "recommendation": "verify step 3 assumptions"
  }
}
```
