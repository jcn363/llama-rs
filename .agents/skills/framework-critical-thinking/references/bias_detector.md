# Bias Detector

Detection of cognitive bias in the reasoning process and mitigation strategies.

## Bias Types

### Confirmation Bias
- **Symptom**: Seeking evidence that confirms existing hypothesis, ignoring contrary data
- **Mitigation**: "List 3 reasons why the opposite might be true"
- **Detection**: Check if negative results were considered

### Anchoring Bias
- **Symptom**: Over-reliance on first piece of information encountered
- **Mitigation**: Generate independent estimates before revealing reference points
- **Detection**: Compare multiple independently generated values

### Availability Heuristic
- **Symptom**: Overweighting recent or memorable examples
- **Mitigation**: Request statistical baselines, not anecdotes
- **Detection**: Check if reasoning relies on single vivid example

### Framing Effects
- **Symptom**: Different responses based on how problem is presented
- **Mitigation**: Re-frame the problem from multiple perspectives
- **Detection**: Test with rephrased problem statement

### Recency Bias
- **Symptom**: Overweighting recent information in sequence
- **Mitigation**: Evaluate all evidence independently before synthesis
- **Detection**: Check if earlier context is dropped

## Implementation

```python
class BiasDetector:
    def detect(self, thoughts: list[Thought]) -> list[Bias]:
        biases = []
        if self.confirmation_bias(thoughts):
            biases.append(Bias("confirmation", severity="medium"))
        if self.anchoring_bias(thoughts):
            biases.append(Bias("anchoring", severity="high"))
        if self.availability_bias(thoughts):
            biases.append(Bias("availability", severity="medium"))
        if self.framing_effect(thoughts):
            biases.append(Bias("framing", severity="low"))
        if self.recency_bias(thoughts):
            biases.append(Bias("recency", severity="medium"))
        return biases

    def mitigate(self, thoughts: list[Thought], biases: list[Bias]) -> list[Thought]:
        for bias in biases:
            strategy = self.get_strategy(bias)
            thoughts = strategy.apply(thoughts)
        return thoughts
```

## Mitigation Strategies

| Bias | Strategy |
|------|----------|
| Confirmation | Deliberate counter-evidence search |
| Anchoring | Multiple independent estimates |
| Availability | Request base rates, statistical context |
| Framing | Rephrase problem from N perspectives |
| Recency | Summarize all evidence before concluding |
