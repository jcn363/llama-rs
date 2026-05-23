# Self-Verification

Chain-of-Verification (CoVe) and self-verification techniques to validate outputs before delivery.

## Chain-of-Verification (CoVe)

```
Generated Output
    │
    ├─ Step 1: Identify verifiable claims
    │   Extract factual assertions from output
    │
    ├─ Step 2: Generate verification questions
    │   "Is X true?" "Does Y imply Z?"
    │
    ├─ Step 3: Answer independently
    │   Answer each question without seeing original output
    │
    └─ Step 4: Cross-check and refine
        Compare answers → flag mismatches → regenerate
```

### Implementation

```python
class ChainOfVerification:
    def verify(self, output: str) -> VerifiedOutput:
        claims = self.extract_claims(output)
        questions = [self.to_question(c) for c in claims]
        answers = [self.answer_independently(q) for q in questions]
        mismatches = self.cross_check(claims, answers)
        if mismatches:
            return self.refine(output, mismatches)
        return VerifiedOutput(output, verified=True)
```

## Self-Refine Loop

1. **Generate**: Produce initial output
2. **Critique**: Self-evaluate against criteria
3. **Refine**: Improve based on critique
4. **Repeat**: Until criteria met or budget exhausted

## Backward Verification

For math/logic: work backward from answer to verify premises.

```
Forward: Premises → Reasoning → Conclusion
Backward: Conclusion → Inverse reasoning → Should match premises
```

## Cross-Verification

When external sources are available:
- Compare factual claims against trusted sources
- Flag unsupported assertions
- Request citation for any novel claim
