---
name: framework-critical-thinking
description: "Use when building AI agents for complex multi-step reasoning, implementing self-correction and error detection, selecting between CoT/ToT/GoT reasoning methods, adding metacognitive monitoring to improve reliability, or reducing hallucination in AI outputs - provides architectural components for reasoning routing, metacognitive monitoring, self-verification, bias detection, producer-critic orchestration, and memory curation with reference implementations."
---

# Framework Critical Thinking (FCT)

## Overview

Critical Thinking Framework provides architectural components for building AI agents with metacognition and self-correction capabilities. Integrates Chain-of-Thought, Tree-of-Thoughts, Graph-of-Thoughts, self-verification (CoVe), and producer-critic patterns for reliable, transparent reasoning.

## When to Use

- Building AI agents for complex tasks requiring multi-step reasoning
- Implementing self-correction and error detection in agent workflows
- Selecting the appropriate reasoning method (CoT vs ToT vs GoT) based on task characteristics
- Adding metacognitive monitoring to improve reliability
- Reducing hallucination and reasoning errors in AI outputs

**Triggers:** "build an agent that can self-correct", "implement better reasoning", "reduce hallucination", "choose between CoT/ToT/GoT", "add error detection to agent workflow"

## Core Components

### 1. Reasoning Router
**File:** `references/reasoning_router.md`

Detects problem complexity and routes to the optimal reasoning method (CoT/ToT/GoT/Self-Consistency).

| Task Type | Method |
|-----------|--------|
| Straightforward, single solution path | Chain-of-Thought (CoT) |
| Complex, multiple viable paths | Tree-of-Thoughts (BFS/DFS) |
| Interconnected reasoning | Graph-of-Thoughts (GoT) |
| Critical, needs verification | Self-Consistency |

### 2. Metacognitive Monitor
**File:** `references/metacognitive_monitor.md`

Self-assessment and error detection in the reasoning process. Implements the Producer-Critic pattern for continuous quality control.

- Confidence scoring per reasoning step
- Anomaly detection in thought patterns
- Reflection trigger conditions
- Human handoff protocols

### 3. Self-Verification
**File:** `references/self_verification.md`

Implementation of Chain-of-Verification (CoVe) and self-verification techniques to validate outputs before delivery.

- Chain-of-Verification (CoVe)
- Self-Refine loops
- Backward verification (math/logic)
- Cross-verification with external sources

### 4. Bias Detector
**File:** `references/bias_detector.md`

Detection of cognitive bias in the reasoning process and mitigation strategies.

- Confirmation bias
- Anchoring bias
- Availability heuristic
- Framing effects
- Recency bias

### 5. Producer-Critic Orchestrator
**File:** `references/producer_critic_orchestrator.md`

Pattern for orchestrating Generate-Critique-Refine cycles in agent workflows.

- Master Agent (orchestrator)
- Producer Agent (generation)
- Critic Agent (evaluation)
- Refinement loops with budget constraints

### 6. Memory Curator
**File:** `references/memory_curator.md`

Management of episodic memory with quality weighting to prevent memory pollution from bad episodes.

- Quality-weighted memory storage
- Experience replay for learning
- Memory consolidation strategies
- Selective retention policies

### 7. Reasoning Validator
**File:** `references/reasoning_validator.md`

Logical consistency checker and structural validation for reasoning chains.

- Logical consistency checks
- Structural completeness
- Assumption validation
- Contradiction detection

### 8. Reflection Trigger
**File:** `references/reflection_trigger.md`

Rule-based triggers to activate self-correction loops based on specific conditions.

- Confidence threshold violations
- Repeated action patterns
- Latency spikes
- Complexity indicators

## Workflow Decision Tree

```
User Request: Build/improve AI agent with critical thinking

├── Step 1: Analyze Task Complexity
│   ├── Simple, single-path        → CoT
│   ├── Complex, multi-path        → ToT
│   ├── Interconnected             → GoT
│   └── Critical, needs verification → Self-Consistency
│
├── Step 2: Implement Metacognitive Layer
│   ├── Add confidence scoring
│   ├── Set up reflection triggers
│   └── Configure human handoff thresholds
│
├── Step 3: Add Self-Verification
│   ├── Implement CoVe for factual claims
│   ├── Add backward verification for math/logic
│   └── Setup cross-verification if external sources available
│
├── Step 4: Integrate Bias Detection
│   ├── Check for confirmation bias
│   ├── Validate assumption diversity
│   └── Apply mitigation strategies
│
└── Step 5: Setup Memory & Learning
    ├── Configure episodic memory
    ├── Setup quality weighting
    └── Implement experience replay
```

## Quick Reference: Reasoning Method Selection

| Task Characteristic | Recommended Method | Cost | Accuracy |
|--------------------|-------------------|------|----------|
| Simple, linear | CoT | Low | Good |
| Complex planning | ToT-BFS | High | Very Good |
| Deep reasoning | ToT-DFS | High | Very Good |
| Interconnected | GoT | Very High | Excellent |
| Critical decisions | Self-Consistency | Very High | Excellent |
| Factual claims | CoVe | Medium | Good |

## Implementation Example

```python
class CriticalThinkingAgent:
    def __init__(self):
        self.reasoning_router = ReasoningRouter()
        self.metacognitive_monitor = MetacognitiveMonitor()
        self.self_verifier = SelfVerification()
        self.bias_detector = BiasDetector()

    async def solve(self, problem):
        # Step 1: Route to appropriate method
        method = self.reasoning_router.select(problem)

        # Step 2: Generate with monitoring
        thoughts = []
        for step in method.generate(problem):
            confidence = self.metacognitive_monitor.assess(step)
            if confidence < THRESHOLD:
                step = self.reflect_and_improve(step)
            thoughts.append(step)

        # Step 3: Self-verification
        verified = self.self_verifier.verify(thoughts)

        # Step 4: Bias check
        if self.bias_detector.detect(verified):
            verified = self.bias_detector.mitigate(verified)

        return verified
```

## Resources

### references/
- `reasoning_router.md` — Reasoning method selection (P0)
- `metacognitive_monitor.md` — Self-assessment and monitoring (P0)
- `self_verification.md` — Output verification techniques (P0)
- `bias_detector.md` — Bias detection and mitigation (P0)
- `producer_critic_orchestrator.md` — Generate-critique-refine pattern (P1)
- `memory_curator.md` — Memory management (P1)
- `reasoning_validator.md` — Logical validation (P1)
- `reflection_trigger.md` — Trigger conditions (P1)
- `uncertainty_quantifier.md` — Confidence calibration (P2)
- `fallback_handler.md` — Graceful degradation (P2)

## Sources

- [Tree of Thoughts: Branching Reasoning for LLMs](https://www.emergentmind.com/topics/tree-of-thoughts-tot)
- [AI Agents: Metacognition for Self-Aware Intelligence - Microsoft](https://techcommunity.microsoft.com/blog/educatordeveloperblog/ai-agents-metacognition-for-self-aware-intelligence---part-9/4402253)
- [Self-Verification-Based LLMs](https://www.emergentmind.com/topics/self-verification-based-llms)
- [Cognitive Architecture in AI](https://sema4.ai/learning-center/cognitive-architecture-ai/)
