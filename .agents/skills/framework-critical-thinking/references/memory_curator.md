# Memory Curator

Management of episodic memory with quality weighting to prevent memory pollution from bad episodes.

## Quality-Weighted Memory

Each memory episode is stored with a quality score that affects retrieval priority.

```python
@dataclass
class MemoryEpisode:
    id: str
    input: str
    output: str
    confidence: float        # 0-1, metacognitive score
    outcome_score: float     # 0-1, actual result quality
    retrieval_count: int
    timestamp: datetime

class MemoryCurator:
    def store(self, episode: MemoryEpisode):
        quality = self.compute_quality(episode)
        if quality > STORAGE_THRESHOLD:
            self.db.insert(episode)

    def retrieve(self, query: str, top_k: int = 5) -> list[MemoryEpisode]:
        candidates = self.db.similarity_search(query, k=top_k*2)
        # Re-rank by quality score
        candidates.sort(key=lambda e: e.confidence * e.outcome_score, reverse=True)
        return candidates[:top_k]
```

## Experience Replay

Periodically re-process high-quality past episodes to reinforce learning:

1. Sample top 10% of episodes by quality score
2. Re-run through reasoning pipeline
3. Compare current output with stored output
4. Update confidence scores based on consistency

## Memory Consolidation

| Frequency | Action |
|-----------|--------|
| Per session | Quality-weight new episodes |
| Daily | Replay high-quality episodes |
| Weekly | Prune episodes below retention threshold |
| Monthly | Aggregate patterns into semantic memory |

## Selective Retention Policy

- **Keep**: High-confidence proven episodes, frequently retrieved episodes
- **Prune**: Low-confidence episodes, never-retrieved episodes after 30 days
- **Archive**: Medium-confidence episodes to compressed long-term storage
