# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_100_samples.json
- Mode: real
- Records: 200
- Agents: react, reflexion
- Token accounting: All run-level token totals came from provider usage metadata.

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.95 | 1.0 | 0.05 |
| Avg attempts | 1 | 1.04 | 0.04 |
| Avg token estimate | 531.16 | 568.24 | 37.08 |
| Avg prompt tokens | 483.9 | 517.13 | 33.23 |
| Avg completion tokens | 47.26 | 51.11 | 3.85 |
| Avg latency (ms) | 2309.78 | 3531.12 | 1221.34 |

## Token sources
```json
{
  "react": {
    "api_usage": 100
  },
  "reflexion": {
    "api_usage": 100
  }
}
```

## Failure modes
```json
{
  "react": {
    "none": 95,
    "wrong_final_answer": 5
  },
  "reflexion": {
    "none": 100
  }
}
```

## Extensions implemented
- adaptive_max_attempts
- benchmark_report_json
- reflection_memory
- structured_evaluator

## Discussion
Reflexion helps when the first attempt stops after the first hop or drifts to a wrong second-hop entity. The tradeoff is higher attempts, token cost, and latency. This report also distinguishes token totals that came from provider usage metadata from fallback estimates so benchmark cost analysis stays interpretable. In a real report, students should explain when the reflection memory was useful, which failure modes remained, and whether evaluator quality limited gains.
