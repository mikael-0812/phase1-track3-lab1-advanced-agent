from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from .schemas import ReportPayload, RunRecord

def token_source_breakdown(records: list[RunRecord]) -> dict[str, dict[str, int]]:
    grouped: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        grouped[record.agent_type][record.token_source] += 1
    return {agent: dict(counter) for agent, counter in grouped.items()}


def summarize(records: list[RunRecord]) -> dict:
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        grouped[record.agent_type].append(record)
    summary: dict[str, dict] = {}
    for agent_type, rows in grouped.items():
        source_counts = Counter(r.token_source for r in rows)
        summary[agent_type] = {
            "count": len(rows),
            "em": round(mean(1.0 if r.is_correct else 0.0 for r in rows), 4),
            "avg_attempts": round(mean(r.attempts for r in rows), 4),
            "avg_token_estimate": round(mean(r.token_estimate for r in rows), 2),
            "avg_prompt_tokens": round(mean(r.prompt_tokens for r in rows), 2),
            "avg_completion_tokens": round(mean(r.completion_tokens for r in rows), 2),
            "avg_latency_ms": round(mean(r.latency_ms for r in rows), 2),
            "token_source_breakdown": dict(source_counts),
        }
    if "react" in summary and "reflexion" in summary:
        summary["delta_reflexion_minus_react"] = {
            "em_abs": round(summary["reflexion"]["em"] - summary["react"]["em"], 4),
            "attempts_abs": round(summary["reflexion"]["avg_attempts"] - summary["react"]["avg_attempts"], 4),
            "tokens_abs": round(summary["reflexion"]["avg_token_estimate"] - summary["react"]["avg_token_estimate"], 2),
            "prompt_tokens_abs": round(summary["reflexion"]["avg_prompt_tokens"] - summary["react"]["avg_prompt_tokens"], 2),
            "completion_tokens_abs": round(summary["reflexion"]["avg_completion_tokens"] - summary["react"]["avg_completion_tokens"], 2),
            "latency_abs": round(summary["reflexion"]["avg_latency_ms"] - summary["react"]["avg_latency_ms"], 2),
        }
    return summary

def failure_breakdown(records: list[RunRecord]) -> dict:
    grouped: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        grouped[record.agent_type][record.failure_mode] += 1
    return {agent: dict(counter) for agent, counter in grouped.items()}

def build_report(
    records: list[RunRecord],
    dataset_name: str,
    mode: str = "mock",
    extensions: list[str] | None = None,
) -> ReportPayload:
    examples = [
        {
            "qid": r.qid,
            "agent_type": r.agent_type,
            "gold_answer": r.gold_answer,
            "predicted_answer": r.predicted_answer,
            "is_correct": r.is_correct,
            "attempts": r.attempts,
            "failure_mode": r.failure_mode,
            "reflection_count": len(r.reflections),
            "token_estimate": r.token_estimate,
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "latency_ms": r.latency_ms,
            "token_source": r.token_source,
        }
        for r in records
    ]
    token_sources = token_source_breakdown(records)
    estimated_records = sum(1 for r in records if r.token_source == "estimated")
    mixed_records = sum(1 for r in records if r.token_source == "mixed")
    api_usage_records = sum(1 for r in records if r.token_source == "api_usage")
    if estimated_records == 0 and mixed_records == 0:
        token_note = "All run-level token totals came from provider usage metadata."
    elif api_usage_records == 0:
        token_note = "All run-level token totals used fallback estimates because provider usage metadata was unavailable."
    else:
        token_note = "Some run-level token totals came from provider usage metadata while others used fallback estimates."
    report_extensions = extensions or []
    return ReportPayload(
        meta={
            "dataset": dataset_name,
            "mode": mode,
            "num_records": len(records),
            "agents": sorted({r.agent_type for r in records}),
            "token_source_breakdown": token_sources,
            "token_accounting_note": token_note,
        },
        summary=summarize(records),
        failure_modes=failure_breakdown(records),
        examples=examples,
        extensions=report_extensions,
        discussion="Reflexion helps when the first attempt stops after the first hop or drifts to a wrong second-hop entity. The tradeoff is higher attempts, token cost, and latency. This report also distinguishes token totals that came from provider usage metadata from fallback estimates so benchmark cost analysis stays interpretable. In a real report, students should explain when the reflection memory was useful, which failure modes remained, and whether evaluator quality limited gains.",
    )

def save_report(report: ReportPayload, out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    json_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    s = report.summary
    react = s.get("react", {})
    reflexion = s.get("reflexion", {})
    delta = s.get("delta_reflexion_minus_react", {})
    ext_lines = "\n".join(f"- {item}" for item in report.extensions)
    md = f"""# Lab 16 Benchmark Report

## Metadata
- Dataset: {report.meta['dataset']}
- Mode: {report.meta['mode']}
- Records: {report.meta['num_records']}
- Agents: {', '.join(report.meta['agents'])}
- Token accounting: {report.meta.get('token_accounting_note', 'N/A')}

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | {react.get('em', 0)} | {reflexion.get('em', 0)} | {delta.get('em_abs', 0)} |
| Avg attempts | {react.get('avg_attempts', 0)} | {reflexion.get('avg_attempts', 0)} | {delta.get('attempts_abs', 0)} |
| Avg token estimate | {react.get('avg_token_estimate', 0)} | {reflexion.get('avg_token_estimate', 0)} | {delta.get('tokens_abs', 0)} |
| Avg prompt tokens | {react.get('avg_prompt_tokens', 0)} | {reflexion.get('avg_prompt_tokens', 0)} | {delta.get('prompt_tokens_abs', 0)} |
| Avg completion tokens | {react.get('avg_completion_tokens', 0)} | {reflexion.get('avg_completion_tokens', 0)} | {delta.get('completion_tokens_abs', 0)} |
| Avg latency (ms) | {react.get('avg_latency_ms', 0)} | {reflexion.get('avg_latency_ms', 0)} | {delta.get('latency_abs', 0)} |

## Token sources
```json
{json.dumps(report.meta.get('token_source_breakdown', {}), indent=2)}
```

## Failure modes
```json
{json.dumps(report.failure_modes, indent=2)}
```

## Extensions implemented
{ext_lines}

## Discussion
{report.discussion}
"""
    md_path.write_text(md, encoding="utf-8")
    return json_path, md_path
