from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

from . import runtime
from .schemas import AttemptTrace, CallMetrics, QAExample, ReflectionEntry, RunRecord


def _merge_token_source(metrics: list[CallMetrics]) -> Literal["api_usage", "estimated", "mixed"]:
    sources = {metric.token_source for metric in metrics}
    if not sources:
        return "estimated"
    if len(sources) == 1:
        return sources.pop()
    return "mixed"

@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1

    def effective_max_attempts(self, example: QAExample) -> int:
        return self.max_attempts

    def implemented_extensions(self) -> list[str]:
        return []

    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        attempt_limit = self.effective_max_attempts(example)
        for attempt_id in range(1, attempt_limit + 1):
            answer, actor_metrics = runtime.actor_answer_with_metrics(
                example,
                attempt_id,
                self.agent_type,
                reflection_memory,
            )
            judge, evaluator_metrics = runtime.evaluator_with_metrics(example, answer)
            attempt_metrics = [actor_metrics, evaluator_metrics]
            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=answer,
                score=judge.score,
                reason=judge.reason,
                token_estimate=sum(metric.total_tokens for metric in attempt_metrics),
                prompt_tokens=sum(metric.prompt_tokens for metric in attempt_metrics),
                completion_tokens=sum(metric.completion_tokens for metric in attempt_metrics),
                latency_ms=sum(metric.latency_ms for metric in attempt_metrics),
                token_source=_merge_token_source(attempt_metrics),
                call_metrics=attempt_metrics.copy(),
            )
            final_answer = answer
            final_score = judge.score
            if judge.score == 1:
                traces.append(trace)
                break

            if self.agent_type == "reflexion" and attempt_id < attempt_limit:
                reflection, reflection_metrics = runtime.reflector_with_metrics(
                    example,
                    attempt_id,
                    judge,
                )
                reflections.append(reflection)
                reflection_memory.append(
                    f"Lesson: {reflection.lesson} Strategy: {reflection.next_strategy}"
                )
                trace.reflection = reflection
                trace.call_metrics.append(reflection_metrics)
                trace.token_estimate += reflection_metrics.total_tokens
                trace.prompt_tokens += reflection_metrics.prompt_tokens
                trace.completion_tokens += reflection_metrics.completion_tokens
                trace.latency_ms += reflection_metrics.latency_ms
                trace.token_source = _merge_token_source(trace.call_metrics)

            traces.append(trace)

        total_tokens = sum(t.token_estimate for t in traces)
        total_prompt_tokens = sum(t.prompt_tokens for t in traces)
        total_completion_tokens = sum(t.completion_tokens for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        all_metrics = [metric for trace in traces for metric in trace.call_metrics]
        failure_mode = (
            "none"
            if final_score == 1
            else runtime.failure_mode_by_qid().get(example.qid, "wrong_final_answer")
        )
        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type=self.agent_type,
            predicted_answer=final_answer,
            is_correct=bool(final_score),
            attempts=len(traces),
            token_estimate=total_tokens,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            latency_ms=total_latency,
            token_source=_merge_token_source(all_metrics),
            failure_mode=failure_mode,
            reflections=reflections,
            traces=traces,
        )

class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)


class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3, adaptive_max_attempts: bool = True) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts)
        self.adaptive_max_attempts = adaptive_max_attempts

    def effective_max_attempts(self, example: QAExample) -> int:
        if not self.adaptive_max_attempts:
            return self.max_attempts

        difficulty_budget = {
            "easy": min(self.max_attempts, 2),
            "medium": self.max_attempts,
            "hard": self.max_attempts,
        }
        return max(1, difficulty_budget.get(example.difficulty, self.max_attempts))

    def implemented_extensions(self) -> list[str]:
        extensions = ["reflection_memory"]
        if self.adaptive_max_attempts:
            extensions.append("adaptive_max_attempts")
        return extensions
