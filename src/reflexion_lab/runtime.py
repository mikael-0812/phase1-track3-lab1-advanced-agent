from __future__ import annotations

import json
import os
import time
from types import ModuleType

from dotenv import load_dotenv

from .schemas import CallMetrics, JudgeResult, QAExample, ReflectionEntry

load_dotenv()


def use_mock() -> bool:
    return os.getenv("USE_MOCK", "true").lower() == "true"


def _runtime_module() -> ModuleType:
    if use_mock():
        from . import mock_runtime as runtime_module
    else:
        from . import llm_runtime as runtime_module
    return runtime_module


def failure_mode_by_qid() -> dict[str, str]:
    return _runtime_module().FAILURE_MODE_BY_QID


def _estimate_tokens(text: str) -> int:
    cleaned = text.strip()
    if not cleaned:
        return 0
    return max(1, (len(cleaned) + 3) // 4)


def _fallback_metrics(
    stage: str,
    prompt_text: str,
    output_text: str,
    latency_ms: int,
) -> CallMetrics:
    prompt_tokens = _estimate_tokens(prompt_text)
    completion_tokens = _estimate_tokens(output_text)
    return CallMetrics(
        stage=stage,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        latency_ms=latency_ms,
        token_source="estimated",
    )


def actor_answer(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str],
) -> str:
    return actor_answer_with_metrics(
        example,
        attempt_id,
        agent_type,
        reflection_memory,
    )[0]


def actor_answer_with_metrics(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str],
) -> tuple[str, CallMetrics]:
    module = _runtime_module()
    if hasattr(module, "actor_answer_with_metrics"):
        return module.actor_answer_with_metrics(
            example,
            attempt_id,
            agent_type,
            reflection_memory,
        )

    start = time.perf_counter()
    answer = module.actor_answer(
        example,
        attempt_id,
        agent_type,
        reflection_memory,
    )
    latency_ms = int(round((time.perf_counter() - start) * 1000))
    context_text = "\n".join(f"{chunk.title}\n{chunk.text}" for chunk in example.context)
    memory_text = "\n".join(reflection_memory)
    prompt_text = (
        f"Question: {example.question}\n"
        f"Agent type: {agent_type}\n"
        f"Attempt: {attempt_id}\n"
        f"Context:\n{context_text}\n"
        f"Reflection memory:\n{memory_text}"
    )
    return answer, _fallback_metrics("actor", prompt_text, answer, latency_ms)


def evaluator(example: QAExample, answer: str) -> JudgeResult:
    return evaluator_with_metrics(example, answer)[0]


def evaluator_with_metrics(
    example: QAExample,
    answer: str,
) -> tuple[JudgeResult, CallMetrics]:
    module = _runtime_module()
    if hasattr(module, "evaluator_with_metrics"):
        return module.evaluator_with_metrics(example, answer)

    start = time.perf_counter()
    judge = module.evaluator(example, answer)
    latency_ms = int(round((time.perf_counter() - start) * 1000))
    prompt_text = (
        f"Question: {example.question}\n"
        f"Gold answer: {example.gold_answer}\n"
        f"Predicted answer: {answer}"
    )
    output_text = json.dumps(judge.model_dump(), ensure_ascii=True)
    return judge, _fallback_metrics("evaluator", prompt_text, output_text, latency_ms)


def reflector(
    example: QAExample,
    attempt_id: int,
    judge: JudgeResult,
) -> ReflectionEntry:
    return reflector_with_metrics(example, attempt_id, judge)[0]


def reflector_with_metrics(
    example: QAExample,
    attempt_id: int,
    judge: JudgeResult,
) -> tuple[ReflectionEntry, CallMetrics]:
    module = _runtime_module()
    if hasattr(module, "reflector_with_metrics"):
        return module.reflector_with_metrics(example, attempt_id, judge)

    start = time.perf_counter()
    reflection = module.reflector(example, attempt_id, judge)
    latency_ms = int(round((time.perf_counter() - start) * 1000))
    context_text = "\n".join(f"{chunk.title}\n{chunk.text}" for chunk in example.context)
    prompt_text = (
        f"Question: {example.question}\n"
        f"Context:\n{context_text}\n"
        f"Judge reason: {judge.reason}\n"
        f"Missing evidence: {judge.missing_evidence}\n"
        f"Spurious claims: {judge.spurious_claims}"
    )
    output_text = json.dumps(reflection.model_dump(), ensure_ascii=True)
    return reflection, _fallback_metrics("reflector", prompt_text, output_text, latency_ms)
