from __future__ import annotations
import json
import os
import time
from json import JSONDecodeError

from dotenv import load_dotenv

from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import CallMetrics, QAExample, JudgeResult, ReflectionEntry

load_dotenv()
_CLIENT = None
_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
_MODEL = os.getenv(
    "LLM_MODEL",
    "gemini-2.5-flash" if _PROVIDER == "gemini" else "gpt-4o-mini",
)
FAILURE_MODE_BY_QID = {}

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "enum": [0, 1]},
        "reason": {"type": "string"},
        "missing_evidence": {
            "type": "array",
            "items": {"type": "string"},
        },
        "spurious_claims": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["score", "reason", "missing_evidence", "spurious_claims"],
}

_REFLECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "failure_reason": {"type": "string"},
        "lesson": {"type": "string"},
        "next_strategy": {"type": "string"},
    },
    "required": ["failure_reason", "lesson", "next_strategy"],
}


def _get_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _provider() -> str:
    if _PROVIDER not in {"openai", "gemini"}:
        raise ValueError(
            f"Unsupported LLM_PROVIDER={_PROVIDER!r}. Use 'openai' or 'gemini'."
        )
    return _PROVIDER


def _get_client():
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    provider = _provider()
    if provider == "openai":
        from openai import OpenAI

        api_key = _get_env("LLM_API_KEY", "OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing OpenAI API key. Set LLM_API_KEY or OPENAI_API_KEY before using real mode."
            )

        _CLIENT = OpenAI(
            api_key=api_key,
            base_url=os.getenv("LLM_BASE_URL") or None,
        )
        return _CLIENT

    from google import genai

    api_key = _get_env("GEMINI_API_KEY", "GOOGLE_API_KEY", "LLM_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing Gemini API key. Set GEMINI_API_KEY, GOOGLE_API_KEY, or LLM_API_KEY before using real mode."
        )

    _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT

def _context_text(example: QAExample) -> str:
    return "\n\n".join(f"[{c.title}]\n{c.text}" for c in example.context)


def _estimate_tokens(text: str) -> int:
    cleaned = text.strip()
    if not cleaned:
        return 0
    return max(1, (len(cleaned) + 3) // 4)


def _build_call_metrics(
    stage: str,
    prompt_text: str,
    output_text: str,
    usage: object | None,
    latency_ms: int,
) -> CallMetrics:
    def usage_value(*names: str) -> int | None:
        if usage is None:
            return None
        for name in names:
            value = getattr(usage, name, None)
            if value is not None:
                return value
            if isinstance(usage, dict):
                value = usage.get(name)
                if value is not None:
                    return value
        return None

    prompt_tokens = usage_value("prompt_tokens", "prompt_token_count", "promptTokenCount")
    completion_tokens = usage_value(
        "completion_tokens",
        "candidates_token_count",
        "completionTokenCount",
        "candidatesTokenCount",
    )
    total_tokens = usage_value("total_tokens", "total_token_count", "totalTokenCount")
    if total_tokens is None:
        total_tokens = 0
    if prompt_tokens is None:
        prompt_tokens = 0
    if completion_tokens is None:
        completion_tokens = 0

    if total_tokens > 0 or prompt_tokens > 0 or completion_tokens > 0:
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens
        return CallMetrics(
            stage=stage,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            token_source="api_usage",
        )

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


def _chat_with_metrics(
    stage: str,
    system: str,
    user: str,
    response_format: dict[str, str] | None = None,
    json_schema: dict | None = None,
) -> tuple[str, CallMetrics]:
    if _provider() == "gemini":
        return _gemini_chat_with_metrics(stage, system, user, json_schema=json_schema)

    return _openai_chat_with_metrics(stage, system, user, response_format=response_format)


def _openai_chat_with_metrics(
    stage: str,
    system: str,
    user: str,
    response_format: dict[str, str] | None = None,
) -> tuple[str, CallMetrics]:
    request: dict = {
        "model": _MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if response_format is not None:
        request["response_format"] = response_format

    start = time.perf_counter()
    resp = _get_client().chat.completions.create(
        **request,
    )
    latency_ms = int(round((time.perf_counter() - start) * 1000))
    output_text = (resp.choices[0].message.content or "").strip()
    prompt_text = f"System:\n{system}\n\nUser:\n{user}"
    metrics = _build_call_metrics(
        stage=stage,
        prompt_text=prompt_text,
        output_text=output_text,
        usage=getattr(resp, "usage", None),
        latency_ms=latency_ms,
    )
    return output_text, metrics


def _gemini_chat_with_metrics(
    stage: str,
    system: str,
    user: str,
    json_schema: dict | None = None,
) -> tuple[str, CallMetrics]:
    from google.genai import types

    config_kwargs = {
        "system_instruction": system,
        "temperature": 0,
    }
    if json_schema is not None:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_json_schema"] = json_schema

    start = time.perf_counter()
    resp = _get_client().models.generate_content(
        model=_MODEL,
        contents=user,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    latency_ms = int(round((time.perf_counter() - start) * 1000))
    output_text = (getattr(resp, "text", None) or "").strip()
    prompt_text = f"System:\n{system}\n\nUser:\n{user}"
    metrics = _build_call_metrics(
        stage=stage,
        prompt_text=prompt_text,
        output_text=output_text,
        usage=getattr(resp, "usage_metadata", None),
        latency_ms=latency_ms,
    )
    return output_text, metrics


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except JSONDecodeError:
        return _extract_json(text)


def _chat_json_with_metrics(
    system: str,
    user: str,
    stage: str,
    json_schema: dict | None = None,
) -> tuple[dict, CallMetrics]:
    try:
        text, metrics = _chat_with_metrics(
            stage,
            system,
            user,
            response_format={"type": "json_object"},
            json_schema=json_schema,
        )
        return _parse_json(text), metrics
    except Exception:
        text, metrics = _chat_with_metrics(stage, system, user, json_schema=json_schema)
        return _parse_json(text), metrics


def _normalize_judge_payload(payload: dict) -> dict:
    score = payload.get("score")
    if isinstance(score, str) and score.isdigit():
        payload["score"] = int(score)

    payload.setdefault("reason", "")
    payload.setdefault("missing_evidence", [])
    payload.setdefault("spurious_claims", [])
    return payload


def _normalize_reflection_payload(payload: dict, judge: JudgeResult) -> dict:
    payload.setdefault("failure_reason", judge.reason)
    payload.setdefault("lesson", "Validate the missing hop before answering.")
    payload.setdefault("next_strategy", "Ground the final answer in the relevant context.")
    return payload


def _extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Model did not return JSON: {text}")
    return json.loads(text[start:end + 1])


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
    memory = "\n".join(f"- {m}" for m in reflection_memory) if reflection_memory else "None"
    user = f"""Question: {example.question}

Context:
{_context_text(example)}

Agent type: {agent_type}
Attempt: {attempt_id}

Reflection memory:
{memory}

Return only the final answer."""
    return _chat_with_metrics("actor", ACTOR_SYSTEM, user)


def evaluator(example: QAExample, answer: str) -> JudgeResult:
    return evaluator_with_metrics(example, answer)[0]


def evaluator_with_metrics(
    example: QAExample,
    answer: str,
) -> tuple[JudgeResult, CallMetrics]:
    user = f"""Question: {example.question}
Gold answer: {example.gold_answer}
Predicted answer: {answer}

Return JSON only."""
    payload, metrics = _chat_json_with_metrics(
        EVALUATOR_SYSTEM,
        user,
        "evaluator",
        json_schema=_JUDGE_SCHEMA,
    )
    payload = _normalize_judge_payload(payload)
    return JudgeResult.model_validate(payload), metrics


def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    return reflector_with_metrics(example, attempt_id, judge)[0]


def reflector_with_metrics(
    example: QAExample,
    attempt_id: int,
    judge: JudgeResult,
) -> tuple[ReflectionEntry, CallMetrics]:
    user = f"""Question: {example.question}
Context:
{_context_text(example)}

Judge reason: {judge.reason}
Missing evidence: {judge.missing_evidence}
Spurious claims: {judge.spurious_claims}

Return JSON only."""
    payload, metrics = _chat_json_with_metrics(
        REFLECTOR_SYSTEM,
        user,
        "reflector",
        json_schema=_REFLECTION_SCHEMA,
    )
    payload = _normalize_reflection_payload(payload, judge)
    payload["attempt_id"] = attempt_id
    return ReflectionEntry.model_validate(payload), metrics
