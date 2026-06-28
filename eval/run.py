"""Eval harness — Ragas evaluation for the AI PDF Assistant RAG pipeline.

Metrics (Ragas standard):
  faithfulness      — is the answer grounded in the retrieved context?
  answer_relevancy  — does the answer address the question?
  context_precision — are retrieved chunks ranked precisely for the question?
  context_recall    — do the chunks cover the reference answer?

LLM judge: Groq llama-3.3-70b via its OpenAI-compatible API endpoint.
Embeddings: local bge-small-en-v1.5 (already in the project venv).

Workaround: ragas 0.4.x unconditionally imports ChatVertexAI from a module
removed in langchain-community 0.3+. A stub is injected before ragas loads.
This stub is never used — it only satisfies the import.

Usage:
    python -m eval.run                   # run all 30 questions
    python -m eval.run --limit 5         # quick smoke-test
    python -m eval.run --save-baseline   # persist results as baseline
    API_URL=https://... python -m eval.run
"""
from __future__ import annotations

# ── Ragas compatibility shim ──────────────────────────────────────────────────
# ragas 0.4.x has a hard import of ChatVertexAI from a path removed in
# langchain-community 0.3+. Inject a stub so the import resolves without
# pulling in the full Google Cloud VertexAI stack.
import sys
from types import ModuleType as _ModuleType

_vertexai_stub = _ModuleType("langchain_community.chat_models.vertexai")
class _ChatVertexAIStub:  # noqa: N801
    pass
_vertexai_stub.ChatVertexAI = _ChatVertexAIStub  # type: ignore[attr-defined]
sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_stub
# ── end shim ──────────────────────────────────────────────────────────────────

import argparse
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any

import httpx
from datasets import Dataset
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (  # old-style API — supports LangchainLLMWrapper
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

EVAL_DIR = Path(__file__).parent
GOLD_FILE = EVAL_DIR / "gold.jsonl"
BASELINE_FILE = EVAL_DIR / "baseline.json"

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

# Judge LLM: prefer OpenRouter (higher limits); fall back to Groq
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if OPENROUTER_API_KEY:
    JUDGE_API_KEY = OPENROUTER_API_KEY
    JUDGE_BASE_URL = "https://openrouter.ai/api/v1"
    JUDGE_MODEL = "meta-llama/llama-3.3-70b-instruct"
else:
    JUDGE_API_KEY = GROQ_API_KEY
    JUDGE_BASE_URL = "https://api.groq.com/openai/v1"
    JUDGE_MODEL = "llama-3.3-70b-versatile"

REGRESSION_THRESHOLD = 0.10   # 10 pp drop triggers a warning
LATENCY_REGRESSION = 0.20     # 20% slowdown triggers a warning


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def query_api(question: str, timeout: int = 60) -> dict[str, Any]:
    """Call POST /query and return answer + citation snippets + wall-clock time."""
    start = time.perf_counter()
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(f"{API_URL}/query", json={"query": question})
        resp.raise_for_status()
    elapsed = round(time.perf_counter() - start, 3)
    data = resp.json()
    return {
        "answer": data.get("answer", ""),
        "contexts": [c["snippet"] for c in data.get("citations", [])],
        "latency_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def _mean(scores: list | float) -> float:
    """Average a list of scores; handle NaN and pass-through floats."""
    import math
    if isinstance(scores, (int, float)):
        return round(float(scores), 4)
    valid = [s for s in scores if s is not None and not math.isnan(s)]
    return round(sum(valid) / len(valid), 4) if valid else 0.0


def run_eval(limit: int | None = None) -> dict[str, Any]:
    if not JUDGE_API_KEY:
        print("ERROR: set OPENROUTER_API_KEY or GROQ_API_KEY")
        raise SystemExit(1)

    gold = [json.loads(ln) for ln in GOLD_FILE.read_text().splitlines() if ln.strip()]
    if limit:
        gold = gold[:limit]

    provider = "OpenRouter" if OPENROUTER_API_KEY else "Groq"
    print(f"Running Ragas eval on {len(gold)} questions → {API_URL}")
    print(f"Judge: {provider} / {JUDGE_MODEL}")
    print("-" * 60)

    # --- Set up Ragas LLM judge ---
    # Old-style ragas.metrics API (accepts LangchainLLMWrapper).
    # ragas.metrics.collections requires InstructorLLM — incompatible here.
    groq_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=JUDGE_MODEL,
            api_key=JUDGE_API_KEY,  # type: ignore[arg-type]
            base_url=JUDGE_BASE_URL,
            temperature=0.0,
        )
    )
    # Local bge-small for answer_relevancy embeddings — no extra API key needed
    local_embeddings = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # --- Query the RAG pipeline for each gold question ---
    rows: list[dict] = []
    latencies: list[float] = []

    for i, item in enumerate(gold, 1):
        q = item["question"]
        ref = item["reference_answer"]
        print(f"[{i:02d}/{len(gold)}] {q[:70]}...")

        try:
            result = query_api(q)
        except Exception as exc:
            print(f"         API ERROR: {exc}")
            rows.append({
                "question": q,
                "answer": "",
                "contexts": [],
                "reference": ref,
                "latency_s": 0.0,
                "error": True,
            })
            continue

        answer = result["answer"]
        contexts = result["contexts"]
        latency = result["latency_s"]
        latencies.append(latency)

        no_answer = "don't have enough information" in answer.lower()
        status = "NO_ANSWER" if no_answer else f"latency={latency:.1f}s ctx={len(contexts)}"
        print(f"         → {status}")

        rows.append({
            "question": q,
            "answer": answer,
            "contexts": contexts if contexts else [""],
            "reference": ref,
            "latency_s": latency,
            "error": False,
        })

    # --- Build Ragas dataset (only answered questions) ---
    answered = [r for r in rows if not r.get("error") and r["contexts"] != [""]]
    if not answered:
        print("No answered questions — cannot run Ragas evaluation.")
        raise SystemExit(1)

    dataset = Dataset.from_list([
        {
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["contexts"],
            "reference": r["reference"],
        }
        for r in answered
    ])

    print(f"\nRunning Ragas on {len(answered)} answered questions…")
    ragas_result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=groq_llm,
        embeddings=local_embeddings,
    )

    # --- Aggregate ---
    sorted_lat = sorted(latencies)
    n = len(sorted_lat)
    summary: dict[str, Any] = {
        "n_questions": len(gold),
        "n_answered": len(answered),
        "no_answer_rate": round((len(gold) - len(answered)) / len(gold), 4),
        "faithfulness": _mean(ragas_result["faithfulness"]),
        "answer_relevancy": _mean(ragas_result["answer_relevancy"]),
        "context_precision": _mean(ragas_result["context_precision"]),
        "context_recall": _mean(ragas_result["context_recall"]),
        "latency_p50_s": round(sorted_lat[n // 2], 3) if n else 0.0,
        "latency_p95_s": round(sorted_lat[int(n * 0.95)], 3) if n else 0.0,
        "details": rows,
    }
    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_summary(s: dict[str, Any]) -> None:
    print()
    print("=" * 60)
    print("RAGAS EVAL RESULTS")
    print("=" * 60)
    print(f"  Questions:        {s['n_questions']}")
    print(f"  Answered:         {s['n_answered']}")
    print(f"  No-answer rate:   {s['no_answer_rate']:.1%}")
    print()
    print(f"  Faithfulness:     {s['faithfulness']:.3f}")
    print(f"  Answer relevancy: {s['answer_relevancy']:.3f}")
    print(f"  Context precision:{s['context_precision']:.3f}")
    print(f"  Context recall:   {s['context_recall']:.3f}")
    print()
    print(f"  Latency p50:      {s['latency_p50_s']:.2f}s")
    print(f"  Latency p95:      {s['latency_p95_s']:.2f}s")
    print("=" * 60)


def compare_to_baseline(current: dict[str, Any]) -> list[str]:
    if not BASELINE_FILE.exists():
        return []
    baseline = json.loads(BASELINE_FILE.read_text())
    warnings_list = []
    for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        delta = current[metric] - baseline.get(metric, 0.0)
        if delta < -REGRESSION_THRESHOLD:
            warnings_list.append(
                f"REGRESSION {metric}: {baseline[metric]:.3f} → {current[metric]:.3f} (Δ{delta:+.3f})"
            )
    for metric in ("latency_p50_s", "latency_p95_s"):
        base_val = baseline.get(metric, 0.0)
        curr_val = current[metric]
        if base_val > 0 and (curr_val - base_val) / base_val > LATENCY_REGRESSION:
            warnings_list.append(
                f"LATENCY REGRESSION {metric}: {base_val:.3f}s → {curr_val:.3f}s "
                f"({(curr_val - base_val) / base_val * 100:+.0f}%)"
            )
    return warnings_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--label", type=str, default=None, help="Optional label suffix for the results file")
    args = parser.parse_args()

    summary = run_eval(limit=args.limit)
    print_summary(summary)

    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    label = f"_{args.label}" if args.label else ""
    results_file = EVAL_DIR / "runs" / f"{ts}{label}.json"
    results_file.parent.mkdir(exist_ok=True)
    results_file.write_text(json.dumps(summary, indent=2))
    print(f"\nFull results → {results_file}")

    regressions = compare_to_baseline(summary)
    if regressions:
        print("\nREGRESSIONS DETECTED:")
        for r in regressions:
            print(f"  {r}")

    if args.save_baseline:
        baseline_data = {k: v for k, v in summary.items() if k != "details"}
        BASELINE_FILE.write_text(json.dumps(baseline_data, indent=2))
        print(f"Baseline saved → {BASELINE_FILE}")

    if regressions:
        raise SystemExit(1)
