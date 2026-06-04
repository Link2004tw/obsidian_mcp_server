"""Retrieval evaluation benchmark — precision@k, recall@k, MRR.

Usage:
    python -c "from obsidian_ai.eval import run_eval; results = run_eval(); print(results)"
"""

import json
import os
from pathlib import Path

from . import ranker
from .logger import get_logger

log = get_logger(__name__)

BENCHMARK_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "eval_queries.json"


def load_benchmark(path: str | None = None) -> list[dict]:
    """Load eval queries from JSON.

    Expected format::

        [
            {
                "query": "ESP32 motor control",
                "expected": ["Project-MotorControl", "ESP32-Notes"],
                "description": "Find notes about ESP32 motor control"
            },
            ...
        ]

    Args:
        path: path to the JSON file.  Defaults to ``data/eval_queries.json``.

    Returns:
        List of query dicts.
    """
    path = path or str(BENCHMARK_PATH)
    if not os.path.isfile(path):
        log.warning("eval benchmark file not found: %s", path)
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _extract_titles(paths: list[str]) -> set[str]:
    """Extract note titles (basename without .md) from full paths."""
    return {os.path.splitext(os.path.basename(p))[0].casefold() for p in paths}


def _precision_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    """Precision@k = (relevant in top-k) / k."""
    top = retrieved[:k]
    if not top:
        return 0.0
    relevant = sum(1 for p in top if os.path.splitext(os.path.basename(p))[0].casefold() in expected)
    return relevant / k


def _recall_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    """Recall@k = (relevant in top-k) / total_expected."""
    if not expected:
        return 0.0
    top = retrieved[:k]
    relevant = sum(1 for p in top if os.path.splitext(os.path.basename(p))[0].casefold() in expected)
    return relevant / len(expected)


def _mrr(retrieved: list[str], expected: set[str]) -> float:
    """Mean Reciprocal Rank — first relevant result's rank reciprocal."""
    expected_titles = {e.casefold() for e in expected}
    for rank, path in enumerate(retrieved, start=1):
        title = os.path.splitext(os.path.basename(path))[0].casefold()
        if title in expected_titles:
            return 1.0 / rank
    return 0.0


def run_eval(
    queries: list[dict] | None = None,
    top_k: int = 5,
    use_graph: bool = False,
    use_summaries: bool = False,
    expand_entities: bool = False,
    use_community_boost: bool = False,
) -> dict:
    """Run retrieval evaluation against a benchmark set.

    Args:
        queries: list of query dicts, or ``None`` to load from ``data/eval_queries.json``.
        top_k: number of results to retrieve per query.
        use_graph: if True, enable graph traversal in search.
        use_summaries: if True, enable summary-first retrieval.
        expand_entities: if True, enable entity-relationship expansion.
        use_community_boost: if True, enable community-aware boosting.

    Returns:
        Dict with ``precision_at_k``, ``recall_at_k``, ``mrr``,
        ``per_query`` (list of per-query results), and ``total_queries``.
    """
    if queries is None:
        queries = load_benchmark()
    if not queries:
        return {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "per_query": [],
            "total_queries": 0,
        }

    per_query: list[dict] = []
    precision_sum = 0.0
    recall_sum = 0.0
    mrr_sum = 0.0

    for q in queries:
        query_text = q.get("query", "")
        expected_raw = q.get("expected", [])
        expected = {e.casefold() for e in expected_raw}
        desc = q.get("description", "")

        if not query_text or not expected:
            continue

        results = ranker.search(
            query=query_text,
            n=top_k,
            use_graph=use_graph,
            use_summaries=use_summaries,
            expand_entities=expand_entities,
            use_community_boost=use_community_boost,
        )

        retrieved = [r["path"] for r in results]

        prec = _precision_at_k(retrieved, expected, top_k)
        rec = _recall_at_k(retrieved, expected, top_k)
        mr = _mrr(retrieved, expected)

        precision_sum += prec
        recall_sum += rec
        mrr_sum += mr

        per_query.append({
            "query": query_text,
            "description": desc,
            "expected": sorted(expected),
            "retrieved": retrieved,
            "precision_at_k": round(prec, 4),
            "recall_at_k": round(rec, 4),
            "mrr": round(mr, 4),
        })

    n = len(per_query)
    return {
        "precision_at_k": round(precision_sum / n, 4) if n else 0.0,
        "recall_at_k": round(recall_sum / n, 4) if n else 0.0,
        "mrr": round(mrr_sum / n, 4) if n else 0.0,
        "per_query": per_query,
        "total_queries": n,
    }


def format_results(results: dict) -> str:
    """Format eval results as a human-readable string."""
    lines = [
        "=" * 50,
        "Retrieval Evaluation Results",
        "=" * 50,
        f"Total queries:  {results['total_queries']}",
        f"Precision@{5}:  {results['precision_at_k']:.4f}",
        f"Recall@{5}:     {results['recall_at_k']:.4f}",
        f"MRR:            {results['mrr']:.4f}",
        "-" * 50,
    ]
    for q in results.get("per_query", []):
        lines.append(f"\nQuery: {q['query']}")
        if q.get("description"):
            lines.append(f"  Desc:  {q['description']}")
        lines.append(f"  P@{5}:  {q['precision_at_k']:.4f}  R@{5}:  {q['recall_at_k']:.4f}  MRR: {q['mrr']:.4f}")
        lines.append(f"  Expected: {', '.join(q['expected'])}")
        lines.append(f"  Got:      {', '.join(q['retrieved'][:5])}")
    return "\n".join(lines)
