"""Offline evaluation harness.

Supports the evaluation methodology set out for Chapter Four: measuring
retrieval quality against relevance judgements, and comparing personalised
against non-personalised ranking over an identical candidate set.

The comparison is run with ``allow_discovery=False`` throughout, so that the
corpus is held fixed across conditions. If discovery were permitted, the index
would differ between the baseline and personalised runs and the comparison would
be confounded.

This module computes metrics. It contains no results: the numbers reported in
Chapter Four must come from running it over a real judged collection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .engine import IntelliVerse
from .models import ScoredContent


@dataclass
class Query:
    """One evaluation query with its relevance judgements.

    ``relevance`` maps ``content_id`` to a graded judgement (0 = not relevant,
    higher = more relevant). Binary judgements are the special case of using
    only 0 and 1; NDCG uses the grades directly, while precision and recall
    treat anything above ``binary_threshold`` as relevant.
    """

    query_id: str
    text: str
    relevance: Mapping[str, float] = field(default_factory=dict)
    user_id: str | None = None

    def relevant_ids(self, threshold: float = 0.5) -> set[str]:
        return {cid for cid, grade in self.relevance.items() if grade >= threshold}


# --- metrics -------------------------------------------------------------

def precision_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the top k results that are relevant."""
    if k <= 0:
        return 0.0
    top = ranked_ids[:k]
    if not top:
        return 0.0
    return sum(1 for cid in top if cid in relevant) / min(k, len(top))


def recall_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of all relevant items that appear in the top k."""
    if not relevant:
        return 0.0
    return sum(1 for cid in ranked_ids[:k] if cid in relevant) / len(relevant)


def dcg_at_k(ranked_ids: Sequence[str], grades: Mapping[str, float], k: int) -> float:
    """Discounted cumulative gain, using the standard log2 position discount."""
    return sum(
        grades.get(cid, 0.0) / math.log2(position + 2)
        for position, cid in enumerate(ranked_ids[:k])
    )


def ndcg_at_k(ranked_ids: Sequence[str], grades: Mapping[str, float], k: int) -> float:
    """DCG normalised by the best achievable ordering.

    Returns 0.0 when no relevant item exists, since perfect and worthless
    rankings are indistinguishable in that case.
    """
    ideal_order = sorted(grades.values(), reverse=True)[:k]
    ideal = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_order))
    if ideal <= 0:
        return 0.0
    return dcg_at_k(ranked_ids, grades, k) / ideal


def reciprocal_rank(ranked_ids: Sequence[str], relevant: set[str]) -> float:
    """1/rank of the first relevant result; 0 if none is retrieved."""
    for position, cid in enumerate(ranked_ids, start=1):
        if cid in relevant:
            return 1.0 / position
    return 0.0


# --- aggregation ---------------------------------------------------------

@dataclass
class QueryResult:
    query_id: str
    condition: str
    precision: dict[int, float] = field(default_factory=dict)
    recall: dict[int, float] = field(default_factory=dict)
    ndcg: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    retrieved: int = 0
    personalised: bool = False


@dataclass
class EvaluationReport:
    condition: str
    per_query: list[QueryResult] = field(default_factory=list)

    def mean(self, metric: str, k: int | None = None) -> float:
        if not self.per_query:
            return 0.0
        if metric == "mrr":
            values = [r.mrr for r in self.per_query]
        else:
            values = [getattr(r, metric).get(k, 0.0) for r in self.per_query]
        return sum(values) / len(values)

    def summary(self, cutoffs: Sequence[int] = (1, 5, 10)) -> dict[str, float]:
        out: dict[str, float] = {}
        for k in cutoffs:
            out[f"P@{k}"] = self.mean("precision", k)
            out[f"R@{k}"] = self.mean("recall", k)
            out[f"NDCG@{k}"] = self.mean("ndcg", k)
        out["MRR"] = self.mean("mrr")
        return out


class Evaluator:
    def __init__(
        self,
        engine: IntelliVerse,
        *,
        cutoffs: Sequence[int] = (1, 5, 10),
        binary_threshold: float = 0.5,
    ) -> None:
        self.engine = engine
        self.cutoffs = tuple(cutoffs)
        self.binary_threshold = binary_threshold

    def _score_ranking(
        self, query: Query, results: Sequence[ScoredContent], condition: str, personalised: bool
    ) -> QueryResult:
        ranked_ids = [r.item.content_id for r in results]
        relevant = query.relevant_ids(self.binary_threshold)
        outcome = QueryResult(
            query_id=query.query_id,
            condition=condition,
            mrr=reciprocal_rank(ranked_ids, relevant),
            retrieved=len(ranked_ids),
            personalised=personalised,
        )
        for k in self.cutoffs:
            outcome.precision[k] = precision_at_k(ranked_ids, relevant, k)
            outcome.recall[k] = recall_at_k(ranked_ids, relevant, k)
            outcome.ndcg[k] = ndcg_at_k(ranked_ids, query.relevance, k)
        return outcome

    def run(
        self,
        queries: Iterable[Query],
        *,
        personalise: bool,
        condition: str | None = None,
    ) -> EvaluationReport:
        """Evaluate one condition over a query set."""
        condition = condition or ("personalised" if personalise else "semantic-baseline")
        report = EvaluationReport(condition=condition)
        limit = max(self.cutoffs)

        for query in queries:
            response = self.engine.search(
                query.text,
                user_id=query.user_id,
                personalise=personalise,
                allow_discovery=False,  # hold the corpus fixed
                limit=limit,
            )
            report.per_query.append(
                self._score_ranking(query, response.results, condition, response.personalised)
            )
        return report

    def compare(self, queries: Sequence[Query]) -> dict[str, EvaluationReport]:
        """Run both conditions over the same queries and corpus.

        This is the central experiment: identical retrieval, ranked with and
        without the user profile, so any difference is attributable to
        personalisation alone.
        """
        return {
            "baseline": self.run(queries, personalise=False),
            "personalised": self.run(queries, personalise=True),
        }


def format_comparison(
    reports: Mapping[str, EvaluationReport], cutoffs: Sequence[int] = (1, 5, 10)
) -> str:
    """Render a comparison as a plain-text table suitable for a report."""
    summaries = {name: report.summary(cutoffs) for name, report in reports.items()}
    metrics = list(next(iter(summaries.values())).keys())
    names = list(summaries)

    header = f"{'Metric':<12}" + "".join(f"{n:>18}" for n in names) + f"{'Δ':>12}"
    lines = [header, "-" * len(header)]
    for metric in metrics:
        values = [summaries[n][metric] for n in names]
        delta = values[-1] - values[0] if len(values) > 1 else 0.0
        row = f"{metric:<12}" + "".join(f"{v:>18.4f}" for v in values) + f"{delta:>+12.4f}"
        lines.append(row)
    return "\n".join(lines)
