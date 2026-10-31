"""Information Retrieval evaluation metrics calculator class."""

import math


class IRMetricsCalculator:
    """
    Calculator for Information Retrieval ranking metrics:
    Precision@k, Recall@k, MRR (Reciprocal Rank), DCG@k, NDCG@k, and Latency percentiles.
    """

    @staticmethod
    def _count_relevant_hits(
        retrieved_ids: list[str],
        relevant_docs: dict[str, int],
        cutoff: int,
    ) -> int:
        """Count how many documents in the top-cutoff slice have relevance > 0."""
        return sum(1 for doc_id in retrieved_ids[:cutoff] if relevant_docs.get(doc_id, 0) > 0)

    def precision_at_k(
        self,
        retrieved_ids: list[str],
        relevant_docs: dict[str, int],
        k: int = 10,
    ) -> float:
        """Compute Precision@k: fraction of top-k retrieved documents that are relevant (rel > 0)."""
        if k <= 0 or not retrieved_ids:
            return 0.0

        relevant_hits = self._count_relevant_hits(retrieved_ids, relevant_docs, k)
        return round(relevant_hits / k, 4)

    def recall_at_k(
        self,
        retrieved_ids: list[str],
        relevant_docs: dict[str, int],
        k: int = 10,
    ) -> float:
        """Compute Recall@k: fraction of all relevant documents (rel > 0) found in top-k."""
        total_relevant = sum(1 for rel in relevant_docs.values() if rel > 0)
        if k <= 0 or total_relevant == 0 or not retrieved_ids:
            return 0.0

        relevant_hits = self._count_relevant_hits(retrieved_ids, relevant_docs, k)
        return round(relevant_hits / total_relevant, 4)

    def reciprocal_rank(
        self,
        retrieved_ids: list[str],
        relevant_docs: dict[str, int],
        k: int | None = None,
    ) -> float:
        """Compute Reciprocal Rank (RR): 1 / rank of the first relevant document (rel > 0)."""
        candidates = retrieved_ids[:k] if k is not None else retrieved_ids
        for idx, doc_id in enumerate(candidates, start=1):
            if relevant_docs.get(doc_id, 0) > 0:
                return round(1.0 / idx, 4)
        return 0.0

    def dcg_at_k(self, relevances: list[int], k: int = 10) -> float:
        """
        Compute Discounted Cumulative Gain at k using exponential gain:
        DCG@k = sum_{i=1..k} (2^{rel_i} - 1) / log2(i + 1)
        """
        if k <= 0 or not relevances:
            return 0.0

        score = 0.0
        for idx, rel in enumerate(relevances[:k], start=1):
            if rel > 0:
                score += ((2**rel) - 1.0) / math.log2(idx + 1.0)
        return score

    def ndcg_at_k(
        self,
        retrieved_ids: list[str],
        relevant_docs: dict[str, int],
        k: int = 10,
    ) -> float:
        """Compute Normalized Discounted Cumulative Gain (NDCG@k): DCG@k / IDCG@k."""
        if k <= 0 or not retrieved_ids or not relevant_docs:
            return 0.0

        actual_relevances = [relevant_docs.get(doc_id, 0) for doc_id in retrieved_ids[:k]]
        ideal_relevances = sorted((rel for rel in relevant_docs.values() if rel > 0), reverse=True)

        idcg = self.dcg_at_k(ideal_relevances, k=k)
        if idcg == 0.0:
            return 0.0

        dcg = self.dcg_at_k(actual_relevances, k=k)
        return round(dcg / idcg, 4)

    @staticmethod
    def percentile(values: list[float], p: float) -> float:
        """Compute linear-interpolated percentile p (in [0, 100]) for a list of floats."""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        if len(sorted_vals) == 1:
            return round(sorted_vals[0], 2)

        rank = (p / 100.0) * (len(sorted_vals) - 1)
        low = int(math.floor(rank))
        high = int(math.ceil(rank))
        if low == high:
            return round(sorted_vals[low], 2)
        weight = rank - low
        interpolated = sorted_vals[low] * (1.0 - weight) + sorted_vals[high] * weight
        return round(interpolated, 2)
