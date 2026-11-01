"""Streamlit dashboard for side-by-side search comparison and benchmark analytics."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from elasticsearch import AsyncElasticsearch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import get_settings
from src.repositories.jobs_repository import ElasticsearchJobRepository
from src.schemas.evaluation import BenchmarkQuery, StrategyEvaluationReport
from src.schemas.search import JobHit, SearchRequest, SearchResponse
from src.services.embedder import JobEmbedder
from src.services.fusion import (
    RankFusionStrategy,
    ReciprocalRankFusion,
    WeightedScoreFusion,
)
from src.services.ingestion.seniority import detect_seniority, get_seniority_badge
from src.services.ingestion.skill_extractor import SKILL_PATTERNS
from src.services.search_service import SearchService

EVALUATION_RESULTS_PATH = PROJECT_ROOT / "data" / "evaluation_results.json"
GROUND_TRUTH_PATH = PROJECT_ROOT / "data" / "ground_truth.json"

COMMON_CITIES = [
    "All",
    "Berlin",
    "München",
    "Hamburg",
    "Stuttgart",
    "Frankfurt am Main",
    "Köln",
    "Erlangen",
    "Karlsruhe, Baden",
    "Dresden",
    "Nürnberg, Mittelfranken",
]


@st.cache_resource(show_spinner="Loading embedding model...")
def get_cached_embedder() -> JobEmbedder:
    """Load and warm up the SentenceTransformer model once per process."""
    embedder = JobEmbedder()
    embedder.embed_query("warmup")
    return embedder


def load_evaluation_reports(path: Path = EVALUATION_RESULTS_PATH) -> list[StrategyEvaluationReport]:
    """Load precomputed benchmark evaluation reports from disk."""
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [StrategyEvaluationReport.model_validate(item) for item in raw]


def load_benchmark_queries(path: Path = GROUND_TRUTH_PATH) -> list[BenchmarkQuery]:
    """Load benchmark queries from ground_truth.json."""
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [BenchmarkQuery.model_validate(item) for item in raw]


def build_metrics_dataframe(reports: list[StrategyEvaluationReport]) -> pd.DataFrame:
    """Convert StrategyEvaluationReport list into a summary DataFrame."""
    rows: list[dict[str, Any]] = []
    for r in reports:
        rows.append(
            {
                "Strategy": r.strategy,
                f"NDCG@{r.k}": r.mean_ndcg_at_k,
                "MRR": r.mean_mrr,
                f"Recall@{r.k}": r.mean_recall_at_k,
                f"Precision@{r.k}": r.mean_precision_at_k,
                "Mean (ms)": r.mean_latency_ms,
                "p50 (ms)": r.p50_latency_ms,
                "p95 (ms)": r.p95_latency_ms,
            }
        )
    return pd.DataFrame(rows)


class ArenaDashboard:
    """Interactive UI for comparing retrieval strategies and inspecting IR evaluation metrics."""

    def __init__(self, embedder: JobEmbedder) -> None:
        self.embedder = embedder
        self.settings = get_settings()

    async def _run_three_strategies(
        self,
        request: SearchRequest,
        fusion_strategy: RankFusionStrategy,
    ) -> tuple[SearchResponse, SearchResponse, SearchResponse]:
        es_cfg = self.settings.elasticsearch
        client = AsyncElasticsearch(
            hosts=[es_cfg.url],
            request_timeout=es_cfg.timeout_seconds,
            max_retries=es_cfg.max_retries,
            retry_on_timeout=True,
        )
        try:
            repo = ElasticsearchJobRepository(client=client)
            service = SearchService(
                repository=repo,
                embedder=self.embedder,
                fusion_strategy=fusion_strategy,
            )
            bm25_res, knn_res, hybrid_res = await asyncio.gather(
                service.search_bm25(request),
                service.search_knn(request),
                service.search_hybrid(request, fusion_strategy=fusion_strategy),
            )
            return bm25_res, knn_res, hybrid_res
        finally:
            await client.close()

    def execute_arena_search(
        self,
        request: SearchRequest,
        fusion_strategy: RankFusionStrategy,
    ) -> tuple[SearchResponse, SearchResponse, SearchResponse]:
        """Run BM25, kNN, and Hybrid search concurrently inside an event loop."""
        return asyncio.run(self._run_three_strategies(request, fusion_strategy))

    @staticmethod
    def _render_hit_card(hit: JobHit, relevant_docs: dict[str, int] | None = None) -> None:
        location = hit.city or "Unspecified"
        remote_badge = " | Remote" if hit.is_remote else ""
        rel_badge = ""
        if relevant_docs is not None and hit.id in relevant_docs:
            grade = relevant_docs[hit.id]
            rel_badge = f" | Grade: {grade}/3"

        seniority_code = getattr(hit, "seniority", None) or detect_seniority(hit.title, hit.description)
        seniority_badge = get_seniority_badge(seniority_code)
        apply_url = hit.source_url or f"https://www.arbeitsagentur.de/jobsuche/jobdetail/{hit.id}"

        with st.container(border=True):
            st.markdown(f"**#{hit.rank} {hit.title}**")
            st.caption(
                f"{seniority_badge} | {hit.company or 'Unknown Company'} | {location}{remote_badge} "
                f"| Score: `{hit.score:.4f}`{rel_badge}"
            )
            if hit.skills:
                skills_preview = ", ".join(hit.skills[:8])
                st.markdown(f"Skills: `{skills_preview}`")
            if hit.details and "ranks" in hit.details:
                ranks = hit.details["ranks"]
                bm25_r = ranks.get("bm25", "-")
                knn_r = ranks.get("knn", "-")
                st.caption(f"Source ranks: BM25 #{bm25_r}, kNN #{knn_r}")

            st.link_button("🔗 Apply on Arbeitsagentur", apply_url, use_container_width=True)

            with st.expander("📄 Full job description"):
                if hit.published_date:
                    st.caption(f"Published date: {hit.published_date}")
                st.caption(f"Seniority level: **{seniority_badge}**")
                if hit.skills:
                    st.markdown("**All extracted skills:** " + ", ".join(hit.skills))
                st.markdown("---")
                if hit.description:
                    st.write(hit.description)
                else:
                    st.info("No detailed description available.")

    def render_arena_tab(self, benchmark_queries: list[BenchmarkQuery]) -> None:
        """Render the side-by-side search comparison interface."""
        st.subheader("Side-by-Side Search Comparison")

        preset_options = ["Custom query"] + [
            f"{q.id}: {q.query}" for q in benchmark_queries
        ]
        selected_preset = st.selectbox("Benchmark preset (optional)", options=preset_options)

        active_bq: BenchmarkQuery | None = None
        default_query = "Senior Python Backend Engineer FastAPI PostgreSQL"
        if selected_preset and selected_preset != "Custom query":
            qid = selected_preset.split(":", 1)[0]
            active_bq = next((q for q in benchmark_queries if q.id == qid), None)
            if active_bq:
                default_query = active_bq.query

        col_q, col_limit = st.columns([4, 1])
        with col_q:
            query_text = st.text_input("Search query", value=default_query)
        with col_limit:
            limit = int(st.number_input("Top-K", min_value=1, max_value=50, value=10))

        col_f1, col_f2, col_f3, col_f4 = st.columns([2, 1, 2, 3])
        with col_f1:
            default_city_idx = 0
            if active_bq and active_bq.city in COMMON_CITIES:
                default_city_idx = COMMON_CITIES.index(active_bq.city)
            city_choice = st.selectbox("City filter", options=COMMON_CITIES, index=default_city_idx)
        with col_f2:
            remote_options = ["Any", "Remote only", "On-site only"]
            default_remote_idx = 1 if (active_bq and active_bq.is_remote is True) else 0
            remote_choice = st.selectbox("Work mode", options=remote_options, index=default_remote_idx)
        with col_f3:
            seniority_options = [
                "All levels",
                "Junior (inkl. Berufseinsteiger)",
                "Mid-Level",
                "Senior",
                "Lead / Architect",
            ]
            seniority_choice = st.selectbox("Seniority level", options=seniority_options, index=0)
        with col_f4:
            all_skills = sorted(SKILL_PATTERNS.keys())
            default_skills = (
                [s for s in (active_bq.skills or []) if s in all_skills] if active_bq else []
            )
            selected_skills = st.multiselect("Required skills", options=all_skills, default=default_skills)

        col_fusion, col_param = st.columns([2, 3])
        with col_fusion:
            fusion_mode = st.radio(
                "Hybrid Fusion Strategy",
                options=["Reciprocal Rank Fusion (RRF)", "Weighted Score Fusion"],
                horizontal=True,
            )
        with col_param:
            if fusion_mode == "Reciprocal Rank Fusion (RRF)":
                rrf_k = int(st.slider("RRF rank constant (k)", min_value=1, max_value=120, value=60))
                fusion_strategy: RankFusionStrategy = ReciprocalRankFusion(k=rrf_k)
            else:
                bm25_weight = st.slider("BM25 weight", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
                knn_weight = round(1.0 - bm25_weight, 2)
                if bm25_weight == 0.0 and knn_weight == 0.0:
                    bm25_weight = 0.5
                    knn_weight = 0.5
                fusion_strategy = WeightedScoreFusion(
                    weights={"bm25": max(bm25_weight, 0.01), "knn": max(knn_weight, 0.01)}
                )

        if not query_text.strip():
            st.warning("Enter a search query to run comparison.")
            return

        is_remote_val: bool | None = None
        if remote_choice == "Remote only":
            is_remote_val = True
        elif remote_choice == "On-site only":
            is_remote_val = False

        seniority_map = {
            "Junior (inkl. Berufseinsteiger)": "junior",
            "Mid-Level": "mid",
            "Senior": "senior",
            "Lead / Architect": "lead",
        }
        seniority_val = seniority_map.get(seniority_choice) if seniority_choice else None

        request = SearchRequest(
            query=query_text.strip(),
            limit=limit,
            city=None if city_choice == "All" else city_choice,
            is_remote=is_remote_val,
            skills=selected_skills or None,
            seniority=seniority_val,
        )

        try:
            bm25_res, knn_res, hybrid_res = self.execute_arena_search(request, fusion_strategy)
        except Exception as exc:
            st.error(f"Failed to query Elasticsearch: {exc}")
            return

        rel_docs = active_bq.relevant_docs if active_bq else None
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(f"### BM25 (`{bm25_res.took_ms:.1f} ms`)")
            st.caption(f"Total hits: {bm25_res.total}")
            for hit in bm25_res.items:
                self._render_hit_card(hit, rel_docs)

        with c2:
            st.markdown(f"### Vector kNN (`{knn_res.took_ms:.1f} ms`)")
            st.caption(f"Total hits: {knn_res.total}")
            for hit in knn_res.items:
                self._render_hit_card(hit, rel_docs)

        with c3:
            st.markdown(f"### Hybrid `{hybrid_res.strategy}` (`{hybrid_res.took_ms:.1f} ms`)")
            st.caption(f"Returned fused hits: { len(hybrid_res.items) }")
            for hit in hybrid_res.items:
                self._render_hit_card(hit, rel_docs)

    def render_benchmark_tab(
        self,
        reports: list[StrategyEvaluationReport],
        benchmark_queries: list[BenchmarkQuery],
    ) -> None:
        """Render evaluation benchmark tables, charts, and per-query breakdowns."""
        st.subheader("Information Retrieval Benchmark Results")

        if not reports:
            st.info(
                "No evaluation report found at `data/evaluation_results.json`. "
                "Run `uv run python scripts/evaluate.py` first."
            )
            return

        df = build_metrics_dataframe(reports)
        by_strat = {r.strategy: r for r in reports}

        if "hybrid_rrf" in by_strat and "bm25" in by_strat:
            rrf_rep = by_strat["hybrid_rrf"]
            bm25_rep = by_strat["bm25"]
            ndcg_lift = ((rrf_rep.mean_ndcg_at_k - bm25_rep.mean_ndcg_at_k) / bm25_rep.mean_ndcg_at_k) * 100
            mrr_lift = ((rrf_rep.mean_mrr - bm25_rep.mean_mrr) / bm25_rep.mean_mrr) * 100

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Queries Evaluated", rrf_rep.num_queries)
            m2.metric(f"Best NDCG@{rrf_rep.k} (RRF)", f"{rrf_rep.mean_ndcg_at_k:.4f}", f"{ndcg_lift:+.1f}% vs BM25")
            m3.metric("Best MRR (RRF)", f"{rrf_rep.mean_mrr:.4f}", f"{mrr_lift:+.1f}% vs BM25")
            m4.metric(f"Best Recall@{rrf_rep.k} (RRF)", f"{rrf_rep.mean_recall_at_k:.4f}")

        st.dataframe(df, use_container_width=True, hide_index=True)

        k_val = reports[0].k
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("#### Ranking Quality Metrics")
            quality_df = df.set_index("Strategy")[
                [f"NDCG@{k_val}", "MRR", f"Recall@{k_val}", f"Precision@{k_val}"]
            ]
            st.bar_chart(quality_df)

        with chart_col2:
            st.markdown("#### Latency Distribution (ms)")
            latency_df = df.set_index("Strategy")[["Mean (ms)", "p50 (ms)", "p95 (ms)"]]
            st.bar_chart(latency_df)

        st.subheader("Per-Query Breakdown")
        query_options = [f"{q.id}: {q.query}" for q in benchmark_queries]
        if not query_options:
            return

        selected_q = st.selectbox("Select benchmark query", options=query_options)
        if not selected_q:
            return
        selected_qid = selected_q.split(":", 1)[0]

        query_rows: list[dict[str, Any]] = []
        for r in reports:
            q_res = next((qr for qr in r.query_results if qr.query_id == selected_qid), None)
            if q_res:
                query_rows.append(
                    {
                        "Strategy": r.strategy,
                        f"NDCG@{r.k}": q_res.ndcg_at_k,
                        "MRR": q_res.mrr,
                        f"Recall@{r.k}": q_res.recall_at_k,
                        f"Precision@{r.k}": q_res.precision_at_k,
                        "Latency (ms)": q_res.took_ms,
                        "Top Retrieved IDs": ", ".join(q_res.retrieved_ids[:5]),
                    }
                )

        if query_rows:
            st.dataframe(pd.DataFrame(query_rows), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="IR-Arena | Hybrid Search Benchmark",
        layout="wide",
    )
    st.title("IR-Arena: Hybrid Search & Retrieval Evaluation")
    st.caption(
        "Comparing Lexical BM25, Semantic Vector kNN, and Hybrid Rank Fusion on German IT job listings"
    )

    embedder = get_cached_embedder()
    dashboard = ArenaDashboard(embedder=embedder)
    benchmark_queries = load_benchmark_queries()
    reports = load_evaluation_reports()

    tab_arena, tab_bench = st.tabs(["Search Arena", "Benchmark Analytics"])
    with tab_arena:
        dashboard.render_arena_tab(benchmark_queries)
    with tab_bench:
        dashboard.render_benchmark_tab(reports, benchmark_queries)


if __name__ == "__main__":
    main()
