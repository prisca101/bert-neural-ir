# src/retrieval/cross_encoder_rerank.py
"""
Cross-Encoder reranking utilities using sentence-transformers.
Main model used: cross-encoder/ms-marco-MiniLM-L-6-v2
"""

from typing import List, Tuple, Optional
from pathlib import Path

from sentence_transformers import CrossEncoder
from tqdm import tqdm
import torch


class Reranker:
    """
    Wrapper around CrossEncoder that handles:
    - lazy loading
    - batch scoring
    - device selection
    - optional model caching / pre-loading
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = None,           # 'cuda', 'cpu', or None → auto
        max_length: int = 512,
        batch_size: int = 32,
        cache_dir: Optional[str | Path] = None
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.cache_dir = cache_dir

        self._model: CrossEncoder | None = None

    def load(self) -> None:
        """Load the model if not already loaded."""
        if self._model is not None:
            return

        print(f"Loading Cross-Encoder: {self.model_name} ...")
        print(f"Device: {self.device or 'auto'} | max_length: {self.max_length}")

        self._model = CrossEncoder(
            self.model_name,
            device=self.device,
            max_length=self.max_length,
            cache_folder=str(self.cache_dir) if self.cache_dir else None
        )
        print("Cross-Encoder loaded.")

    @property
    def model(self) -> CrossEncoder:
        """Get the loaded model (loads lazily if needed)."""
        if self._model is None:
            self.load()
        return self._model

    def score_pairs(
        self,
        queries: List[str],
        documents: List[str],
        show_progress: bool = True
    ) -> List[float]:
        """
        Score a list of (query, document) pairs.

        Args:
            queries: List of query strings (can be repeated for many docs per query)
            documents: List of document strings (same length as queries)
            show_progress: Show tqdm progress bar

        Returns:
            List of relevance scores (higher = more relevant)
        """
        if len(queries) != len(documents):
            raise ValueError("queries and documents must have the same length")

        pairs = list(zip(queries, documents))

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=show_progress
        )

        return scores.tolist()

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, float]],     # (doc_id, initial_score)
        doc_id_to_text: dict[str, str],
        top_k: int = 100,
        show_progress: bool = False
    ) -> List[Tuple[str, float]]:
        """
        Rerank a list of candidate (doc_id, bm25_score) pairs for one query.

        Args:
            query: The search query
            candidates: List of (doc_id, bm25_score) from first-stage retrieval
            doc_id_to_text: Mapping from doc_id → full document text
            top_k: How many results to return after reranking
            show_progress: Show progress bar during scoring

        Returns:
            List of (doc_id, cross_encoder_score) sorted descending
        """
        if len(candidates) == 0:
            return []

        # Get texts for the candidates
        doc_ids = [doc_id for doc_id, _ in candidates]
        texts = [doc_id_to_text.get(doc_id, "") for doc_id in doc_ids]

        # Prepare input pairs
        query_repeated = [query] * len(texts)

        # Score
        ce_scores = self.score_pairs(query_repeated, texts, show_progress=show_progress)

        # Combine with original ids and sort
        ranked = list(zip(doc_ids, ce_scores))
        ranked.sort(key=lambda x: x[1], reverse=True)

        return ranked[:top_k]


# Convenience global instance (most common usage pattern)
default_reranker = Reranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    batch_size=32,
    # device="cuda" if torch.cuda.is_available() else "cpu"   # uncomment if you have cuda
)


if __name__ == "__main__":
    # Quick test
    from src.data.load_dataset import load_nfcorpus_full_corpus

    print("=== Cross-Encoder reranker quick test ===\n")

    doc_id_to_text, _ = load_nfcorpus_full_corpus()

    # Fake candidates (normally from BM25)
    fake_candidates = [
        ("MED-10", 12.5),
        ("MED-14", 11.8),
        ("MED-301", 10.2),
    ]

    query = "statin use breast cancer survival"

    reranker = Reranker()
    # or just use: default_reranker

    top_reranked = reranker.rerank(
        query=query,
        candidates=fake_candidates,
        doc_id_to_text=doc_id_to_text,
        top_k=3,
        show_progress=True
    )

    print("Reranked results:")
    for doc_id, score in top_reranked:
        print(f"  {doc_id:12} → {score:9.4f}")