# src/retrieval/feature_extraction.py
"""
Feature extraction for Learning-to-Rank:
- BM25 first-stage retrieval
- Cross-Encoder second-stage reranking scores
- Simple hand-crafted features
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.data.load_dataset import load_nfcorpus_full_corpus, load_nfcorpus_splits
from src.retrieval.bm25 import retrieve_bm25
from src.retrieval.cross_encoder_rerank import default_reranker, Reranker


def build_features_for_split(
    split: str = "train",                   # "train", "test", "dev"
    bm25_top_k: int = 300,
    rerank_top_k: int = 100,
    output_dir: str | Path = "data/processed",
    save_features: bool = True,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    batch_size: int = 32,
    normalize_scores: bool = True,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Extract LTR features for one split (train / test).

    Returns:
        DataFrame with columns: qid, doc_id, bm25_score, bert_score, doc_length, query_length, label (0 if test)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Extracting features for {split.upper()} split ===")

    # ─── Load corpus & split ────────────────────────────────────────
    doc_id_to_text, doc_ids = load_nfcorpus_full_corpus()
    dataset_train, dataset_test = load_nfcorpus_splits()

    dataset = dataset_train if split == "train" else dataset_test
    queries = list(dataset.queries_iter())

    # ─── Load reranker (shared across queries) ──────────────────────
    reranker = Reranker(model_name=model_name, batch_size=batch_size)
    reranker.load()  # ensure loaded

    # ─── Prepare qrels lookup (only for train) ──────────────────────
    qrels_dict: Dict[str, Dict[str, int]] = {}
    if split == "train":
        for qrel in dataset.qrels_iter():
            qid, did, rel = qrel.query_id, qrel.doc_id, qrel.relevance
            if qid not in qrels_dict:
                qrels_dict[qid] = {}
            qrels_dict[qid][did] = rel

    # ─── Prepare result containers ──────────────────────────────────
    feature_rows = []
    group_sizes = []      # for XGBoost group info
    labels = []

    for query in tqdm(queries, desc=f"Processing {split} queries"):
        qid = query.query_id
        qtext = query.text

        # 1. BM25 first-stage retrieval
        bm25_results = retrieve_bm25(
            bm25=None,  # we'll load it inside if needed - or pass loaded one
            query=qtext,
            doc_ids=doc_ids,
            top_k=bm25_top_k,
            lowercase=True
        )

        if not bm25_results:
            continue

        top_doc_ids = [did for did, _ in bm25_results]
        top_bm25_scores = np.array([score for _, score in bm25_results])

        # Optional: min-max normalize BM25 scores per query
        if normalize_scores and len(top_bm25_scores) > 0:
            min_s, max_s = top_bm25_scores.min(), top_bm25_scores.max()
            if max_s > min_s:
                top_bm25_scores = (top_bm25_scores - min_s) / (max_s - min_s + 1e-8)

        # 2. Cross-Encoder reranking
        candidate_texts = [doc_id_to_text.get(did, "") for did in top_doc_ids]
        pairs = [[qtext, text] for text in candidate_texts]

        with torch.no_grad():
            bert_scores = reranker.model.predict(
                pairs,
                batch_size=batch_size,
                show_progress_bar=False
            )

        # 3. Build feature rows
        query_len = len(qtext.split())

        for rank, (did, bm25_score) in enumerate(zip(top_doc_ids, top_bm25_scores)):
            doc_text = doc_id_to_text.get(did, "")
            doc_len = len(doc_text.split())

            feature_rows.append({
                "qid": qid,
                "doc_id": did,
                "bm25_score": float(bm25_score),
                "bert_score": float(bert_scores[rank]),
                "doc_length": doc_len,
                "query_length": query_len,
            })

            # Label (0 if no judgment / test set)
            label = qrels_dict.get(qid, {}).get(did, 0) if split == "train" else 0
            labels.append(label)

        group_sizes.append(len(top_doc_ids))

    # ─── Build DataFrame ────────────────────────────────────────────
    df = pd.DataFrame(feature_rows)
    df["label"] = labels

    print(f"Features extracted: {len(df)} rows, {df['qid'].nunique()} queries")

    if save_features:
        parquet_path = output_dir / f"features_{split}.parquet"
        df.to_parquet(parquet_path, index=False)
        print(f"Saved features → {parquet_path}")

        # Optional: save group info for XGBoost
        group_path = output_dir / f"groups_{split}.txt"
        with open(group_path, "w") as f:
            for size in group_sizes:
                f.write(f"{size}\n")
        print(f"Saved group sizes → {group_path}")

    return df


def load_features(
    split: str = "train",
    features_dir: str | Path = "data/processed"
) -> pd.DataFrame:
    """Quick utility to load previously saved features."""
    path = Path(features_dir) / f"features_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Features not found: {path}")
    return pd.read_parquet(path)


if __name__ == "__main__":
    # Example usage
    print("=== Feature extraction demo ===\n")

    # For train (with labels)
    train_df = build_features_for_split(
        split="train",
        bm25_top_k=300,
        rerank_top_k=100,
        output_dir="data/processed",
        save_features=True
    )

    print(train_df.head(8))