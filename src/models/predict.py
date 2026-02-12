# src/models/predict.py
"""
Generate predictions for all model variants and evaluate them using ir_measures.
Variants:
- BM25 baseline
- Cross-Encoder reranker (on BM25 top-k)
- XGBoost reranked with BM25 features
- XGBoost reranked with BERT features
- XGBoost reranked with combined features
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import xgboost as xgb
import ir_measures
from ir_measures import nDCG, R, AP, RR

from src.data.load_dataset import load_nfcorpus_full_corpus, load_nfcorpus_splits
from src.retrieval.bm25 import load_bm25_index, retrieve_bm25
from src.retrieval.cross_encoder_rerank import Reranker
from src.retrieval.feature_extraction import load_features  # optional, only if reusing precomputed


def load_models(
    model_dir: str | Path = "models"
) -> Dict[str, Optional[object]]:
    """Load trained XGBoost rankers."""
    model_dir = Path(model_dir)
    
    models = {
        "BM25":           None,
        "BERT_rerank":    None,
        "BM25_XGB":       xgb.XGBRanker().load_model(model_dir / "bm25_only_xgb.json"),
        "BERT_XGB":       xgb.XGBRanker().load_model(model_dir / "bert_only_xgb.json"),
        "Combined_XGB":   xgb.XGBRanker().load_model(model_dir / "combined_xgb.json"),
    }
    print("Loaded XGBoost models:", [k for k in models if models[k] is not None])
    return models


def generate_run_on_split(
    split: str = "test",                # "val" or "test"
    bm25_top_k: int = 300,
    output_dir: str | Path = "results/runs",
    model_dir: str | Path = "models",
    batch_size: int = 32,
    use_precomputed_features: bool = False
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Generate prediction runs for all model variants on a given split.

    Returns:
        runs: dict[method → qid → docid → score]
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Generating predictions for {split.upper()} split ===")

    # ─── Load shared resources ──────────────────────────────────────
    doc_id_to_text, doc_ids = load_nfcorpus_full_corpus()
    _, dataset_test = load_nfcorpus_splits()
    dataset = dataset_test if split == "test" else None  # for val you'd need filtered train

    if split == "val":
        raise NotImplementedError("For val set use pre-split features or adapt logic")

    queries = list(dataset.queries_iter())
    qrels = {q.query_id: {} for q in queries}  # we'll fill only if needed for metrics

    bm25, _ = load_bm25_index("models/bm25_index.pkl")
    reranker = Reranker(batch_size=batch_size)
    reranker.load()

    rankers = load_models(model_dir)

    # Feature columns per method
    feature_sets = {
        "BM25":           None,
        "BERT_rerank":    None,
        "BM25_XGB":       ["bm25_score", "doc_length", "query_length"],
        "BERT_XGB":       ["bert_score", "doc_length", "query_length"],
        "Combined_XGB":   ["bm25_score", "bert_score", "doc_length", "query_length"],
    }

    runs = defaultdict(lambda: defaultdict(dict))  # method → qid → did → score

    for query in tqdm(queries, desc=f"Predicting {split} queries"):
        qid = query.query_id
        qtext = query.text

        # 1. BM25 first-stage
        bm25_results = retrieve_bm25(
            bm25=bm25,
            query=qtext,
            doc_ids=doc_ids,
            top_k=bm25_top_k
        )

        if not bm25_results:
            continue

        top_doc_ids = [did for did, _ in bm25_results]
        top_bm25_raw = np.array([score for _, score in bm25_results])

        # Normalize BM25 scores (same as training)
        if len(top_bm25_raw) > 1 and top_bm25_raw.max() > top_bm25_raw.min():
            top_bm25 = (top_bm25_raw - top_bm25_raw.min()) / (top_bm25_raw.max() - top_bm25_raw.min() + 1e-8)
        else:
            top_bm25 = top_bm25_raw.astype(float)

        # 2. Cross-Encoder scores
        candidate_texts = [doc_id_to_text.get(did, "") for did in top_doc_ids]
        pairs = [[qtext, text] for text in candidate_texts]
        bert_scores = reranker.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

        # 3. Build mini feature DF for this query
        feat_df = pd.DataFrame({
            "bm25_score": top_bm25,
            "bert_score": bert_scores,
            "doc_length": [len(text.split()) for text in candidate_texts],
            "query_length": len(qtext.split()),
        }, index=top_doc_ids)

        # 4. Generate scores for each method
        for method in feature_sets:
            if method in ["BM25", "BERT_rerank"]:
                scores = feat_df["bm25_score"] if method == "BM25" else feat_df["bert_score"]
            else:
                X = feat_df[feature_sets[method]].values
                scores = rankers[method].predict(X)

            for did, score in zip(top_doc_ids, scores):
                runs[method][qid][did] = float(score)

    # Optional: save runs in TREC format
    for method in runs:
        run_path = output_dir / f"run_{method}_{split}.trec"
        with open(run_path, "w") as f:
            for qid in sorted(runs[method]):
                ranked = sorted(runs[method][qid].items(), key=lambda x: x[1], reverse=True)
                for rank, (did, score) in enumerate(ranked, 1):
                    f.write(f"{qid} Q0 {did} {rank} {score:.6f} {method}\n")
        print(f"Saved TREC run: {run_path}")

    return runs


def evaluate_runs(
    runs: Dict[str, Dict[str, Dict[str, float]]],
    split: str = "test",
    metrics: list = [nDCG@10, R@100]
):
    """Compute metrics for all runs using ir_measures."""
    _, dataset_test = load_nfcorpus_splits()
    qrels = list(dataset_test.qrels_iter())

    print(f"\n{'='*50}\nEvaluation results on {split.upper()} ({len(qrels)} qrels)\n")

    results = {}
    for method in runs:
        run = runs[method]
        measures = ir_measures.iter_calc(metrics, qrels, run)
        agg = ir_measures.Aggregate(measures).mean()
        results[method] = agg

        print(f"{method:15}", end="")
        for m in metrics:
            print(f"  {m}={agg[m]:.4f}", end="")
        print()

    # Optional: save results
    with open(f"results/metrics_{split}.json", "w") as f:
        json.dump({str(k): {str(m): v for m, v in v.items()} for k, v in results.items()}, f, indent=2)

    return results


if __name__ == "__main__":
    print("=== Prediction & Evaluation ===\n")

    # Generate predictions on test set
    test_runs = generate_run_on_split(
        split="test",
        bm25_top_k=300,
        output_dir="results/runs",
        model_dir="models"
    )

    # Evaluate
    evaluate_runs(test_runs, split="test")