# src/models/train_xgboost.py
"""
Train XGBoost Learning-to-Rank models on extracted features.
Supports multiple feature sets (BM25-only, BERT-only, combined).
Uses group-aware splitting and NDCG@10 early stopping.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

from src.data.load_dataset import load_nfcorpus_splits  # only needed for val qrels


def load_features_and_groups(
    features_dir: str | Path = "data/processed",
    split: str = "train"
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load features parquet + prepare groups and labels."""
    features_dir = Path(features_dir)
    df = pd.read_parquet(features_dir / f"features_{split}.parquet")

    groups = df.groupby("qid").size().to_numpy()
    y = df["label"].values.astype(np.float32)

    print(f"Loaded {split} features: {len(df):,} candidates, {len(groups)} queries")
    return df, groups, y


def prepare_validation_qrels(
    val_qids: set[str],
    dataset_train=None  # optional: pass if already loaded
) -> list:
    """Extract qrels only for queries present in validation set."""
    if dataset_train is None:
        _, dataset_train = load_nfcorpus_splits()  # slow — better to pass it

    val_qrels = []
    for qrel in dataset_train.qrels_iter():
        if qrel.query_id in val_qids:
            val_qrels.append(
                (qrel.query_id, qrel.doc_id, qrel.relevance)
            )
    print(f"Validation qrels extracted: {len(val_qrels)} judgments")
    return val_qrels


def train_rankers(
    df_train: pd.DataFrame,
    groups_train: np.ndarray,
    y_train: np.ndarray,
    df_val: pd.DataFrame,
    groups_val: np.ndarray,
    y_val: np.ndarray,
    output_dir: str | Path = "models",
    n_estimators: int = 800,
    early_stopping_rounds: int = 40
) -> Dict[str, xgb.XGBRanker]:
    """
    Train multiple XGBRanker models with different feature subsets.
    Returns dict of trained models.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define feature groups to compare
    feature_sets = {
        "bm25_only":   ["bm25_score", "doc_length", "query_length"],
        "bert_only":   ["bert_score", "doc_length", "query_length"],
        "combined":    ["bm25_score", "bert_score", "doc_length", "query_length"],
    }

    rankers = {}
    params_base = {
        "objective": "rank:ndcg",
        "tree_method": "hist",
        "learning_rate": 0.05,
        "max_depth": 6,
        "random_state": 42,
        "early_stopping_rounds": early_stopping_rounds,
        "eval_metric": ["ndcg@10", "map@100"],
    }

    for name, cols in feature_sets.items():
        print(f"\n{'='*40}\nTraining {name.upper()} ...")

        X_train = df_train[cols].values
        X_val   = df_val[cols].values

        ranker = xgb.XGBRanker(
            n_estimators=n_estimators,
            **params_base
        )

        ranker.fit(
            X_train, y_train,
            group=groups_train,
            eval_set=[(X_val, y_val)],
            eval_group=[groups_val],
            verbose=20
        )

        rankers[name] = ranker

        # Save model
        model_path = output_dir / f"{name}_xgb.json"
        ranker.save_model(model_path)
        print(f"Model saved → {model_path}")
        print(f"Best iteration: {ranker.best_iteration}")
        print(f"Best NDCG@10:   {ranker.best_score['validation_0']['ndcg@10']:.4f}")

    return rankers


def show_feature_importances(
    rankers: Dict[str, xgb.XGBRanker],
    feature_sets: Dict[str, List[str]]
):
    """Print readable feature importance for each model."""
    for name, ranker in rankers.items():
        print(f"\nFeature importances ─ {name.upper()}")
        importance = pd.DataFrame({
            "feature": [f"f{i}" for i in range(ranker.n_features_in_)],
            "gain": ranker.feature_importances_
        }).sort_values("gain", ascending=False)

        cols = feature_sets[name]
        importance["real_feature"] = importance["feature"].apply(
            lambda x: cols[int(x[1:])] if x.startswith("f") else x
        )

        print(importance[["real_feature", "gain"]].round(4))
        print()


if __name__ == "__main__":
    print("=== XGBoost LTR Training ===\n")

    # ─── Load data ──────────────────────────────────────────────────
    features_dir = Path("data/processed")

    df_all, groups_all, y_all = load_features_and_groups(features_dir, "train")

    # ─── Group-aware train/val split by query ───────────────────────
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_idx, val_idx = next(
        splitter.split(
            X=np.arange(len(df_all)),
            groups=df_all["qid"].values
        )
    )

    df_train = df_all.iloc[train_idx].reset_index(drop=True)
    df_val   = df_all.iloc[val_idx].reset_index(drop=True)

    groups_train = df_train.groupby("qid").size().to_numpy()
    groups_val   = df_val.groupby("qid").size().to_numpy()

    y_train = df_train["label"].values.astype(np.float32)
    y_val   = df_val["label"].values.astype(np.float32)

    print(f"Train: {len(df_train):,} candidates | {df_train['qid'].nunique()} queries")
    print(f"  Val: {len(df_val):,} candidates | {df_val['qid'].nunique()} queries")

    # ─── Train all variants ─────────────────────────────────────────
    rankers = train_rankers(
        df_train, groups_train, y_train,
        df_val,   groups_val,   y_val,
        output_dir="models",
        n_estimators=800,
        early_stopping_rounds=40
    )

    # ─── Show results ───────────────────────────────────────────────
    feature_sets = {
        "bm25_only":   ["bm25_score", "doc_length", "query_length"],
        "bert_only":   ["bert_score", "doc_length", "query_length"],
        "combined":    ["bm25_score", "bert_score", "doc_length", "query_length"],
    }

    show_feature_importances(rankers, feature_sets)

    print("\nTraining complete. Models saved in 'models/' folder.")