# app/utils.py
import streamlit as st
from sentence_transformers import CrossEncoder
import xgboost as xgb
from rank_bm25 import BM25Okapi
import numpy as np
import pickle
import os

@st.cache_resource(show_spinner="Loading models… this takes ~10–30 seconds the first time")
def load_models_and_index():
    base_dir = os.path.dirname(os.path.dirname(__file__))  # go up from app/ → root

    # 1. Load pre-built BM25 index + corpus mapping
    bm25_path = os.path.join(base_dir, "models", "bm25_index.pkl")
    corpus_path = os.path.join(base_dir, "models", "corpus_dict.pkl")

    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)
    with open(corpus_path, "rb") as f:
        doc_id_to_text = pickle.load(f)
        doc_ids = list(doc_id_to_text.keys())

    # 2. BERT reranker
    reranker_path = os.path.join(base_dir, "models", "bert_reranker_ms_marco_mini_lm")
    reranker = CrossEncoder(reranker_path, device='cpu')   # change to 'cuda' if you have GPU locally

    # 3. XGBoost ranker (bert-only)
    xgb_path = os.path.join(base_dir, "models", "bert_xgb.json")
    ranker = xgb.XGBRanker()
    ranker.load_model(xgb_path)

    return reranker, ranker, bm25, doc_id_to_text, doc_ids


def search(query: str, bm25, doc_ids, doc_id_to_text, reranker, ranker, top_k_retrieve=150, top_k_show=10):
    if not query.strip():
        return []

    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # Get top candidates
    top_idx = np.argsort(bm25_scores)[::-1][:top_k_retrieve]
    top_doc_ids = [doc_ids[i] for i in top_idx]
    top_bm25_scores = bm25_scores[top_idx]

    # Normalize BM25 (same as training)
    if top_bm25_scores.max() > top_bm25_scores.min():
        top_bm25_norm = (top_bm25_scores - top_bm25_scores.min()) / \
                        (top_bm25_scores.max() - top_bm25_scores.min() + 1e-8)
    else:
        top_bm25_norm = top_bm25_scores.astype(float)

    # BERT scores
    pairs = [[query, doc_id_to_text[did]] for did in top_doc_ids]
    bert_scores = reranker.predict(pairs)

    # Build feature matrix (same as training)
    features = []
    for be, did in zip(bert_scores, top_doc_ids):
        doc_len = len(doc_id_to_text[did].split())
        q_len = len(tokenized_query)
        features.append([be, doc_len, q_len])          # ← only these 3

    X = np.array(features)

    # XGBoost inference (using bert_xgb model)
    xgb_scores = ranker.predict(X)

    # Build results list
    results = []
    for i, did in enumerate(top_doc_ids):
        results.append({
            "doc_id": did,
            "text": doc_id_to_text[did][:400] + "…" if len(doc_id_to_text[did]) > 400 else doc_id_to_text[did],
            "bm25_score": float(top_bm25_scores[i]),
            "bert_score": float(bert_scores[i]),
            "xgb_score": float(xgb_scores[i]),
        })

    # Sort by chosen method
    return sorted(results, key=lambda x: x["xgb_score"], reverse=True)[:top_k_show]