# src/retrieval/bm25.py
"""
BM25 indexing and retrieval utilities for NFCorpus / BEIR-style datasets.
"""

from typing import List, Dict, Optional, Tuple
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi
from tqdm import tqdm


def build_bm25_index(
    doc_texts: List[str],
    lowercase: bool = True,
    split_on_whitespace: bool = True,
    save_path: Optional[str | Path] = None
) -> BM25Okapi:
    """
    Build a BM25Okapi index from a list of document texts.

    Args:
        doc_texts: List of raw document texts (in the same order as doc_ids)
        lowercase: Whether to lowercase all text before tokenization
        split_on_whitespace: Simple whitespace tokenization (can be replaced later)
        save_path: If provided, saves the BM25 object + metadata using pickle

    Returns:
        Trained BM25Okapi instance
    """
    print("Tokenizing corpus for BM25...")

    if lowercase:
        tokenized_corpus = [
            text.lower().split()
            for text in tqdm(doc_texts, desc="Tokenizing")
        ]
    else:
        tokenized_corpus = [
            text.split()
            for text in tqdm(doc_texts, desc="Tokenizing")
        ]

    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save both the BM25 object and some metadata
        metadata = {
            "num_docs": len(doc_texts),
            "lowercase": lowercase,
            "tokenization": "whitespace"
        }
        
        with open(save_path, "wb") as f:
            pickle.dump({
                "bm25": bm25,
                "metadata": metadata
            }, f)
        
        print(f"BM25 index saved to: {save_path}")

    print("BM25 index ready.")
    return bm25


def load_bm25_index(load_path: str | Path) -> Tuple[BM25Okapi, Dict]:
    """
    Load a previously saved BM25 index from pickle.

    Args:
        load_path: Path to the saved .pkl file

    Returns:
        (bm25 object, metadata dict)
    """
    load_path = Path(load_path)
    if not load_path.exists():
        raise FileNotFoundError(f"BM25 index not found at {load_path}")

    print(f"Loading BM25 index from {load_path}...")
    with open(load_path, "rb") as f:
        data = pickle.load(f)
    
    bm25 = data["bm25"]
    metadata = data["metadata"]
    
    print(f"Loaded BM25 index with {metadata['num_docs']} documents")
    return bm25, metadata


def retrieve_bm25(
    bm25: BM25Okapi,
    query: str,
    doc_ids: List[str],
    top_k: int = 300,
    lowercase: bool = True
) -> List[Tuple[str, float]]:
    """
    Retrieve top-k documents for a single query using BM25.

    Args:
        bm25: Trained BM25Okapi instance
        query: Raw query string
        doc_ids: List of document ids in the same order as the corpus used for indexing
        top_k: Number of documents to return
        lowercase: Whether to lowercase the query (should match indexing)

    Returns:
        List of (doc_id, score) tuples, sorted descending by score
    """
    if lowercase:
        tokenized_query = query.lower().split()
    else:
        tokenized_query = query.split()

    scores = bm25.get_scores(tokenized_query)
    
    # Get top-k indices and scores
    top_indices = scores.argsort()[-top_k:][::-1]
    results = [(doc_ids[i], scores[i]) for i in top_indices]
    
    return results


if __name__ == "__main__":
    # Quick test / example usage
    from src.data.load_dataset import load_nfcorpus_full_corpus

    print("=== BM25 quick test ===\n")
    
    doc_id_to_text, doc_ids = load_nfcorpus_full_corpus()
    doc_texts = [doc_id_to_text[doc_id] for doc_id in doc_ids]

    bm25 = build_bm25_index(
        doc_texts,
        save_path="models/bm25_index.pkl"
    )

    # Example retrieval
    example_query = "statin use and breast cancer survival"
    top_docs = retrieve_bm25(bm25, example_query, doc_ids, top_k=5)
    
    print("Top 5 results for example query:")
    for doc_id, score in top_docs:
        print(f"{doc_id:12} | score: {score:.4f}")