# src/data/load_dataset.py
"""
Utilities for loading and preparing the BEIR NFCorpus dataset.
"""

import random
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import ir_datasets


def load_nfcorpus_full_corpus() -> Tuple[Dict[str, str], List[str]]:
    """
    Load the full NFCorpus document collection (same for train/test/dev).
    
    Returns:
        doc_id_to_text: dict mapping doc_id → full text
        doc_ids: list of all document ids (order preserved)
    """
    print("Loading NFCorpus full corpus...")
    all_docs = list(ir_datasets.load("beir/nfcorpus").docs_iter())
    
    doc_id_to_text = {doc.doc_id: doc.text for doc in all_docs}
    doc_ids = list(doc_id_to_text.keys())
    
    print(f"Corpus size: {len(doc_ids)} documents")
    return doc_id_to_text, doc_ids


def load_nfcorpus_splits() -> Tuple[ir_datasets.Dataset, ir_datasets.Dataset]:
    """
    Load train and test splits of NFCorpus.
    
    Returns:
        (dataset_train, dataset_test)
    """
    print("Loading NFCorpus train/test splits...")
    dataset_train = ir_datasets.load("beir/nfcorpus/train")
    dataset_test = ir_datasets.load("beir/nfcorpus/test")
    return dataset_train, dataset_test


def filter_queries_by_relevance(
    qrels_iter,
    min_rel: int = 5,
    max_rel: int = 64,
    random_seed: int = 42
) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """
    Filter queries based on number of relevant documents.
    Downsample queries that have more than max_rel positives.
    
    Args:
        qrels_iter: iterator over qrels (from dataset.qrels_iter())
        min_rel:   minimum number of relevant docs required
        max_rel:   maximum number of relevant docs (downsample if more)
        random_seed: for reproducible downsampling
    
    Returns:
        valid_qids: set of kept query ids
        filtered_qrels: dict query_id → set of relevant doc_ids
    """
    random.seed(random_seed)
    
    # 1. Collect all relevant docs per query
    qrels_dict = defaultdict(list)
    for qrel in qrels_iter:
        qrels_dict[qrel.query_id].append(qrel.doc_id)
    
    # 2. Filter
    valid_qids = set()
    filtered_qrels = {}
    downsampled = 0
    
    for qid, docs in qrels_dict.items():
        if len(docs) >= min_rel:
            valid_qids.add(qid)
            
            if len(docs) > max_rel:
                docs = random.sample(docs, max_rel)
                downsampled += 1
                
            filtered_qrels[qid] = set(docs)   # set → fast membership check later
    
    print(f"Kept {len(valid_qids)} out of {len(qrels_dict)} queries")
    print(f"Downsampled {downsampled} queries to <= {max_rel} positives")
    
    return valid_qids, filtered_qrels


def load_filtered_train_data(
    min_rel: int = 5,
    max_rel: int = 64,
    random_seed: int = 42
) -> Tuple[Dict[str, str], List[str], Dict[str, Set[str]], Set[str]]:
    """
    Convenience function: load corpus + filtered train queries.
    
    Returns:
        doc_id_to_text, doc_ids, filtered_train_qrels, valid_train_qids
    """
    # Load corpus once
    doc_id_to_text, doc_ids = load_nfcorpus_full_corpus()
    
    # Load train split
    dataset_train, _ = load_nfcorpus_splits()
    
    # Filter train queries
    valid_train_qids, filtered_train_qrels = filter_queries_by_relevance(
        dataset_train.qrels_iter(),
        min_rel=min_rel,
        max_rel=max_rel,
        random_seed=random_seed
    )
    
    return doc_id_to_text, doc_ids, filtered_train_qrels, valid_train_qids


if __name__ == "__main__":
    # Quick test / demo when running the file directly
    print("Running quick test of data loading...")
    docs, ids, qrels, qids = load_filtered_train_data()
    print(f"Example query: {next(iter(qrels))}")
    print(f"Has {len(qrels[next(iter(qrels))])} relevant docs")