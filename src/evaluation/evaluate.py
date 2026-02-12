# src/evaluation/evaluate.py
"""
Evaluate IR runs using ir_measures.
Supports:
- TREC-format run files (*.trec)
- In-memory run dictionaries (method → qid → docid → score)
Produces console tables, CSV, and JSON output.
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Union

import ir_measures
import pandas as pd
from ir_measures import nDCG, R, AP, RR, P

from src.data.load_dataset import load_nfcorpus_splits


def load_trec_run(
    run_path: str | Path
) -> Dict[str, Dict[str, float]]:
    """
    Load a TREC-format run file into memory.
    Returns: {qid: {docid: score, ...}, ...}
    """
    run_path = Path(run_path)
    if not run_path.exists():
        raise FileNotFoundError(f"Run file not found: {run_path}")

    run = defaultdict(dict)
    with open(run_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, _, docid, _, score, _ = parts
            run[qid][docid] = float(score)
    print(f"Loaded TREC run: {run_path} ({len(run)} queries)")
    return dict(run)


def load_qrels(
    split: str = "test"
) -> List[ir_measures.Qrel]:
    """Load ground-truth qrels for the given split."""
    _, dataset_test = load_nfcorpus_splits()
    dataset = dataset_test if split == "test" else None

    if dataset is None:
        raise ValueError("Only 'test' split currently supported for qrels loading")

    qrels = [
        ir_measures.Qrel(q.query_id, q.doc_id, q.relevance)
        for q in dataset.qrels_iter()
    ]
    print(f"Loaded {len(qrels)} qrels for {split} split")
    return qrels


def evaluate_run(
    run: Union[Dict[str, Dict[str, float]], str, Path],
    qrels: List[ir_measures.Qrel],
    metrics: List = [nDCG@10, R@100, AP, RR@10, P@5, P@10],
    method_name: Optional[str] = None
) -> Dict:
    """
    Evaluate a single run (in-memory or file path).
    Returns dict of metric → value
    """
    if isinstance(run, (str, Path)):
        run_dict = load_trec_run(run)
        method_name = method_name or Path(run).stem
    else:
        run_dict = run
        method_name = method_name or "unnamed_run"

    flat_run = [
        ir_measures.ScoredDoc(qid, did, score)
        for qid, docs in run_dict.items()
        for did, score in docs.items()
    ]

    agg = ir_measures.calc_aggregate(metrics, qrels, flat_run)

    results = {str(m): agg[m] for m in metrics}
    print(f"{method_name:20}", end="")
    for m in metrics:
        print(f"  {str(m):8} {agg[m]:.4f}", end="")
    print()

    return {"method": method_name, **results}


def evaluate_all_runs(
    run_paths_or_dicts: List[Union[str, Path, Dict]],
    split: str = "test",
    metrics: List = [nDCG@10, R@100, AP, RR@10, P@5, P@10],
    output_dir: str | Path = "results",
    save_csv: bool = True,
    save_json: bool = True
) -> pd.DataFrame:
    """
    Evaluate multiple runs and return a summary DataFrame.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qrels = load_qrels(split)

    print(f"\n{'='*70}")
    print(f"Evaluation on {split.upper()} set – {len(qrels)} qrels")
    print(f"Metrics: {[str(m) for m in metrics]}")
    print(f"{'-'*70}")
    print(f"{'Method':20}  {'nDCG@10':8}  {'R@100':8}  {'AP':8}  {'RR@10':8}  {'P@5':8}  {'P@10':8}")
    print(f"{'-'*70}")

    results_list = []

    for item in run_paths_or_dicts:
        if isinstance(item, dict):
            # in-memory run dictionary
            res = evaluate_run(item, qrels, metrics, method_name="in-memory-run")
        else:
            # file path
            res = evaluate_run(item, qrels, metrics)
        results_list.append(res)

    df = pd.DataFrame(results_list)
    df = df.set_index("method")

    print(f"{'-'*70}")

    # Save results
    if save_csv:
        csv_path = output_dir / f"metrics_{split}.csv"
        df.to_csv(csv_path)
        print(f"Saved CSV → {csv_path}")

    if save_json:
        json_path = output_dir / f"metrics_{split}.json"
        df.round(4).to_json(json_path, orient="index", indent=2)
        print(f"Saved JSON → {json_path}")

    return df


if __name__ == "__main__":
    print("=== Running evaluation examples ===\n")

    # Option 1: evaluate from saved TREC run files
    run_dir = Path("results/runs")
    run_files = list(run_dir.glob("run_*_test.trec"))

    if run_files:
        print("Found TREC run files:")
        for f in run_files:
            print(f"  - {f.name}")
        print()

        evaluate_all_runs(
            run_files,
            split="test",
            metrics=[nDCG@10, R@100, AP, RR@10, P@5, P@10],
            output_dir="results",
        )

    else:
        print("No TREC run files found in results/runs/")
        print("Run predict.py first to generate them.")