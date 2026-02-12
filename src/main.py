# run.py
"""
Main entry point to run the full Neural IR pipeline on NFCorpus.
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Neural IR pipeline (BM25 + Cross-Encoder + XGBoost LTR)")
    parser.add_argument("--step", choices=["all", "index", "features", "train", "predict", "eval"],
                        default="all", help="Which step to run")
    parser.add_argument("--split", default="train", choices=["train", "test"],
                        help="Dataset split for features/eval")
    args = parser.parse_args()

    root = Path(__file__).parent

    if args.step in ("all", "index"):
        print("\n=== 1. Building BM25 index ===")
        from src.data.load_dataset import load_nfcorpus_full_corpus
        from src.retrieval.bm25 import build_bm25_index
        doc_id_to_text, doc_ids = load_nfcorpus_full_corpus()
        texts = [doc_id_to_text[did] for did in doc_ids]
        build_bm25_index(texts, save_path="models/bm25_index.pkl")

    if args.step in ("all", "features"):
        print(f"\n=== 2. Extracting features ({args.split}) ===")
        from src.retrieval.feature_extraction import build_features_for_split
        build_features_for_split(split=args.split, output_dir="data/processed")

    if args.step in ("all", "train"):
        print("\n=== 3. Training XGBoost models ===")
        from src.models.train_xgboost import main as train_main
        # If you made train_xgboost.py runnable → or just call the function
        # train_main()   # if you add def main() there
        import subprocess
        subprocess.run(["python", str(root / "src/models/train_xgboost.py")], check=True)

    if args.step in ("all", "predict"):
        print(f"\n=== 4. Predicting on {args.split} ===")
        import subprocess
        subprocess.run(["python", str(root / "src/models/predict.py")], check=True)

    if args.step in ("all", "eval"):
        print("\n=== 5. Evaluating runs ===")
        import subprocess
        subprocess.run(["python", str(root / "src/evaluation/evaluate.py")], check=True)

    print("\nDone.")

if __name__ == "__main__":
    main()