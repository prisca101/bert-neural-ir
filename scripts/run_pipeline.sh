#!/usr/bin/env bash
set -euo pipefail

echo "=== Neural IR Pipeline on NFCorpus (BEIR) ==="

# 1. Install dependencies (if needed)
# pip install -r requirements.txt

# 2. (Re-)build BM25 index (only needed once or if corpus changes)
echo "Step 1: Building/loading BM25 index..."
python -c "from src.retrieval.bm25 import build_bm25_index; \
           from src.data.load_beir import load_nfcorpus_full_corpus; \
           doc_id_to_text, doc_ids = load_nfcorpus_full_corpus(); \
           texts = [doc_id_to_text[did] for did in doc_ids]; \
           build_bm25_index(texts, save_path='models/bm25_index.pkl')"

# 3. Extract features for train
echo "Step 2: Extracting training features..."
python src/retrieval/feature_extraction.py \
    --split train \
    --bm25_top_k 300 \
    --output_dir data/processed

# 4. Train XGBoost models
echo "Step 3: Training XGBoost LTR models..."
python src/models/train_xgboost.py

# 5. Predict + evaluate on test set
echo "Step 4: Generating predictions and evaluating on test set..."
python src/models/predict.py

# 6. Show final metrics table
echo "Step 5: Final evaluation summary"
python src/evaluation/evaluate.py

echo -e "\nPipeline finished! Check:"
echo "  - models/           → trained XGBoost models"
echo "  - results/runs/     → TREC-format run files"
echo "  - results/          → metrics_*.csv + metrics_*.json"