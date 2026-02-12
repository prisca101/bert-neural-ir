# Evaluation Results

Final performance on the **BEIR NFCorpus test set** :

| Model                  | nDCG@10  | R@100   | Δ nDCG@10 vs BM25 | Notes                              |
|------------------------|----------|---------|-------------------|------------------------------------|
| BM25 (baseline)        | 0.2630   | 0.1988  | —                 | Standard first-stage retrieval     |
| Cross-Encoder rerank   | **0.3099** | **0.2192** | +17.8%          | BM25 top-300 → MiniLM-L-6-v2 rerank |
| BM25 features + XGBoost| 0.2643   | 0.2011  | +0.5%             | Very small lift                    |
| BERT features + XGBoost| **0.3114** | **0.2196** | +18.4%          | Strongest single reranker signal   |
| Combined + XGBoost     | 0.3092   | 0.2190  | +17.6%            | Slightly below pure BERT rerank    |

**Best result**: BERT-only features + XGBoost → **nDCG@10 = 0.3114** (+18.4% over BM25)

> These numbers place the best model roughly in line with or slightly above many published zero-shot BEIR results for nfcorpus using similar-sized rerankers (MiniLM-L-6-v2).

## Quick Interpretation of Metrics

1. **Cross-encoder reranking is very powerful**  
   Even a relatively small model (ms-marco-MiniLM-L-6-v2) gives a massive lift over BM25 alone (~+18% nDCG@10). This confirms how effective dense late-interaction reranking is on scientific/medical text.

2. **XGBoost LTR adds only marginal gains here**  
   When using **pure BERT reranker scores** as the main feature, the tree-based LTR model could not meaningfully improve over the raw cross-encoder ordering. 
   → In this dataset & setting, the cross-encoder already produces near-optimal rankings within the top-300 candidates.

3. **Combining BM25 + BERT signals did not help**  
   The combined feature set actually performed slightly **worse** than BERT-only.  
   Possible reasons:  
   - BM25 score is noisy / weakly correlated after good reranking  
   - Feature scale differences or multicollinearity  
   - Not enough signal left to learn from after a strong reranker

4. **Simple hand-crafted features were not decisive**  
   Adding `doc_length` and `query_length` did not make a noticeable difference, likely because document lengths in nfcorpus are relatively homogeneous (scientific abstracts).

## 💡 What I would try next (if continuing the project)

- Add more features (e.g. query-document term overlap, position signals, title-specific signals)
- Add query-document **dense similarity** (e.g. ColBERT)
- Experiment with **negative sampling** strategies inside XGBoost training

## Summary – one-liner takeaway

> A lightweight cross-encoder reranker already captures most of the gains possible in this setting.  
> Adding tree-based LTR on top brings only marginal (or even negative) benefit when the reranker is already strong.