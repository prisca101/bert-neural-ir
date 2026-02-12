# app/app.py
import streamlit as st
from utils import load_models_and_index, search

st.set_page_config(
    page_title="Neural IR & LTR Demo — NFCorpus",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Neural Information Retrieval") 

with st.expander("🌸  What is this demo?", expanded=True):
    st.markdown(
        """
        This small web app demonstrates a two-stage retrieval pipeline:
        
        1. First stage: **BM25** retrieves top ~300 candidates (classic sparse method)  
        2. Second stage: **MiniLM cross-encoder** re-ranks them (neural)  
        3. Optional third stage: **XGBoost LTR** re-scores using hand-crafted + neural features
        
        All models were trained on the **NFCorpus** dataset (nutrition & health domain, part of BEIR benchmark).
        
        ✨ Try queries like:  
        • health benefits of omega-3  
        • vitamin D deficiency symptoms  
        • Mediterranean diet evidence
        """
    ) 

st.caption(
    "BM25 retrieval → MiniLM cross-encoder reranking → XGBoost pointwise/listwise LTR  "
    "— evaluated on NFCorpus (BEIR benchmark)"
)

st.sidebar.header("Ranking Method")
method = st.sidebar.radio(
    "Show results ranked by",
    options=[
        "BM25 (classic baseline)",
        "BERT reranker (cross-encoder)",
        "XGBoost LTR (BERT features)"
    ],
    index=2,
    captions=[
        "Traditional sparse retrieval – fast but keyword-based",
        "Neural re-ranking on BM25 candidates – better semantic understanding",
        "Learned ranking with BERT + document/query features"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption("Dataset: NFCorpus (BEIR)")
st.sidebar.caption("Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2")
st.sidebar.caption("LTR model: XGBoost rank:ndcg")



# Load everything once
reranker, ranker, bm25, doc_id_to_text, doc_ids = load_models_and_index()
query = st.text_input("Enter your search query:", placeholder="e.g. health benefits of omega-3 fatty acids")

if query:
    with st.spinner("Searching…"):
        results = search(
            query,
            bm25=bm25,
            doc_ids=doc_ids,
            doc_id_to_text=doc_id_to_text,
            reranker=reranker,
            ranker=ranker,
            top_k_retrieve=150,
            top_k_show=10
        )

    if not results:
        st.warning("No results found.")
    else:
        method_key = {
            "BM25 (classic baseline)": "bm25_score",
            "BERT reranker (cross-encoder)": "bert_score",
            "XGBoost LTR (BERT features)": "xgb_score"
        }[method]

        # Re-sort displayed results by chosen score (optional – already sorted by xgb above)
        displayed = sorted(results, key=lambda x: x[method_key], reverse=True)

        st.markdown(f"**Top {len(displayed)} results — sorted by {method}**")

        for i, res in enumerate(displayed, 1):
            with st.container(border=True):
                col1, col2 = st.columns([8, 2])
                with col1:
                    st.markdown(f"**{i}.** {res['text']}")
                with col2:
                    st.metric(
                        label=method.split(" ")[0],    # "BM25" / "BERT" / "XGBoost"
                        value=f"{res[method_key]:.4f}",
                        delta=None
                    )

                st.caption(
                    f"Doc ID: {res['doc_id']}  "
                    f"BM25: {res['bm25_score']:.3f}  "
                    f"BERT: {res['bert_score']:.3f}  "
                    f"XGB: {res['xgb_score']:.3f}"
                )



st.markdown("---")
st.caption(
    "Built with Streamlit • sentence-transformers • rank_bm25 • XGBoost  •  "
    "Trained on NFCorpus (BEIR)  •  "
    "Last updated: Feb 2026"
)