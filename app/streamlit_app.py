"""Minimal Streamlit front-end for DocuMind RAG. Run with: streamlit run app/streamlit_app.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from app.rag_chain import RagPipeline

st.set_page_config(page_title="DocuMind RAG", page_icon="📚")
st.title("📚 DocuMind — RAG over your documents")
st.caption("Retrieval-Augmented Generation demo. Default mode runs fully offline (no API key needed).")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

with st.sidebar:
    st.header("Configuration")
    folder = st.text_input("Documents folder", value="data/sample_docs")
    llm_kind = st.selectbox("LLM", ["fake (offline demo)", "openai", "anthropic"], index=0)
    llm_key = {"fake (offline demo)": "fake", "openai": "openai", "anthropic": "anthropic"}[llm_kind]
    if st.button("Build / rebuild index"):
        pipeline = RagPipeline(embedder_kind="tfidf", llm_kind=llm_key)
        n = pipeline.ingest_folder(folder)
        st.session_state.pipeline = pipeline
        st.success(f"Indexed {n} chunks from {folder}")

question = st.text_input("Ask a question about the indexed documents")
if st.button("Ask") and question:
    if st.session_state.pipeline is None:
        st.warning("Build the index first (sidebar).")
    else:
        result = st.session_state.pipeline.query(question)
        st.subheader("Answer")
        st.write(result["answer"])
        st.subheader("Sources")
        for s in result["sources"]:
            st.markdown(f"**{s['source']}** (score {s['score']}) — _{s['excerpt']}..._")
