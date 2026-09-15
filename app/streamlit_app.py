"""Phase 8 deliverable: a chat frontend for DocuMind's FastAPI backend.

Deliberately a thin HTTP client, not a direct import of the pipeline --
the whole point of Phase 8 is separating the API from the UI so either
could be swapped or scaled independently (the same "swappable modules"
principle the project plan's §4 architecture decision applied to
ingestion/retrieval from day one, now applied one layer up).

Run:
    uvicorn api.main:app --reload      # in one terminal
    streamlit run app/streamlit_app.py # in another
"""

import requests
import streamlit as st

st.set_page_config(page_title="DocuMind", page_icon="📄")

with st.sidebar:
    st.header("Configuration")
    api_url = st.text_input("API URL", "http://localhost:8000")
    strategy = st.selectbox("Chunking strategy", ["recursive", "fixed_size", "semantic", "sentence_window"], index=0)
    embedder = st.selectbox("Embedder", ["nomic_embed_text", "sentence_transformers"], index=0)
    k = st.slider("Top-k chunks", 1, 10, 5)
    token_budget = st.number_input("Token budget", min_value=0, value=2000, step=100)
    hybrid = st.checkbox("Hybrid search (BM25 + dense)")
    rerank = st.checkbox("Cross-encoder re-rank")
    prompt_variant = st.radio("Prompt variant", ["guarded", "baseline"])
    if st.button("Clear conversation"):
        st.session_state.history = []

st.title("📄 DocuMind")
st.caption("Ask a question about the ingested corpus. Answers are grounded in retrieved chunks and cited.")

if "history" not in st.session_state:
    st.session_state.history = []


def render_sources(sources: list[dict], context_tokens: int) -> None:
    if not sources:
        return
    with st.expander(f"{len(sources)} source(s), {context_tokens} context tokens"):
        for source in sources:
            page = f", p.{source['page_start']}" if source.get("page_start") else ""
            st.markdown(f"**[source: {source['citation_index']}] {source['source_path']}{page}**")
            st.caption(source["text_preview"])


for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        render_sources(turn["sources"], turn["context_tokens"])

question = st.chat_input("Ask a question...")
if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{api_url}/ask",
                    json={
                        "question": question,
                        "strategy": strategy,
                        "embedder": embedder,
                        "k": k,
                        "token_budget": token_budget,
                        "hybrid": hybrid,
                        "rerank": rerank,
                        "prompt_variant": prompt_variant,
                    },
                    timeout=180,
                )
                response.raise_for_status()
                data = response.json()
            except requests.RequestException as e:
                st.error(f"Request to the API failed: {e}")
                st.stop()

        st.write(data["answer"])
        render_sources(data["sources"], data["context_tokens"])

    st.session_state.history.append(
        {"question": question, "answer": data["answer"], "sources": data["sources"], "context_tokens": data["context_tokens"]}
    )
