
import os, sys
import streamlit as st

CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from loader import load_text_from_file
from embeddings import LocalEmbedder, OpenAIAnswerer
from vector_store import FaissVectorStore
from prompt_templates import QA_PROMPT

st.set_page_config(page_title="Intelligent Document Assistant", layout="centered")
st.title("Intelligent Document Assistant (RAG Starter)")

uploaded = st.file_uploader("Upload a .txt file", type=["txt"])
answer_backend = st.selectbox("Answer generation backend", ["None (show prompt only)", "OpenAI (answers)"])

if uploaded:
    text = load_text_from_file(uploaded)
    st.text_area("Document Preview", text, height=220)

    if st.button("Index Document"):
        with st.spinner("Embedding chunks and building FAISS index..."):
            embedder = LocalEmbedder()
            chunks = embedder.chunk_text(text)
            embs = embedder.embed_texts(chunks)
            store = FaissVectorStore(embs, chunks)
        st.success("Index created. Ask a question below.")

        query = st.text_input("Ask your question about this document")
        if query:
            with st.spinner("Retrieving relevant context..."):
                q_vec = embedder.embed_query(query)
                results = store.search(q_vec, k=4)
                context = "\n\n".join(results)
                prompt = QA_PROMPT.format(context=context, question=query)

            st.subheader("Constructed Prompt")
            st.code(prompt)

            if answer_backend == "OpenAI (answers)":
                answerer = OpenAIAnswerer()
                if not answerer.is_ready():
                    st.warning("OPENAI_API_KEY not set. Showing prompt only.")
                else:
                    with st.spinner("Calling OpenAI for grounded answer..."):
                        answer = answerer.answer(prompt)
                    st.subheader("Answer")
                    st.write(answer)
else:
    st.info("Tip: try the included sample file at `sample_data/sample.txt`.")
