# Intelligent Document Assistant (LangChain-style RAG starter)
A portfolio-ready starter showing an **AI document Q&A assistant** with Streamlit, SentenceTransformers embeddings, and FAISS vector search.

## Quick start
```bash
python -m venv venv
source venv/bin/activate   # mac/linux
venv\Scripts\activate    # windows

pip install -r requirements.txt
streamlit run app/app.py
```
Upload `sample_data/sample.txt` and ask questions.

## Optional OpenAI answers
Set `OPENAI_API_KEY` env var and choose **OpenAI (answers)** in the UI.
