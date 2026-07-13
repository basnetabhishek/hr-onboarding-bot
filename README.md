# HR Onboarding Assistant

A Streamlit chatbot that answers new-hire questions from a company's own HR
documents. Upload the employee handbook, benefits guide, and policy PDFs once;
the app embeds them into a FAISS index saved on disk and answers questions using
retrieval-augmented generation (RAG) with source citations.

Every answer is grounded in the uploaded documents. If something isn't covered,
the assistant says so and points the employee to HR instead of guessing.

## How it works

1. **Ingest** — PDFs are read page by page with `pypdf`, keeping page numbers.
2. **Chunk** — pages are split into overlapping chunks for retrieval.
3. **Embed & store** — chunks are embedded with OpenAI and saved to a FAISS
   index on disk, so they don't need reprocessing next launch.
4. **Retrieve** — each question is rephrased against the chat history, then used
   to pull the most relevant chunks.
5. **Answer** — `gpt-4o-mini` answers from those chunks only, and the app shows
   which document and page each answer came from.

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your OpenAI API key
cp .env.example .env             # then edit .env and paste your key
```

Get a key at https://platform.openai.com/api-keys. Keep it in `.env` only —
never paste it into the code.

## Run

```bash
streamlit run app.py
```

Then upload your HR PDFs in the sidebar, click **Process documents**, and start
asking questions. The index is saved to `faiss_index/` and reloaded
automatically the next time you launch.

## Project layout

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI and app flow |
| `rag_pipeline.py` | PDF ingestion, FAISS index, retrieval chain |
| `ui.py` | Header and citation styling |
| `requirements.txt` | Dependencies |
| `.env.example` | Template for your API key |

## Configuration

Defaults live at the top of `rag_pipeline.py`: chunk size, overlap, number of
retrieved chunks (`TOP_K`), and the embedding and chat model names. Adjust to
trade off cost, speed, and answer detail.

## Notes

- The saved FAISS index is loaded with `allow_dangerous_deserialization=True`.
  This is safe here because the app only ever loads an index it created locally;
  don't point it at index files from untrusted sources.
- Costs scale with document size (embedding, one-time) and usage (each question).
  `text-embedding-3-small` and `gpt-4o-mini` are inexpensive choices.
