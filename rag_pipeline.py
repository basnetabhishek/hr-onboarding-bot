"""Retrieval-augmented pipeline for the HR onboarding assistant.

Responsibilities:
- read uploaded PDFs into page-level documents (keeping page numbers for citations)
- split them into overlapping chunks
- embed the chunks and store them in a FAISS index that is saved to disk
- build a history-aware conversational retrieval chain that answers only from
  the indexed documents

The OpenAI API key is read from the environment (OPENAI_API_KEY) and never
stored in code. Load it from a .env file via python-dotenv in the app.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- Configuration -----------------------------------------------------------

INDEX_DIR = Path("faiss_index")
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4


def _embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


# --- Ingestion ---------------------------------------------------------------

def load_pdf_documents(uploaded_files) -> list[Document]:
    """Read uploaded PDFs into page-level documents.

    Each page becomes one Document carrying its file name and page number in
    metadata, which is what lets the app cite sources later.
    """
    documents: list[Document] = []
    for file in uploaded_files:
        reader = PdfReader(file)
        name = getattr(file, "name", "document.pdf")
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": name, "page": page_number},
                )
            )
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Split page documents into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


# --- Vector store ------------------------------------------------------------

def build_vector_store(uploaded_files) -> FAISS:
    """Build a FAISS store from uploaded PDFs and persist it to disk."""
    raw_docs = load_pdf_documents(uploaded_files)
    if not raw_docs:
        raise ValueError("No readable text was found in the uploaded PDFs.")
    chunks = split_documents(raw_docs)
    store = FAISS.from_documents(chunks, _embeddings())
    store.save_local(str(INDEX_DIR))
    return store


def load_vector_store() -> FAISS | None:
    """Load a previously saved FAISS store, or None if none exists."""
    if not index_exists():
        return None
    return FAISS.load_local(
        str(INDEX_DIR),
        _embeddings(),
        # Safe here: we only ever load an index this app created locally.
        allow_dangerous_deserialization=True,
    )


def index_exists() -> bool:
    return (INDEX_DIR / "index.faiss").exists()


# --- Conversational chain ----------------------------------------------------

CONTEXTUALIZE_PROMPT = (
    "Given the chat history and the latest user question, rewrite the question "
    "so that it can be understood without the history. Do not answer it. If the "
    "question is already self-contained, return it unchanged."
)

ANSWER_PROMPT = (
    "You are an HR onboarding assistant helping new employees. Answer the "
    "question using only the context below, which is drawn from the company's "
    "HR documents. If the answer is not in the context, say you do not have that "
    "information in the documents and suggest contacting the HR team directly. "
    "Do not invent policies. Keep answers clear and concise.\n\n"
    "Context:\n{context}"
)


def build_conversational_chain(store: FAISS):
    """Assemble a history-aware retrieval chain over the given store."""
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    retriever = store.as_retriever(search_kwargs={"k": TOP_K})

    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ANSWER_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    document_chain = create_stuff_documents_chain(llm, answer_prompt)

    return create_retrieval_chain(history_aware_retriever, document_chain)


# --- Helpers used by the UI --------------------------------------------------

def format_history(messages: Iterable[dict]) -> list:
    """Convert stored UI messages into LangChain message objects."""
    history: list = []
    for msg in messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        else:
            history.append(AIMessage(content=msg["content"]))
    return history


def format_sources(context_docs) -> list[str]:
    """Return unique 'file (p.N)' citation labels from retrieved documents."""
    seen: list[str] = []
    for doc in context_docs:
        source = doc.metadata.get("source", "document")
        page = doc.metadata.get("page")
        label = f"{source} (p.{page})" if page else source
        if label not in seen:
            seen.append(label)
    return seen
