"""HR Onboarding Assistant.

A Streamlit chatbot that answers new-hire questions from the company's own HR
PDFs using retrieval-augmented generation. Documents are embedded once, saved to
a FAISS index on disk, and reused on the next launch.

Run with:  streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv

import rag_pipeline as rag
from ui import CUSTOM_CSS, PAGE_ICON, PAGE_TITLE, header_html, sources_html


def init_state() -> None:
    """Set up session state and auto-load any persisted index on first run."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chain" not in st.session_state:
        st.session_state.chain = None

    if st.session_state.chain is None and rag.index_exists():
        store = rag.load_vector_store()
        if store is not None:
            st.session_state.chain = rag.build_conversational_chain(store)


def render_sidebar() -> None:
    with st.sidebar:
        st.subheader("HR documents")
        if rag.index_exists():
            st.caption("A saved document index was found and loaded.")

        uploaded = st.file_uploader(
            "Upload PDFs (handbook, policies, benefits)",
            type="pdf",
            accept_multiple_files=True,
        )

        if st.button("Process documents", use_container_width=True):
            if not uploaded:
                st.warning("Please upload at least one PDF first.")
            else:
                with st.spinner("Reading and indexing documents..."):
                    try:
                        store = rag.build_vector_store(uploaded)
                        st.session_state.chain = rag.build_conversational_chain(store)
                        st.session_state.messages = []
                        st.success("Documents indexed. Ask away below.")
                    except Exception as exc:
                        st.error(f"Could not process documents: {exc}")

        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def render_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                st.markdown(sources_html(msg["sources"]), unsafe_allow_html=True)


def handle_query(question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    history = rag.format_history(st.session_state.messages[:-1])
    with st.chat_message("assistant"):
        with st.spinner("Looking through the documents..."):
            try:
                result = st.session_state.chain.invoke(
                    {"input": question, "chat_history": history}
                )
                answer = result["answer"]
                sources = rag.format_sources(result.get("context", []))
            except Exception as exc:
                answer = f"Something went wrong while answering: {exc}"
                sources = []
        st.markdown(answer)
        if sources:
            st.markdown(sources_html(sources), unsafe_allow_html=True)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(header_html(), unsafe_allow_html=True)

    init_state()
    render_sidebar()

    if st.session_state.chain is None:
        st.info(
            "Upload your HR PDFs in the sidebar and click **Process documents** "
            "to get started."
        )
        return

    render_history()
    question = st.chat_input("Ask about onboarding, benefits, leave policy...")
    if question:
        handle_query(question)


if __name__ == "__main__":
    main()
