import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from pdf_processor import extract_text_from_pdfs, chunk_text
from vector_store import create_index_if_not_exists, upsert_chunks
from rag_chain import RAGChain

load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Research Rush",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Research Rush")
st.caption("Upload PDFs and ask questions — powered by Gemini + Pinecone RAG")

# ---------------------------------------------------------------------------
# Session state — persists across reruns within the same browser session
# ---------------------------------------------------------------------------
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None        # RAGChain instance (has memory)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []       # list of (question, answer) for display

if "pdfs_processed" not in st.session_state:
    st.session_state.pdfs_processed = False  # whether PDFs have been indexed

# ---------------------------------------------------------------------------
# Sidebar — PDF upload and processing
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📂 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    process_btn = st.button("⚡ Process PDFs", use_container_width=True)

    if process_btn and uploaded_files:
        with st.spinner("Extracting text from PDFs..."):
            # Save uploaded files to a temp directory so pdf_processor can read them
            temp_dir = tempfile.mkdtemp()
            pdf_paths = []
            for uploaded_file in uploaded_files:
                path = os.path.join(temp_dir, uploaded_file.name)
                with open(path, "wb") as f:
                    f.write(uploaded_file.read())
                pdf_paths.append(path)

            text = extract_text_from_pdfs(pdf_paths)
            chunks = chunk_text(text)

        st.success(f"Extracted text — {len(chunks)} chunks created.")

        with st.spinner("Generating embeddings and indexing in Pinecone..."):
            index = create_index_if_not_exists()
            upsert_chunks(index, chunks, source_name="uploaded_pdfs")

        with st.spinner("Setting up RAG chain..."):
            st.session_state.rag_chain = RAGChain(index)
            st.session_state.chat_history = []
            st.session_state.pdfs_processed = True

        st.success("✅ PDFs indexed! You can now ask questions.")

    if process_btn and not uploaded_files:
        st.warning("Please upload at least one PDF first.")

    # Show status and clear button if PDFs are already processed
    if st.session_state.pdfs_processed:
        st.divider()
        st.success("PDFs are indexed and ready.")

        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.chat_history = []
            if st.session_state.rag_chain:
                st.session_state.rag_chain.clear_history()
            st.rerun()

# ---------------------------------------------------------------------------
# Main area — chat interface
# ---------------------------------------------------------------------------
if not st.session_state.pdfs_processed:
    st.info("👈 Upload your PDFs in the sidebar and click **Process PDFs** to get started.")
else:
    # Display existing conversation history
    for question, answer in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)

    # Chat input — appears at the bottom
    user_input = st.chat_input("Ask a question about your PDFs...")

    if user_input:
        # Show the user's message immediately
        with st.chat_message("user"):
            st.write(user_input)

        # Generate answer with a spinner
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.rag_chain.ask(user_input)
            st.write(answer)

        # Store in display history
        st.session_state.chat_history.append((user_input, answer))