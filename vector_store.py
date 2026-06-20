import os
import time

import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

INDEX_NAME = "reviewrush"
EMBEDDING_DIMENSION = 3072  # gemini-embedding-001 outputs 3072-dim vectors


# ---------------------------------------------------------------------------
# API key helpers — reads from Streamlit secrets on cloud, .env locally
# ---------------------------------------------------------------------------
def get_google_api_key():
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        return os.getenv("GOOGLE_API_KEY")


def get_pinecone_api_key():
    try:
        return st.secrets["PINECONE_API_KEY"]
    except Exception:
        return os.getenv("PINECONE_API_KEY")


# ---------------------------------------------------------------------------
# Clients — initialized lazily so secrets are read at runtime, not import
# ---------------------------------------------------------------------------
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=get_google_api_key()
    )


def get_pinecone_client():
    return Pinecone(api_key=get_pinecone_api_key())


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------
def create_index_if_not_exists():
    """Create the Pinecone index once. Safe to call every run — does
    nothing if the index already exists."""
    pc = get_pinecone_client()
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        print(f"[INFO] Creating Pinecone index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            print("[INFO] Waiting for index to be ready...")
            time.sleep(2)
        print("[INFO] Index created and ready.")
    else:
        print(f"[INFO] Index '{INDEX_NAME}' already exists — reusing it.")

    return pc.Index(INDEX_NAME)


def clear_index(index):
    """
    Delete all vectors from the index to free up storage.
    Called before upserting new PDFs so storage stays flat regardless
    of how many times the app is used.
    """
    index.delete(delete_all=True)
    print("[INFO] Index cleared — ready for new documents.")


# ---------------------------------------------------------------------------
# Embedding + upsert
# ---------------------------------------------------------------------------
def embed_chunks(chunks, batch_size=50):
    """Embed chunks in small batches with a delay to respect rate limits."""
    embeddings = get_embeddings()
    print(f"[INFO] Generating embeddings for {len(chunks)} chunks...")
    all_vectors = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"[INFO] Embedding batch {i // batch_size + 1} ({len(batch)} chunks)...")
        vectors = embeddings.embed_documents(batch)
        all_vectors.extend(vectors)

        if i + batch_size < len(chunks):
            time.sleep(10)

    return all_vectors


def upsert_chunks(index, chunks, source_name="document"):
    """Embed each chunk and upsert into Pinecone with metadata."""
    vectors = embed_chunks(chunks)

    records = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        records.append({
            "id": f"{source_name}-chunk-{i}",
            "values": vector,
            "metadata": {
                "text": chunk,
                "source": source_name
            }
        })

    print(f"[INFO] Upserting {len(records)} vectors into Pinecone...")
    batch_size = 100
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        index.upsert(vectors=batch)

    print("[INFO] Upsert complete.")


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
def query_similar_chunks(index, query_text, top_k=5):
    """Embed a query and return the top_k most similar chunks from Pinecone."""
    embeddings = get_embeddings()
    query_vector = embeddings.embed_query(query_text)

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    return results["matches"]
