import os
import time

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()  # reads GOOGLE_API_KEY and PINECONE_API_KEY from .env

INDEX_NAME = "researchrush"
EMBEDDING_DIMENSION = 3072   # output size of Google's embedding-001 model

# ---------------------------------------------------------------------------
# Embeddings client (Gemini)
# ---------------------------------------------------------------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
# ---------------------------------------------------------------------------
# Pinecone client
# ---------------------------------------------------------------------------
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


def create_index_if_not_exists():
    """Create the Pinecone index once. Safe to call every run — does
    nothing if the index already exists."""
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        print(f"[INFO] Creating Pinecone index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        # Serverless index creation is async — wait until it's ready
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            print("[INFO] Waiting for index to be ready...")
            time.sleep(2)
        print("[INFO] Index created and ready.")
    else:
        print(f"[INFO] Index '{INDEX_NAME}' already exists — reusing it.")

    return pc.Index(INDEX_NAME)


import time

def embed_chunks(chunks, batch_size=50):
    """Embed chunks in small batches with a delay to respect rate limits."""
    print(f"[INFO] Generating embeddings for {len(chunks)} chunks...")
    all_vectors = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"[INFO] Embedding batch {i // batch_size + 1} ({len(batch)} chunks)...")
        vectors = embeddings.embed_documents(batch)
        all_vectors.extend(vectors)

        # Wait between batches to stay under 100 requests/min rate limit
        if i + batch_size < len(chunks):
            time.sleep(10)

    return all_vectors


def upsert_chunks(index, chunks, source_name="document"):
    """
    Embed each chunk and upsert into Pinecone.
    Each vector gets a unique ID and stores the original text + source
    filename as metadata, so we can retrieve the readable text later.
    """
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
    # Pinecone recommends batching upserts in groups of ~100
    batch_size = 100
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        index.upsert(vectors=batch)

    print("[INFO] Upsert complete.")


def query_similar_chunks(index, query_text, top_k=5):
    """Embed a query and return the top_k most similar chunks from Pinecone."""
    query_vector = embeddings.embed_query(query_text)

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    return results["matches"]
