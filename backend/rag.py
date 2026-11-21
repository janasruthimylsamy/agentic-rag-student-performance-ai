
import os
from pathlib import Path

import faiss
import numpy as np
from google import genai


BASE_DIR = Path(__file__).resolve().parent.parent

POLICY_DIR = Path(
    os.getenv("POLICY_DATA_DIR", BASE_DIR)
)

POLICY_FILES = [
    "grading_policy.md",
    "attendance_policy.md",
    "alert_rules.md",
    "progress_milestones.md",
    "faq.md",
]

EMBEDDING_MODEL = "gemini-embedding-001"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

policy_chunks = []
policy_sources = []
faiss_index = None
google_client = None


def initialize_rag():
    """Load policy files, create embeddings, and build the FAISS index."""
    global policy_chunks, policy_sources
    global faiss_index, google_client

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is missing. Add it to your environment secrets."
        )

    google_client = genai.Client(api_key=api_key)

    policy_chunks = []
    policy_sources = []

    for filename in POLICY_FILES:
        file_path = POLICY_DIR / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Policy document not found: {file_path}"
            )

        content = file_path.read_text(encoding="utf-8")

        for chunk in create_chunks(content):
            policy_chunks.append(chunk)
            policy_sources.append(filename)

    if not policy_chunks:
        raise ValueError("No policy text was found.")

    embeddings = []

    for start in range(0, len(policy_chunks), 20):
        batch = policy_chunks[start:start + 20]

        response = google_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch
        )

        embeddings.extend(
            item.values for item in response.embeddings
        )

    embedding_matrix = np.asarray(
        embeddings,
        dtype="float32"
    )

    faiss.normalize_L2(embedding_matrix)

    dimension = embedding_matrix.shape[1]

    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(embedding_matrix)

    return {
        "policy_files": len(POLICY_FILES),
        "policy_chunks": len(policy_chunks),
        "embedding_dimension": dimension,
    }


def create_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split policy text into overlapping word-based chunks."""
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def retrieve_policy(query, top_k=3):
    """Find the most relevant policy passages for a question."""
    if google_client is None or faiss_index is None:
        raise RuntimeError(
            "RAG is not initialized. Call initialize_rag() first."
        )

    if not query or not query.strip():
        return []

    response = google_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query
    )

    query_embedding = np.asarray(
        [response.embeddings[0].values],
        dtype="float32"
    )

    faiss.normalize_L2(query_embedding)

    result_count = min(top_k, faiss_index.ntotal)

    scores, indices = faiss_index.search(
        query_embedding,
        result_count
    )

    results = []

    for score, index in zip(scores[0], indices[0]):
        if index == -1:
            continue

        results.append({
            "source": policy_sources[index],
            "score": float(score),
            "text": policy_chunks[index],
        })

    return results
