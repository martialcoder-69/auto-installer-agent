"""
rag/retriever.py
────────────────
Query ChromaDB for the most relevant documentation chunks
given a natural-language query.
"""

from __future__ import annotations

import chromadb


from app.config import Config

# ADD:
from sentence_transformers import SentenceTransformer
_embedder = SentenceTransformer(Config.EMBEDDING_MODEL)

def _get_embedding(text: str) -> list[float]:
    return _embedder.encode(text, show_progress_bar=False).tolist()



def _chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(Config.VECTORSTORE_DIR))


def retrieve(
    query: str,
    collection_name: str,
    top_k: int | None = None,
) -> list[dict]:
    """
    Returns a list of the top-k most relevant chunks.

    Each dict has:
        - text   (str)  : the chunk content
        - url    (str)  : source URL
        - score  (float): distance score (lower = more relevant)
    """
    k = top_k or Config.TOP_K
    db = _chroma_client()

    try:
        collection = db.get_collection(name=collection_name)
    except Exception:
        return []   # collection doesn't exist yet

    query_embedding = _get_embedding(query)
    results = collection.query(
    query_embeddings=[query_embedding],
    n_results=min(k, collection.count()),
    include=["documents", "metadatas", "distances"],
)

    chunks: list[dict] = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for text, meta, dist in zip(docs, metas, distances):
        chunks.append(
            {
                "text": text,
                "url": meta.get("url", ""),
                "score": round(dist, 4),
            }
        )

    return chunks


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a single context string for the LLM."""
    if not chunks:
        return "No documentation context available."
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Chunk {i} | source: {chunk['url']}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)
