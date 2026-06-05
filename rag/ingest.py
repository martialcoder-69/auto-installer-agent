"""
rag/ingest.py
─────────────
Given a software name (or explicit URL), this module:
  1. Uses LLM to discover the official docs URL.
  2. Scrapes the page and follows relevant sub-links (BFS, capped).
  3. Splits text into overlapping chunks.
  4. Embeds & stores them in ChromaDB (one collection per software).
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import Callable
from urllib.parse import urljoin, urlparse

import chromadb
import requests
from bs4 import BeautifulSoup



from app.config import Config

# ── LLM Client (HF) ────────────────────────────────────────────────────────
from groq import Groq
_client = Groq(api_key=Config.GROQ_API_KEY)

# ADD:
from sentence_transformers import SentenceTransformer
_embedder = None

def _get_embeddings(texts: list[str]) -> list[list[float]]:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(Config.EMBEDDING_MODEL)
    return _embedder.encode(texts, show_progress_bar=False).tolist()
def _chat(messages, max_tokens=200, temperature=0.1):
    """Groq chat wrapper."""
    resp = _client.chat.completions.create(
        model=Config.LLM_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()
def _chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(Config.VECTORSTORE_DIR))


def _collection_name(software: str) -> str:
    """Normalise software name → valid ChromaDB collection name."""
    name = re.sub(r"[^a-z0-9_-]", "_", software.lower().strip())
    return name[:60] or "default"


# ── Step 1 – Discover docs URL ─────────────────────────────────────────────

def discover_docs_url(software: str) -> str:
    """Ask model for official installation-docs URL."""
    raw = _chat(
        messages=[
            {
                "role": "user",
                "content": (
                    f"What is the official documentation URL for installing '{software}'?\n"
                    "Return ONLY a valid URL.\n"
                    "No explanation.\n"
                    "Example: https://nodejs.org/en/docs"
                ),
            }
        ],
        max_tokens=200,
        temperature=0.0,
    )

    raw = raw.strip().strip('"').strip("'")

    # Extract first valid URL (HF models may add extra text)
    match = re.search(r"https?://[^\s]+", raw)
    if match:
        url = match.group(0)
    else:
        raise ValueError(f"Model returned invalid URL: {raw!r}")

    return url


# ── Step 2 – Scrape pages ──────────────────────────────────────────────────

def _fetch_text(url: str) -> str:
    """Fetch a URL and return visible text."""
    headers = {"User-Agent": "AutoInstallerBot/1.0 (educational project)"}
    resp = requests.get(url, headers=headers, timeout=Config.REQUEST_TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove noise
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    return soup.get_text(separator="\n", strip=True)


def _find_sub_links(base_url: str, html: str, max_links: int = 20) -> list[str]:
    """Extract same-domain links that look like install/setup docs."""
    parsed = urlparse(base_url)
    base_domain = parsed.netloc

    soup = BeautifulSoup(html, "html.parser")

    install_keywords = re.compile(
        r"install|setup|getting.started|download|quickstart|guide", re.I
    )

    links: list[str] = []

    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])

        if urlparse(href).netloc != base_domain:
            continue

        if install_keywords.search(href) or install_keywords.search(a.get_text()):
            links.append(href)

        if len(links) >= max_links:
            break

    return list(dict.fromkeys(links))  # deduplicate


def scrape_docs(start_url: str, on_log: Callable[[str], None] | None = None) -> list[dict]:
    """BFS scrape starting from *start_url*."""
    log = on_log or (lambda x: None)

    visited: set[str] = set()
    queue = [start_url]
    pages: list[dict] = []

    while queue and len(pages) < Config.MAX_SCRAPE_PAGES:
        url = queue.pop(0)

        if url in visited:
            continue

        visited.add(url)

        try:
            log(f"Scraping: {url}")

            headers = {"User-Agent": "AutoInstallerBot/1.0"}
            resp = requests.get(url, headers=headers, timeout=Config.REQUEST_TIMEOUT)
            resp.raise_for_status()

            html = resp.text
            text = _fetch_text(url)

            pages.append({"url": url, "text": text})

            sub_links = _find_sub_links(url, html)
            queue.extend(lnk for lnk in sub_links if lnk not in visited)

            time.sleep(0.5)

        except Exception as exc:
            log(f"  ⚠ Skipping {url}: {exc}")

    return pages


# ── Step 3 – Chunk text ────────────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks."""
    size = Config.CHUNK_SIZE
    overlap = Config.CHUNK_OVERLAP

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap

    return [c.strip() for c in chunks if c.strip()]


# ── Step 4 – Embed & store ─────────────────────────────────────────────────

def ingest(
    software: str,
    docs_url: str | None = None,
    on_log: Callable[[str], None] | None = None,
) -> str:
    """
    Full pipeline: discover URL → scrape → chunk → embed → store.
    """
    log = on_log or (lambda x: None)

    Config.bootstrap()

    url = docs_url or discover_docs_url(software)
    log(f"📄 Docs URL: {url}")

    pages = scrape_docs(url, on_log=log)
    log(f"✅ Scraped {len(pages)} page(s)")

    col_name = _collection_name(software)
    db = _chroma_client()

    # Reset collection
    try:
        db.delete_collection(col_name)
    except Exception:
        pass

    collection = db.get_or_create_collection(name=col_name)

    doc_ids, doc_texts, doc_metas = [], [], []

    for page in pages:
        chunks = chunk_text(page["text"])

        for i, chunk in enumerate(chunks):
            uid = hashlib.md5(f"{page['url']}_{i}".encode()).hexdigest()

            doc_ids.append(uid)
            doc_texts.append(chunk)
            doc_metas.append({"url": page["url"], "chunk_index": i})

    if doc_texts:
        batch = 500

        for i in range(0, len(doc_texts), batch):
            batch_texts = doc_texts[i : i+batch]
            embeddings = _get_embeddings(batch_texts)
            collection.add(
                ids=doc_ids[i : i+batch],
                documents=batch_texts,  # use original text, not embeddings --- IGNORE ---
                metadatas=doc_metas[i : i+batch],
                embeddings=embeddings,
              )

        log(f"💾 Stored {len(doc_texts)} chunks in '{col_name}'")
    else:
        log("⚠ No text extracted.")

    return col_name