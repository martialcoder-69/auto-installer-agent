import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent


class Config:
    # ── Hugging Face ───────────────────────────────────────────────────────
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    LLM_MODEL: str = "llama-3.3-70b-versatile"   # free on Groq
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Alternative options:
    # "mistralai/Mistral-7B-Instruct-v0.3"
    # "meta-llama/Meta-Llama-3-70B-Instruct"

    # ── Paths ──────────────────────────────────────────────────────────────
    DOCS_DIR: Path = BASE_DIR / "data" / "docs"
    VECTORSTORE_DIR: Path = BASE_DIR / "data" / "vectorstore"
    PROMPTS_DIR: Path = BASE_DIR / "prompts"

    # ── RAG ────────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 600          # characters per chunk
    CHUNK_OVERLAP: int = 80        # overlap between chunks
    TOP_K: int = 6                 # chunks to retrieve per query

    # ── Agent ──────────────────────────────────────────────────────────────
    MAX_RETRIES: int = 3           # self-heal retries before giving up
    COMMAND_TIMEOUT: int = 120     # seconds per shell command
    USE_DOCKER: bool = False       # set True to sandbox commands in Docker

    # ── Scraping ───────────────────────────────────────────────────────────
    MAX_SCRAPE_PAGES: int = 10     # max sub-pages to follow per doc site
    REQUEST_TIMEOUT: int = 15      # HTTP request timeout in seconds

    # ── Ensure directories exist ───────────────────────────────────────────
    @classmethod
    def bootstrap(cls) -> None:
        for d in [cls.DOCS_DIR, cls.VECTORSTORE_DIR]:
            d.mkdir(parents=True, exist_ok=True)