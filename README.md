# 🤖 Auto-Installer Agent

A chat-based AI agent that installs software automatically by:
1. **Scraping** the official documentation of the requested software
2. **Indexing** it with RAG (ChromaDB + sentence-transformers)
3. **Planning** the right shell commands using Claude
4. **Executing** them on your system (or in a Docker sandbox)
5. **Self-healing** any errors automatically

---

## Project Structure

```
auto-installer-agent/
├── app/
│   ├── main.py         # Streamlit chat UI
│   └── config.py       # All settings
│
├── rag/
│   ├── ingest.py       # Scrape → chunk → embed → store
│   └── retriever.py    # Query ChromaDB for relevant chunks
│
├── agent/
│   ├── planner.py      # Claude-based command planner
│   ├── executor.py     # Main orchestration pipeline
│   └── self_heal.py    # Error analysis & fix generation
│
├── tools/
│   ├── shell.py        # subprocess wrapper (host)
│   └── docker_runner.py# Docker sandbox execution
│
├── data/
│   ├── docs/           # (reserved for saved doc text)
│   └── vectorstore/    # ChromaDB persistent storage
│
├── prompts/
│   ├── command_prompt.txt  # System prompt for planner
│   └── fix_prompt.txt      # System prompt for self-healer
│
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone & install dependencies

```bash
git clone <your-repo>
cd auto-installer-agent
pip install -r requirements.txt
```

### 2. Set your Anthropic API key

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run the app

```bash
streamlit run app/main.py
```

---

## Configuration (`app/config.py`)

| Setting | Default | Description |
|---|---|---|
| `MODEL` | `claude-sonnet-4-20250514` | Claude model to use |
| `CHUNK_SIZE` | `600` | Characters per RAG chunk |
| `TOP_K` | `6` | Doc chunks retrieved per query |
| `MAX_RETRIES` | `3` | Self-heal retry attempts |
| `COMMAND_TIMEOUT` | `120` | Seconds per shell command |
| `USE_DOCKER` | `False` | Sandbox commands in Docker |
| `MAX_SCRAPE_PAGES` | `10` | Pages to scrape per doc site |

---

## Usage Examples

- *"Install Node.js"*
- *"Set up Docker on Ubuntu"*
- *"Download and install PostgreSQL"*
- *"Install the latest version of Python"*
- *"Set up nginx web server"*

---

## Safety

- A blocklist prevents obviously destructive commands (`rm -rf /`, fork bombs, etc.)
- Enable **Docker sandbox** mode in the sidebar to run all commands in an isolated container
- Always review the planned commands before they execute (coming soon: approval step)

---

## Architecture

```
User Query
    │
    ▼
Identify Software  ──► Claude
    │
    ▼
Scrape Docs  ──► requests + BeautifulSoup
    │
    ▼
Ingest to ChromaDB  ──► sentence-transformers embeddings
    │
    ▼
Retrieve Chunks  ──► top-k similarity search
    │
    ▼
Plan Commands  ──► Claude + prompts/command_prompt.txt
    │
    ▼
Execute  ──► shell.py | docker_runner.py
    │
    ├── ✅ Success → Summarise
    │
    └── ❌ Failure → Self-Heal
              │
              ▼
         Analyse Error  ──► Claude + prompts/fix_prompt.txt
              │
              ▼
         Run Fix Commands
              │
              ▼
         Retry (up to MAX_RETRIES)
```
