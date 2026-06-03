"""
Auto-Installer Agent — Streamlit Chat Interface
Run:  streamlit run app/main.py
"""

import sys
from pathlib import Path

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from app.config import Config
from agent.executor import AgentExecutor

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Auto-Installer Agent",
    page_icon="🤖",
    layout="wide",
)

# ── Bootstrap dirs & executor ──────────────────────────────────────────────────
Config.bootstrap()

@st.cache_resource(show_spinner="Initialising agent…")
def get_executor() -> AgentExecutor:
    return AgentExecutor()

executor = get_executor()

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 Hi! I'm your **Auto-Installer Agent**.\n\n"
                "Tell me what you want to install — e.g.:\n"
                "- *Install Node.js*\n"
                "- *Set up Docker on Ubuntu*\n"
                "- *Download and install PostgreSQL*\n\n"
                "I'll scrape the official docs, figure out the right commands, "
                "run them, and fix any errors automatically."
            ),
        }
    ]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    use_docker = st.toggle(
        "Sandbox in Docker",
        value=Config.USE_DOCKER,
        help="Run commands inside a Docker container instead of your host system.",
    )
    Config.USE_DOCKER = use_docker

    st.divider()
    st.caption("**Model:** " + Config.LLM_MODEL)
    st.caption("**Max retries:** " + str(Config.MAX_RETRIES))

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ── Chat history ───────────────────────────────────────────────────────────────
st.title("🤖 Auto-Installer Agent")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input ──────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("What would you like to install?"):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream agent response
    with st.chat_message("assistant"):
        status_box = st.empty()   # live status updates
        output_box = st.empty()   # growing log

        log_lines: list[str] = []

        def on_status(msg: str) -> None:
            status_box.info(f"⏳ {msg}")

        def on_log(line: str) -> None:
            log_lines.append(line)
            output_box.code("\n".join(log_lines), language="bash")

        try:
            result = executor.run(
                user_query=prompt,
                on_status=on_status,
                on_log=on_log,
            )
        except Exception as exc:
            result = f"❌ Unexpected error: {exc}"

        status_box.empty()   # clear spinner

        # Final human-readable summary from agent
        with st.container():
            st.markdown(result)

        full_response = result
        if log_lines:
            full_response = "```bash\n" + "\n".join(log_lines) + "\n```\n\n" + result

    st.session_state.messages.append({"role": "assistant", "content": full_response})
