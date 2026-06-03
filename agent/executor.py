"""
agent/executor.py
─────────────────
Orchestrates the full pipeline:
  User query → identify software → RAG ingest → retrieve → plan → execute → (self-heal)
"""

from __future__ import annotations

import re
from typing import Callable



from app.config import Config
from rag.ingest import ingest, _collection_name
from rag.retriever import retrieve, format_context
from agent.planner import plan
from agent.self_heal import heal
from tools.shell import run_command
from tools.docker_runner import run_in_docker

# ── HF Client ─────────────────────────────────────────────────────────────
from groq import Groq
_client = Groq(api_key=Config.GROQ_API_KEY)
_NOP = lambda _: None   # no-op callback


def _chat(messages, max_tokens=200, temperature=0.1):
    """Groq chat wrapper."""
    resp = _client.chat.completions.create(
        model=Config.LLM_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


def _identify_software(query: str) -> str:
    """Extract the primary software name from the user query."""
    response = _chat(
        messages=[
            {
                "role": "user",
                "content": (
                    f"From this request: \"{query}\"\n"
                    "What is the exact name of the software/tool to install?\n"
                    "Reply with ONLY the software name, nothing else.\n"
                    "Examples: Node.js, Docker, PostgreSQL"
                ),
            }
        ],
        max_tokens=60,
        temperature=0.0,  # deterministic extraction
    )

    return response.strip().strip('"').strip("'")


def _summarise(
    software: str,
    commands: list[str],
    results: list[dict],
    success: bool,
) -> str:
    """Generate a friendly final summary."""
    lines = []
    for cmd, res in zip(commands, results):
        status = "✅" if res["success"] else "❌"
        lines.append(f"{status} `{cmd}`")

        if res.get("stdout"):
            lines.append(f"  stdout: {res['stdout'][:300]}")
        if res.get("stderr") and not res["success"]:
            lines.append(f"  stderr: {res['stderr'][:300]}")

    summary_input = "\n".join(lines)
    outcome = "succeeded" if success else "encountered errors even after self-healing"

    prompt = (
        f"The installation of {software} {outcome}.\n\n"
        f"Command results:\n{summary_input}\n\n"
        "Write a concise, friendly 2-4 sentence summary for the user. "
        "If successful, mention what they can do next. "
        "If failed, suggest manual steps or where to get help."
    )

    return _chat(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.5,
    )


class AgentExecutor:
    """Main agent orchestrator."""

    def run(
        self,
        user_query: str,
        on_status: Callable[[str], None] = _NOP,
        on_log: Callable[[str], None] = _NOP,
    ) -> str:

        # ── 1. Identify software ───────────────────────────────────────────
        on_status("Identifying software…")
        software = _identify_software(user_query)
        on_log(f"🔍 Identified software: {software}")

        # ── 2. Ingest docs ─────────────────────────────────────────────────
        on_status(f"Scraping {software} documentation…")
        col_name = ingest(software, on_log=on_log)

        # ── 3. Retrieve relevant chunks ────────────────────────────────────
        on_status("Retrieving relevant documentation chunks…")
        chunks = retrieve(user_query, col_name)
        context = format_context(chunks)
        on_log(f"📚 Retrieved {len(chunks)} chunk(s) from docs")

        # ── 4. Plan commands ───────────────────────────────────────────────
        on_status("Planning installation commands…")
        try:
            commands = plan(user_query, context)
        except ValueError as exc:
            return f"❌ Planner failed: {exc}"

        on_log(f"📋 Plan ({len(commands)} step(s)):")
        for i, cmd in enumerate(commands, 1):
            on_log(f"  {i}. {cmd}")

        # ── 5. Execute commands ────────────────────────────────────────────
        runner = run_in_docker if Config.USE_DOCKER else run_command

        all_results: list[dict] = []
        overall_success = True

        for i, cmd in enumerate(commands, 1):
            on_status(f"Running step {i}/{len(commands)}: {cmd[:60]}…")
            on_log(f"\n▶ [{i}/{len(commands)}] {cmd}")

            result = runner(cmd, timeout=Config.COMMAND_TIMEOUT)

            if result["stdout"]:
                on_log(result["stdout"])
            if result["stderr"]:
                on_log(f"STDERR: {result['stderr']}")

            if result["success"]:
                on_log(f"✅ Step {i} succeeded (exit {result['returncode']})")
                all_results.append(result)
                continue

            # ── 5a. Self-heal ──────────────────────────────────────────────
            on_log(f"⚠️  Step {i} failed — attempting self-heal…")
            on_status(f"Self-healing step {i}…")

            healed, heal_results = heal(
                failed_command=cmd,
                stdout=result["stdout"],
                stderr=result["stderr"],
                doc_context=context,
                runner=runner,
                on_log=on_log,
            )

            all_results.append(result)
            all_results.extend(heal_results)

            if not healed:
                on_log(f"❌ Step {i} could not be healed after {Config.MAX_RETRIES} retries.")
                overall_success = False
                break

            on_log(f"✅ Step {i} healed successfully!")

        # ── 6. Summarise ───────────────────────────────────────────────────
        on_status("Generating summary…")
        return _summarise(software, commands, all_results, overall_success)