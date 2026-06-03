"""
agent/planner.py
────────────────
Given a user query + RAG context, generate
a structured installation plan (list of shell commands).
"""

from __future__ import annotations

import json
import platform
import re

from groq import Groq

from app.config import Config

# ── Groq Client ────────────────────────────────────────────────────────────
_client = Groq(api_key=Config.GROQ_API_KEY)
MAX_COMMANDS = 6


def _load_prompt(name: str) -> str:
    path = Config.PROMPTS_DIR / name
    if path.exists():
        return path.read_text()
    return ""


def _detect_os() -> str:
    system = platform.system()
    if system == "Linux":
        try:
            with open("/etc/os-release") as f:
                info = f.read()
            if "ubuntu" in info.lower():
                return "Ubuntu Linux"
            if "debian" in info.lower():
                return "Debian Linux"
            if "fedora" in info.lower():
                return "Fedora Linux"
            if "arch" in info.lower():
                return "Arch Linux"
        except OSError:
            pass
        return "Linux"
    return system


def plan(user_query: str, doc_context: str) -> list[str]:
    os_info = _detect_os()
    base_prompt = _load_prompt("command_prompt.txt")

    resp = _client.chat.completions.create(
        model=Config.LLM_MODEL,
        messages=[
            {"role": "system", "content": base_prompt},
            {
                "role": "user",
                "content": (
                    f"OS: {os_info}\n"
                    f"Request: {user_query}\n"
                    f"Docs:\n{doc_context[:1500]}\n\n"
                    "Output the JSON array now:"
                ),
            },
        ],
        max_tokens=300,
        temperature=0.1,
    )

    raw = resp.choices[0].message.content.strip()

    # Strip markdown fences
    raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw).strip()
    raw = re.sub(r"\n?```$", "", raw).strip()

    # Extract just the [...] array if there's extra prose
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        raw = match.group(0)

    # Close truncated array (token cutoff recovery)
    if "[" in raw and not raw.rstrip().endswith("]"):
        raw = raw.rstrip().rstrip(",") + '"]'

    try:
        commands = json.loads(raw)
    except json.JSONDecodeError:
        commands = re.findall(r'"((?:[^"\\]|\\.)*)"', raw)

    if not commands:
        raise ValueError(f"Could not extract commands from LLM output:\n{raw}")

    # Deduplicate and cap
    seen, result = set(), []
    for cmd in commands:
        cmd = cmd.strip()
        if cmd and cmd not in seen:
            seen.add(cmd)
            result.append(cmd)
        if len(result) >= MAX_COMMANDS:
            break

    return result