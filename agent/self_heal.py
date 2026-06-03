"""
agent/self_heal.py
──────────────────
When a shell command fails:
  1. Ask LLM for fix commands
  2. Execute them
  3. Retry original command
"""

from __future__ import annotations

import json
import re
from typing import Callable



from app.config import Config

# ── HF Client ─────────────────────────────────────────────────────────────
from groq import Groq
_client = Groq(api_key=Config.GROQ_API_KEY)

_NOP = lambda _: None


def _chat(messages, system=None, max_tokens=600, temperature=0.3):
    """HF chat wrapper."""
    if system:
        messages = [{"role": "system", "content": system}] + messages

    resp = _client.chat.completions.create(
        model=Config.LLM_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return resp.choices[0].message.content.strip()


def _load_fix_prompt() -> str:
    path = Config.PROMPTS_DIR / "fix_prompt.txt"
    if path.exists():
        return path.read_text()
    return ""


def _extract_json(raw: str) -> str:
    """Extract JSON array safely from model output."""
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        return match.group(0)
    return raw


def _ask_for_fix(
    failed_command: str,
    stdout: str,
    stderr: str,
    doc_context: str,
    previous_fixes: list[str],
) -> list[str]:
    """
    Ask HF model how to fix the failed command.
    Returns list of commands.
    """
    base_prompt = _load_fix_prompt()

    system_prompt = base_prompt or (
        "You are an expert systems engineer.\n"
        "A shell command failed. Your job is to fix it.\n\n"
        "STRICT RULES:\n"
        "1. Output ONLY a JSON array\n"
        "2. No explanations\n"
        "3. No markdown or backticks\n"
        "4. Each item must be a valid shell command\n"
        "5. Commands must directly fix the error\n\n"
        "Example:\n"
        "[\"sudo apt-get update\", \"sudo apt-get install -f\"]"
    )

    prev_fix_str = ""
    if previous_fixes:
        prev_fix_str = "\n\nPrevious failed fixes:\n" + "\n".join(
            f"- {c}" for c in previous_fixes
        )

    user_message = (
        f"Failed command:\n{failed_command}\n\n"
        f"STDOUT:\n{stdout or '(empty)'}\n\n"
        f"STDERR:\n{stderr or '(empty)'}\n\n"
        f"Documentation context:\n{doc_context[:2000]}"
        f"{prev_fix_str}\n\n"
        "Return ONLY a JSON array of fix commands."
    )

    raw = _chat(
        messages=[{"role": "user", "content": user_message}],
        system=system_prompt,
        max_tokens=600,
        temperature=0.3,
    )

    raw = raw.strip()

    # Remove markdown fences
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    # Extract JSON safely
    raw_json = _extract_json(raw)

    try:
        cmds = json.loads(raw_json)
        if isinstance(cmds, list) and all(isinstance(c, str) for c in cmds):
            return [c.strip() for c in cmds if c.strip()]
    except json.JSONDecodeError:
        pass

    return []


def heal(
    failed_command: str,
    stdout: str,
    stderr: str,
    doc_context: str,
    runner: Callable[[str, int], dict],
    on_log: Callable[[str], None] = _NOP,
) -> tuple[bool, list[dict]]:
    """
    Attempt to self-heal a failed command.
    """
    all_results: list[dict] = []
    previous_fixes: list[str] = []

    for attempt in range(1, Config.MAX_RETRIES + 1):
        on_log(f"  🔧 Self-heal attempt {attempt}/{Config.MAX_RETRIES}…")

        fix_cmds = _ask_for_fix(
            failed_command=failed_command,
            stdout=stdout,
            stderr=stderr,
            doc_context=doc_context,
            previous_fixes=previous_fixes,
        )

        if not fix_cmds:
            on_log("  ⚠ Model couldn't suggest a fix.")
            break

        on_log(f"  Fix plan: {fix_cmds}")

        fix_success = True

        for cmd in fix_cmds:
            on_log(f"  🔨 Fix cmd: {cmd}")
            res = runner(cmd, Config.COMMAND_TIMEOUT)
            all_results.append(res)

            if res["stdout"]:
                on_log(f"     stdout: {res['stdout'][:400]}")
            if res["stderr"]:
                on_log(f"     stderr: {res['stderr'][:400]}")

            if not res["success"]:
                fix_success = False
                on_log(f"  ❌ Fix failed: {cmd}")
                previous_fixes.append(cmd)

        if not fix_success:
            stdout = all_results[-1].get("stdout", "")
            stderr = all_results[-1].get("stderr", "")
            continue

        # Retry original command
        on_log(f"  ♻️ Retrying original command: {failed_command}")
        retry_res = runner(failed_command, Config.COMMAND_TIMEOUT)
        all_results.append(retry_res)

        if retry_res["stdout"]:
            on_log(retry_res["stdout"])
        if retry_res["stderr"]:
            on_log(f"  stderr: {retry_res['stderr'][:400]}")

        if retry_res["success"]:
            return True, all_results

        stdout = retry_res.get("stdout", "")
        stderr = retry_res.get("stderr", "")
        previous_fixes.extend(fix_cmds)

    return False, all_results