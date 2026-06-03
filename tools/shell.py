"""
tools/shell.py
──────────────
Safe subprocess wrapper for running shell commands on the host system.
"""

from __future__ import annotations

import subprocess
import shlex
import os


# Commands that are too dangerous to ever run
# ADD at top:
import platform


_BLOCKLIST = {
    "rm -rf /",
    "rm -rf /*",
    ":(){ :|:& };:",   # fork bomb
    "mkfs",
    "dd if=/dev/zero of=/dev/sda",
}


def _is_safe(command: str) -> bool:
    """Basic safety check — blocks obviously destructive commands."""
    cmd_lower = command.strip().lower()
    for blocked in _BLOCKLIST:
        if blocked in cmd_lower:
            return False
    return True


def run_command(command: str, timeout: int = 120) -> dict:
    """
    Run *command* in a subprocess shell.

    Returns a dict:
        success    (bool)   : True if exit code == 0
        returncode (int)    : process return code
        stdout     (str)    : captured stdout
        stderr     (str)    : captured stderr
        command    (str)    : the command that was run
    """
    if not _is_safe(command):
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"BLOCKED: command deemed unsafe — '{command}'",
            "command": command,
        }

    env = os.environ.copy()
    # Ensure non-interactive installs don't hang waiting for user input
    env.setdefault("DEBIAN_FRONTEND", "noninteractive")

    try:
        if platform.system() == "Windows":
            proc = subprocess.run(
            f'powershell -ExecutionPolicy Bypass -Command "{command}"',
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        else:
            proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        return {
            "success": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "command": command,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s: {command}",
            "command": command,
        }
    except Exception as exc:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Unexpected error running command: {exc}",
            "command": command,
        }
