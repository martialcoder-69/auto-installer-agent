"""
tools/docker_runner.py
──────────────────────
Run shell commands inside a Docker container for sandboxed execution.
Requires Docker to be installed and running on the host.

The agent will create (or reuse) a persistent container named
'auto-installer-sandbox'. This lets state (installed packages, etc.)
persist across consecutive commands in the same session.
"""

from __future__ import annotations

import subprocess

CONTAINER_NAME = "auto-installer-sandbox"
BASE_IMAGE = "ubuntu:22.04"


def _container_exists() -> bool:
    result = subprocess.run(
        ["docker", "inspect", "--type=container", CONTAINER_NAME],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _container_running() -> bool:
    result = subprocess.run(
        ["docker", "inspect", "--format={{.State.Running}}", CONTAINER_NAME],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() == "true"


def _ensure_container() -> None:
    """Create the sandbox container if it doesn't exist, or start it if stopped."""
    if not _container_exists():
        subprocess.run(
            [
                "docker", "run",
                "-d",                   # detached
                "--name", CONTAINER_NAME,
                "--rm",                 # remove when stopped
                BASE_IMAGE,
                "sleep", "infinity",   # keep alive
            ],
            check=True,
        )
    elif not _container_running():
        subprocess.run(["docker", "start", CONTAINER_NAME], check=True)


def run_in_docker(command: str, timeout: int = 120) -> dict:
    """
    Execute *command* inside the sandbox container.

    Returns the same dict structure as tools/shell.py::run_command.
    """
    try:
        _ensure_container()
    except subprocess.CalledProcessError as exc:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Failed to start Docker container: {exc}",
            "command": command,
        }

    docker_cmd = [
        "docker", "exec",
        "-e", "DEBIAN_FRONTEND=noninteractive",
        CONTAINER_NAME,
        "bash", "-c", command,
    ]

    try:
        proc = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
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
            "stderr": f"Docker command timed out after {timeout}s.",
            "command": command,
        }
    except FileNotFoundError:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": "Docker is not installed or not in PATH.",
            "command": command,
        }


def stop_container() -> None:
    """Gracefully stop the sandbox container."""
    if _container_running():
        subprocess.run(["docker", "stop", CONTAINER_NAME], check=False)
