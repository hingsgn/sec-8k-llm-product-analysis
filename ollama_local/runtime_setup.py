"""Start Ollama in a notebook's runtime, then download the selected local model."""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import requests


def ensure_ollama(model="llama3.2:3b", base_url="http://127.0.0.1:11434", log_dir=".", install_linux=False):
    """Reuse a live server, or start a local server and wait until it is ready.

    Automatic system installation is restricted to an explicitly enabled Linux
    environment such as a Colab VM. Mac/Windows users install the Ollama app first.
    A server running on a laptop is not localhost inside a Colab VM.
    """
    session = requests.Session()
    session.trust_env = False
    base_url = base_url.rstrip("/")
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    def ready():
        try:
            response = session.get(base_url + "/api/tags", timeout=2)
            response.raise_for_status()
            return True
        except requests.RequestException:
            return False

    if not ready():
        binary = os.getenv("ASSIGNMENT_OLLAMA_BINARY") or shutil.which("ollama")
        if not binary:
            if platform.system() != "Linux" or not install_linux:
                raise RuntimeError("Install Ollama from https://ollama.com/download, open the app or run 'ollama serve', then rerun this cell. On Colab enable automatic Linux installation.")
            # Current official Linux packages use zstd compression.
            prefix = [] if hasattr(os, "geteuid") and os.geteuid() == 0 else ["sudo"]
            if not shutil.which("zstd"):
                subprocess.run(prefix + ["apt-get", "update", "-qq"], check=True)
                subprocess.run(prefix + ["apt-get", "install", "-y", "-qq", "zstd"], check=True)
            # Download then execute the official installer, without shell interpolation.
            response = requests.get("https://ollama.com/install.sh", timeout=60)
            response.raise_for_status()
            with tempfile.TemporaryDirectory(prefix="assignment2_ollama_") as temp:
                installer = Path(temp) / "install.sh"
                installer.write_bytes(response.content)
                subprocess.run(["sh", str(installer)], check=True)
            binary = shutil.which("ollama")
        if not binary:
            raise RuntimeError("Ollama installation completed but the binary is missing.")
        env = os.environ.copy()
        env["OLLAMA_HOST"] = "127.0.0.1:11434"
        # Keep the process object on the returned handle; logs survive notebook cells.
        log_path = log_dir / "ollama_server.log"
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen([binary, "serve"], stdout=log, stderr=subprocess.STDOUT, env=env)
        for _ in range(60):
            if ready():
                break
            if process.poll() is not None:
                raise RuntimeError(f"Ollama exited with code {process.returncode}. Read {log_path} for the cause.")
            time.sleep(1)
        else:
            raise RuntimeError(f"Ollama did not become ready. Read {log_path}.")
    else:
        process = None
    response = session.get(base_url + "/api/tags", timeout=10)
    response.raise_for_status()
    installed = {m.get("name") for m in response.json().get("models", [])}
    if model not in installed and model + ":latest" not in installed:
        print(f"Downloading {model} into this runtime. This requires several GB of free space.", flush=True)
        # Streaming shows progress and avoids one long silent download timeout.
        last_print = 0.0
        with session.post(base_url + "/api/pull", json={"model": model, "stream": True},
                          stream=True, timeout=(15, 180)) as pull:
            pull.raise_for_status()
            for line in pull.iter_lines():
                if not line:
                    continue
                item = json.loads(line)
                if item.get("error"):
                    raise RuntimeError(item["error"])
                if time.monotonic() - last_print > 5 or item.get("status") == "success":
                    total = item.get("total", 0)
                    suffix = f" {100 * item.get('completed', 0) / total:.1f}%" if total else ""
                    print(item.get("status", "Downloading") + suffix, flush=True)
                    last_print = time.monotonic()
    print(f"Ollama is ready: {model} at {base_url}", flush=True)
    return process
