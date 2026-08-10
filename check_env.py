#!/usr/bin/env python3
"""Environment preflight check for Dominion -- NOT a setuptools build
script (deliberately not named setup.py: a real pyproject.toml now lives
here too -- see its own comment on the dependency tradeoff this fork
makes -- and setuptools' build backend will exec a root-level setup.py
as part of resolving build requirements, which used to break `pip install
-e .` the moment this file existed alongside it).

This just:

  1. Confirms the running Python is new enough.
  2. Checks whether the `ollama` CLI is on PATH, and if so, pulls the
     models the live agents use (dominion.agents.llm_agent's TEXT_MODELS).
  3. Prints the exact commands to install the package and start the
     server.

Ollama is optional: without it (or with it but skipped), the show still
runs end-to-end using the scripted fallback agent -- see
src/dominion/agents/scripted_agent.py and the fallback behavior in
src/dominion/agents/llm_agent.py. This script never fails hard on a
missing or unreachable Ollama; it just tells you what you'll be missing.
For the llama.cpp backend instead of/alongside Ollama, see README.md.

Usage:
    python check_env.py
    python3 check_env.py   (if `python` isn't Python 3 on your system)
"""
from __future__ import annotations
import shutil
import subprocess
import sys
from pathlib import Path

MIN_PYTHON = (3, 10)
REPO_ROOT = Path(__file__).parent.resolve()
SRC_DIR = REPO_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


def check_python_version() -> bool:
    ok = sys.version_info[:2] >= MIN_PYTHON
    have = f"{sys.version_info.major}.{sys.version_info.minor}"
    want = f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}"
    if ok:
        print(f"[ok]   Python {have} (need {want}+)")
    else:
        print(f"[FAIL] Python {have} is too old -- need {want}+")
    return ok


def check_ollama() -> bool:
    path = shutil.which("ollama")
    if path:
        print(f"[ok]   ollama found at {path}")
        return True
    print("[skip] ollama not found on PATH.")
    print("       The show still runs fine without it -- players just use the")
    print("       scripted fallback agent instead of live model decisions.")
    print("       To enable live agents later: https://ollama.com/download")
    return False


def pull_models() -> None:
    from dominion.agents.llm_agent import TEXT_MODELS  # noqa: E402

    print(f"\nPulling {len(TEXT_MODELS)} model(s) used by the live Ollama agents "
          f"(skips any already downloaded)...")
    failed = []
    for model in TEXT_MODELS:
        print(f"\n--- ollama pull {model} ---")
        result = subprocess.run(["ollama", "pull", model])
        if result.returncode != 0:
            failed.append(model)
    if failed:
        print(f"\n[warn] Could not pull: {', '.join(failed)}. "
              f"Players assigned these models will use the scripted fallback "
              f"until you pull them manually (`ollama pull <name>`).")
    else:
        print("\n[ok]   All live-agent Ollama models are ready.")
    print("\nOptional: `ollama pull llava:7b` additionally enables the "
          "vision-verified image fetch in "
          "src/dominion/engine/fetch_images.py --vision (pre-production "
          "only, not needed to play).")


def main() -> None:
    print("Dominion setup check\n" + "=" * 21)
    py_ok = check_python_version()
    if not py_ok:
        sys.exit(1)

    if check_ollama():
        pull_models()

    print("\nSetup check complete. Install the package, then start the server:\n")
    print(f'    "{sys.executable}" -m pip install -e ".[dev]"')
    print(f'    "{sys.executable}" -m uvicorn dominion.server.app:app --port 8765\n')
    print("Then open http://localhost:8765 and click Start Show.")
    print("(Set DOMINION_SCRIPTED_ONLY=1 first to skip live model calls "
          "entirely and run near-instantly.)")


if __name__ == "__main__":
    main()
