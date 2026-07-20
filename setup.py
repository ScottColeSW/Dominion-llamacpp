#!/usr/bin/env python3
"""One-time setup check for Dominion. Not a package installer -- the project
has no pip dependencies (see requirements.txt) -- this just:

  1. Confirms the running Python is new enough.
  2. Checks whether the `ollama` CLI is on PATH, and if so, pulls the models
     the live agents use (engine/ollama_agent.py's TEXT_MODELS).
  3. Prints the exact command to start the server, using this same Python
     interpreter, so there's no "python vs python3" guessing afterward.

Ollama is optional: without it (or with it but skipped), the show still
runs end-to-end using the scripted fallback agent -- see
prototype/engine/agents.py and the OllamaAgent fallback behavior in
prototype/engine/ollama_agent.py. This script never fails hard on a missing
or unreachable Ollama; it just tells you what you'll be missing.

Usage:
    python setup.py
    python3 setup.py   (if `python` isn't Python 3 on your system)
"""
from __future__ import annotations
import shutil
import subprocess
import sys
from pathlib import Path

MIN_PYTHON = (3, 9)
REPO_ROOT = Path(__file__).parent.resolve()
PROTOTYPE_DIR = REPO_ROOT / "prototype"

sys.path.insert(0, str(PROTOTYPE_DIR))


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
    from engine.ollama_agent import TEXT_MODELS  # noqa: E402

    print(f"\nPulling {len(TEXT_MODELS)} model(s) used by the live agents "
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
        print("\n[ok]   All live-agent models are ready.")
    print("\nOptional: `ollama pull llava:7b` additionally enables the "
          "vision-verified image fetch in engine/fetch_images.py --vision "
          "(pre-production only, not needed to play).")


def main() -> None:
    print("Dominion setup check\n" + "=" * 21)
    py_ok = check_python_version()
    if not py_ok:
        sys.exit(1)

    if check_ollama():
        pull_models()

    server_path = PROTOTYPE_DIR / "server.py"
    print("\nSetup check complete. Start the server with:\n")
    print(f'    "{sys.executable}" "{server_path}"\n')
    print("Then open http://localhost:8765 and click Start Show.")
    print("(Set DOMINION_SCRIPTED_ONLY=1 first to skip live Ollama calls "
          "entirely and run near-instantly.)")


if __name__ == "__main__":
    main()
