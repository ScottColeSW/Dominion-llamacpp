"""Installed-model catalog + system memory, backing the pre-show model
picker: choose which installed models populate tonight's roster, with a
live memory budget so picks that would no longer fit grey out.

"Installed" means the same thing on both backends: actually present on
disk, not TEXT_MODELS/LLAMACPP_MODELS's fixed lists (those stay the
picker's pre-checked defaults, not the ceiling on what's selectable).
Ollama's own /api/tags already reports each installed model's real
on-disk size in bytes; llama.cpp has no equivalent API, so this scans
LLAMACPP_MODELS_DIR (inference/config.py) for *.gguf files directly.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

import psutil

from ..inference.config import settings

OLLAMA_TAGS_TIMEOUT = 3.0


def _fetch_json(url: str, timeout: float) -> Optional[dict]:
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError):
        return None


def _ollama_installed_models() -> Dict[str, Any]:
    data = _fetch_json(f"{settings.ollama_url}/api/tags", OLLAMA_TAGS_TIMEOUT)
    reachable = data is not None
    models = []
    for entry in (data or {}).get("models", []):
        name = entry.get("name") or entry.get("model")
        size = entry.get("size")
        if name and isinstance(size, (int, float)):
            models.append({"name": name, "size_bytes": int(size)})
    return {"reachable": reachable, "models": models}


def _llamacpp_installed_models() -> Dict[str, Any]:
    models_dir = Path(settings.llamacpp_models_dir)
    if not models_dir.is_dir():
        return {"reachable": False, "models": []}
    models = []
    for gguf_path in sorted(models_dir.glob("*.gguf")):
        try:
            size = gguf_path.stat().st_size
        except OSError:
            continue
        # .stem, not the full filename -- llama-server's router mode
        # addresses a model by GGUF basename with no extension (see
        # llamacpp_client.py's module docstring), so this is the exact
        # string a request's "model" field needs to select it.
        models.append({"name": gguf_path.stem, "size_bytes": size})
    return {"reachable": True, "models": models}


def get_model_catalog() -> Dict[str, Any]:
    """Cross-platform via psutil, not a platform-specific syscall --
    available_bytes is what the picker's frontend budgets selections
    against (see web/index.html's model-picker logic): a conservative
    "if every selected model were simultaneously resident" sum, not a
    simulation of either backend's actual load/unload/swap behavior,
    which genuinely varies by runtime config (Ollama's idle-unload timeout,
    llama-server's --models-max). A budgeting aid, not a hard guarantee.
    """
    vm = psutil.virtual_memory()
    return {
        "ollama": _ollama_installed_models(),
        "llamacpp": _llamacpp_installed_models(),
        "system_memory": {"total_bytes": vm.total, "available_bytes": vm.available},
    }
