"""server/model_catalog.py -- pure logic against a temp directory of fake
.gguf files (no real llama-server/Ollama needed) and psutil's real system
memory (read-only, always available)."""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from dominion.server.model_catalog import _llamacpp_installed_models, get_model_catalog


class LlamaCppInstalledModelsTest(unittest.TestCase):
    def test_missing_directory_reports_unreachable(self) -> None:
        with patch("dominion.server.model_catalog.settings") as mock_settings:
            mock_settings.llamacpp_models_dir = "/no/such/directory/at/all"
            result = _llamacpp_installed_models()
        self.assertEqual(result, {"reachable": False, "models": []})

    def test_scans_gguf_files_with_real_sizes_stripping_extension(self) -> None:
        with TemporaryDirectory() as tmp:
            gguf_path = Path(tmp) / "qwen2.5-3b-instruct.gguf"
            gguf_path.write_bytes(b"x" * 1024)
            (Path(tmp) / "not-a-model.txt").write_text("ignore me")

            with patch("dominion.server.model_catalog.settings") as mock_settings:
                mock_settings.llamacpp_models_dir = tmp
                result = _llamacpp_installed_models()

        self.assertTrue(result["reachable"])
        self.assertEqual(result["models"], [{"name": "qwen2.5-3b-instruct", "size_bytes": 1024}])


class ModelCatalogTest(unittest.TestCase):
    def test_includes_real_system_memory(self) -> None:
        # _fetch_json (the one real network call) mocked to unreachable --
        # this test is about the catalog's shape/psutil wiring, not
        # actually needing a live Ollama in whatever environment runs it.
        with patch("dominion.server.model_catalog._fetch_json", return_value=None):
            catalog = get_model_catalog()
        self.assertIn("system_memory", catalog)
        self.assertGreater(catalog["system_memory"]["total_bytes"], 0)
        self.assertGreaterEqual(catalog["system_memory"]["available_bytes"], 0)
        self.assertEqual(catalog["ollama"], {"reachable": False, "models": []})
        self.assertIn("llamacpp", catalog)


if __name__ == "__main__":
    unittest.main()
