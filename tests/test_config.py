"""Tests for config.py.

Run from the repo root:  python -m unittest discover -s tests
"""
import importlib
import os
import sys
import tempfile
import unittest
from unittest import mock

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402


class TestLoadConfig(unittest.TestCase):
    def test_loads_valid_yaml(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("database:\n  chroma_path: ./x\n")
            path = f.name
        try:
            data = config.load_config(path)
            self.assertEqual(data["database"]["chroma_path"], "./x")
        finally:
            os.unlink(path)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            config.load_config("/nonexistent/nope.yaml")

    def test_malformed_yaml_raises(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("key: [unclosed\n  bad: :::")
            path = f.name
        try:
            with self.assertRaises(yaml.YAMLError):
                config.load_config(path)
        finally:
            os.unlink(path)

    def test_empty_file_returns_none(self):
        # safe_load of an empty document is None; callers must handle it.
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            self.assertIsNone(config.load_config(path))
        finally:
            os.unlink(path)


class TestModuleConstants(unittest.TestCase):
    """The module-level constants must mirror the real config.yaml."""

    def test_database_constants(self):
        self.assertEqual(config.CHROMA_PATH, config.cfg["database"]["chroma_path"])
        self.assertEqual(config.COLLECTION_NAME, config.cfg["database"]["collection_name"])

    def test_storage_nodes_path_has_default(self):
        expected = config.cfg["database"].get("storage_nodes_path", "./storage_nodes")
        self.assertEqual(config.STORAGE_NODES_PATH, expected)

    def test_model_constants(self):
        self.assertEqual(config.MODEL_NAME, config.cfg["models"]["llm_model"])
        self.assertTrue(config.OLLAMA_URL)

    def test_splitter_constants_are_sane(self):
        self.assertIsInstance(config.CHUNK_SIZE, int)
        self.assertIsInstance(config.CHUNK_OVERLAP, int)
        self.assertGreater(config.CHUNK_SIZE, config.CHUNK_OVERLAP)

    def test_debug_defaults_when_missing(self):
        # .get() defaults must not blow up even if the "debug" section vanishes.
        with mock.patch.object(config, "cfg", {"debug": {}}):
            self.assertFalse(config.cfg.get("debug", {}).get("enabled", False))


class TestOllamaUrlEnvOverride(unittest.TestCase):
    def test_env_var_overrides_yaml(self):
        try:
            with mock.patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://override:9999"}):
                importlib.reload(config)
                self.assertEqual(config.OLLAMA_URL, "http://override:9999")
        finally:
            os.environ.pop("OLLAMA_BASE_URL", None)
            importlib.reload(config)  # restore pristine module state

    def test_yaml_value_used_without_env(self):
        self.assertNotIn("OLLAMA_BASE_URL", os.environ)
        self.assertEqual(config.OLLAMA_URL, config.cfg["models"]["ollama_url"])


class TestGetEmbeddingModel(unittest.TestCase):
    def test_builds_model_from_config(self):
        sentinel = object()
        with mock.patch.object(config, "HuggingFaceEmbedding",
                               return_value=sentinel) as hf:
            result = config.get_embedding_model()
        self.assertIs(result, sentinel)
        _, kwargs = hf.call_args
        self.assertEqual(kwargs["model_name"], config.cfg["models"]["embedding_model"])

    def test_construction_error_propagates(self):
        with mock.patch.object(config, "HuggingFaceEmbedding",
                               side_effect=OSError("no such model")):
            with self.assertRaises(OSError):
                config.get_embedding_model()


if __name__ == "__main__":
    unittest.main()
