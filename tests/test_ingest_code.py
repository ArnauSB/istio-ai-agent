"""Tests for ingest_code.py.

Run from the repo root:  python -m unittest discover -s tests
"""
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ingest_code  # noqa: E402
import config  # noqa: E402


class TestLoadExclusions(unittest.TestCase):
    def test_reads_patterns_from_yaml(self):
        data = "github:\n  exclude_patterns:\n    - '*.log'\n    - '**/vendor/**'\n"
        with mock.patch("builtins.open", mock.mock_open(read_data=data)):
            self.assertEqual(ingest_code.load_exclusions(), ["*.log", "**/vendor/**"])

    def test_missing_keys_return_empty_list(self):
        with mock.patch("builtins.open", mock.mock_open(read_data="app: {}\n")):
            self.assertEqual(ingest_code.load_exclusions(), [])

    def test_unreadable_file_returns_empty_list(self):
        with mock.patch("builtins.open", side_effect=IOError("denied")):
            self.assertEqual(ingest_code.load_exclusions(), [])


class TestIsExcluded(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(
            ingest_code, "EXCLUDE_PATTERNS",
            ["**/vendor/**", "**/*_test.go", "**/testdata/**", "*.log"],
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_vendor_directory_excluded(self):
        self.assertTrue(ingest_code.is_excluded("repo/vendor/lib/x.go"))

    def test_test_files_excluded(self):
        self.assertTrue(ingest_code.is_excluded("repo/pkg/foo_test.go"))

    def test_basename_pattern_excluded(self):
        self.assertTrue(ingest_code.is_excluded("deep/nested/build.log"))

    def test_regular_source_not_excluded(self):
        self.assertFalse(ingest_code.is_excluded("repo/pkg/main.go"))
        self.assertFalse(ingest_code.is_excluded("docs/setup.md"))

    def test_empty_patterns_exclude_nothing(self):
        with mock.patch.object(ingest_code, "EXCLUDE_PATTERNS", []):
            self.assertFalse(ingest_code.is_excluded("repo/vendor/x.go"))


class TestResetDatabase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.chroma = os.path.join(self.tmp, "chroma_db")
        self.nodes = os.path.join(self.tmp, "storage_nodes")

    def _patch_paths(self):
        return (mock.patch.object(config, "CHROMA_PATH", self.chroma),
                mock.patch.object(config, "STORAGE_NODES_PATH", self.nodes))

    def test_deletes_existing_folders(self):
        os.makedirs(self.chroma)
        os.makedirs(self.nodes)
        p1, p2 = self._patch_paths()
        with p1, p2:
            ingest_code.reset_database()
        self.assertFalse(os.path.exists(self.chroma))
        self.assertFalse(os.path.exists(self.nodes))

    def test_missing_folders_are_fine(self):
        p1, p2 = self._patch_paths()
        with p1, p2:
            ingest_code.reset_database()  # must not raise

    def test_rmtree_error_does_not_abort(self):
        os.makedirs(self.chroma)
        p1, p2 = self._patch_paths()
        with p1, p2, mock.patch.object(ingest_code.shutil, "rmtree",
                                       side_effect=OSError("busy")):
            ingest_code.reset_database()  # must not raise
        self.assertTrue(os.path.exists(self.chroma))  # nothing deleted

    def test_debug_dir_cleaned_only_when_enabled(self):
        debug_dir = os.path.join(self.tmp, "debug_converted")
        os.makedirs(debug_dir)
        p1, p2 = self._patch_paths()
        with p1, p2, mock.patch.object(ingest_code, "DEBUG_ENABLED", False):
            ingest_code.reset_database()
        self.assertTrue(os.path.exists(debug_dir))  # untouched when disabled

        with p1, p2, mock.patch.object(ingest_code, "DEBUG_ENABLED", True), \
                mock.patch.object(ingest_code, "DEBUG_OUTPUT_DIR", debug_dir):
            ingest_code.reset_database()
        self.assertFalse(os.path.exists(debug_dir))


class TestSaveDebugFile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_noop_when_debug_disabled(self):
        with mock.patch.object(ingest_code, "DEBUG_ENABLED", False), \
                mock.patch.object(ingest_code, "DEBUG_OUTPUT_DIR", self.tmp):
            ingest_code.save_debug_file("content", "a/b.md")
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "a", "b.md")))

    def test_writes_nested_file_when_enabled(self):
        with mock.patch.object(ingest_code, "DEBUG_ENABLED", True), \
                mock.patch.object(ingest_code, "DEBUG_OUTPUT_DIR", self.tmp):
            ingest_code.save_debug_file("# converted", "1.28/envoy-docs/x.md")
        path = os.path.join(self.tmp, "1.28", "envoy-docs", "x.md")
        with open(path) as f:
            self.assertEqual(f.read(), "# converted")


class TestConverters(unittest.TestCase):
    def test_rst_conversion_success(self):
        with mock.patch.object(ingest_code.pypandoc, "convert_file",
                               return_value="# Title") as pc:
            self.assertEqual(ingest_code.convert_rst_to_md_pypandoc("doc.rst"), "# Title")
        args, kwargs = pc.call_args
        self.assertEqual(kwargs.get("format", args[2] if len(args) > 2 else None), "rst")

    def test_rst_conversion_failure_returns_empty(self):
        with mock.patch.object(ingest_code.pypandoc, "convert_file",
                               side_effect=RuntimeError("pandoc missing")):
            self.assertEqual(ingest_code.convert_rst_to_md_pypandoc("doc.rst"), "")

    def test_proto_wrapped_in_code_fence(self):
        out = ingest_code.convert_proto_to_md("message Foo {}", "foo.proto")
        self.assertTrue(out.startswith("# foo.proto"))
        self.assertIn("```protobuf\nmessage Foo {}\n```", out)

    def test_proto_empty_content(self):
        out = ingest_code.convert_proto_to_md("", "empty.proto")
        self.assertIn("```protobuf\n\n```", out)


class TestIngestCodeOrchestration(unittest.TestCase):
    """Run ingest_code() end-to-end with all external boundaries mocked."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self._cwd = os.getcwd()
        os.chdir(self.tmp)  # keep data_versions/ writes off the real repo
        self.addCleanup(os.chdir, self._cwd)

        patchers = [
            mock.patch.object(ingest_code, "reset_database"),
            mock.patch.object(ingest_code, "chromadb"),
            mock.patch.object(ingest_code, "ChromaVectorStore"),
            mock.patch.object(ingest_code, "StorageContext"),
            mock.patch.object(ingest_code, "VectorStoreIndex"),
            mock.patch.object(ingest_code, "Settings"),
            mock.patch.object(config, "get_embedding_model"),
            mock.patch.object(config, "STORAGE_NODES_PATH",
                              os.path.join(self.tmp, "storage_nodes")),
        ]
        for p in patchers:
            p.start()
            self.addCleanup(p.stop)

    def _run_with_cfg(self, repositories):
        cfg = {
            "system": {"active_versions": ["1.28"], "default_version": "1.28"},
            "github": {"repositories": repositories, "extensions": [".md"]},
        }
        with mock.patch.object(config, "cfg", cfg):
            ingest_code.ingest_code()

    def test_repo_without_branch_mapping_is_skipped(self):
        with mock.patch.object(ingest_code, "Repo") as repo:
            self._run_with_cfg([{"name": "no-mapping", "url": "https://example.com/x.git"}])
        repo.clone_from.assert_not_called()
        # Indexing still runs, with zero documents.
        (nodes,), _ = ingest_code.VectorStoreIndex.call_args
        self.assertEqual(list(nodes), [])

    def test_clone_uses_mapped_branch(self):
        repo_conf = {
            "name": "mapped", "url": "https://example.com/x.git",
            "version_maps": {"1.28": "release-1.28"},
        }
        with mock.patch.object(ingest_code, "Repo") as repo:
            # clone_from is mocked, so no directory appears -> scan path missing -> skip
            self._run_with_cfg([repo_conf])
        repo.clone_from.assert_called_once()
        _, kwargs = repo.clone_from.call_args
        self.assertEqual(kwargs.get("branch"), "release-1.28")
        self.assertEqual(kwargs.get("depth"), 1)

    def test_clone_failure_is_survived(self):
        repo_conf = {
            "name": "broken", "url": "https://example.com/x.git",
            "version_maps": {"1.28": "release-1.28"},
        }
        with mock.patch.object(ingest_code, "Repo") as repo:
            repo.clone_from.side_effect = Exception("network down")
            self._run_with_cfg([repo_conf])  # must not raise
        repo.clone_from.assert_called_once()

    def test_existing_checkout_reads_standard_files(self):
        base = os.path.join(self.tmp, "data_versions", "1.28", "local")
        os.makedirs(base)
        with open(os.path.join(base, "guide.md"), "w") as f:
            f.write("# istio guide\n\nHello.")
        with open(os.path.join(base, "ignore.txt"), "w") as f:
            f.write("not an allowed extension")

        repo_conf = {"name": "local", "url": "unused",
                     "version_maps": {"1.28": "release-1.28"},
                     "category": "istio-documentation"}
        with mock.patch.object(ingest_code, "Repo") as repo:
            self._run_with_cfg([repo_conf])
        repo.clone_from.assert_not_called()  # folder already existed

        (nodes,), _ = ingest_code.VectorStoreIndex.call_args
        nodes = list(nodes)
        self.assertGreater(len(nodes), 0)
        meta = nodes[0].metadata
        self.assertEqual(meta["repo_name"], "local")
        self.assertEqual(meta["system_version"], "1.28")
        self.assertEqual(meta["git_branch"], "release-1.28")
        self.assertEqual(meta["category"], "istio-documentation")
        self.assertEqual(meta["file_path"], "local/guide.md")
        # .txt is not in the allowed extensions -> must not be ingested
        self.assertTrue(all("ignore.txt" not in n.metadata["file_path"] for n in nodes))


if __name__ == "__main__":
    unittest.main()
