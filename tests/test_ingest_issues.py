"""Tests for ingest_issues.py.

Run from the repo root:  python -m unittest discover -s tests
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from github import GithubException, RateLimitExceededException  # noqa: E402

import ingest_issues  # noqa: E402
import config  # noqa: E402


def _comment(login="alice", body="try restarting istiod"):
    c = mock.MagicMock()
    c.user.login = login
    c.created_at = datetime(2026, 1, 1)
    c.body = body
    return c


def _issue(number=1, title="Bug", n_comments=1, comments=None):
    issue = mock.MagicMock()
    issue.title = title
    issue.created_at = datetime(2026, 1, 1)
    issue.state = "open"
    issue.body = f"body of issue {number}"
    issue.comments = n_comments
    issue.get_comments.return_value = (
        comments if comments is not None else [_comment() for _ in range(n_comments)]
    )
    issue.html_url = f"https://github.com/istio/istio/issues/{number}"
    return issue


def _search_results(issues):
    results = mock.MagicMock()
    results.totalCount = len(issues)
    results.__iter__ = lambda self: iter(issues)
    return results


class TestGetRepoSlugFromConfig(unittest.TestCase):
    def test_finds_istio_core_in_real_config(self):
        self.assertEqual(ingest_issues.get_repo_slug_from_config("istio-core"),
                         "istio/istio")

    def test_unknown_repo_returns_none(self):
        self.assertIsNone(ingest_issues.get_repo_slug_from_config("does-not-exist"))

    def test_empty_repository_list(self):
        with mock.patch.object(config, "cfg", {"github": {}}):
            self.assertIsNone(ingest_issues.get_repo_slug_from_config("istio-core"))


class TestFetchGithubIssues(unittest.TestCase):
    def setUp(self):
        # Never actually sleep in tests (rate-limit handlers sleep 60s).
        patcher = mock.patch.object(ingest_issues.time, "sleep")
        patcher.start()
        self.addCleanup(patcher.stop)

        self.gh_patcher = mock.patch.object(ingest_issues, "Github")
        self.auth_patcher = mock.patch.object(ingest_issues, "Auth")
        self.gh = self.gh_patcher.start()
        self.auth_patcher.start()
        self.addCleanup(self.gh_patcher.stop)
        self.addCleanup(self.auth_patcher.stop)
        self.client = self.gh.return_value

    def test_downloads_issues_with_comments(self):
        issues = [_issue(1, title="Sidecar crash", n_comments=2),
                  _issue(2, title="mTLS handshake fails", n_comments=1)]
        self.client.search_issues.return_value = _search_results(issues)

        data = ingest_issues.fetch_github_issues("tok", "istio/istio")

        self.assertEqual(len(data), 2)
        first = data[0]
        self.assertIn("Title: Sidecar crash", first["text"])
        self.assertIn("Comment by alice", first["text"])
        self.assertIn("https://github.com/istio/istio/issues/1", first["text"])
        self.assertEqual(first["metadata"]["type"], "github_issue")
        self.assertEqual(first["metadata"]["system_version"], "any")
        self.assertEqual(first["metadata"]["state"], "open")

    def test_issue_without_comments_skips_comment_section(self):
        self.client.search_issues.return_value = _search_results(
            [_issue(1, n_comments=0)])
        data = ingest_issues.fetch_github_issues("tok", "istio/istio")
        self.assertEqual(len(data), 1)
        self.assertNotIn("CONVERSATION / COMMENTS", data[0]["text"])

    def test_limit_is_enforced(self):
        issues = [_issue(i) for i in range(5)]
        self.client.search_issues.return_value = _search_results(issues)
        data = ingest_issues.fetch_github_issues("tok", "istio/istio", limit=2)
        self.assertEqual(len(data), 2)

    def test_long_threads_are_truncated(self):
        issue = _issue(1, n_comments=60, comments=[_comment() for _ in range(60)])
        self.client.search_issues.return_value = _search_results([issue])
        data = ingest_issues.fetch_github_issues("tok", "istio/istio")
        self.assertIn("truncated long thread", data[0]["text"])

    def test_search_rate_limit_returns_partial_results(self):
        self.client.search_issues.side_effect = RateLimitExceededException(
            403, {"message": "rate limit"}, {})
        data = ingest_issues.fetch_github_issues("tok", "istio/istio")
        self.assertEqual(data, [])

    def test_broken_issue_is_skipped_others_survive(self):
        broken = _issue(1, n_comments=3)
        broken.get_comments.side_effect = GithubException(
            500, {"message": "server error"}, {})
        ok = _issue(2, title="Good issue", n_comments=1)
        self.client.search_issues.return_value = _search_results([broken, ok])

        data = ingest_issues.fetch_github_issues("tok", "istio/istio")

        self.assertEqual(len(data), 1)
        self.assertIn("Good issue", data[0]["text"])

    def test_unexpected_search_error_returns_empty(self):
        self.client.search_issues.side_effect = ValueError("bad query")
        data = ingest_issues.fetch_github_issues("tok", "istio/istio")
        self.assertEqual(data, [])

    def test_query_targets_repo_and_deep_discussions(self):
        self.client.search_issues.return_value = _search_results([])
        ingest_issues.fetch_github_issues("tok", "istio/istio", days_back=10)
        _, kwargs = self.client.search_issues.call_args
        query = kwargs["query"]
        self.assertIn("repo:istio/istio", query)
        self.assertIn("comments:>5", query)


class TestRunIngestion(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self._cwd = os.getcwd()
        os.chdir(self.tmp)  # cache path is relative -> keep it off the repo
        os.makedirs("data_versions")
        self.addCleanup(os.chdir, self._cwd)

    def test_missing_repo_in_config_aborts(self):
        with mock.patch.object(config, "cfg", {"github": {"repositories": []}}), \
                mock.patch.object(ingest_issues, "chromadb") as chroma:
            ingest_issues.run_ingestion()
        chroma.PersistentClient.assert_not_called()

    def test_no_cache_and_no_token_aborts(self):
        with mock.patch.object(config, "GITHUB_TOKEN", None), \
                mock.patch.object(ingest_issues, "chromadb") as chroma:
            ingest_issues.run_ingestion()
        chroma.PersistentClient.assert_not_called()

    def test_no_cache_and_zero_downloaded_aborts(self):
        with mock.patch.object(config, "GITHUB_TOKEN", "tok"), \
                mock.patch.object(ingest_issues, "fetch_github_issues",
                                  return_value=[]), \
                mock.patch.object(ingest_issues, "chromadb") as chroma:
            ingest_issues.run_ingestion()
        chroma.PersistentClient.assert_not_called()
        self.assertFalse(os.path.exists("data_versions/istio_issues_cache.json"))

    def _write_cache(self, records):
        with open("data_versions/istio_issues_cache.json", "w") as f:
            json.dump(records, f)

    def test_cached_issues_are_indexed(self):
        self._write_cache([
            {"text": "Title: Bug\nbody", "metadata": {"source": "http://u/1",
                                                      "type": "github_issue"}},
            {"text": None, "metadata": {"source": "http://u/2",
                                        "type": "github_issue"}},  # null text path
        ])
        nodes_path = os.path.join(self.tmp, "no_such_docstore")
        with mock.patch.object(ingest_issues, "chromadb"), \
                mock.patch.object(ingest_issues, "ChromaVectorStore"), \
                mock.patch.object(ingest_issues, "StorageContext"), \
                mock.patch.object(ingest_issues, "Settings"), \
                mock.patch.object(ingest_issues.VectorStoreIndex,
                                  "from_documents") as from_docs, \
                mock.patch.object(config, "get_embedding_model"), \
                mock.patch.object(config, "STORAGE_NODES_PATH", nodes_path):
            ingest_issues.run_ingestion()

        (documents,), _ = from_docs.call_args
        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0].text, "Title: Bug\nbody")
        self.assertEqual(documents[1].text, "No content")  # null text handled
        self.assertEqual(documents[0].metadata["source"], "http://u/1")

    def test_fresh_download_writes_cache(self):
        fetched = [{"text": "t", "metadata": {"source": "http://u/1"}}]
        nodes_path = os.path.join(self.tmp, "no_such_docstore")
        with mock.patch.object(config, "GITHUB_TOKEN", "tok"), \
                mock.patch.object(ingest_issues, "fetch_github_issues",
                                  return_value=fetched), \
                mock.patch.object(ingest_issues, "chromadb"), \
                mock.patch.object(ingest_issues, "ChromaVectorStore"), \
                mock.patch.object(ingest_issues, "StorageContext"), \
                mock.patch.object(ingest_issues, "Settings"), \
                mock.patch.object(ingest_issues.VectorStoreIndex,
                                  "from_documents"), \
                mock.patch.object(config, "get_embedding_model"), \
                mock.patch.object(config, "STORAGE_NODES_PATH", nodes_path):
            ingest_issues.run_ingestion()

        with open("data_versions/istio_issues_cache.json") as f:
            self.assertEqual(json.load(f), fetched)

    def test_corrupt_docstore_falls_back_to_fresh_context(self):
        self._write_cache([{"text": "t", "metadata": {"source": "http://u/1"}}])
        nodes_path = os.path.join(self.tmp, "existing_docstore")
        os.makedirs(nodes_path)  # exists -> triggers the load-existing branch
        with mock.patch.object(ingest_issues, "chromadb"), \
                mock.patch.object(ingest_issues, "ChromaVectorStore"), \
                mock.patch.object(ingest_issues, "StorageContext") as sc, \
                mock.patch.object(ingest_issues, "Settings"), \
                mock.patch.object(ingest_issues.VectorStoreIndex,
                                  "from_documents"), \
                mock.patch.object(config, "get_embedding_model"), \
                mock.patch.object(config, "STORAGE_NODES_PATH", nodes_path):
            # First from_defaults call (load existing) blows up; the fallback
            # call must succeed and ingestion must complete.
            sc.from_defaults.side_effect = [ValueError("corrupt docstore"),
                                            mock.MagicMock()]
            ingest_issues.run_ingestion()  # must not raise
        self.assertEqual(sc.from_defaults.call_count, 2)


if __name__ == "__main__":
    unittest.main()
