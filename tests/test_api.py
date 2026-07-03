"""Tests for api.py.

Run from the repo root:  python -m unittest discover -s tests
"""
import asyncio
import io
import json
import os
import sys
import time
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# api.py instantiates SentenceTransformerRerank at import time, which would
# download/load the reranker model. Patch it before importing the module.
_rerank_patcher = mock.patch("llama_index.core.postprocessor.SentenceTransformerRerank")
_rerank_patcher.start()
import api  # noqa: E402
_rerank_patcher.stop()

from fastapi import HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from llama_index.core.retrievers import BaseRetriever  # noqa: E402
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle  # noqa: E402

import config  # noqa: E402


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------
class TestSessionStore(unittest.TestCase):
    def test_creates_and_reuses_memory(self):
        store = api.SessionStore(ttl=100, max_sessions=10)
        m1 = store.get_or_create("a")
        m2 = store.get_or_create("a")
        self.assertIs(m1, m2)
        self.assertEqual(len(store), 1)

    def test_distinct_sessions_get_distinct_memory(self):
        store = api.SessionStore(ttl=100, max_sessions=10)
        self.assertIsNot(store.get_or_create("a"), store.get_or_create("b"))
        self.assertEqual(len(store), 2)

    def test_remove_existing_and_missing(self):
        store = api.SessionStore(ttl=100, max_sessions=10)
        store.get_or_create("a")
        self.assertTrue(store.remove("a"))
        self.assertFalse(store.remove("a"))
        self.assertFalse(store.remove("never-existed"))
        self.assertEqual(len(store), 0)

    def test_lru_eviction_respects_recent_use(self):
        store = api.SessionStore(ttl=100, max_sessions=3)
        for sid in ("a", "b", "c"):
            store.get_or_create(sid)
        store.get_or_create("a")  # touch "a" -> "b" becomes LRU
        store.get_or_create("d")  # over cap -> evicts "b"
        self.assertEqual(len(store), 3)
        self.assertFalse(store.remove("b"))
        for sid in ("a", "c", "d"):
            self.assertTrue(store.remove(sid), sid)

    def test_ttl_expires_idle_sessions(self):
        store = api.SessionStore(ttl=0, max_sessions=10)
        store.get_or_create("old")
        time.sleep(0.01)
        store.get_or_create("new")  # purge runs on access
        self.assertFalse(store.remove("old"))
        self.assertTrue(store.remove("new"))

    def test_expired_session_gets_fresh_memory(self):
        store = api.SessionStore(ttl=0, max_sessions=10)
        m1 = store.get_or_create("s")
        time.sleep(0.01)
        m2 = store.get_or_create("s")
        self.assertIsNot(m1, m2)


# ---------------------------------------------------------------------------
# VersionFilteredRetriever
# ---------------------------------------------------------------------------
class _FakeInner(BaseRetriever):
    def __init__(self, nodes):
        self._nodes = nodes
        super().__init__()

    def _retrieve(self, query_bundle):
        return self._nodes

    async def _aretrieve(self, query_bundle):
        return self._nodes


def _node(i, version=None):
    metadata = {} if version is None else {"system_version": version}
    return NodeWithScore(node=TextNode(text=f"doc{i}", metadata=metadata), score=1.0)


class TestVersionFilteredRetriever(unittest.TestCase):
    def setUp(self):
        self.nodes = [
            _node(0, "1.26"),
            _node(1, "1.27"),
            _node(2, "1.28"),
            _node(3, "any"),
            _node(4, "1.28"),
            _node(5, version=None),  # missing metadata -> must be dropped
        ]
        self.retriever = api.VersionFilteredRetriever(_FakeInner(self.nodes), "1.28")
        self.qb = QueryBundle(query_str="q")

    def _versions(self, nodes):
        return sorted(n.node.metadata.get("system_version", "MISSING") for n in nodes)

    def test_sync_filters_other_versions(self):
        result = self.retriever.retrieve(self.qb)
        self.assertEqual(self._versions(result), ["1.28", "1.28", "any"])

    def test_async_filters_other_versions(self):
        result = asyncio.run(self.retriever.aretrieve(self.qb))
        self.assertEqual(self._versions(result), ["1.28", "1.28", "any"])

    def test_empty_input(self):
        retriever = api.VersionFilteredRetriever(_FakeInner([]), "1.28")
        self.assertEqual(retriever.retrieve(self.qb), [])

    def test_no_matches(self):
        retriever = api.VersionFilteredRetriever(_FakeInner(self.nodes), "9.99")
        result = retriever.retrieve(self.qb)
        self.assertEqual(self._versions(result), ["any"])


# ---------------------------------------------------------------------------
# detect_version_intent
# ---------------------------------------------------------------------------
class TestDetectVersionIntent(unittest.TestCase):
    """Uses the real config.yaml: active 1.26-1.28, default 1.28."""

    def test_anchored_istio_keyword(self):
        self.assertEqual(api.detect_version_intent("how does istio 1.27 handle mTLS"),
                         ("1.27", None))

    def test_anchored_version_keyword(self):
        self.assertEqual(api.detect_version_intent("istio version 1.26 please"),
                         ("1.26", None))

    def test_anchored_upgrade_keyword(self):
        self.assertEqual(api.detect_version_intent("upgrade from 1.26 to 1.28"),
                         ("1.26", None))

    def test_anchored_with_v_prefix(self):
        self.assertEqual(api.detect_version_intent("whats new in istio v1.28"),
                         ("1.28", None))

    def test_anchored_ancient_version_warns_and_uses_oldest(self):
        version, warning = api.detect_version_intent("anything about istio 1.15?")
        self.assertEqual(version, "1.26")
        self.assertIn("1.15", warning)
        self.assertIn("1.26", warning)

    def test_anchored_unknown_newer_version_falls_back_with_note(self):
        version, warning = api.detect_version_intent("istio 1.99 features")
        self.assertEqual(version, "1.28")
        self.assertIsNotNone(warning)

    def test_bare_supported_version(self):
        self.assertEqual(api.detect_version_intent("how do I configure a Gateway in 1.27?"),
                         ("1.27", None))

    def test_bare_unsupported_version_ignored_silently(self):
        # Stray version numbers must not trigger a "using default" warning.
        self.assertEqual(api.detect_version_intent("I run kubernetes 1.30, does it matter?"),
                         ("1.28", None))

    def test_semver_noise_in_pasted_config_ignored(self):
        msg = "apiVersion: networking.istio.io/v1 image: envoy:1.5"
        self.assertEqual(api.detect_version_intent(msg), ("1.28", None))

    def test_no_version_returns_default(self):
        self.assertEqual(api.detect_version_intent("what is a VirtualService?"),
                         ("1.28", None))

    def test_malformed_active_versions_hits_fallback_branch(self):
        broken = {"system": {"active_versions": ["abc"], "default_version": "abc"}}
        with mock.patch.object(config, "cfg", broken):
            self.assertEqual(api.detect_version_intent("no version here"), ("abc", None))


# ---------------------------------------------------------------------------
# Endpoints (called directly; lifespan/vector index intentionally NOT started)
# ---------------------------------------------------------------------------
def _upload(data: bytes, filename: str) -> UploadFile:
    return UploadFile(io.BytesIO(data), filename=filename)


def _fake_engine(tokens, nodes):
    """Chat engine double: astream_chat -> stream with token gen + source nodes."""
    stream = SimpleNamespace(source_nodes=nodes)

    async def gen():
        for t in tokens:
            yield t

    stream.async_response_gen = gen
    return SimpleNamespace(astream_chat=mock.AsyncMock(return_value=stream))


def _source_node(score=0.0, **metadata):
    return SimpleNamespace(metadata=metadata, score=score)


async def _collect(resp: StreamingResponse) -> str:
    chunks = []
    async for chunk in resp.body_iterator:
        chunks.append(chunk)
    return "".join(chunks)


class TestEndpoints(unittest.TestCase):
    def test_read_root_serves_index(self):
        result = asyncio.run(api.read_root())
        self.assertIsInstance(result, FileResponse)
        self.assertIn("index.html", result.path)

    def test_favicon(self):
        result = asyncio.run(api.favicon())
        self.assertIsInstance(result, FileResponse)

    def test_health_reports_session_count(self):
        result = asyncio.run(api.health_check())
        self.assertEqual(result["status"], "ok")
        self.assertIsInstance(result["active_sessions"], int)

    def test_reset_clears_existing_session(self):
        api.session_store.get_or_create("reset-me")
        self.assertEqual(asyncio.run(api.reset_chat(session_id="reset-me")),
                         {"status": "memory_cleared"})

    def test_reset_unknown_session(self):
        self.assertEqual(asyncio.run(api.reset_chat(session_id="ghost")),
                         {"status": "no_session_found"})

    def test_chat_rejects_oversized_file(self):
        big = _upload(b"0" * (api.MAX_FILE_SIZE + 1), "big.yaml")
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(api.chat_endpoint(message="hi", session_id="s", file=big))
        self.assertEqual(ctx.exception.status_code, 413)

    def test_chat_rejects_bad_extension(self):
        bad = _upload(b"MZ", "evil.exe")
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(api.chat_endpoint(message="hi", session_id="s", file=bad))
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn(".exe", ctx.exception.detail)

    def test_chat_unavailable_before_startup(self):
        # vector_index is None because the lifespan never ran in tests.
        self.assertIsNone(api.vector_index)
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(api.chat_endpoint(message="hi", session_id="s", file=None))
        self.assertEqual(ctx.exception.status_code, 503)

    def test_chat_engine_error_maps_to_500(self):
        engine = SimpleNamespace(astream_chat=mock.AsyncMock(side_effect=RuntimeError("boom")))
        with mock.patch.object(api, "get_chat_engine", return_value=engine):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(api.chat_endpoint(message="hi", session_id="s", file=None))
        self.assertEqual(ctx.exception.status_code, 500)

    def test_chat_streams_tokens_and_sources(self):
        nodes = [
            _source_node(score=0.0, repo_name="istio-core",
                         file_path="istio-core/pilot/model.go", source=""),
            # duplicate path -> must be deduplicated
            _source_node(score=2.0, repo_name="istio-core",
                         file_path="istio-core/pilot/model.go", source=""),
            _source_node(score=1.0, repo_name="istio/istio", type="github_issue",
                         title="Sidecar bug", source="https://github.com/istio/istio/issues/1"),
        ]
        engine = _fake_engine(["Hello", " world"], nodes)
        with mock.patch.object(api, "get_chat_engine", return_value=engine):
            resp = asyncio.run(api.chat_endpoint(message="explain pilot",
                                                 session_id="stream-test", file=None))
            self.assertIsInstance(resp, StreamingResponse)
            body = asyncio.run(_collect(resp))

        text, sources_json = body.split("__SOURCES__:")
        self.assertIn("Hello world", text)
        sources = json.loads(sources_json)
        self.assertEqual(len(sources), 2)  # duplicate collapsed
        self.assertAlmostEqual(sources[0]["score"], 0.5)  # sigmoid(0) = 0.5
        self.assertEqual(sources[1]["file"], "Issue: Sidecar bug")
        self.assertEqual(sources[1]["url"], "https://github.com/istio/istio/issues/1")

    def test_chat_streams_version_warning_first(self):
        engine = _fake_engine(["answer"], [])
        with mock.patch.object(api, "get_chat_engine", return_value=engine):
            resp = asyncio.run(api.chat_endpoint(message="tell me about istio 1.15",
                                                 session_id="warn-test", file=None))
            body = asyncio.run(_collect(resp))
        self.assertTrue(body.startswith("Note:"), body[:60])
        self.assertIn("1.26", body)

    def test_chat_undecodable_file_yields_warning_not_crash(self):
        engine = _fake_engine(["answer"], [])
        broken = _upload(b"\xff\xfe\xfa\x00", "conf.yaml")  # invalid utf-8
        with mock.patch.object(api, "get_chat_engine", return_value=engine):
            resp = asyncio.run(api.chat_endpoint(message="check my file",
                                                 session_id="decode-test", file=broken))
            body = asyncio.run(_collect(resp))
        self.assertIn("Could not read file conf.yaml", body)
        # The message itself must still reach the engine (without file context).
        (query,), _ = engine.astream_chat.call_args
        self.assertEqual(query, "check my file")

    def test_chat_valid_file_is_embedded_in_query(self):
        engine = _fake_engine(["ok"], [])
        upload = _upload(b"kind: Gateway", "gw.yaml")
        with mock.patch.object(api, "get_chat_engine", return_value=engine):
            resp = asyncio.run(api.chat_endpoint(message="is this valid?",
                                                 session_id="file-test", file=upload))
            asyncio.run(_collect(resp))
        (query,), _ = engine.astream_chat.call_args
        self.assertIn("[Attached File: gw.yaml]", query)
        self.assertIn("kind: Gateway", query)
        self.assertIn("is this valid?", query)


if __name__ == "__main__":
    unittest.main()
