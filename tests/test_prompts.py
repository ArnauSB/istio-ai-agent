"""Tests for prompts.py.

Run from the repo root:  python -m unittest discover -s tests
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import prompts  # noqa: E402


class TestIstioSystemPrompt(unittest.TestCase):
    def test_is_nonempty_string(self):
        self.assertIsInstance(prompts.ISTIO_SYSTEM_PROMPT, str)
        self.assertGreater(len(prompts.ISTIO_SYSTEM_PROMPT.strip()), 100)

    def test_mentions_domain(self):
        self.assertIn("Istio", prompts.ISTIO_SYSTEM_PROMPT)

    def test_references_context_categories_used_by_ingestion(self):
        # The prompt teaches the model how to weigh categories; those names
        # must exactly match what config.yaml / ingest_code.py stamp on docs
        # and the `type` that ingest_issues.py uses.
        for label in ("source_code", "istio-documentation",
                      "practical-examples", "github_issue"):
            self.assertIn(label, prompts.ISTIO_SYSTEM_PROMPT, label)

    def test_categories_match_config_yaml(self):
        # Every category configured for ingestion must be explained in the prompt.
        import config
        for repo in config.cfg["github"]["repositories"]:
            category = repo.get("category")
            if category:
                self.assertIn(category, prompts.ISTIO_SYSTEM_PROMPT, category)

    def test_stale_category_names_removed(self):
        # These never matched any ingested document; they must not come back.
        self.assertNotIn("official_documentation", prompts.ISTIO_SYSTEM_PROMPT)
        self.assertNotIn("practical_examples", prompts.ISTIO_SYSTEM_PROMPT)

    def test_no_template_placeholders_left(self):
        self.assertNotIn("{", prompts.ISTIO_SYSTEM_PROMPT)
        self.assertNotIn("}", prompts.ISTIO_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
