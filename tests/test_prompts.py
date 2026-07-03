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
        # The prompt teaches the model how to weigh categories; those category
        # names must stay in sync with what the ingestion scripts stamp on docs.
        for category in ("source_code",):
            self.assertIn(category, prompts.ISTIO_SYSTEM_PROMPT, category)

    def test_no_template_placeholders_left(self):
        self.assertNotIn("{", prompts.ISTIO_SYSTEM_PROMPT)
        self.assertNotIn("}", prompts.ISTIO_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
