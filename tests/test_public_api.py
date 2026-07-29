"""Smoke tests for the documented package-level public API."""

from __future__ import annotations

import unittest

import motif_cumulants


class PublicAPITests(unittest.TestCase):
    def test_every_declared_public_name_is_importable(self) -> None:
        missing = [
            name for name in motif_cumulants.__all__
            if not hasattr(motif_cumulants, name)
        ]
        self.assertEqual(missing, [])

    def test_version_matches_current_feature_release(self) -> None:
        self.assertEqual(motif_cumulants.__version__, "0.6.0")


if __name__ == "__main__":
    unittest.main()
