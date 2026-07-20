import unittest

from trace_ace.semantic_cache import compact_objective_context


class SemanticCacheTests(unittest.TestCase):
    def test_compaction_prioritizes_ending_and_retains_opening(self) -> None:
        lines = ["[OBJECTIVE] fraction multiplication"] + [
            f"[STUDENT] evidence turn {index}" for index in range(20)
        ]
        compact = compact_objective_context("\n".join(lines))
        self.assertIn("evidence turn 19", compact)
        self.assertIn("evidence turn 0", compact)
        self.assertLess(compact.index("evidence turn 19"), compact.index("evidence turn 0"))

    def test_short_context_keeps_every_line(self) -> None:
        text = "[OBJECTIVE] decimals\n[TUTOR] explain tenths\n[STUDENT] ten tenths make one"
        compact = compact_objective_context(text)
        self.assertIn("explain tenths", compact)
        self.assertIn("ten tenths make one", compact)


if __name__ == "__main__":
    unittest.main()
