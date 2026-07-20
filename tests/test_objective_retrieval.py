import unittest

from trace_ace.objective_retrieval import retrieve_objective_context


class ObjectiveRetrievalTests(unittest.TestCase):
    def test_retrieval_selects_matching_turn_and_neighbor(self) -> None:
        text = "\n".join(
            [
                "[TUTOR] Hello and welcome.",
                "[STUDENT] Hello.",
                "[TUTOR] Let us multiply fractions using a diagram.",
                "[STUDENT] I multiply the numerators first.",
                "[TUTOR] Great explanation.",
                "[STUDENT] Goodbye.",
            ]
        )
        context, stats = retrieve_objective_context(
            text, "Multiplying two fractions using visual diagrams", top_lines=1
        )
        self.assertIn("multiply fractions", context)
        self.assertIn("numerators", context)
        self.assertGreater(stats["retrieval_term_coverage"], 0)

    def test_retrieval_fallback_is_deterministic(self) -> None:
        text = "\n".join(f"[TUTOR] unrelated turn {index}" for index in range(50))
        first, first_stats = retrieve_objective_context(text, "calculating circumference")
        second, second_stats = retrieve_objective_context(text, "calculating circumference")
        self.assertEqual(first, second)
        self.assertEqual(first_stats, second_stats)
        self.assertEqual(first_stats["retrieval_positive_lines"], 0)


if __name__ == "__main__":
    unittest.main()
