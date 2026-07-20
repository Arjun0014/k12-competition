import unittest

from trace_ace.nli_cross_encoder_cache import mastery_hypothesis


class NliCrossEncoderCacheTests(unittest.TestCase):
    def test_mastery_hypothesis_is_deterministic_and_well_formed(self) -> None:
        value = mastery_hypothesis("  Adding   fractions. ")
        self.assertEqual(
            value,
            "The student demonstrates mastery of this learning objective: Adding fractions.",
        )


if __name__ == "__main__":
    unittest.main()
