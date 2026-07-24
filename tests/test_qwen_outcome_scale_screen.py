from trace_ace import qwen_outcome_screen as screen
from trace_ace.qwen_outcome_scale_screen import configure_e500


def test_e500_configuration_isolated_by_artifact_stem() -> None:
    original = (screen.HIDDEN, screen.ARTIFACT_STEM, screen.CANDIDATE)
    configure_e500()
    try:
        assert screen.HIDDEN == 1536
        assert screen.ARTIFACT_STEM == "qwen_outcome_e500"
        assert screen.CANDIDATE == "E500_qwen2_5_1_5b_outcome"
    finally:
        screen.HIDDEN, screen.ARTIFACT_STEM, screen.CANDIDATE = original
