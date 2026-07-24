from __future__ import annotations

from collections.abc import Iterable

from trace_ace import qwen_outcome_screen as screen


def configure_e500() -> None:
    screen.MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    screen.MODEL_DIRECTORY = "Qwen2.5-1.5B-Instruct"
    screen.MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    screen.MODEL_SHA256 = (
        "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
    )
    screen.CONFIG_SHA256 = (
        "98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670"
    )
    screen.TOKENIZER_SHA256 = (
        "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
    )
    screen.HIDDEN = 1536
    screen.ARTIFACT_STEM = "qwen_outcome_e500"
    screen.CANDIDATE = "E500_qwen2_5_1_5b_outcome"
    screen.RUN_SUFFIX = "qwen2_5_1_5b_outcome"


def main(argv: Iterable[str] | None = None) -> None:
    configure_e500()
    screen.main(argv)


if __name__ == "__main__":
    main()
