from __future__ import annotations

import pandas as pd

from trace_ace.misconception_atlas_discovery import (
    _normalized_question,
    prototype_text,
    query_text,
)


def test_query_omits_correct_answer() -> None:
    row = type(
        "Row",
        (),
        {
            "Question": "What is 2 + 3?",
            "Incorrect_Answer": "6",
            "Correct_Answer": "5",
            "Explanation": "I multiplied.",
        },
    )()
    rendered = query_text(row)
    assert "Student answer: 6" in rendered
    assert "I multiplied." not in rendered
    assert "Correct" not in rendered
    assert "\n5" not in rendered


def test_query_omits_educator_explanation() -> None:
    row = type(
        "Row",
        (),
        {
            "Question": "What is 2 + 3?",
            "Incorrect_Answer": "6",
            "Correct_Answer": "5",
            "Explanation": "The learner multiplied instead of adding.",
        },
    )()
    rendered = query_text(row)
    assert "Student answer: 6" in rendered
    assert "multiplied" not in rendered
    assert "explanation" not in rendered.casefold()


def test_prototype_excludes_duplicate_normalized_question() -> None:
    frame = pd.DataFrame(
        {
            "Question": ["What is 2+3?", "What is 2 + 3?", "What is 4+5?"],
            "Incorrect_Answer": ["6", "7", "20"],
            "Explanation": ["", "", ""],
            "normalized_question": [
                _normalized_question("What is 2+3?"),
                _normalized_question("What is 2 + 3?"),
                _normalized_question("What is 4+5?"),
            ],
        }
    )
    text, count = prototype_text(
        "Adds incorrectly.",
        frame,
        _normalized_question("What is 2+3?"),
    )
    assert count == 1
    assert "What is 4+5?" in text
    assert "What is 2+3?" not in text
