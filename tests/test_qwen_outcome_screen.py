from trace_ace.qwen_outcome_screen import MAX_LENGTH, prompt_token_ids


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return f"SYS:{messages[0]['content']} USER:{messages[1]['content']} ASSISTANT:"

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))


def test_prompt_truncates_only_evidence() -> None:
    ids = prompt_token_ids(FakeTokenizer(), "add fractions", "word " * 1000)
    assert len(ids) == MAX_LENGTH
