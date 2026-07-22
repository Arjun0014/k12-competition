from __future__ import annotations

import contextlib
import io
import json
import math
import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer


WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)
DIGIT_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
ROLE_PREFIX_RE = re.compile(r"^\[[A-Z_]+\]\s*")
AFFIRM_RE = re.compile(
    r"\b(?:correct|exactly|excellent|great|good|well done|yes|right|brilliant|perfect|super)\b",
    re.IGNORECASE,
)
CORRECTION_RE = re.compile(
    r"\b(?:incorrect|wrong|not quite|try again|almost|check that|check your)\b",
    re.IGNORECASE,
)
UNCERTAINTY_RE = re.compile(
    r"\b(?:i don't know|i do not know|not sure|don't understand|do not understand|confused|help)\b",
    re.IGNORECASE,
)
REASONING_RE = re.compile(
    r"\b(?:because|therefore|so|think|explain|why|how|reason)\b",
    re.IGNORECASE,
)

# These cue patterns are versioned separately because the promoted ordered-feedback
# component was validated with a deliberately broader tutoring-trajectory schema.
ORDERED_AFFIRM_RE = re.compile(
    r"\b(?:correct|exactly|excellent|great|good|well done|yes|right|brilliant|perfect|super)\b",
    re.IGNORECASE,
)
ORDERED_CORRECTION_RE = re.compile(
    r"\b(?:incorrect|wrong|not quite|try again|almost|check(?: that| your)?|remember)\b",
    re.IGNORECASE,
)
ORDERED_UNCERTAINTY_RE = re.compile(
    r"\b(?:don'?t know|not sure|confused|can'?t|cannot|unsure|maybe|guess)\b",
    re.IGNORECASE,
)
ORDERED_REASONING_RE = re.compile(
    r"\b(?:because|therefore|so|means|equals|i think|first|then|next)\b",
    re.IGNORECASE,
)

POSITIVE_FEEDBACK = (
    "correct", "well done", "good job", "exactly", "that's right",
    "thats right", "excellent", "brilliant",
)
CORRECTIVE_FEEDBACK = (
    "not quite", "try again", "have another", "check", "mistake", "remember", "almost",
)
STUDENT_REASONING = ("because", "so ", "therefore", "i think", "means", "equals")
STUDENT_UNCERTAINTY = (
    "don't know", "dont know", "not sure", "confused", "i can't", "i cant",
)

STOPWORDS = {
    "able", "about", "after", "again", "against", "all", "also", "and", "another",
    "any", "are", "around", "been", "before", "being", "between", "both", "can",
    "could", "different", "each", "find", "first", "for", "from", "given", "have",
    "identify", "into", "know", "knowing", "learn", "learning", "make", "more",
    "numbers", "number", "objects", "other", "recognise", "represent", "solve", "than",
    "that", "the", "their", "them", "then", "these", "they", "this", "through",
    "understand", "understanding", "using", "value", "what", "when", "where", "which",
    "with", "within", "work", "working", "would",
}


def load_asset(root: Path) -> dict[str, object]:
    with (root / "assets" / "objective_prior.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def predict_objective_prior(features: pd.DataFrame, asset: dict[str, object]) -> np.ndarray:
    required = {"response_id", "learning_objective_id"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Missing test feature columns: {sorted(missing)}")
    if features["response_id"].duplicated().any():
        raise ValueError("Duplicate response_id values in test features.")
    global_prior = float(asset["global_prior"])
    clip = float(asset.get("probability_clip", 1e-6))
    priors = {str(key): float(value) for key, value in asset["objective_priors"].items()}
    probability = features["learning_objective_id"].map(priors).fillna(global_prior).to_numpy(float)
    return np.clip(probability, clip, 1.0 - clip)


# Backward-compatible name retained for the objective-prior unit tests.
predict = predict_objective_prior


def _count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def _read_transcript(transcript_path: Path) -> dict[str, object]:
    transcript = pd.read_csv(
        transcript_path,
        dtype={"session_id": "string", "role": "string", "content": "string"},
    )
    required = {"utterance_id", "role", "content"}
    missing = required.difference(transcript.columns)
    if missing:
        raise ValueError(f"Missing transcript columns: {sorted(missing)}")
    transcript["utterance_id"] = pd.to_numeric(transcript["utterance_id"], errors="raise")
    transcript = transcript.sort_values("utterance_id", kind="stable")
    roles = transcript["role"].fillna("").str.lower().tolist()
    contents = transcript["content"].fillna("").astype(str).tolist()
    n_rows = len(contents)
    closing_start = max(0, math.floor(n_rows * 0.75))
    opening_end = min(n_rows, max(1, math.ceil(n_rows * 0.25)))

    tutor_parts = [text for role, text in zip(roles, contents) if role == "tutor"]
    student_parts = [text for role, text in zip(roles, contents) if role == "student"]
    closing_tutor_parts = [
        text
        for role, text in zip(roles[closing_start:], contents[closing_start:])
        if role == "tutor"
    ]
    closing_student_parts = [
        text
        for role, text in zip(roles[closing_start:], contents[closing_start:])
        if role == "student"
    ]
    full_text = "\n".join(
        f"[{role.upper()}] {text}" for role, text in zip(roles, contents)
    )
    tutor_text = "\n".join(tutor_parts)
    student_text = "\n".join(student_parts)
    closing_tutor = "\n".join(closing_tutor_parts)
    closing_student = "\n".join(closing_student_parts)
    opening_text = "\n".join(
        f"[{role.upper()}] {text}"
        for role, text in zip(roles[:opening_end], contents[:opening_end])
    )
    tutor_words = _count_words(tutor_text)
    student_words = _count_words(student_text)
    behavior = np.asarray(
        [
            sum(left != right for left, right in zip(roles, roles[1:])),
            tutor_words,
            student_words,
            tutor_text.count("?"),
            student_text.count("?"),
            len(AFFIRM_RE.findall(tutor_text)),
            len(CORRECTION_RE.findall(tutor_text)),
            len(UNCERTAINTY_RE.findall(student_text)),
            len(REASONING_RE.findall(student_text)),
            len(REASONING_RE.findall(tutor_text)),
            len(DIGIT_RE.findall(student_text)),
            len(DIGIT_RE.findall(tutor_text)),
            student_words / max(1, len(student_parts)),
            tutor_words / max(1, len(tutor_parts)),
            _count_words(closing_student),
            _count_words(closing_tutor),
        ],
        dtype=np.float64,
    )
    return {
        "dialogue_lines": list(zip(roles, contents)),
        "full_text": full_text,
        "student_text": student_text,
        "tutor_text": tutor_text,
        "closing_student_text": closing_student,
        "closing_tutor_text": closing_tutor,
        "opening_text": opening_text,
        "behavior": behavior,
    }


def transcript_text(transcript_path: Path) -> str:
    return str(_read_transcript(transcript_path)["full_text"])


def _stem(token: str) -> str:
    if len(token) > 6 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 5 and token.endswith("ied"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 5 and token.endswith("es"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token


def _tokens(text: str) -> list[str]:
    return [_stem(token.lower()) for token in TOKEN_RE.findall(text)]


def _objective_terms(text: str) -> tuple[list[str], list[tuple[str, str]]]:
    raw = _tokens(text)
    terms = [token for token in raw if len(token) >= 3 and token not in STOPWORDS]
    unique_terms = list(dict.fromkeys(terms))
    bigrams = [
        (left, right)
        for left, right in zip(raw, raw[1:])
        if left not in STOPWORDS and right not in STOPWORDS and len(left) >= 3 and len(right) >= 3
    ]
    return unique_terms, list(dict.fromkeys(bigrams))


def _compact(text: str, max_words: int) -> str:
    return " ".join(str(text).split()[:max_words])


def _line_token_features(
    lines: list[tuple[str, str]],
) -> list[tuple[set[str], set[tuple[str, str]]]]:
    features = []
    for _, content in lines:
        tokens = _tokens(content)
        features.append((set(tokens), set(zip(tokens, tokens[1:]))))
    return features


def _objective_score_from_features(
    token_set: set[str],
    line_bigrams: set[tuple[str, str]],
    term_set: set[str],
    objective_bigrams: set[tuple[str, str]],
) -> float:
    overlap = term_set.intersection(token_set)
    score = sum(1.0 + min(len(token), 10) / 20.0 for token in overlap)
    if objective_bigrams and line_bigrams:
        score += 1.5 * len(objective_bigrams.intersection(line_bigrams))
    return float(score)


def _feedback_windows(lines: list[tuple[str, str]], objective: str) -> str:
    terms, bigrams = _objective_terms(objective)
    term_set = set(terms)
    bigram_set = set(bigrams)
    cached_features = _line_token_features(lines)
    dialogue = [
        (index, role, content)
        for index, (role, content) in enumerate(lines)
        if role in {"student", "tutor"}
    ]
    windows: list[tuple[int, float, str]] = []
    for (left_index, left_role, left), (right_index, right_role, right) in zip(
        dialogue, dialogue[1:]
    ):
        if left_role == right_role:
            continue
        left_tokens, left_bigrams = cached_features[left_index]
        right_tokens, right_bigrams = cached_features[right_index]
        objective_score = _objective_score_from_features(
            left_tokens, left_bigrams, term_set, bigram_set
        ) + _objective_score_from_features(
            right_tokens, right_bigrams, term_set, bigram_set
        )
        if left_role == "student":
            student, tutor = left, right
            rendered = (
                f"[ANSWER] {_compact(left, 14)}\n"
                f"[FEEDBACK] {_compact(right, 14)}"
            )
        else:
            student, tutor = right, left
            rendered = (
                f"[PROMPT] {_compact(left, 14)}\n"
                f"[RESPONSE] {_compact(right, 14)}"
            )
        student_lower = student.lower()
        tutor_lower = tutor.lower()
        cue_bonus = 0.0
        cue_bonus += sum(cue in tutor_lower for cue in POSITIVE_FEEDBACK)
        cue_bonus += sum(cue in tutor_lower for cue in CORRECTIVE_FEEDBACK)
        cue_bonus += 0.5 * sum(cue in student_lower for cue in STUDENT_REASONING)
        cue_bonus += 0.5 * sum(cue in student_lower for cue in STUDENT_UNCERTAINTY)
        windows.append((max(left_index, right_index), objective_score + cue_bonus, rendered))

    selected: set[int] = set()
    ranked = sorted(
        enumerate(windows), key=lambda item: (item[1][1], item[1][0]), reverse=True
    )
    selected.update(index for index, (_, score, _) in ranked[:5] if score > 0)
    selected.update(range(max(0, len(windows) - 2), len(windows)))
    if windows and len(selected) < min(4, len(windows)):
        positions = np.linspace(0, len(windows) - 1, min(4, len(windows))).astype(int)
        selected.update(int(position) for position in positions)
    rendered = [
        windows[index][2]
        for index in sorted(selected, key=lambda item: windows[item][0])
    ]
    return "\n".join([f"[OBJECTIVE] {objective.strip()}", *rendered])


def _ordered_feature_values(
    lines: list[tuple[str, str]], objective: str
) -> np.ndarray:
    terms, _ = _objective_terms(objective)
    term_set = set(terms)
    term_denominator = max(1, len(term_set))
    dialogue: list[dict[str, object]] = []
    for role, content in lines:
        if role not in {"student", "tutor"}:
            continue
        tokens = _tokens(content)
        dialogue.append(
            {
                "role": role,
                "tokens": tokens,
                "overlap": float(len(term_set.intersection(tokens)) / term_denominator),
                "words": float(len(str(content).split())),
                "affirm": float(bool(ORDERED_AFFIRM_RE.search(content))) if role == "tutor" else 0.0,
                "correction": float(bool(ORDERED_CORRECTION_RE.search(content))) if role == "tutor" else 0.0,
                "uncertainty": float(bool(ORDERED_UNCERTAINTY_RE.search(content))) if role == "student" else 0.0,
                "reasoning": float(bool(ORDERED_REASONING_RE.search(content))) if role == "student" else 0.0,
            }
        )

    count = len(dialogue)
    for index, turn in enumerate(dialogue):
        turn["position"] = float(index / max(1, count - 1))
    student = [turn for turn in dialogue if turn["role"] == "student"]
    tutor = [turn for turn in dialogue if turn["role"] == "tutor"]
    early = [turn for turn in dialogue if float(turn["position"]) <= 1.0 / 3.0]
    late = [turn for turn in dialogue if float(turn["position"]) >= 2.0 / 3.0]

    def mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    def cue_count(turns: list[dict[str, object]], name: str) -> float:
        return float(sum(float(turn[name]) for turn in turns))

    def role_values(
        turns: list[dict[str, object]], role: str, field: str
    ) -> list[float]:
        return [float(turn[field]) for turn in turns if turn["role"] == role]

    student_overlap = role_values(dialogue, "student", "overlap")
    tutor_overlap = role_values(dialogue, "tutor", "overlap")
    student_early_overlap = role_values(early, "student", "overlap")
    student_late_overlap = role_values(late, "student", "overlap")
    tutor_early_overlap = role_values(early, "tutor", "overlap")
    tutor_late_overlap = role_values(late, "tutor", "overlap")

    affirm_early = cue_count(early, "affirm")
    affirm_late = cue_count(late, "affirm")
    correction_early = cue_count(early, "correction")
    correction_late = cue_count(late, "correction")
    uncertainty_early = cue_count(early, "uncertainty")
    uncertainty_late = cue_count(late, "uncertainty")
    reasoning_early = cue_count(early, "reasoning")
    reasoning_late = cue_count(late, "reasoning")

    feedback_pairs = 0.0
    affirm_pairs = 0.0
    correction_pairs = 0.0
    objective_affirm_pairs = 0.0
    objective_correction_pairs = 0.0
    uncertain_correction_pairs = 0.0
    uncertain_affirm_pairs = 0.0
    reasoning_affirm_pairs = 0.0
    repair_opportunities = 0.0
    repair_overlap_improved = 0.0
    repair_later_affirm = 0.0
    repair_deltas: list[float] = []
    for index in range(max(0, count - 1)):
        answer = dialogue[index]
        feedback = dialogue[index + 1]
        if answer["role"] != "student" or feedback["role"] != "tutor":
            continue
        feedback_pairs += 1.0
        affirmation = float(feedback["affirm"])
        correction = float(feedback["correction"])
        objective_linked = float(answer["overlap"]) > 0.0
        affirm_pairs += affirmation
        correction_pairs += correction
        objective_affirm_pairs += affirmation * objective_linked
        objective_correction_pairs += correction * objective_linked
        uncertain_correction_pairs += float(answer["uncertainty"]) * correction
        uncertain_affirm_pairs += float(answer["uncertainty"]) * affirmation
        reasoning_affirm_pairs += float(answer["reasoning"]) * affirmation
        if not correction:
            continue
        repair_opportunities += 1.0
        next_students = [
            later for later in dialogue[index + 2 :] if later["role"] == "student"
        ]
        if next_students:
            delta = float(next_students[0]["overlap"]) - float(answer["overlap"])
            repair_deltas.append(delta)
            repair_overlap_improved += float(delta > 0.0)
        repair_later_affirm += float(
            any(float(later["affirm"]) > 0.0 for later in dialogue[index + 2 :])
        )

    role_switches = float(
        sum(left["role"] != right["role"] for left, right in zip(dialogue, dialogue[1:]))
    )
    last_affirm = max(
        (float(turn["position"]) for turn in tutor if float(turn["affirm"]) > 0.0),
        default=-1.0,
    )
    last_correction = max(
        (float(turn["position"]) for turn in tutor if float(turn["correction"]) > 0.0),
        default=-1.0,
    )
    student_words_early = mean(role_values(early, "student", "words"))
    student_words_late = mean(role_values(late, "student", "words"))
    tutor_words_early = mean(role_values(early, "tutor", "words"))
    tutor_words_late = mean(role_values(late, "tutor", "words"))
    student_objective_early = mean(student_early_overlap)
    student_objective_late = mean(student_late_overlap)
    tutor_objective_early = mean(tutor_early_overlap)
    tutor_objective_late = mean(tutor_late_overlap)
    repair_rate = (
        (repair_overlap_improved + repair_later_affirm) / (2.0 * repair_opportunities)
        if repair_opportunities else 0.0
    )
    values = np.asarray(
        [
            float(count), float(len(student)), float(len(tutor)), role_switches,
            float(len(student) / max(1, count)), affirm_early, affirm_late,
            affirm_late - affirm_early, correction_early, correction_late,
            correction_late - correction_early, uncertainty_early, uncertainty_late,
            uncertainty_late - uncertainty_early, reasoning_early, reasoning_late,
            reasoning_late - reasoning_early, mean(student_overlap),
            max(student_overlap, default=0.0), student_overlap[0] if student_overlap else 0.0,
            student_overlap[-1] if student_overlap else 0.0, student_objective_early,
            student_objective_late, student_objective_late - student_objective_early,
            float(np.mean(np.asarray(student_overlap) > 0.0)) if student_overlap else 0.0,
            mean(tutor_overlap), max(tutor_overlap, default=0.0),
            tutor_overlap[-1] if tutor_overlap else 0.0,
            tutor_objective_late - tutor_objective_early,
            float(np.mean(np.asarray(tutor_overlap) > 0.0)) if tutor_overlap else 0.0,
            feedback_pairs, affirm_pairs, correction_pairs, objective_affirm_pairs,
            objective_correction_pairs, uncertain_correction_pairs, uncertain_affirm_pairs,
            reasoning_affirm_pairs, repair_opportunities, repair_overlap_improved,
            repair_later_affirm, mean(repair_deltas), repair_rate, student_words_early,
            student_words_late, student_words_late - student_words_early, tutor_words_early,
            tutor_words_late, tutor_words_late - tutor_words_early, last_affirm,
            last_correction,
        ],
        dtype=np.float64,
    )
    if values.shape != (51,) or not np.isfinite(values).all():
        raise RuntimeError("Ordered feedback feature extraction failed its schema contract.")
    return values


def retrieve_objective_context(full_text: str, objective: str) -> tuple[str, np.ndarray]:
    lines = [line.strip() for line in full_text.splitlines() if line.strip()]
    terms, bigrams = _objective_terms(objective)
    term_set = set(terms)
    scores = np.zeros(len(lines), dtype=np.float64)
    matched_union: set[str] = set()
    for index, line in enumerate(lines):
        line_tokens = _tokens(ROLE_PREFIX_RE.sub("", line))
        overlap = term_set.intersection(line_tokens)
        if overlap:
            matched_union.update(overlap)
            scores[index] += sum(1.0 + min(len(token), 10) / 20.0 for token in overlap)
        if bigrams and line_tokens:
            line_bigrams = set(zip(line_tokens, line_tokens[1:]))
            scores[index] += 1.5 * sum(pair in line_bigrams for pair in bigrams)
    positive = np.flatnonzero(scores > 0)
    if len(positive):
        ranked = positive[np.argsort(scores[positive])[::-1][:24]]
        selected: set[int] = set()
        for index in ranked:
            selected.update(range(max(0, index - 1), min(len(lines), index + 2)))
        selected_indices = sorted(selected)
        if len(selected_indices) > 72:
            priority = sorted(selected_indices, key=lambda item: scores[item], reverse=True)
            selected_indices = sorted(priority[:72])
    else:
        fallback = min(18, len(lines))
        selected_indices = sorted(
            set(range(fallback)).union(range(max(0, len(lines) - fallback), len(lines)))
        )
    if len(positive):
        first_position = positive.min() / max(1, len(lines) - 1)
        last_position = positive.max() / max(1, len(lines) - 1)
        mean_score = scores[positive].mean()
        max_score = scores[positive].max()
    else:
        first_position = last_position = -1.0
        mean_score = max_score = 0.0
    stats = np.asarray(
        [
            len(lines), len(positive), len(selected_indices), len(term_set), len(matched_union),
            len(matched_union) / max(1, len(term_set)), first_position, last_position,
            mean_score, max_score,
        ],
        dtype=np.float64,
    )
    context = "\n".join(
        [f"[OBJECTIVE] {objective.strip()}", *(lines[index] for index in selected_indices)]
    )
    return context, stats


def compact_objective_context(text: str) -> str:
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines:
        return ""
    objective = lines[0].replace("[OBJECTIVE]", "Objective:", 1)
    discussion = lines[1:]
    selected = discussion if len(discussion) <= 12 else [*discussion[-8:], *discussion[:4]]
    compact_lines = [" ".join(line.split()[:16]) for line in selected]
    ending_count = min(8, len(discussion))
    parts = [objective]
    if compact_lines[:ending_count]:
        parts.append("End of objective discussion: " + " ".join(compact_lines[:ending_count]))
    if compact_lines[ending_count:]:
        parts.append("Earlier objective discussion: " + " ".join(compact_lines[ending_count:]))
    return "\n".join(parts)


def _hasher(n_features: int) -> HashingVectorizer:
    return HashingVectorizer(
        analyzer="word", ngram_range=(1, 2), n_features=n_features,
        alternate_sign=False, norm="l2", lowercase=True, dtype=np.float32,
    )


def _row_cosine(left: sparse.csr_matrix, right: sparse.csr_matrix) -> np.ndarray:
    return np.asarray(left.multiply(right).sum(axis=1)).ravel()


def _unit_hstack(blocks: list[tuple[sparse.csr_matrix, float]]) -> sparse.csr_matrix:
    norm = math.sqrt(sum(weight * weight for _, weight in blocks))
    return sparse.hstack(
        [matrix * np.float32(weight / norm) for matrix, weight in blocks],
        format="csr", dtype=np.float32,
    )


def _behavior_features(raw: np.ndarray) -> np.ndarray:
    values = np.log1p(np.maximum(raw, 0.0))
    tutor_words = raw[:, 1]
    student_words = raw[:, 2]
    closing_student = raw[:, 14]
    closing_tutor = raw[:, 15]
    eps = 1.0
    ratios = np.column_stack(
        [
            student_words / (tutor_words + eps),
            closing_student / (student_words + eps),
            closing_tutor / (tutor_words + eps),
            raw[:, 4] / (student_words + eps),
            raw[:, 5] / (tutor_words + eps),
            raw[:, 6] / (tutor_words + eps),
            raw[:, 7] / (student_words + eps),
            raw[:, 8] / (student_words + eps),
        ]
    )
    return np.column_stack([values, ratios])


def _load_encoder(model_dir: Path):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    import transformers
    from sentence_transformers import SentenceTransformer
    transformers.logging.set_verbosity_error()
    transformers.logging.disable_progress_bar()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        encoder = SentenceTransformer(str(model_dir), device=device)
    encoder.max_seq_length = 256
    return encoder, device


def predict_final_ensemble(
    features: pd.DataFrame,
    artifact: dict[str, object],
    transcript_dir: Path,
    encoder_dir: Path,
) -> np.ndarray:
    required = {"response_id", "session_id", "learning_objective_id", "learning_objective"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Missing test feature columns: {sorted(missing)}")
    if features["response_id"].duplicated().any():
        raise ValueError("Duplicate response_id values in test features.")

    sessions = pd.Index(features["session_id"].drop_duplicates())
    records = [_read_transcript(transcript_dir / f"{session_id}.csv") for session_id in sessions]
    lookup = pd.Series(np.arange(len(sessions)), index=sessions)
    response_rows = features["session_id"].map(lookup).to_numpy(dtype=np.int64)
    objectives = features["learning_objective"].fillna("").astype(str).tolist()

    full_session = _hasher(2**17).transform([record["full_text"] for record in records]).tocsr()
    full_response = full_session[response_rows]
    full_probability = artifact["full_model"].predict_proba(full_response)[:, 1]

    student = _hasher(2**16).transform([record["student_text"] for record in records]).tocsr()[response_rows]
    tutor = _hasher(2**16).transform([record["tutor_text"] for record in records]).tocsr()[response_rows]
    opening = _hasher(2**16).transform([record["opening_text"] for record in records]).tocsr()[response_rows]
    closing_student = _hasher(2**15).transform(
        [record["closing_student_text"] for record in records]
    ).tocsr()[response_rows]
    closing_tutor = _hasher(2**15).transform(
        [record["closing_tutor_text"] for record in records]
    ).tocsr()[response_rows]
    objective_64k = _hasher(2**16).transform(objectives).tocsr()
    objective_32k = _hasher(2**15).transform(objectives).tocsr()

    behavior_raw = np.vstack([record["behavior"] for record in records])[response_rows]
    behavior = _behavior_features(behavior_raw)
    role_alignment = np.column_stack(
        [
            _row_cosine(student, objective_64k), _row_cosine(tutor, objective_64k),
            _row_cosine(opening, objective_64k), _row_cosine(closing_student, objective_32k),
            _row_cosine(closing_tutor, objective_32k),
        ]
    )
    role_dense = artifact["role_scaler"].transform(
        np.column_stack([behavior, role_alignment])
    ).astype(np.float32)
    role_sparse = _unit_hstack(
        [(student, 0.75), (tutor, 0.55), (objective_64k, 1.0)]
    )
    role_matrix = sparse.hstack(
        [role_sparse, sparse.csr_matrix(role_dense * np.float32(0.2))], format="csr"
    )
    role_probability = artifact["role_model"].predict_proba(role_matrix)[:, 1]

    contexts: list[str] = []
    retrieval_stats: list[np.ndarray] = []
    for row_index, objective in enumerate(objectives):
        record = records[response_rows[row_index]]
        context, stats = retrieve_objective_context(str(record["full_text"]), objective)
        contexts.append(compact_objective_context(context))
        retrieval_stats.append(stats)
    retrieval = np.vstack(retrieval_stats)
    retrieval[:, [0, 1, 2, 3, 4, 8, 9]] = np.log1p(
        np.maximum(retrieval[:, [0, 1, 2, 3, 4, 8, 9]], 0.0)
    )

    encoder, device = _load_encoder(encoder_dir)
    batch_size = 128 if device == "cuda" else 32
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        context_embedding = encoder.encode(
            contexts, batch_size=batch_size, show_progress_bar=False,
            normalize_embeddings=True, convert_to_numpy=True,
        ).astype(np.float32)
        objective_embedding = encoder.encode(
            objectives, batch_size=batch_size, show_progress_bar=False,
            normalize_embeddings=True, convert_to_numpy=True,
        ).astype(np.float32)
    semantic_features = np.column_stack(
        [
            context_embedding * np.float32(0.7),
            objective_embedding * np.float32(0.7),
            context_embedding * objective_embedding * np.float32(6.0),
            np.abs(context_embedding - objective_embedding) * np.float32(0.7),
        ]
    )
    semantic_cosine = np.sum(context_embedding * objective_embedding, axis=1, keepdims=True)
    semantic_dense = artifact["semantic_scaler"].transform(
        np.column_stack([behavior, retrieval, semantic_cosine])
    ).astype(np.float32)
    semantic_matrix = np.column_stack(
        [semantic_features, semantic_dense * np.float32(0.08)]
    )
    semantic_probability = artifact["semantic_model"].predict_proba(semantic_matrix)[:, 1]

    weights = artifact["ensemble_weights"]
    probability = (
        float(weights["full_transcript"]) * full_probability
        + float(weights["role_objective_dense"]) * role_probability
        + float(weights["semantic_interaction_dense"]) * semantic_probability
    )
    clip = float(artifact.get("probability_clip", 1e-6))
    return np.clip(probability, clip, 1.0 - clip)


def predict_feedback_upgrade(
    features: pd.DataFrame,
    artifact: dict[str, object],
    transcript_dir: Path,
    base_probability: np.ndarray,
) -> np.ndarray:
    required_keys = {
        "feedback_model", "feedback_scaler", "feedback_word_features",
        "feedback_dense_weight", "feedback_blend_weight",
        "feedback_ordered_feature_names",
    }
    missing_keys = required_keys.difference(artifact)
    if missing_keys:
        raise ValueError(f"Feedback artifact is missing keys: {sorted(missing_keys)}")
    if len(artifact["feedback_ordered_feature_names"]) != 51:
        raise ValueError("Feedback artifact ordered feature schema has an invalid width.")

    sessions = pd.Index(features["session_id"].drop_duplicates())
    records = [_read_transcript(transcript_dir / f"{session_id}.csv") for session_id in sessions]
    lookup = pd.Series(np.arange(len(sessions)), index=sessions)
    response_rows = features["session_id"].map(lookup).to_numpy(dtype=np.int64)
    objectives = features["learning_objective"].fillna("").astype(str).tolist()

    feedback_texts: list[str] = []
    ordered_rows: list[np.ndarray] = []
    for row_index, objective in enumerate(objectives):
        record = records[response_rows[row_index]]
        lines = record["dialogue_lines"]
        feedback_texts.append(_feedback_windows(lines, objective))
        ordered_rows.append(_ordered_feature_values(lines, objective))
    word_matrix = _hasher(int(artifact["feedback_word_features"])).transform(
        feedback_texts
    ).tocsr()
    ordered = np.vstack(ordered_rows)
    dense_scaled = artifact["feedback_scaler"].transform(ordered).astype(np.float32)
    feedback_matrix = sparse.hstack(
        [
            word_matrix,
            sparse.csr_matrix(
                dense_scaled * np.float32(artifact["feedback_dense_weight"])
            ),
        ],
        format="csr",
    )
    feedback_probability = artifact["feedback_model"].predict_proba(feedback_matrix)[:, 1]
    weight = float(artifact["feedback_blend_weight"])
    probability = (1.0 - weight) * base_probability + weight * feedback_probability
    clip = float(artifact.get("probability_clip", 1e-6))
    return np.clip(probability, clip, 1.0 - clip)


def predict_nbsvm_feedback(
    features: pd.DataFrame,
    artifact: dict[str, object],
    transcript_dir: Path,
) -> np.ndarray:
    required_keys = {
        "nbsvm_model",
        "nbsvm_scaler",
        "nbsvm_log_count_ratio",
        "nbsvm_word_features",
        "nbsvm_dense_weight",
        "nbsvm_ordered_feature_names",
    }
    missing_keys = required_keys.difference(artifact)
    if missing_keys:
        raise ValueError(f"NB-SVM artifact is missing keys: {sorted(missing_keys)}")
    if len(artifact["nbsvm_ordered_feature_names"]) != 51:
        raise ValueError("NB-SVM ordered feature schema has an invalid width.")

    sessions = pd.Index(features["session_id"].drop_duplicates())
    records = [_read_transcript(transcript_dir / f"{session_id}.csv") for session_id in sessions]
    lookup = pd.Series(np.arange(len(sessions)), index=sessions)
    response_rows = features["session_id"].map(lookup).to_numpy(dtype=np.int64)
    objectives = features["learning_objective"].fillna("").astype(str).tolist()
    feedback_texts: list[str] = []
    ordered_rows: list[np.ndarray] = []
    for row_index, objective in enumerate(objectives):
        lines = records[response_rows[row_index]]["dialogue_lines"]
        feedback_texts.append(_feedback_windows(lines, objective))
        ordered_rows.append(_ordered_feature_values(lines, objective))

    word_matrix = _hasher(int(artifact["nbsvm_word_features"])).transform(
        feedback_texts
    ).tocsr()
    ratio = np.asarray(artifact["nbsvm_log_count_ratio"], dtype=np.float32)
    if ratio.shape != (word_matrix.shape[1],) or not np.isfinite(ratio).all():
        raise ValueError("NB-SVM ratio violates its feature contract.")
    word_matrix = word_matrix.multiply(ratio).tocsr()
    ordered = np.vstack(ordered_rows)
    dense = artifact["nbsvm_scaler"].transform(ordered).astype(np.float32)
    matrix = sparse.hstack(
        [
            word_matrix,
            sparse.csr_matrix(
                dense * np.float32(artifact["nbsvm_dense_weight"])
            ),
        ],
        format="csr",
    )
    probability = artifact["nbsvm_model"].predict_proba(matrix)[:, 1]
    clip = float(artifact.get("probability_clip", 1e-6))
    return np.clip(probability, clip, 1.0 - clip)


def _mastery_hypothesis(objective: str) -> str:
    return (
        "The student demonstrates mastery of this learning objective: "
        f"{str(objective).strip()}."
    )


def _load_supervised_model(model_dir: Path, delta_path: Path):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    import transformers
    from safetensors.torch import load_file
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    transformers.logging.set_verbosity_error()
    transformers.logging.disable_progress_bar()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            num_labels=1,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )
        delta = load_file(str(delta_path), device="cpu")
        result = model.load_state_dict(delta, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(
            f"Unexpected supervised delta keys: {sorted(result.unexpected_keys)}"
        )
    if not {"classifier.weight", "classifier.bias"}.issubset(delta):
        raise RuntimeError("Supervised delta is missing its classifier tensors.")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_supervised_component(
    features: pd.DataFrame,
    artifact: dict[str, object],
    transcript_dir: Path,
    encoder_dir: Path,
    delta_path: Path,
) -> np.ndarray:
    import torch

    sessions = pd.Index(features["session_id"].drop_duplicates())
    records = [_read_transcript(transcript_dir / f"{session_id}.csv") for session_id in sessions]
    lookup = pd.Series(np.arange(len(sessions)), index=sessions)
    response_rows = features["session_id"].map(lookup).to_numpy(dtype=np.int64)
    objectives = features["learning_objective"].fillna("").astype(str).tolist()
    contexts: list[str] = []
    for row_index, objective in enumerate(objectives):
        record = records[response_rows[row_index]]
        context, _ = retrieve_objective_context(str(record["full_text"]), objective)
        contexts.append(context)

    tokenizer, model, device = _load_supervised_model(encoder_dir, delta_path)
    batch_size = 128 if device == "cuda" else 32
    max_length = int(artifact["supervised_max_length"])
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            stop = min(len(features), start + batch_size)
            encoded = tokenizer(
                contexts[start:stop],
                [_mastery_hypothesis(value) for value in objectives[start:stop]],
                padding="max_length",
                truncation="only_first",
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(device)
                for key, value in encoded.items()
                if key in {"input_ids", "attention_mask", "token_type_ids"}
            }
            probability = torch.sigmoid(model(**inputs).logits.squeeze(-1))
            batches.append(probability.cpu().numpy().astype(np.float64))
    result = np.concatenate(batches)
    if len(result) != len(features) or not np.isfinite(result).all():
        raise RuntimeError("Supervised component produced invalid probabilities.")
    clip = float(artifact.get("probability_clip", 1e-6))
    return np.clip(result, clip, 1.0 - clip)


def main() -> None:
    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    features = pd.read_csv(
        data_dir / "test_features.csv",
        dtype={
            "response_id": "string", "session_id": "string",
            "learning_objective_id": "string", "learning_objective": "string",
        },
    )
    submission_format = pd.read_csv(
        data_dir / "submission_format.csv", dtype={"response_id": "string"}
    )
    if list(submission_format.columns) != ["response_id", "probability"]:
        raise ValueError("submission_format.csv has unexpected columns.")
    if submission_format["response_id"].duplicated().any():
        raise ValueError("Duplicate response_id values in submission format.")
    if set(submission_format["response_id"]) != set(features["response_id"]):
        raise ValueError("Test features and submission format response IDs do not match.")

    rank_path = root / "assets" / "final_ensemble_v04_rank.joblib"
    feedback_path = root / "assets" / "final_ensemble_v03_feedback.joblib"
    bge_backup_path = root / "assets" / "final_ensemble_v05_bge_backup.joblib"
    final_path = (
        rank_path
        if rank_path.exists()
        else (
            feedback_path
            if feedback_path.exists()
            else (
                bge_backup_path
                if bge_backup_path.exists()
                else root / "assets" / "final_ensemble_v02.joblib"
            )
        )
    )
    if final_path.exists():
        artifact = joblib.load(final_path)
        if artifact.get("build_sklearn_version") != sklearn.__version__:
            raise RuntimeError(
                f"Model sklearn mismatch: artifact={artifact.get('build_sklearn_version')}, "
                f"runtime={sklearn.__version__}"
            )
        encoder_name = str(artifact.get("semantic_encoder", "BAAI/bge-small-en-v1.5")).rsplit("/", 1)[-1]
        base_probability = predict_final_ensemble(
            features, artifact, data_dir / "test_transcripts",
            root / "assets" / encoder_name,
        )
        if final_path == rank_path:
            nb_probability = predict_nbsvm_feedback(
                features, artifact, data_dir / "test_transcripts"
            )
            nb_weight = float(artifact["nbsvm_component_weight"])
            combined_probability = (
                (1.0 - nb_weight) * base_probability + nb_weight * nb_probability
            )
            supervised_probability = predict_supervised_component(
                features,
                artifact,
                data_dir / "test_transcripts",
                root / "assets" / "bge-small-en-v1.5",
                root / "assets" / str(artifact["supervised_delta_file"]),
            )
            supervised_weight = float(artifact["supervised_component_weight"])
            probability = (
                (1.0 - supervised_weight) * combined_probability
                + supervised_weight * supervised_probability
            )
            probability = np.clip(
                probability,
                float(artifact.get("probability_clip", 1e-6)),
                1.0 - float(artifact.get("probability_clip", 1e-6)),
            )
        elif final_path == feedback_path:
            probability = predict_feedback_upgrade(
                features, artifact, data_dir / "test_transcripts", base_probability
            )
        else:
            probability = base_probability
    else:
        probability = predict_objective_prior(features, load_asset(root))
    predictions = pd.DataFrame(
        {"response_id": features["response_id"], "probability": probability}
    )
    output = submission_format[["response_id"]].merge(
        predictions, on="response_id", how="left", validate="one_to_one"
    )
    if output["probability"].isna().any():
        raise ValueError("Missing probabilities after aligning submission rows.")
    output.to_csv(root / "submission.csv", index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
