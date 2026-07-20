from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trace_ace.config import load_config
from trace_ace.io import discover_project_paths
from trace_ace.metrics import binary_metrics, fold_metric_rows


LEGAL_METADATA_COLUMNS = [
    "n_utterances",
    "duration_seconds",
    "content_chars",
    "tutor_utterances",
    "student_utterances",
    "background_utterances",
    "other_role_utterances",
    "unclear_utterances",
    "empty_utterances",
    "objective_chars",
    "objective_words",
    "student_fraction",
    "tutor_fraction",
    "background_fraction",
    "unclear_fraction",
]


def _logistic(c: float, max_iter: int) -> LogisticRegression:
    return LogisticRegression(
        C=c,
        solver="liblinear",
        max_iter=max_iter,
        random_state=0,
    )


def global_prior_oof(frame: pd.DataFrame) -> np.ndarray:
    predictions = np.empty(len(frame), dtype=np.float64)
    for fold in sorted(frame["fold"].unique()):
        validation_mask = frame["fold"].eq(fold).to_numpy()
        prior = float(frame.loc[~validation_mask, "target"].mean())
        predictions[validation_mask] = prior
    return predictions


def objective_prior_oof(frame: pd.DataFrame, alpha: float) -> np.ndarray:
    predictions = np.empty(len(frame), dtype=np.float64)
    for fold in sorted(frame["fold"].unique()):
        validation_mask = frame["fold"].eq(fold)
        train = frame.loc[~validation_mask]
        validation = frame.loc[validation_mask]
        global_prior = float(train["target"].mean())
        grouped = train.groupby("learning_objective_id", observed=True)["target"].agg(["sum", "count"])
        posterior = (grouped["sum"] + alpha * global_prior) / (grouped["count"] + alpha)
        predictions[validation_mask.to_numpy()] = (
            validation["learning_objective_id"].map(posterior).fillna(global_prior).to_numpy()
        )
    return predictions


def metadata_logistic_oof(frame: pd.DataFrame, c: float, max_iter: int) -> np.ndarray:
    predictions = np.empty(len(frame), dtype=np.float64)
    x = frame[LEGAL_METADATA_COLUMNS].astype(np.float64)
    y = frame["target"].to_numpy()
    for fold in sorted(frame["fold"].unique()):
        validation_mask = frame["fold"].eq(fold).to_numpy()
        pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            _logistic(c=c, max_iter=max_iter),
        )
        pipeline.fit(x.loc[~validation_mask], y[~validation_mask])
        predictions[validation_mask] = pipeline.predict_proba(x.loc[validation_mask])[:, 1]
    return predictions


def objective_text_oof(frame: pd.DataFrame, c: float, max_iter: int) -> np.ndarray:
    predictions = np.empty(len(frame), dtype=np.float64)
    y = frame["target"].to_numpy()
    for fold in sorted(frame["fold"].unique()):
        validation_mask = frame["fold"].eq(fold).to_numpy()
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=30_000,
            sublinear_tf=True,
            strip_accents="unicode",
            dtype=np.float32,
        )
        x_train = vectorizer.fit_transform(frame.loc[~validation_mask, "learning_objective"])
        x_validation = vectorizer.transform(frame.loc[validation_mask, "learning_objective"])
        model = _logistic(c=c, max_iter=max_iter)
        model.fit(x_train, y[~validation_mask])
        predictions[validation_mask] = model.predict_proba(x_validation)[:, 1]
    return predictions


def _session_response_matrix(
    vectorizer: TfidfVectorizer,
    session_texts: pd.Series,
    response_session_ids: pd.Series,
):
    unique_session_ids = pd.Index(response_session_ids.drop_duplicates())
    missing = unique_session_ids.difference(session_texts.index)
    if len(missing):
        raise ValueError(f"Missing cached text for {len(missing)} sessions.")
    unique_matrix = vectorizer.transform(session_texts.loc[unique_session_ids])
    lookup = pd.Series(np.arange(len(unique_session_ids)), index=unique_session_ids)
    row_indices = response_session_ids.map(lookup).to_numpy(dtype=np.int64)
    return unique_matrix[row_indices]


def transcript_text_oof(
    frame: pd.DataFrame,
    session_texts: pd.DataFrame,
    text_column: str,
    analyzer: str,
    ngram_range: tuple[int, int],
    min_df: int,
    max_features: int,
    c: float,
    max_iter: int,
    checkpoint_path: Path | None = None,
) -> np.ndarray:
    predictions = np.full(len(frame), np.nan, dtype=np.float64)
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = pd.read_parquet(checkpoint_path)
        if list(checkpoint["response_id"]) != list(frame["response_id"]):
            raise ValueError(f"Checkpoint response IDs do not match: {checkpoint_path}")
        predictions = checkpoint["probability"].to_numpy(dtype=np.float64)
        print(f"Loaded checkpoint {checkpoint_path.name}.", flush=True)
    y = frame["target"].to_numpy()
    indexed_texts = session_texts.set_index("session_id")[text_column]

    for fold in sorted(frame["fold"].unique()):
        validation_mask = frame["fold"].eq(fold).to_numpy()
        if np.isfinite(predictions[validation_mask]).all():
            print(
                f"Skipping completed {text_column}/{analyzer} fold {int(fold) + 1}.",
                flush=True,
            )
            continue
        train_rows = frame.loc[~validation_mask]
        validation_rows = frame.loc[validation_mask]
        train_sessions = pd.Index(train_rows["session_id"].drop_duplicates())

        transcript_vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=max_features,
            sublinear_tf=True,
            strip_accents="unicode",
            dtype=np.float32,
        )
        transcript_vectorizer.fit(indexed_texts.loc[train_sessions])
        x_train_transcript = _session_response_matrix(
            transcript_vectorizer, indexed_texts, train_rows["session_id"]
        )
        x_validation_transcript = _session_response_matrix(
            transcript_vectorizer, indexed_texts, validation_rows["session_id"]
        )

        objective_vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=2,
            max_features=min(20_000, max_features),
            sublinear_tf=True,
            strip_accents="unicode",
            dtype=np.float32,
        )
        x_train_objective = objective_vectorizer.fit_transform(train_rows["learning_objective"])
        x_validation_objective = objective_vectorizer.transform(
            validation_rows["learning_objective"]
        )

        x_train = hstack([x_train_transcript, x_train_objective], format="csr")
        x_validation = hstack(
            [x_validation_transcript, x_validation_objective], format="csr"
        )
        model = _logistic(c=c, max_iter=max_iter)
        model.fit(x_train, y[~validation_mask])
        predictions[validation_mask] = model.predict_proba(x_validation)[:, 1]
        print(
            f"Completed {text_column}/{analyzer} fold {int(fold) + 1}: "
            f"{x_train.shape[1]} features.",
            flush=True,
        )
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_frame = pd.DataFrame(
                {"response_id": frame["response_id"], "probability": predictions}
            )
            temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
            checkpoint_frame.to_parquet(temporary, index=False)
            temporary.replace(checkpoint_path)
    if not np.isfinite(predictions).all():
        raise RuntimeError(f"Incomplete text predictions for {text_column}/{analyzer}.")
    return predictions


def fit_objective_prior_asset(frame: pd.DataFrame, alpha: float, clip: float) -> dict[str, object]:
    global_prior = float(frame["target"].mean())
    grouped = frame.groupby("learning_objective_id", observed=True)["target"].agg(["sum", "count"])
    posterior = (grouped["sum"] + alpha * global_prior) / (grouped["count"] + alpha)
    return {
        "artifact_version": "1.0.0",
        "model_type": "smoothed_objective_prior",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "alpha": alpha,
        "probability_clip": clip,
        "global_prior": global_prior,
        "objective_priors": {str(key): float(value) for key, value in posterior.items()},
        "training_rows": int(len(frame)),
        "training_sessions": int(frame["session_id"].nunique()),
    }


def _write_experiment_outputs(
    paths,
    frame: pd.DataFrame,
    prediction_columns: list[str],
    clip: float,
    run_id: str,
) -> tuple[Path, pd.DataFrame]:
    run_dir = paths.experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    oof_columns = ["response_id", "session_id", "fold", "target", *prediction_columns]
    frame[oof_columns].to_parquet(run_dir / "oof_predictions.parquet", index=False)

    metric_rows: list[dict[str, object]] = []
    for prediction_column in prediction_columns:
        model_name = prediction_column.removeprefix("pred_")
        for row in fold_metric_rows(frame, prediction_column, clip=clip):
            metric_rows.append({"model": model_name, **row})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(run_dir / "metrics.csv", index=False, lineterminator="\n")

    pooled = metrics.loc[metrics["fold"].eq("pooled")].copy()
    if pooled.empty:
        pooled = metrics.loc[metrics["fold"].astype(str).eq("pooled")].copy()
    report = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(frame)),
        "folds": int(frame["fold"].nunique()),
        "models": pooled.to_dict(orient="records"),
    }
    with (run_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    ledger_path = paths.experiments_dir / "experiment_ledger.csv"
    ledger_rows = []
    for row in pooled.itertuples(index=False):
        ledger_rows.append(
            {
                "run_id": run_id,
                "timestamp_utc": report["generated_at_utc"],
                "model": row.model,
                "fold_scheme": "5-fold StratifiedGroupKFold(session_id)",
                "log_loss": row.log_loss,
                "roc_auc": row.roc_auc,
                "brier_score": row.brier_score,
                "ece_10": row.ece_10,
                "status": "completed",
            }
        )
    ledger_new = pd.DataFrame(ledger_rows)
    if ledger_path.exists():
        ledger = pd.concat([pd.read_csv(ledger_path), ledger_new], ignore_index=True)
    else:
        ledger = ledger_new
    ledger.to_csv(ledger_path, index=False, lineterminator="\n")
    return run_dir, metrics


def run_baselines(project_root: str | Path, include_text: bool = True) -> dict[str, object]:
    paths = discover_project_paths(project_root)
    config = load_config(paths.root)
    modeling_path = paths.cache_dir / "modeling_base.parquet"
    texts_path = paths.cache_dir / "session_texts.parquet"
    if not modeling_path.exists() or not texts_path.exists():
        raise FileNotFoundError("Foundation caches are missing. Run build_foundation.py first.")

    frame = pd.read_parquet(modeling_path).reset_index(drop=True)
    text_config = config["text_baseline"]
    c = float(text_config["logistic_c"])
    max_iter = int(text_config["max_iter"])
    alpha = float(config["objective_smoothing"])
    clip = float(config["probability_clip"])

    frame["pred_global_prior"] = global_prior_oof(frame)
    frame["pred_metadata_logistic"] = metadata_logistic_oof(frame, c=c, max_iter=max_iter)
    frame["pred_objective_prior"] = objective_prior_oof(frame, alpha=alpha)
    frame["pred_objective_text"] = objective_text_oof(frame, c=c, max_iter=max_iter)
    prediction_columns = [
        "pred_global_prior",
        "pred_metadata_logistic",
        "pred_objective_prior",
        "pred_objective_text",
    ]

    if include_text:
        session_texts = pd.read_parquet(texts_path)
        checkpoint_dir = paths.experiments_dir / "checkpoints"
        word_checkpoint = checkpoint_dir / (
            f"word_full_seed{int(config['fold_seed'])}_"
            f"max{int(text_config['word_max_features'])}_"
            f"mindf{int(text_config['word_min_df'])}.parquet"
        )
        char_checkpoint = checkpoint_dir / (
            f"char_closing_seed{int(config['fold_seed'])}_"
            f"max{int(text_config['char_max_features'])}_"
            f"mindf{int(text_config['char_min_df'])}.parquet"
        )
        frame["pred_word_full"] = transcript_text_oof(
            frame=frame,
            session_texts=session_texts,
            text_column="full_text",
            analyzer="word",
            ngram_range=(1, 2),
            min_df=int(text_config["word_min_df"]),
            max_features=int(text_config["word_max_features"]),
            c=c,
            max_iter=max_iter,
            checkpoint_path=word_checkpoint,
        )
        frame["pred_char_closing"] = transcript_text_oof(
            frame=frame,
            session_texts=session_texts,
            text_column="closing_text",
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=int(text_config["char_min_df"]),
            max_features=int(text_config["char_max_features"]),
            c=c,
            max_iter=max_iter,
            checkpoint_path=char_checkpoint,
        )
        frame["pred_word_char_blend"] = 0.5 * frame["pred_word_full"] + 0.5 * frame[
            "pred_char_closing"
        ]
        prediction_columns.extend(
            ["pred_word_full", "pred_char_closing", "pred_word_char_blend"]
        )

    paths.models_dir.mkdir(parents=True, exist_ok=True)
    asset = fit_objective_prior_asset(frame, alpha=alpha, clip=clip)
    asset_path = paths.models_dir / "objective_prior.json"
    with asset_path.open("w", encoding="utf-8") as handle:
        json.dump(asset, handle, indent=2, sort_keys=True)
        handle.write("\n")
    submission_asset = paths.root / "submission_src" / "assets" / "objective_prior.json"
    submission_asset.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(asset_path, submission_asset)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_baselines")
    run_dir, metrics = _write_experiment_outputs(
        paths=paths,
        frame=frame,
        prediction_columns=prediction_columns,
        clip=clip,
        run_id=run_id,
    )
    pooled = metrics.loc[metrics["fold"].astype(str).eq("pooled")]
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "pooled_metrics": pooled.to_dict(orient="records"),
        "objective_asset": str(asset_path),
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train grouped Trace the Ace baselines.")
    parser.add_argument("--project-root", default=".", help="Path to the project root.")
    parser.add_argument(
        "--skip-text",
        action="store_true",
        help="Run only fast non-transcript baselines.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run_baselines(args.project_root, include_text=not args.skip_text)
    print(f"Baseline run complete: {result['run_id']}")
    for row in result["pooled_metrics"]:
        print(
            f"{row['model']}: log_loss={row['log_loss']:.6f}, "
            f"auc={row['roc_auc']:.4f}, brier={row['brier_score']:.6f}"
        )


if __name__ == "__main__":
    main()
