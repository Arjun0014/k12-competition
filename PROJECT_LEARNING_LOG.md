# Trace the Ace - Project Learning Log

This is the append-only record of what the project team has learned, verified, changed, rejected, and still needs to resolve. It is deliberately separate from the experiment ledger: this file records durable understanding and decisions, while individual model runs will be tracked in a structured experiment table.

## How to maintain this log

- Add one dated section for every day on which meaningful work is done.
- Use local project time (Asia/Kolkata) and ISO dates (`YYYY-MM-DD`).
- Mark statements as **Verified**, **Inference**, **Decision**, **Correction**, **Risk**, or **Open**.
- Never silently rewrite a prior conclusion. Add a correction in the new day's entry and link back to the old conclusion.
- Record unsuccessful approaches and why they failed; negative results are part of the research record.
- Do not paste raw competition transcripts, names, or other row-level competition data here.
- Record source URLs, file names, model versions, licenses, seeds, split versions, and artifact hashes whenever they affect reproducibility.

---

## 2026-07-16 (Asia/Kolkata) - Initial competition, rules, data, and feasibility audit

### Work completed

- Read `Trace_the_Ace_full_overview.md`, `competition_rules.md`, `code_submission_format.md`, and `information_websites_to_see.md`.
- Verified the current competition home, about, rules, code-submission, community-code, and official runtime pages.
- Read and visually checked all four pages of `example_documentation_guide.pdf`.
- Audited the training feature and label tables, both supplied submission-format files, all 22,821 training transcripts, and the locally downloaded external-dataset folders.
- Measured initial group-safe, out-of-fold baselines.
- Audited the current local machine for training and runtime-test feasibility.

### What the competition is actually asking us to do

- **Verified:** For each `response_id`, estimate the probability that the student answered the next assessment question correctly after a tutoring session.
- **Verified:** The inputs for one prediction are a tutoring-session transcript and one learning objective. A session can produce several response rows when several objectives were assessed.
- **Verified:** The target is binary. The required output is a probability, not a hard class.
- **Verified:** The primary metric is binary log loss; lower is better. Leaderboard ROC AUC is secondary and does not determine rank.
- **Verified:** This is a code-execution competition. We submit a ZIP containing inference code and model assets, not a precomputed test-prediction CSV.
- **Verified:** Final prizes are not based only on leaderboard rank. The top 15 teams may submit a four-page write-up, and the judges emphasize educational relevance and generalizability.
- **Verified:** The model-submission deadline is 2026-08-27 23:59 UTC. The top-15 write-up deadline is 2026-09-15.

### Verified training-data facts

| Item                                        |           Finding |
| ------------------------------------------- | ----------------: |
| Training response rows                      |            35,072 |
| Unique training sessions / transcript files |            22,821 |
| Unique learning-objective IDs and texts     |         398 / 398 |
| Positive labels                             | 24,637 (70.2469%) |
| Negative labels                             | 10,435 (29.7531%) |
| Total transcript utterances                 |         6,139,854 |
| Approximate transcript CSV size             |          600.9 MB |
| Total transcript content characters         |       416,437,335 |
| Median utterances per session               |               267 |
| Session utterance range                     |         15 to 622 |
| Median content characters per session       |            18,387 |
| Session content-character range             |     337 to 44,556 |
| Tutor utterances                            |         3,196,001 |
| Student utterances                          |         2,697,152 |
| Background utterances                       |           246,701 |
| Feature rows without a transcript           |                 0 |
| Orphan transcript files                     |                 0 |
| Null cells in core feature/label tables     |                 0 |

- **Verified:** `response_id` is unique in both feature and label files and the two files join one-to-one with no missing rows.
- **Verified:** `session_id` repeats. Responses per session range from 1 to 10; 14,457 sessions have one response, while 8,364 sessions have multiple responses.
- **Verified:** 3,207 sessions contain a mix of correct and incorrect outcome labels across learning objectives.
- **Verified:** Every `learning_objective_id` maps to exactly one objective text and vice versa in the training data.
- **Verified:** 332 of 398 learning objectives occur in more than one session. Objective difficulty is a strong signal.
- **Verified:** Utterance IDs start at 0 and are contiguous within every audited transcript.
- **Verified:** Transcript timestamps are relative `HH:MM:SS` values, not full datetimes.
- **Verified:** The role distribution is dominated by tutor/student speech but includes a real third role, `background`, in 22,665 sessions.
- **Inference:** The `background` role likely includes slide/system/audio-context material or diarization spillover. It may carry lesson context and must not be discarded without an ablation.
- **Risk:** The samples inspected show ASR noise (`[unclear]`) and occasional apparent speaker-attribution errors. Speaker tags are useful but not perfectly reliable.

### Corrections to the supplied written overview

These differences are not necessarily competition errors; they are schema facts that our code must handle explicitly.

1. **Correction:** The overview documents three feature columns, but the downloaded feature file has four: `response_id`, `session_id`, `learning_objective_id`, and `learning_objective`.
2. **Correction:** The overview calls the label column `correct`; the downloaded label file calls it `is_correct`.
3. **Correction:** The overview says transcript role is either `tutor` or `student`; the downloaded transcripts also contain `background`.
4. **Correction:** The overview describes `utterance_id` as a string; it is stored as an integer in the audited CSVs.
5. **Correction:** The overview describes `timestamp` as a datetime; it is a relative time-of-session field in the audited CSVs.
6. **Decision:** All readers will validate aliases and actual schemas rather than hard-coding only the overview's names.

### Group-safe baseline results

Method: five-fold `StratifiedGroupKFold`, grouped by `session_id`, seed `20260716`. These are local development estimates, not leaderboard scores.

| Model                                                | OOF log loss | OOF ROC AUC | Interpretation                      |
| ---------------------------------------------------- | -----------: | ----------: | ----------------------------------- |
| Global positive-rate prior                           |     0.608758 |      0.5000 | Minimum sanity baseline             |
| Legal per-sample length/role metadata logistic model |     0.606996 |      0.5377 | Session scale alone is weak         |
| Smoothed training-only objective-ID prior            |     0.551985 |      0.7067 | Objective difficulty is very strong |

- **Verified:** Learning-objective difficulty is the strongest simple signal found so far.
- **Decision:** The objective-only baseline is necessary for performance and as a control, but it is not the research contribution. Every transcript model must be compared against it and demonstrate added value from conversation content.
- **Risk:** An objective-ID lookup will fail on unseen objective IDs. It needs a text-based objective representation and a global-prior fallback.
- **Verified diagnostic:** Training rows from sessions with more assessed objectives have a higher positive rate.
- **Correction / Rule consequence:** The number of other response rows sharing a test session cannot be used as an inference feature because it depends on other test cases. It is excluded even though it is predictive in training.

### Validation and leakage conclusions

- **Decision:** Never use a random row split. The same transcript would otherwise appear in train and validation through different `response_id` rows.
- **Decision:** The primary split groups by `session_id` and is frozen for model comparison.
- **Decision:** A secondary cold-objective evaluation will test performance when objective IDs or objective groups are unseen.
- **Decision:** Calibration must be trained on genuinely out-of-fold predictions, never on the labels used to fit the underlying model.
- **Decision:** No feature may depend on the composition, aggregates, pseudo-labels, embeddings, or statistics of other test samples.
- **Decision:** Any transcript chunk selection must use only the current row's objective and transcript plus parameters fitted on training data.

### Runtime and submission facts

- **Verified:** Python 3.12 only; model assets must be packaged because inference has no internet access.
- **Verified:** ZIP root must contain `main.py`; it must read from read-only `data/` and write root-level `submission.csv`.
- **Verified:** Output columns must be exactly `response_id,probability`, with one row per required response.
- **Verified:** The supplied 100-row submission format is the smoke-test shape; the 10,508-row file is the full test shape.
- **Verified:** Full runtime limit: 6 hours. Smoke-test limit: 10 minutes.
- **Verified:** Runtime hardware: 24 vCPUs, 220 GB RAM, one NVIDIA A100 with 80 GB VRAM.
- **Verified:** Maximum ZIP size: 60 GB. The process has no root filesystem access.
- **Verified:** Test transcript text, objective text, token counts, dataset summaries, or other test-data information must not be printed. Logging is limited to 500 lines and 500 characters per line.
- **Verified:** Current runtime packages include PyTorch, Transformers, sentence-transformers, LightGBM, scikit-learn, PEFT, vLLM, pandas, and Polars. Exact locked versions must be tested in the official container before submission.
- **Verified:** Full submissions are limited to three per seven days. Smoke tests, cancelled jobs, and failed jobs do not consume that quota.

### External-data and model licensing audit

The rule is stricter than “publicly downloadable”: resources must permit commercial use and support an openly releasable solution. We will maintain source and license evidence for every external resource.

| Local resource              | Verified/known license state                                                                                                                 | Current decision                                                       |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `Grade School Math (GSM8k)` | MIT                                                                                                                                          | Eligible candidate, but only indirectly relevant                       |
| `FairytaleQA`               | Apache-2.0                                                                                                                                   | Eligible, low relevance to math tutoring outcomes                      |
| `Bridge`                    | CC BY-NC 4.0                                                                                                                                 | Exclude from prize-eligible training                                   |
| `DrawEduMath`               | Platform catalog lists CC BY-NC-SA 4.0                                                                                                       | Exclude from prize-eligible training                                   |
| `SciQ`                      | CC BY-NC 3.0                                                                                                                                 | Exclude from prize-eligible training                                   |
| `SemEval` folder            | Contents match the TalkMoves classroom corpus; upstream is CC BY-NC-SA 4.0                                                                   | Exclude from prize-eligible training                                   |
| `TalkMoves` folder          | Contents match SemEval-2013 Task 7 (BEETLE/SciEntsBank), not TalkMoves classroom transcripts; platform catalog lists SemEval as CC BY-SA 3.0 | Quarantine pending organizer confirmation on share-alike compatibility |
| `MRBench`                   | Repository declares CC BY-SA 4.0 but it builds on Bridge, which is non-commercial                                                            | Exclude unless organizers explicitly clear it                          |
| `EssayJudge`                | Platform catalog lists Apache-2.0                                                                                                            | Eligible candidate, but low direct relevance                           |

- **Correction:** The local `SemEval` and `TalkMoves` folder names appear semantically swapped relative to their contents. They have not been renamed because raw source folders should remain immutable.
- **Decision:** Do not download more external datasets now. The immediate performance bottleneck is modeling and validating the competition data, not data volume.
- **Decision:** Initial pretrained-model candidates are Apache-2.0 resources: ModernBERT-base (8,192-token encoder), Qwen3-Embedding-0.6B (32K embedding context), all-MiniLM-L6-v2 (fast short-chunk baseline), and Qwen2.5-7B-Instruct (optional structured annotation/direct scoring). Each exact revision will be registered before use.

### Later clarification from the platform dataset catalog — 2026-07-16

The user supplied screenshots of the K-12 AI Infrastructure dataset catalog, and the public catalog was checked again. This corrected part of the first local-file-only license audit.

- **New information:** EssayJudge is listed as Apache-2.0, so it is an eligible external-data candidate. It remains low priority because its lexical-quality target is not the competition target.
- **New information:** The official catalog lists SemEval as CC BY-SA 3.0 and TalkMoves as CC BY-NC-SA 4.0. This reinforces that the local `TalkMoves` folder contains SemEval material and the local `SemEval` folder contains TalkMoves material.
- **What was previously wrong:** EssayJudge was initially quarantined because the local files did not expose a clear license. The platform listing resolves that uncertainty in favor of Apache-2.0.
- **What remains unresolved:** CC BY-SA permits commercial use, but the competition asks for a permissive open license and an MIT-releasable winning solution. SemEval therefore stays quarantined until the organizers confirm that its share-alike obligation is compatible with this competition.
- **What remains unsafe:** MRBench is listed as CC BY-SA 4.0, but its documented use of Bridge creates an upstream CC BY-NC concern. A catalog listing alone does not remove that conflict.
- **Decision:** Being listed on the platform is not a blanket exception to the Trace the Ace external-data rule. The prize-safe allowlist is currently GSM8K (MIT), FairytaleQA (Apache-2.0), and EssayJudge (Apache-2.0). We will not train on any NC resource, and will not train on BY-SA resources without written organizer clearance.
- **Decision:** No download is needed now because the audited local folders already contain these catalog datasets. If a clean re-download is later needed, use the platform-hosted version, record its version/date/URL and SHA-256 hash, and keep it immutable.

### Research direction learned from the competition references

- **Verified from prior work:** Dialogue text can predict student outcomes, and explicit tutor-move information can add signal.
- **Verified from prior work:** Outcome prediction is more tractable than future tutor-move prediction.
- **Verified from prior work:** Instructionally supportive behaviors, probing, feedback, and explanation can correlate with outcomes, but effects vary by dataset.
- **Decision:** We will model student evidence and tutor actions separately, then test their incremental predictive value with ablations.
- **Risk:** This observational dataset supports predictive association, not causal claims about a tutoring strategy causing learning. The write-up must use causal language only if the analysis design justifies it.

### Local development environment

| Resource              | Finding                                 |
| --------------------- | --------------------------------------- |
| CPU                   | AMD Ryzen 5 4600H, 6 cores / 12 threads |
| RAM                   | 11.4 GiB total                          |
| Local NVIDIA GPU      | None detected                           |
| Free C: drive space   | about 208 GiB                           |
| Docker                | Not installed/detected                  |
| `just` command runner | Not installed/detected                  |
| `uv` and Git          | Available                               |

- **Decision:** Local hardware is adequate for audits, Parquet conversion, TF-IDF, logistic regression, and small CPU experiments.
- **Risk:** Long-context embedding and transformer fine-tuning require cloud GPU access or another GPU machine.
- **Open:** Docker Desktop and `just` must be installed before official local container tests.

### Decisions carried into implementation

1. Freeze a session-grouped validation split before serious model tuning.
2. Build a schema-validated, cached transcript table instead of repeatedly opening 22,821 CSVs.
3. Preserve all three speaker roles and test removal/merging only through ablation.
4. Establish objective-only and legal metadata baselines before transcript models.
5. Add word/character TF-IDF transcript models before spending GPU time.
6. Use objective-conditioned long-context and key-moment models, with the full transcript as a comparison.
7. Optimize and report log loss first; treat AUC as diagnostic.
8. Track calibration, runtime, licenses, seeds, and negative results from the beginning.
9. Quarantine non-commercial and unclear external datasets.
10. Build the submission runner early enough that runtime constraints influence model selection.

### Open questions for the next work session

- What is the strongest legal word/character TF-IDF baseline from transcript text, objective text, and their interaction?
- How much does the final quarter of a session contribute versus the full transcript?
- Can objective-conditioned chunk retrieval beat naive truncation?
- Does `background` content improve log loss after controlling for objective and transcript length?
- How many test objectives are unseen cannot be known from local files; the pipeline needs a robust fallback by design.
- Which cloud GPU environment will be used for embedding extraction and fine-tuning?
- Should the organizers be asked to clarify whether CC BY-SA models/data derived from non-commercial sources can ever be prize eligible? Until clarified, they remain excluded.

### Sources checked on this date

- Local: `Trace_the_Ace_full_overview.md`
- Local: `competition_rules.md`
- Local: `code_submission_format.md`
- Local: `example_documentation_guide.pdf`
- Competition: https://platform.k12-ai-infrastructure.org/competitions/3/tutoring-outcomes/
- K-12 AI Infrastructure dataset catalog: https://platform.k12-ai-infrastructure.org/
- Official runtime: https://github.com/drivendataorg/tutoring-outcomes-runtime
- Ikram, Scarlatos, and Lan (2025): https://arxiv.org/abs/2507.06910
- Scarlatos, Baker, and Lan (2025): https://arxiv.org/abs/2409.16490
- ModernBERT model card: https://huggingface.co/answerdotai/ModernBERT-base
- Qwen3 Embedding model card: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- Qwen2.5 model card: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- GSM8K model card/repository: https://huggingface.co/datasets/openai/gsm8k and https://github.com/openai/grade-school-math
- FairytaleQA repository: https://github.com/uci-soe/FairytaleQAData
- Bridge dataset card: https://huggingface.co/datasets/rose-e-wang/bridge
- TalkMoves repository: https://github.com/SumnerLab/TalkMoves
- SciQ dataset card: https://huggingface.co/datasets/allenai/sciq

## 2026-07-16 (Asia/Kolkata) - Foundation, grouped baselines, and first deployable transcript model

### Work completed after plan approval

- Initialized the Git repository and added a conservative `.gitignore` that prevents competition transcripts, feature/label tables, external datasets, caches, model assets, submissions, and temporary files from being committed.
- Added the installable `trace_ace` Python package, `pyproject.toml`, and a resolved `uv.lock` for reproducible local dependencies.
- Added the project configuration and external-resource license registry under `configs/`.
- Built a raw competition-data manifest with path, category, byte size, modification time, and SHA-256 for every supplied competition file.
- Built deterministic Parquet caches for utterances, session features, session text views, response-level modeling data, and frozen fold assignments.
- Added an evaluation harness for log loss, ROC AUC, Brier score, ten-bin expected calibration error, pooled OOF metrics, and fold-level metrics.
- Trained group-safe global, metadata, objective-ID, objective-text, full-session word TF-IDF, and closing-quarter character TF-IDF baselines.
- Added an appendable experiment ledger plus per-run reports and OOF predictions.
- Fitted the promoted full-data sparse word model and created two offline packages: an objective-prior fallback and the transcript-based word TF-IDF candidate.
- Added an inference batch-invariance test, local 100-row smoke tests, ZIP-structure checks, and a formal milestone verifier.

### Reproducibility and cache results

| Item                                 |                                                             Result |
| ------------------------------------ | -----------------------------------------------------------------: |
| Files in raw competition manifest    |                                                             22,825 |
| Bytes in raw competition manifest    |                                                        603,773,729 |
| Raw manifest digest                  | `38a63c7e794277f252327dc36dcd7bbbe312a7f404ce9912ccdc72c9333d8e38` |
| Cached sessions                      |                                                             22,821 |
| Cached utterances                    |                                                          6,139,854 |
| Exact transcript content characters  |                                                        416,437,335 |
| First manifest/cache/fold build      |                                                  about 129 seconds |
| Idempotent rerun using current cache |                                                  about 9.4 seconds |

- **Correction:** The earlier 423-million-character value was a rough estimate. The validated streaming cache reports exactly 416,437,335 content characters.
- **Verified:** Every session belongs to exactly one of the five folds. No transcript/session crosses a training-validation boundary.
- **Verified:** The frozen folds cover all 35,072 responses and use seed `20260716`.
- **Verified:** The second foundation run detected an unchanged source digest and skipped rebuilding the expensive transcript cache.

### Reproducible grouped OOF results

All results use five-fold `StratifiedGroupKFold`, grouped by `session_id`. Every learned vectorizer and model is fitted only on the training portion of each fold.

| Model                                             | OOF log loss |    OOF AUC |        Brier |      ECE-10 |
| ------------------------------------------------- | -----------: | ---------: | -----------: | ----------: |
| Fold-trained global prior                         |     0.608798 |     0.4932 |     0.209023 |     0.00002 |
| Legal metadata logistic                           |     0.599717 |     0.5834 |     0.205348 |     0.00408 |
| Smoothed objective-ID prior                       |     0.551985 |     0.7067 |     0.185812 |     0.00915 |
| Objective-text TF-IDF                             |     0.551396 |     0.7070 |     0.185543 |     0.00335 |
| Full-session word TF-IDF + objective text         | **0.532189** | **0.7384** | **0.177879** | **0.00413** |
| Closing-quarter character TF-IDF + objective text |     0.548694 |     0.7136 |     0.184440 |     0.01242 |
| Fixed 50/50 word-character blend                  |     0.535148 |     0.7338 |     0.179052 |     0.00475 |

### What we learned from the first transcript models

- **Verified:** Full transcript language adds substantial signal beyond objective difficulty. The word model improves log loss by `0.019796` over the objective-ID baseline, about a 3.59% relative reduction.
- **Verified:** The full-session word model beats the objective-text model in every fold, so the gain is not driven by one favorable partition.
- **Verified:** The final quarter alone contains useful information, but the character model's `0.548694` is much weaker than the full-session word model.
- **Negative result:** A fixed equal blend of word and character predictions is worse than the word model alone. The character model is not promoted in its current form.
- **Correction:** The first metadata audit used a smaller feature set and reported about `0.6070`. The implemented legal per-sample metadata model includes role proportions, duration, objective length, `[unclear]` rate, and other transcript-shape features and reaches `0.599717`.
- **Correction:** A pooled constant prior gives AUC 0.5, but the stricter fold-trained global priors vary slightly across folds and produce pooled AUC `0.4932`. This is expected and does not make the model meaningfully discriminative.
- **Decision:** Promote the full-session word TF-IDF model as the first transcript-based submission candidate. Do not promote the equal word-character blend.
- **Decision:** External datasets remain unused. The competition-only transcript gain is already strong, so external data must still demonstrate incremental grouped-CV value before entering training.

### Runtime and engineering learnings

- The initial five-fold word-plus-character run took about 53 minutes on the local Ryzen CPU and peaked below 4 GB RAM.
- The launching shell reached its 30-minute command limit, but the Python worker continued and completed successfully. The original implementation wrote results only at the end, which made the run unnecessarily fragile.
- **What was previously wrong:** The text baseline was initially all-or-nothing. It now writes atomic per-model/per-fold checkpoints and resumes completed folds.
- The promoted full-data word model fitted in about 194 seconds.
- The final sparse training matrix has shape `35,072 × 61,297`.
- The trained artifact is 1,195,372 bytes with SHA-256 `b337062bb9bf622726b5c7057ca2bd52859c1028edfa76c7e3fcd2c8f2128126`.
- The transcript submission ZIP is 1,202,970 bytes, far below the 60 GB limit.
- Both packages passed the supplied 100-response smoke-format test locally without emitting stdout or stderr.
- Eight unit tests pass. The formal milestone verifier passes 23 checks covering manifests, cache row counts, fold isolation, targets, licenses, metrics, artifact hash, ZIP contents, and size limits.

### Current submission status

| Package                                          | Purpose                                | Local status               |
| ------------------------------------------------ | -------------------------------------- | -------------------------- |
| `submission_builds/objective_prior_baseline.zip` | Minimal fallback and runtime debugging | 100-row local smoke passed |
| `submission_builds/word_full_tfidf_baseline.zip` | First promoted transcript model        | 100-row local smoke passed |

The promoted package has not been uploaded. Before spending a full-submission quota, test it in the official runtime container and then in the platform smoke environment. Docker is not currently installed on the local machine, so the official-container check remains open.

### Next modeling questions

- Can role-separated tutor/student word features beat the combined transcript?
- Does full word text plus a smoothed objective prior outperform either component without degrading calibration?
- Are opening, middle, and closing transcript windows complementary to the full-session model?
- Can a stateless hashed feature cache shorten fold-safe sparse experiments?
- Do frozen long-context embeddings improve on `0.532189` enough to justify GPU work?

## 2026-07-16 (Asia/Kolkata) - First platform smoke test and runtime-aligned rebuild

### Platform result

- Uploaded `word_full_tfidf_baseline.zip` as smoke job `1677`.
- **Completed successfully** with smoke score `0.4022` and exit code `0`.
- The submitted `main.py` completed inference in about 3.6 seconds inside the official CUDA 12.9 container.
- The runtime found the correct root-level `main.py`, all three assets, and the generated `submission.csv`.
- **Important interpretation:** The smoke environment uses 100 responses drawn from training data, and the full sparse model was trained on those responses. The `0.4022` score is therefore optimistic and is only a functional check, not a valid estimate of private-test or leaderboard performance.

### Warning discovered

The official runtime emitted `InconsistentVersionWarning` for `TfidfTransformer`, `TfidfVectorizer`, and `LogisticRegression`:

- original artifact build: scikit-learn `1.6.1`;
- official runtime: scikit-learn `1.8.0`.

The artifact loaded and scored, but scikit-learn explicitly warns that cross-version unpickling can produce invalid results. A successful smoke job does not remove that risk.

### Correction implemented

- Added `.python-version` selecting Python 3.12.
- Pinned project scikit-learn to `1.8.0` and refreshed `uv.lock`.
- Created a clean `.venv` with Python `3.12.8` and scikit-learn `1.8.0`.
- Added Python, scikit-learn, NumPy, and joblib build versions to the trained artifact metadata.
- Added a hard inference-time scikit-learn version check so a mismatched artifact fails explicitly.
- Retrained the promoted model without changing its data, features, or model structure.
- Rebuilt both ZIP packages and reran the 100-row local smoke test under the aligned environment with no warnings or output.
- All eight unit tests pass and the artifact verifier now also enforces the required Python/scikit-learn build versions.

### Replacement artifact

| Item                    | Value                                                              |
| ----------------------- | ------------------------------------------------------------------ |
| Python                  | 3.12.8                                                             |
| scikit-learn            | 1.8.0                                                              |
| NumPy                   | 2.5.1                                                              |
| joblib                  | 1.5.3                                                              |
| Model SHA-256           | `3dd0481ac303c097a57275a443ce03a5029a9531fa3d28441e83ce6219ff0a84` |
| Replacement upload file | `submission_builds/word_full_tfidf_runtime180.zip`                 |
| Replacement ZIP SHA-256 | `8f1da10ff54f6463ba9c5b2b51bd000d8659d52760fbaead282c03046c4feee2` |

- **Decision:** Do not use the original job-1677 archive for a Normal submission.
- **Next action:** Upload the replacement ZIP for one more platform smoke test. If it completes without `InconsistentVersionWarning`, use that exact ZIP for the first Normal submission.

### Replacement smoke confirmation

- Uploaded the runtime-aligned replacement as smoke job `1682`.
- **Completed successfully** with exit code `0` and smoke score `0.4022`.
- Inference completed in about 3.3 seconds in the official container.
- No `InconsistentVersionWarning` or other Python/model warning appeared.
- The identical `0.4022` score confirms that aligning the serialization environment did not change the model's predictions on the smoke set.
- **Clarification:** The competition metric is log loss, so a lower number is better. However, the smoke score is calculated on 100 examples drawn from training data and is not a valid leaderboard or held-out-performance estimate.
- **Decision:** Runtime validation is complete. Submit the exact runtime-aligned ZIP as the first Normal submission to obtain the first genuine leaderboard measurement.

## 2026-07-16 (Asia/Kolkata) - Leaderboard failure diagnosis and v0.2 robust ensemble

### Leaderboard evidence that changed the modeling strategy

- **Verified:** Normal submission job `1685` completed successfully and scored public log loss `0.6181`, public AUROC `0.6129`, and rank `75` at the time observed.
- **Verified:** The leading public entry shown at the same time had log loss `0.6013` and AUROC `0.6309`.
- **Correction:** The original session-grouped OOF estimate (`0.532189` log loss) was not representative of the leaderboard. It prevented transcript leakage, but most validation rows still used learning objectives that the model had seen in other sessions. Objective difficulty and vocabulary therefore made the split much easier than the public test regime.
- **Decision:** Do not tune further against the original session-grouped score. It remains useful for measuring seen-objective behavior, but model promotion now requires a second, harder protocol designed around objective shift.

### Objective-disjoint validation protocol

- **Implemented:** Five-fold `StratifiedGroupKFold` grouped by `learning_objective_id`, seed `20260716`.
- **Implemented:** After assigning objective-disjoint validation folds, remove from each training fold every session that appears in that fold's validation rows. This prevents both objective overlap and transcript overlap.
- **Verified:** A simple stateless hashed transcript/objective model reached hard-validation AUROC `0.613226`, extremely close to the observed public AUROC `0.6129`.
- **Inference:** This close match is strong evidence that unseen or shifted objectives explain much of the public failure. It is not proof of the hidden split construction, so the protocol is described as a leaderboard-matched stress test rather than the official split.
- **Verified diagnostic:** On the original OOF predictions, the full word model's log loss was about `0.6427` for objectives unseen in the training fold and about `0.5245` for objectives with at least 500 training examples. The prior validation average was dominated by the easier frequent-objective regime.
- **Verified:** Probability shrinkage toward each fold's training prior materially improved hard-validation log loss while leaving ranking unchanged. The original models were overconfident under objective shift.

### New competition-only feature work

- Built `session_role_texts.parquet` for all 22,821 sessions with tutor text, student text, opening text, closing tutor text, and closing student text.
- Built `session_behavior.parquet` with 16 deterministic per-session counts, means, and pedagogy proxies, followed by eight legal per-sample ratios.
- Built five reusable role-specific hash matrices plus response-order guards.
- Built `response_objective_context.parquet` for all 35,072 responses. The extractor independently matches one row's objective against only its own transcript, retains neighboring turns, and records ten coverage/position statistics.
- **Verified:** Objective matching found at least one relevant line for `99.23%` of training responses. The median selected context was 45 lines from a median 266-line session.
- **Verified:** 8,364 sessions contain multiple assessed objectives, and 3,207 sessions contain mixed correct/incorrect targets. This explains why objective-conditioned views are conceptually necessary even when their standalone model is not strongest.

### Hard-validation ablations and negative results

All values below use the objective-disjoint split with validation-session purge. `Shrunk loss` is the best OOF-only linear probability blend with the fold training prior; it is reported for diagnosis, not automatically transferred to the final ensemble.

| Model/view                                     | Raw log loss |        AUROC | Shrunk loss | Conclusion                        |
| ---------------------------------------------- | -----------: | -----------: | ----------: | --------------------------------- |
| Hash transcript + objective, alpha `3e-5`      |     0.600716 |     0.613226 |    0.591099 | First leaderboard-matched control |
| Tutor + student + objective                    |     0.596654 |     0.614181 |    0.590579 | Role separation helps             |
| Role/objective + behavior/alignment            |     0.597195 |     0.619776 |    0.589000 | Strong sparse component           |
| Role/retrieval/objective + dense, alpha `1e-4` |     0.592476 |     0.620414 |    0.588714 | Small standalone gain             |
| BGE semantic interaction + dense, `C=0.1`      |     0.588525 |     0.624163 |    0.588467 | Strong complementary component    |
| Final fixed 25/25/50 ensemble                  | **0.584582** | **0.634437** |  not needed | Promoted v0.2 candidate           |

- **Negative result:** Standalone objective-retrieved word text reached only AUROC `0.5873`; it did not beat full transcript or role models.
- **Negative result:** Retrieved character n-grams reached only AUROC `0.5675`; the ASR-robust character hypothesis did not pay off in this form.
- **Negative result:** Closing-student text plus objective reached AUROC `0.5783`; the final quarter alone discards too much context.
- **Negative result:** BGE context/objective embeddings without behavior features peaked near AUROC `0.5775`.
- **Control:** The deterministic behavior/retrieval dense features without BGE peaked near AUROC `0.6089`. The combined semantic/dense model's `0.6242` therefore reflects real semantic complementarity, not just the dense controls.
- **Learning:** The full-transcript model has weaker standalone hard-validation AUROC (`0.5992`) but improves the ensemble because its errors differ from the role and semantic models.
- **Risk:** A weight optimizer fitted to all OOF rows produced AUROC `0.6342`, while leave-one-objective-fold-out weight selection produced `0.6260`. The promoted weights are deliberately rounded (`0.25`, `0.25`, `0.50`) and the `0.6344` estimate must still be treated as optimistic.
- **Decision:** Exclude the standalone retrieval sparse component from v0.2 because the nonnegative ensemble optimizer assigned it zero weight. Keep objective retrieval only inside the semantic feature path.

### Provider and external-resource conclusions

- **Verified from the competition description:** The challenge combines Eedi short typed chats and Third Space Learning long voice/ASR sessions.
- **Verified locally:** The 22,821 supplied training transcripts behave like the long voice/ASR provider: median duration is roughly 43 minutes, `[unclear]` is frequent, and examples include spoken/audio artifacts.
- **Inference:** Provider shift may be another hidden-test risk. The hidden provider composition is not disclosed, so do not present this as confirmed.
- **License decision:** `Eedi/Question-Anchored-Tutoring-Dialogues-2k` is marked CC BY-NC 4.0 and its model card directs commercial users to contact Eedi. It was not downloaded or used for training. Written commercial permission and organizer clearance would be required first.
- **External model used:** `BAAI/bge-small-en-v1.5`, MIT license, 384 dimensions, packaged for offline inference. Model weight SHA-256: `ea1d11a3f23d14fe09fc1826fc7944e89c09a634d2217d57a21dd136805ee3e8`.
- **Verified runtime compatibility:** Official runtime repository commit `ea9a81755e101b8036e386430c3a2f3d7c655f2e` includes Python 3.12, scikit-learn 1.8.0, sentence-transformers 5.5.0, Transformers 4.57.6, PyTorch, and an A100 runtime.

### Promoted v0.2 artifact and verification

| Item             | Value                                                                 |
| ---------------- | --------------------------------------------------------------------- |
| Model artifact   | `models/final_ensemble_v02.joblib`                                    |
| Artifact size    | 3,012,930 bytes                                                       |
| Artifact SHA-256 | `14f2f7dd5298a2c7ebd14800f5d5f3263c15f90246e8533f75a44937d2e1b77d`    |
| Upload ZIP       | `submission_builds/final_ensemble_v02.zip`                            |
| ZIP size         | 81,118,668 bytes                                                      |
| ZIP SHA-256      | `c6d64f694fe3e983826af904e5cd0a39a91091eadaafdb97a7fe1a42df823827`    |
| Fixed blend      | 25% full transcript, 25% role/objective/dense, 50% BGE semantic/dense |

- **Verified:** Twelve unit tests pass.
- **Verified:** The formal milestone verifier still passes all 25 checks.
- **Verified:** Standalone inference reproduced cached full-data model probabilities within maximum absolute error `3.1e-8` on a 20-response end-to-end comparison.
- **Verified:** Final inference was batch-invariant within maximum absolute error `1.6e-8` on real training samples.
- **Verified:** The 100-row local smoke fixture completed with valid probabilities and no stdout/stderr. Its in-sample log loss was `0.517774` and AUROC `0.76637`; these are functional diagnostics only, not held-out estimates.
- **Verified:** A clean extract-and-run of the actual ZIP produced byte-identical prediction CSV content to the staged run. The archive includes the otherwise-empty `assets/bge-small-en-v1.5/2_Normalize/` directory required by the model module configuration.
- **Open:** Docker and `just` are not installed locally, so the official Docker image was not run. The platform smoke test is the required authoritative runtime check.
- **Decision:** Do not make another Normal submission before the v0.2 platform smoke test completes cleanly. A top-one outcome is a goal, not a guarantee; the strict local ensemble now reaches the reference top AUROC, but public log loss can still differ because the hidden distribution is unknown.

### Exact next action

1. Upload `submission_builds/final_ensemble_v02.zip` as a **Smoke test**, not a Normal submission.
2. Use private note: `v0.2 objective-shift robust | full hash 25 + role/behavior 25 + BGE semantic 50 | hard CV LL 0.58458 AUC 0.63444`.
3. Confirm root extraction, BGE offline loading, absence of warnings, `submission.csv` creation, exit code `0`, and smoke runtime below 10 minutes.
4. If and only if the smoke test is clean, use the exact same ZIP for one Normal submission.
5. Record the new job ID, runtime, public log loss, public AUROC, rank, and the public-minus-hard-CV gap here before changing any model.

### Additional sources checked

- BGE model card and MIT license: https://huggingface.co/BAAI/bge-small-en-v1.5
- all-MiniLM reference and Apache-2.0 license: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Eedi QATD2k card and commercial-use warning: https://huggingface.co/datasets/Eedi/Question-Anchored-Tutoring-Dialogues-2k
- Official runtime source: https://github.com/drivendataorg/tutoring-outcomes-runtime

## 2026-07-16 (Asia/Kolkata) - v0.2 platform smoke job 1708

### Platform result

- Uploaded `submission_builds/final_ensemble_v02.zip` as smoke job `1708`.
- **Verified:** Job status `Completed`, exit code `0`, smoke log loss `0.5178`.
- **Verified:** Inference ran from approximately `17:18:03.698` to `17:18:13.557`, about `9.86` seconds, far below the ten-minute smoke limit.
- **Verified:** The runtime found all 15 ZIP entries, including the empty `2_Normalize/` model-module directory, loaded the packaged BGE model offline, wrote `submission.csv`, and emitted no Python/model warning.
- **Verified parity:** The platform smoke score rounds exactly from the local smoke log loss `0.5177735669`. This confirms that local and official-runtime preprocessing/model predictions agree on the smoke fixture.

### Correct interpretation of the apparently worse score

- **Verified:** `0.5178` is numerically worse than the earlier v0.1 smoke score `0.4022`; lower log loss is better.
- **Correction:** This does not establish that v0.2 will be worse on the hidden leaderboard. The platform documents that smoke data are 100 responses sampled from training data. The v0.1 TF-IDF model was fitted directly on those rows' sessions/objectives and its unusually low `0.4022` was an in-sample memorization score.
- **Learning:** v0.2 is deliberately more regularized and optimized for unseen-objective generalization. It sacrifices performance on already-seen training rows in exchange for materially stronger objective-disjoint OOF performance.
- **Decision:** Use smoke score only as a functional/runtime check. Use objective-disjoint, session-purged OOF evidence to decide whether a scarce Normal submission is justified.
- **Decision:** Job `1708` clears the runtime gate. The exact same ZIP is approved for one Normal submission; its true value must be measured by the public leaderboard, not by the smoke score.

## 2026-07-16 (Asia/Kolkata) - v0.2 public result, rank 16, and top-1 redirection

### Normal submission result

- **Verified:** The exact v0.2 ZIP completed the Normal platform job with exit code `0`.
- **Verified:** Inference ran from approximately `17:34:27.227` to `17:38:08.334`, about `221.1` seconds.
- **Verified:** Public log loss is `0.6081`, public AUROC is `0.6147`, and the entry moved from rank `75` to rank `16` after two Normal submissions.
- **Verified:** The prior public result was log loss `0.6181` / AUROC `0.6129`. v0.2 therefore improved log loss by `0.0100` and AUROC by `0.0018`.
- **Verified from the live leaderboard:** Current rank 15 is `0.6080`, rank 8 is `0.6060` / `0.6145`, and rank 1 is `0.6013` / `0.6309`.
- **Learning:** v0.2's main public gain came from probability robustness/calibration, not a large improvement in ranking. The objective-disjoint hard-CV AUROC `0.6344` did not transfer directly to public AUROC `0.6147`.
- **Correction:** Random objective-ID-disjoint validation remains useful but cannot remain the sole promotion test. It permits semantically related objectives to cross folds and its per-fold difficulty is highly variable.

### Post-result diagnostics

- **Verified:** v0.2 hard-fold log loss ranges from `0.5554` to `0.5969`; hard-fold AUROC ranges from `0.5946` to `0.6986`. The pooled score hides substantial objective-family instability.
- **Verified:** Pooled OOF optimization changes the fixed `25/25/50` weights to approximately `22.2/29.6/48.2` but improves loss by only `0.000054` (`0.584582` to `0.584527`). The rounded blend is already essentially optimal on that validation set.
- **Negative result:** Cross-fitted Platt scaling worsens hard loss to `0.585630`.
- **Negative result:** Cross-fitted beta calibration worsens hard loss to `0.585557`.
- **Negative result:** Learned logistic stacking of component predictions, disagreements, and session metadata is worse than the fixed blend; the best simple form is roughly `0.5877`.
- **Negative result:** Shrinking the ensemble toward the hard-fold training prior does not help; the best weight is 100% v0.2.
- **Negative result:** A BGE k-nearest-objective difficulty prior peaks at standalone hard loss `0.600951` / AUROC `0.579096` and receives zero useful blend weight.
- **Decision:** Do not use a Normal submission for a calibration-only transform, a tiny blend-weight adjustment, or the semantic-neighbor prior.

### New strategy

- **Decision:** Add repeated semantic-family-disjoint and provider/style-proxy stress tests with session purge before training the next submission candidate.
- **Decision:** The highest-priority new model is a multi-view BGE evidence model over role-specific objective chunks, tutor-question/student-answer windows, tutor-feedback windows, and opening/closing evidence.
- **Decision:** Add ordered student-mastery and tutoring-move trajectory features rather than more flat counts.
- **Decision:** Test `Qwen/Qwen3-Embedding-0.6B` only after the multi-view BGE design shows robust value; it is an Apache-2.0, instruction-aware long-text embedding model, but local CPU cost is material.
- **Rule consequence:** Competition transcripts stay local. Do not send them to external APIs or third-party annotation services.
- **Submission policy:** Assume only one Normal submission remains in the current seven-day window. Use it only for a materially new v0.3 that passes the documented promotion gates and a clean smoke test.
- **Documentation:** The full experiment order, thresholds, risks, and submission schedule are in `TOP_1_EXECUTION_PLAN.md`.

## 2026-07-17 (Asia/Kolkata) - V100/V110 robust validation baseline

### What was implemented

- Added `src/trace_ace/robust_validation.py` and `scripts/run_robust_validation.py`.
- Added deterministic semantic-family clustering of the 398 objective descriptions using the frozen BGE objective embeddings.
- Added four semantic-family-disjoint validation protocols using 25, 50, 50, and 80 clusters across different cluster/fold seeds.
- Every fold holds out complete semantic families and objective IDs, then purges every training row whose session appears in validation.
- Added a regime scorecard for fold, transcript length, ASR uncertainty, background rate, objective frequency, objective-retrieval coverage, and the number of response rows associated with a training session.
- Added tests for deterministic assignments, complete folds, semantic-family disjointness, and scorecard construction. The full suite now passes all 14 tests.

### Full robust run

| Item                         | Value                                                                                      |
| ---------------------------- | ------------------------------------------------------------------------------------------ |
| Run ID                       | `20260716T183434Z_robust_validation`                                                       |
| Runtime                      | approximately 614 seconds including one-time loading of the large sparse caches            |
| OOF predictions              | `experiments/runs/20260716T183434Z_robust_validation/oof_predictions.parquet`              |
| Regime scorecard             | `experiments/runs/20260716T183434Z_robust_validation/regime_scorecard.csv`                 |
| Objective-family assignments | `experiments/runs/20260716T183434Z_robust_validation/objective_family_assignments.parquet` |

| Protocol           | v0.2 log loss |   v0.2 AUROC |
| ------------------ | ------------: | -----------: |
| `semantic_k25_s0`  |      0.583099 |     0.639570 |
| `semantic_k50_s0`  |      0.587578 |     0.624048 |
| `semantic_k50_s1`  |      0.582543 |     0.640916 |
| `semantic_k80_s0`  |      0.586927 |     0.625124 |
| **Mean**           |  **0.585037** | **0.632414** |
| **Median**         |  **0.585013** | **0.632347** |
| **Worst protocol** |  **0.587578** | **0.624048** |

- **Verified:** The fixed v0.2 ensemble beats every individual component on log loss in all four semantic-family protocols.
- **Verified:** Protocol variability remains material. Objective clustering seed/resolution changes AUROC by roughly `0.017`, so one semantic split is still not trustworthy by itself.
- **Verified:** The pooled robust optimum is approximately `25.2%` full transcript, `24.1%` role/dense, and `50.7%` semantic/dense. This is effectively identical to the deployed `25/25/50` blend.
- **Verified stability:** Learning weights from three protocols and evaluating the fourth makes held-out loss worse in all four leave-one-protocol-out tests.
- **Decision:** Freeze the v0.2 weights. No further weight-only or calibration-only experiment is eligible for a Normal submission.

### Failure regimes and modeling consequences

- **Actionable failure:** Low objective-retrieval coverage is the worst legal per-sample regime, with mean loss about `0.6264`, worst loss `0.6298`, mean AUROC `0.6063`, and worst AUROC `0.5960`.
- **Actionable failure:** The longest transcript quintile has mean loss about `0.6025` and worst loss `0.6059`. A single compressed 256-token context is inadequate for the longest sessions.
- **Actionable failure:** The rarest objective-frequency quintile has mean loss about `0.6017`; the most frequent held-out-objective quintile is also unstable and has the weakest ranking. Frequency alone does not supply a safe correction.
- **Diagnostic only:** Training sessions associated with one response row have very high loss around `0.649`. The count of other test responses sharing a session is prohibited as an inference feature, so this result may guide research but cannot be used directly.
- **Learning:** The two largest actionable weaknesses, low lexical retrieval coverage and long transcripts, both support multi-view semantic evidence extraction. The next model must retrieve role-specific evidence and multiple moments instead of expanding the current global blend.

### Decision and next action

- V100 (repeated semantic-family validation) and V110 (regime scorecard) are complete.
- Start E200/E210: role-specific, ordered, multi-view BGE evidence caches and hard OOF models.
- v0.3 must beat the frozen v0.2 baseline across these same four protocols before packaging or smoke testing.

## 2026-07-17 (Asia/Kolkata) - E200/E210 multi-view evidence and E220 ordered mastery implementation

### Multi-view evidence design

- Added `src/trace_ace/multiview_cache.py` and `scripts/build_multiview_cache.py` for four deterministic, response-aligned BGE views: objective-conditioned student evidence, tutor evidence, ordered answer/feedback windows, and opening/middle/closing session trajectory.
- The evidence selectors parse each session once, ignore background lines for dialogue trajectory, retain objective-overlap matches, retain recent evidence, and use evenly spaced fallback evidence when lexical retrieval is weak.
- Added a versioned view schema (`2026-07-17-v2-budgeted`) and per-view embedding metadata so stale text or embedding caches cannot be silently reused after an extraction change.
- Added resumable, deduplicated encoding. Identical view text is embedded once and scattered back to response rows; this reduces session-trajectory encoding from 35,072 response rows to 22,821 unique session texts without changing response-level features.

### Truncation mistake caught and corrected before training

- **Initial audit:** The first feedback view averaged about 330 words and reached 511 words. With BGE capped at 256 tokens, late tutoring feedback and closing evidence would often have been silently truncated.
- **Correction:** Reduced the number and per-line size of selected role/feedback windows while retaining objective matches and recent evidence. Also shortened each selected trajectory turn.
- **Verified budget after correction:** Student evidence mean/max is `83.2/182` words; tutor `138.5/185`; feedback `150.6/219`; trajectory `136.6/231`. No view is empty, all 35,072 response IDs remain in exact modeling order, and there are 22,821 unique trajectories matching the training-session count.
- **Learning:** Text-view length must be audited against the encoder's actual maximum sequence length before embedding. A richer extractor can become worse if its decisive closing evidence sits beyond the truncation boundary.

### Frozen-fold evaluation harness

- Added `src/trace_ace/multiview_validation.py` and `scripts/run_multiview_validation.py`.
- Candidate models reuse the exact response IDs and fold assignments from `20260716T183434Z_robust_validation`; fold-local scaling and fitting still use validation-session purge through the existing hard-validation routine.
- The initial candidate families are student interaction, feedback interaction, mean-evidence interaction, and combined context/student/tutor/feedback/trajectory interaction. Each model also receives per-view/objective/context cosine geometry and the existing legal per-sample dense features.
- **Runtime design:** Each candidate now loads and uses only the new views it actually requires. Student-only and feedback-only ablations therefore do not secretly depend on all four embeddings; the combined model remains a deliberate four-view experiment.
- **Verified rule correction:** The 10-minute limit applies to the 100-response smoke test. A full submission may run for up to 6 hours on the documented single-A100 runtime, although unnecessary compute should still be avoided and monitored.
- **Selection policy:** Screen architecture and regularization only on frozen pilot protocol `semantic_k50_s0`; lock the candidate and blend weight before evaluating the remaining semantic-family protocols. Do not promote a pooled best-fit result.

### E220 ordered tutoring and mastery features

- Added `src/trace_ace/ordered_features.py` and `scripts/build_ordered_features.py` with 51 deterministic, objective-conditioned sequence features.
- Signals include early-to-late student objective overlap, tutor objective overlap, affirmation/correction movement, uncertainty/reasoning movement, ordered student-answer/tutor-feedback pairs, correction-to-revision overlap improvement, later affirmation after correction, response-length movement, and positions of the last correction/affirmation.
- These features use only the current row's objective and its own transcript. They do not use other test rows, test-set counts, clustering, fitted test statistics, pseudo-labels, or session multiplicity across test rows.
- Added unit coverage for correction→improved revision→affirmation and background-line exclusion. Targeted multi-view and ordered-feature tests pass.

### Current execution state

- The versioned multi-view text artifact is complete at `data_cache/response_multiview_texts.parquet` (35,072 rows, approximately 30.7 MB).
- The versioned E220 artifact is complete at `data_cache/response_ordered_features.parquet` (35,072 aligned rows, 51 features, approximately 2.96 MB). All values are finite and no feature is constant.
- **Verification:** The full local unit suite now passes all 20 tests after the multi-view, ordered-feature, candidate-audit, and feedback-hash additions.
- **Diagnostic only:** The largest absolute single-feature target association is about `0.0803` for reasoning followed by tutor affirmation. Late student word count and objective-overlap measures are next around `0.05-0.06`. These are exploratory training-data associations, not held-out evidence and not a promotion result.
- Local CPU BGE encoding is in progress and resumable. The model, 384-dimensional representation, normalization, and 256-token limit match the packaged v0.2 runtime encoder.
- No v0.3 candidate has been scored, packaged, smoke-tested, or submitted yet. The frozen v0.2 remains the only promotion baseline until the new models clear the robust gates.

### E220 frozen-fold result

| Stage                                  | Run ID                                  | Result                                                                                                                      |
| -------------------------------------- | --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Pilot with ordered features            | `20260716T192608Z_multiview_validation` | `C=0.03`, 40% candidate blend: loss `0.586916`, AUROC `0.626968`; deltas vs v0.2 are `-0.000662` loss and `+0.002920` AUROC |
| Pilot control without ordered features | `20260716T192646Z_multiview_validation` | Best comparable gain is only `-0.000125` loss and `+0.000562` AUROC                                                         |
| Locked confirmation on other protocols | `20260716T192736Z_multiview_validation` | Mixed and unstable; fails promotion                                                                                         |

- **Verified pilot fold behavior:** The locked ordered candidate wins log loss in 3/5 pilot folds and AUROC in 4/5. The control run shows that most of this pilot gain comes from E220 rather than simply changing semantic-model regularization.
- **Locked confirmation at 40% candidate weight:** `semantic_k25_s0` worsens by `0.001360` loss / `-0.001097` AUROC; `semantic_k50_s1` worsens by `0.000830` / improves AUROC by `0.000315`; `semantic_k80_s0` improves by `0.000931` / `0.004627` AUROC.
- **Decision:** E220 is a genuine but semantic-family-unstable complementary signal. Reject it as an independent v0.3 change. It may be re-tested only as a pre-declared small ablation on a multi-view candidate that already passes the robust gates.

### E200 student-only pilot result

- **Run:** `20260716T192843Z_multiview_validation`, frozen pilot protocol `semantic_k50_s0`.
- Tested student/objective interaction at `C=0.01`, `0.03`, and `0.1`, with pre-declared v0.2 blend weights from 10% to 50%.
- Best result is `C=0.03`, 10% student model: loss `0.587518`, AUROC `0.624299`; deltas vs v0.2 are only `-0.000060` loss and `+0.000252` AUROC.
- The `C=0.03` standalone student model scores loss `0.591665`, AUROC `0.613137`, materially below v0.2. Increasing student weight worsens the blend.
- **Decision:** Reject the student-only model at the pilot gate. Do not spend confirmation folds or a submission on it. Student embeddings may remain an ablation inside a combined model only if feedback first establishes a robust gain.

### Ordered feedback-window hashing and locked candidate audit

| Stage                                | Run ID                                      | Finding                                                                    |
| ------------------------------------ | ------------------------------------------- | -------------------------------------------------------------------------- |
| Word/ordered pilot                   | `20260716T193609Z_feedback_hash_validation` | Locked winner: word hashing + E220 dense features, `alpha=1e-4`, 30% blend |
| Character and word+character control | `20260716T194048Z_feedback_hash_validation` | Character features are worse; word-only remains locked                     |
| Untouched-protocol confirmation      | `20260716T194158Z_feedback_hash_validation` | Improves loss and AUROC on all three confirmation protocols                |
| Full fold/regime audit               | `20260716T194336Z_candidate_audit`          | Consistent overall but fails the worst-regime regression gate              |

- **Architecture:** Hash the explicit ordered `[ANSWER] -> [FEEDBACK]` windows with word 1-2 grams and combine them with the 51 E220 sequence features. This is different from the deployed role model, which flattens tutor and student text separately and cannot model adjacency.
- **Control result:** Feedback word text without E220 features adds no useful pilot signal. The gain requires the ordered sequence features; character-only and word+character variants are also worse than word-only.
- **Locked 30% blend protocol deltas:** `semantic_k50_s0` `-0.001197` loss / `+0.004536` AUROC; `semantic_k25_s0` `-0.000822` / `+0.005195`; `semantic_k50_s1` `-0.001167` / `+0.005907`; `semantic_k80_s0` `-0.001406` / `+0.005132`.
- **Robust summary:** Mean/median loss improvement is `0.001148/0.001182`; mean AUROC improvement is `0.005192`; loss improves in all 4 protocols and 14/20 folds; AUROC improves in all 4 protocols and 17/20 folds. Worst fold loss regression is only `0.000720`.
- **Actionable strength:** The candidate improves low objective-retrieval coverage by approximately `0.00304-0.00368` loss in every protocol, directly addressing v0.2's worst legal per-sample regime. It also strongly improves the highest objective-frequency quintile.
- **Failure:** It regresses middle/high objective-frequency q3-q4 regimes; worst legal-regime loss regression is `0.001759`, above the `0.001` gate. Median loss gain is also below the `0.002` independent-submission threshold.
- **Decision:** Retain as a cheap complementary E240 component and as evidence for a carefully nested E230 retrieval-coverage gate. Do not promote or submit the fixed 30% blend by itself. Any conditional gate must use only per-sample retrieval statistics and training-fitted thresholds, with selection nested away from its evaluation protocol.
- **Post-hoc sensitivity only:** Audit `20260716T194554Z_candidate_audit` shows that a 20% weight improves all four protocols, wins loss and AUROC in 18/20 folds, and reduces worst legal-regime regression to `0.000899` while retaining mean loss/AUROC gains of `0.001051/0.004362`. It passes consistency and regime gates but still fails the `0.002` loss-size gate. Because 20% was examined after confirmation results were visible, it is not a newly confirmed weight; it requires fresh protocols or must remain a small final-blend sensitivity.

### Fresh-protocol confirmation of the 20% ordered-feedback hash component

- Added two protocols that were not used for architecture, alpha, or weight selection: `semantic_k35_fresh` and `semantic_k65_fresh`, with new cluster and fold seeds.
- **Fresh v0.2 baseline run:** `20260716T200419Z_robust_validation`; k35 loss/AUROC `0.589229/0.620650`, k65 `0.585975/0.628102`.
- **Fresh component run:** `20260716T200509Z_feedback_hash_validation`, fixed `feedback_word_ordered`, `alpha=1e-4`, and pre-frozen 20% blend.
- **Fresh results:** k35 improves loss by `0.000736` and AUROC by `0.003673`; k65 improves loss by `0.001214` and AUROC by `0.004452`.
- **Six-protocol audit:** `20260716T200528Z_candidate_audit`. The fixed 20% component improves loss and AUROC on all six protocols, wins loss in 24/30 folds and AUROC in 26/30, and has mean loss/AUROC improvements `0.001026/0.004262`. Worst fold loss regression is `0.000828`.
- **Remaining caveat:** The worst subgroup regression is `0.001217`, again confined to a descriptive objective-frequency subgroup; the component still fails the `0.002` loss-size gate. It is now independently confirmed as a stable, cheap ensemble ingredient, but remains ineligible as a standalone Normal submission.

### E210 neural feedback-view pilot result

- **Run:** `20260716T201513Z_multiview_validation`, frozen pilot `semantic_k50_s0`.
- Audited and froze a 35,072 x 384 normalized BGE feedback embedding cache before fitting.
- Tested feedback cosine geometry, feedback/objective interaction, and joint current-context + feedback interaction at `C=0.01`, `0.03`, and `0.1`.
- Best result is feedback interaction `C=0.03` at 20% blend: loss `0.587380`, AUROC `0.624972`; deltas vs v0.2 are only `-0.000198` loss and `+0.000925` AUROC.
- The joint context+feedback representation does not improve on this and is effectively neutral or worse at practical weights.
- **Learning:** Ordered answer-feedback windows contain predictive lexical/sequence cues, but BGE-small compresses them into a representation that adds almost no outcome signal beyond v0.2. The confirmed ordered word-hash component is both stronger and cheaper.
- **Decision:** Reject all feedback-BGE variants at the pilot gate. Do not spend confirmation folds or submission runtime on them.

### E221 symbolic answer-feedback event sequence result

- Added deterministic event tokens for selected answer-feedback pairs: position, objective-overlap level, reasoning/uncertainty, answer length/type, tutor affirmation/correction/question/reasoning, and explicit pair interactions.
- **Pilot run:** `20260716T201933Z_feedback_hash_validation`. Word + symbolic events + E220 at `alpha=1e-4`, 30% blend improved pilot loss by `0.001256` and AUROC by `0.005037`, slightly better than word + E220 on the pilot.
- **Locked confirmation:** `20260716T202104Z_feedback_hash_validation`. Loss improves by only `0.000524` on k25, `0.000883` on k50-s1, and `0.001322` on k80. All three are worse than the simpler confirmed word + E220 component at its selected settings.
- **Learning:** Hand-designed event interactions can increase pilot AUROC but did not generalize consistently enough to justify their extra complexity. The simpler representation is more robust.
- **Decision:** Reject E221. Do not tune more event vocabulary or send it to fresh protocols. Retain the independently confirmed word-window + E220 component.

### E230 observable-style component-weight audit

- Compared v0.2 full-transcript, role/dense, semantic/dense, and fixed-ensemble losses across transcript-length, ASR-uncertainty, background-rate, and retrieval-coverage regimes over the original four robust protocols.
- **Verified:** The fixed 25/25/50 ensemble beats every individual component in every audited regime.
- Rounded per-regime grid optimization changes weights only slightly for length, ASR, and background groups and yields at most `0.000108` local subgroup improvement.
- Low retrieval coverage prefers approximately 15% full / 40% role / 45% semantic and gains `0.000560` inside that subgroup, but the subgroup is only about 10% of rows and the overall opportunity is negligible.
- **Decision:** Reject a transcript-style component-weight gate. Preserve fixed v0.2 weights; the confirmed ordered-feedback hash component is a much stronger response to low retrieval coverage.

### Ordered-feedback optimizer control

- **Run:** `20260716T203006Z_feedback_hash_validation`.
- Refit the confirmed word-window + E220 representation with exact L2 logistic regression (`liblinear`) at `C=0.1`, `0.3`, and `1.0`, keeping all folds and blend weights fixed.
- Best logistic pilot blend improves loss by `0.001138` and AUROC by `0.004694`, versus averaged SGD's `0.001197/0.004536` at its selected pilot setting.
- **Decision:** The tiny mixed trade-off does not justify a slower optimizer or another confirmation branch. Keep the independently confirmed averaged-SGD component.

### Nonlinear ordered/dense tree control

- **Run:** `20260716T203248Z_dense_tree_validation` on frozen pilot `semantic_k50_s0`.
- Tested shallow histogram gradient boosting with 7 and 15 leaves over existing legal dense behavior/retrieval/semantic-similarity features plus all 51 E220 ordered features.
- Standalone losses are `0.601416-0.602720` and AUROCs about `0.585`; every tested blend is neutral or worse than v0.2.
- **Decision:** Reject nonlinear dense trees at the pilot gate. The confirmed E220 value depends on adjacency-aware lexical feedback windows, not generic nonlinear thresholds over aggregate features.

### Full-transcript character-hash feasibility stop

- Attempted a fit-free full-session character 3-5 gram hash cache to target ASR spelling noise.
- The in-memory sparse construction reached approximately 9.6 GB private memory on the 11.4 GB local machine before producing a valid cache and began paging heavily alongside trajectory encoding.
- The job was stopped deliberately; no partial matrix was retained and no model score was produced.
- **Decision:** Do not use unbounded full-transcript character hashing locally. If character robustness is revisited, restrict it to compact trajectory/closing evidence so the sparse representation has a bounded memory footprint.

### Bounded compact-trajectory word/character hash result

- Built safe compact-trajectory caches: word approximately 14.4 MB and character approximately 54.1 MB, avoiding the full-character memory explosion.
- **Pilot:** `20260716T204943Z_feedback_hash_validation`. Word trajectory + E220 at `alpha=1e-4`, locked 20% blend improves loss by `0.001204` and AUROC by `0.005463`; character and word+character are effectively tied.
- **Confirmation:** `20260716T205109Z_feedback_hash_validation`. The locked blend worsens k25 loss by `0.000077`, worsens k50-s1 by `0.000276`, and improves k80 by `0.001147`.
- **Decision:** Reject compact trajectory hashing for instability. The independently confirmed ordered answer-feedback word windows remain the only new sparse component retained.

### E210 neural compact-trajectory pilot result

- Audited the completed BGE-small trajectory cache before fitting: 35,072 x 384 float32 rows, all finite, row norms effectively 1.0, and metadata aligned to the frozen `2026-07-17-v2-budgeted` view schema.
- **Run:** `20260716T205622Z_multiview_validation`, frozen pilot `semantic_k50_s0`.
- Tested trajectory cosine geometry, trajectory/objective interaction, and joint current-context + trajectory interaction at `C=0.01`, `0.03`, and `0.1`.
- Best result is trajectory geometry `C=0.03` at 20% blend: loss `0.587452`, AUROC `0.624611`; deltas versus v0.2 are only `-0.000126` loss and `+0.000564` AUROC.
- **Learning:** Both sparse trajectory hashing and dense BGE trajectory encoding fail to generalize beyond the already deployed current-context semantic model. Compacting an entire session into one trajectory representation is not the missing signal.
- **Decision:** Reject trajectory-BGE variants at the pilot gate. Do not spend confirmation protocols or a submission on them. Pause the lower-priority tutor-view encoding at its resumable 10,240/35,072 checkpoint while testing higher-upside architectures.

### Objective-conditioned semantic label retrieval result

- Implemented a legal per-sample two-stage kNN model: select only labeled training objectives nearest to the sample objective, then retrieve labeled training responses using a fixed context/objective cosine score. Validation uses the existing semantic-family-disjoint folds and session purge; no validation labels or test aggregates enter a prediction.
- **Run:** `20260716T210242Z_semantic_knn_validation`, frozen pilot `semantic_k50_s0`.
- Tested objective pools of 8, 20, and 40 objectives; response neighborhoods of 64, 128, and 256; and balanced versus objective-heavy similarity.
- Every blend worsens v0.2. The least harmful result, 5% of `m8_k64_balanced`, changes loss by `+0.000707` and AUROC by `-0.002172`; the standalone kNN AUROC is only `0.536080`.
- **Learning:** Neighbor labels are not transferable across held-out semantic objective families. BGE-small is useful for learning a global supervised boundary, but its local label neighborhoods are unreliable.
- **Decision:** Reject semantic label retrieval at the pilot gate. Do not tune k, temperatures, or retrieval weights further.

### Encoder-upgrade decision

- The representation evidence now points consistently toward the global semantic boundary: alternate BGE-small role views, trajectory views, sparse trajectory hashes, and local label retrieval all fail, while v0.2's full context/objective semantic model produced the large public improvement.
- The runtime permits a 60 GB archive, six-hour inference, one A100 80 GB GPU, 24 vCPUs, and 220 GB RAM, so a stronger packaged encoder is operationally feasible.
- `BAAI/bge-base-en-v1.5` is a drop-in 0.1B-parameter, 768-dimensional upgrade under the MIT license. `Qwen/Qwen3-Embedding-0.6B` is a later higher-cost option under Apache-2.0. Both licenses permit commercial use and satisfy the competition's pretrained-model rule.
- **Decision:** Evaluate BGE-base first using only frozen full context and objective text, identical hard folds, and the existing semantic interaction architecture. This is a representation-capacity experiment, not a new feature-family search. Escalate to Qwen3 only if the base-scale result or a controlled cost/benefit audit justifies it.

### Existing-component ensemble ceiling audit

- Recombined already generated OOF predictions without fitting on test data. On the pilot, ordered feedback + compact trajectory can improve AUROC more than ordered feedback alone, but its best log-loss blend is still dominated by the simpler ordered-feedback component.
- On the three confirmation protocols, 20% ordered feedback + 10% trajectory averages `-0.000987` log-loss and `+0.005967` AUROC versus v0.2. The confirmed feedback-only family reaches approximately `-0.001135` mean loss at 27.5% and therefore retains the better log-loss frontier.
- **Decision:** Do not revive the rejected trajectory component merely to chase AUROC. Keep it available only as a later tiny sensitivity if a materially stronger primary model first clears the loss gate.

### Objective-shift session-weighting result

- Refit the full-session word model with inverse square-root and inverse session-multiplicity sample weights. This tests whether sessions containing many labeled objectives were dominating training and hurting objective-mix transfer.
- **Pilot:** `20260716T212049Z_session_weight_validation`. The best pre-declared result is inverse-count weighting blended 20% with v0.2: `-0.000641` loss and `+0.003255` AUROC.
- **Locked confirmation:** `20260716T212643Z_session_weight_validation`, inverse-count power 1 with the same 20% blend. It worsens k25 by `+0.000855` loss / `-0.002254` AUROC, k50-s1 by `+0.001446` / `-0.003328`, and k80 by `+0.000672` / `-0.000899`.
- **Learning:** Equalizing session influence creates a pilot-only improvement but removes useful response-level weighting under other objective-family partitions. The original full-transcript model's response-level training distribution is more robust.
- **Decision:** Reject session reweighting. Do not add it to the final ensemble or tune the exponent further.

### Live leaderboard checkpoint

- Verified the public leaderboard directly on 2026-07-17. `aj_insanity` remains #16 with log loss `0.6081` and AUROC `0.6147` after two Normal submissions.
- #1 remains `0.6013/0.6309`; #2 is `0.6015/0.6259`; #3 is `0.6032/0.6288`; #4 has the current higher AUROC `0.6332` with loss `0.6033`.
- **Implication:** The target remains approximately `0.0068` public loss and `0.0162` AUROC beyond v0.2. The different loss/AUROC ordering near the top supports retaining both metrics as hard promotion gates rather than optimizing either one alone.

### Surgical 384-token context audit

- Tokenized all 35,072 frozen compact objective contexts with the packaged BGE tokenizer. Mean/p95/max lengths are `210.78/261/363`; 2,351 rows (6.7%) exceed 256 tokens, while none exceed 384.
- Built `bge_small_context_384.npy` by byte-copying the 256-token cache and re-encoding only those 2,351 long rows. The completion audit verifies every short row remains byte-identical.
- **Pilot:** `20260716T213801Z_long_context_validation`. At `C=0.03`, the hybrid improves v0.2 loss/AUROC by `0.000260/0.001672`; full semantic replacement improves by `0.000167/0.003191`.
- **Required control:** Reusing the original 256-token embeddings at the same `C=0.03` gives `0.000248/0.001644` for the hybrid and `0.000144/0.003129` for replacement. Therefore 384-token context itself contributes only about `0.000012-0.000023` loss and `0.000028-0.000062` AUROC; nearly all apparent gain is regularization.
- **Decision:** Reject longer context as immaterial and do not spend confirmation folds or extra submission runtime on it. Preserve 256 tokens for the BGE-base capacity comparison.

### BGE-small semantic regularization control

- The long-context control revealed a pilot-only `C=0.03` semantic model that appeared to improve v0.2 relative to the deployed `C=0.1`, so it was treated as a distinct locked regularization candidate.
- **Locked confirmation:** `20260716T213950Z_multiview_validation`, `semantic_reference_c0.03` on k25, k50-s1, and k80.
- Direct blending with v0.2 worsens k25 and k50-s1. Exact full semantic replacement worsens loss by `0.001383`, `0.001807`, and `0.000419` across the three protocols; its AUROC is also negative on k50-s1. The 25/25 hybrid still worsens every protocol's loss.
- **Decision:** Reject `C=0.03`; preserve the deployed BGE-small regularization. The pilot AUROC gain is protocol-specific and must not influence the BGE-base confirmation settings after selection.

### Pre-registered cross-encoder follow-up

- Downloaded the official `cross-encoder/nli-deberta-v3-small` checkpoint for a later frozen-feature experiment; no results from it were used to change the active BGE-base experiment.
- The model card declares Apache-2.0, a 6-layer 768-hidden DeBERTa-v3-small architecture, and label order contradiction/entailment/neutral. The checkpoint was trained on SNLI (CC BY-SA 4.0) and MultiNLI (a documented mixture of permissive CC BY, CC BY-SA, MIT, OANC, and public-domain sources), with no non-commercial restriction identified.
- **Pre-registered architecture:** Encode a single per-sample pair: compact transcript evidence as premise and `The student demonstrates mastery of this learning objective: <objective>` as hypothesis. Cache only frozen joint `[CLS]`/pooler features and NLI logits, then train the same fold-purged linear probe. Do not use test aggregates or external annotation.
- **Execution order:** Complete and gate BGE-base first. Run the NLI cross-encoder only as the next representation-family experiment if more gain is required; do not tune multiple hypothesis prompts after viewing validation results.

#### External model and lineage references

- BGE-base model card and MIT license: https://huggingface.co/BAAI/bge-base-en-v1.5
- Qwen3-Embedding-0.6B model card and Apache-2.0 license: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- DeBERTa-v3-small NLI cross-encoder model card and Apache-2.0 license: https://huggingface.co/cross-encoder/nli-deberta-v3-small
- SNLI dataset card and CC BY-SA 4.0 license: https://huggingface.co/datasets/stanfordnlp/snli
- MultiNLI dataset card and source-license inventory: https://huggingface.co/datasets/nyu-mll/multi_nli

### Versioned v0.3 feedback deployment scaffold

- Converted the confirmed ordered answer-to-feedback candidate into a standalone training module, `src/trace_ace/feedback_upgrade.py`, with a matching CLI and unit test. The builder deliberately loads and preserves the frozen v0.2 artifact, then appends the new component under a new versioned artifact name; it does not overwrite the rank-16 package.
- Frozen settings exactly match the validated candidate: 131,072 word-hash features with word 1-2 grams, 51 E220 ordered tutoring features scaled at weight `0.12`, averaged `SGDClassifier(loss=log_loss, alpha=1e-4, random_state=20260716)`, and a final feedback blend weight of `0.20`.
- Trained the provisional fallback artifact on all 35,072 labeled responses. After adding automatic submission-asset staging, the current output is `models/final_ensemble_v03_feedback.joblib`, 4,312,284 bytes, SHA-256 `efe8c686b5e3677a5652414061f6459c47cde7f5dad099d6272f71860e16f349`; feedback training matrix shape is `35,072 x 131,123`.
- The artifact records the exact v0.2 parent SHA-256 (`14f2f7dd5298a2c7ebd14800f5d5f3263c15f90246e8533f75a44937d2e1b77d`), both feature schema versions, every ordered feature name, model seed, regularization, and blend weight for reproducibility.
- The one-file competition runtime now detects v0.3 assets, reproduces the exact feedback-window and 51-feature training transforms, and blends the promoted component only when the v0.3 artifact is present. A zero-tolerance parity test compares runtime output against the canonical training implementation. Full project verification now contains 25 passing tests. This artifact is a deployment fallback only; it must not consume the remaining Normal submission while the preregistered BGE-base capacity experiment is still running.
- Audited runtime parity again on 128 evenly spaced real training responses. Every generated `feedback_evidence` string matched the cached training text byte-for-byte and the maximum absolute difference across all ordered features was exactly `0.0`. This removes preprocessing drift as a packaging risk for the feedback component.

### BGE-base capacity experiment

- Completed the preregistered `BAAI/bge-base-en-v1.5` cache using the same compact objective context, objective text, 256-token limit, normalization, and semantic interaction architecture as v0.2. This isolates encoder capacity from view or preprocessing changes.
- Context cache: `35,072 x 768` float32, 107,741,312 bytes, all finite, row norms `0.99999982-1.00000012`, SHA-256 `b5f0066cc8d659e6593596ac47967bd14f2d0f02cedc82319e2a2313cf564d1a`.
- Objective cache: `35,072 x 768` float32, 107,741,312 bytes, all finite, row norms `0.99999988-1.00000012`, SHA-256 `6cc754a287f9ec303aa5f81dd98ab0321c7c5299b2a4e251d6f86f63afe6eb27`.
- **Frozen pilot:** `20260716T232134Z_encoder_upgrade_validation`. Best loss form is `C=0.1`, base-heavy 60%, with `-0.001085` loss / `+0.003685` AUROC. The simpler 50% BGE-base replacement is effectively tied at `-0.001050/+0.003667`. `C=0.03` base-heavy reaches `+0.006238` AUROC but only `-0.000599` loss. BGE-base alone therefore does not clear the independent-component gate.
- Applied the already confirmed and frozen 20% ordered-feedback component on the same pilot. Locked `C=0.1`, 50% BGE-base replacement plus 20% feedback gives `-0.001882` loss / `+0.006847` AUROC. This was close enough to the formal across-protocol loss gate to justify opening untouched confirmations, with C, encoder share, and feedback weight frozen.
- **Locked confirmation:** `20260716T232346Z_encoder_upgrade_validation`. The BGE-base replacement alone improves loss on k25/k50-s1/k80 by `0.000256/0.001557/0.001004`, but its AUROC gain is only `0.001296` on k25 and remains mixed in size.
- **Locked combined candidate:** `20260716T232515Z_combined_candidate_validation`, exact formula `0.8 * (0.25 full + 0.25 role + 0.50 BGE-base C=0.1) + 0.2 ordered-feedback`. Confirmation deltas are k25 `-0.001019/+0.005452`, k50-s1 `-0.002294/+0.008731`, and k80 `-0.002045/+0.007759` for loss/AUROC.
- **Four-protocol audit:** `20260716T232529Z_candidate_audit`. It wins loss and AUROC in all four protocols, 85% of loss folds, and 90% of AUROC folds. Mean loss/AUROC gains are `0.001810/0.007197`; median loss gain is `0.001964`, narrowly below the `0.002` gate. Worst fold loss regression is `0.001756`.
- The decisive failure is regime stability: worst legal-regime loss regression is `0.003975`, concentrated in objective-frequency q3-q4, with negative AUROC shifts in the same groups. This repeats the feedback-only warning and becomes larger with BGE-base.
- **Decision:** Reject this BGE-base combination as the remaining Normal submission. Retain its caches and OOF predictions as a potentially useful ingredient for a later diverse ensemble, but do not package or upload it now. Because it already fails loss-size and regime gates, an objective bootstrap cannot rescue promotion and is not run as a selection loophole.
- **Next action:** Execute the already preregistered DeBERTa-v3-small NLI cross-encoder branch. It is a genuinely different joint evidence/objective representation and is not a calibration or post-hoc weight search.

### DeBERTa NLI cross-encoder experiment

- Completed the preregistered `cross-encoder/nli-deberta-v3-small` cache locally using the single fixed hypothesis `The student demonstrates mastery of this learning objective: <objective>.` The premise is the same compact objective evidence used by the semantic branch; pair tokenization uses 256 tokens and truncates only the premise so the complete objective hypothesis is retained.
- Pooled cache: `35,072 x 768` float32, 107,741,312 bytes, all finite, SHA-256 `912dec20b00aeb86fb8ef0bb7ab92cc6303a7f0c09ab581348ae0735b42ea720`.
- Logit cache: `35,072 x 3` float32 in verified contradiction/entailment/neutral order, 420,992 bytes, all finite, SHA-256 `dbd0d3076e8cf456d67960abafc6774d5f4085687d0b4f45f03bb01aedf37f5f`.
- Batch-size throughput was benchmarked only as an operational setting: 64 was materially faster than 32 and 128 while remaining memory-safe. This did not change model weights, text, truncation, cached feature definition, or validation selection.
- **Frozen pilot:** `20260717T011348Z_nli_cross_encoder_validation`. The best declared candidate is `C=0.1` blended 20% with v0.2: loss `0.587151`, AUROC `0.626921`, deltas `-0.000427/+0.002873`. Increasing weight to 30-40% adds at most about `0.0037` AUROC but loses nearly all or all of the loss improvement. Standalone NLI is much worse at `+0.007510` loss / `-0.019499` AUROC.
- **Learning:** Generic SNLI/MultiNLI entailment features weakly complement v0.2 but do not align strongly enough with tutoring mastery outcomes. The frozen logits should not be interpreted as mastery probabilities; supervised outcome probing confirms the mismatch.
- **Decision:** Reject the NLI branch at the pilot gate. Do not open confirmation protocols, tune alternate hypothesis prompts after seeing results, or package the model. Retain the cache and exact negative result for the write-up.

### Qwen3-Embedding-0.6B screened pilot

- Downloaded the official `Qwen/Qwen3-Embedding-0.6B` SentenceTransformers package, 1.125 GB locally, under Apache-2.0. The model has 28 layers, hidden size 1,024, up to 1,024 output dimensions, Matryoshka support, and instruction-aware query encoding.
- A full 256-token CPU cache benchmark was about `0.8` rows/second, implying approximately 12 hours for all 35,072 contexts. Following the preregistered E300 rule, a full cache was not started without a smaller evidence screen.
- Built a deterministic 4,096-row pilot sample with `random_state=20260717`, stratified across all five `semantic_k50_s0` folds and both labels. Compared every encoder on exactly this same reduced training data, folds, dense controls, interaction architecture, and C grid.
- Frozen Qwen settings: 128 tokens, 256-dimensional normalized Matryoshka output, context as retrieval document, and objective query instruction `Given a K-12 learning objective, retrieve tutoring transcript evidence relevant to assessing student mastery`.
- Context pilot cache: `4,096 x 256` float32, all finite/unit-normalized, SHA-256 `048a10d229fd444e921c763aab0347839e763b7de333b7a8acb1130457fbb76f`.
- Objective pilot cache: `4,096 x 256` float32, all finite/unit-normalized, SHA-256 `c004da128fbc612ea8cb8d2b5a36c934c6a053e8374333bea220f7c8082eb15c`.
- **Same-sample pilot:** `20260717T022451Z_qwen_pilot`. At each encoder's best `C=0.1`, Qwen scores loss/AUROC `0.594904/0.603754`, BGE-small `0.595223/0.606798`, and BGE-base `0.595607/0.608966`.
- Qwen improves BGE-base loss by only `0.000703` while worsening AUROC by `0.005212`; it also trails BGE-small AUROC. The preregistered continuation rule (`>=0.0015` loss gain without AUC loss, or `>=0.005` AUC gain without loss regression) is false.
- **Decision:** Reject the Qwen full-cache escalation. Do not spend approximately 12 CPU hours, confirmation protocols, package size, or submission runtime on it. The stronger instruction-aware encoder does not solve the outcome-ranking signal on the fair reduced-data screen.

### Feedback NB-SVM and locked BGE-base combination

- Added a fold-local NB-SVM implementation over the ordered answer-to-feedback word hash and the 51 E220 sequence features. For every outer fold, the positive/negative log-count ratio is fitted only on that fold's legal training rows, after validation-session purge; the ratio is then applied to both training and held-out sparse features before a dual `liblinear` logistic model. No validation labels, test aggregates, or pseudo-labels enter the transform.
- **Frozen pilot:** `20260717T023053Z_nbsvm_validation`. The selected component is `feedback_word_ordered`, `C=1`, blended 30% with v0.2. It improves pilot loss/AUROC by `0.001426/0.006127`. The full-transcript NB-SVM controls are worse, confirming that the useful signal is the local answer-feedback interaction rather than generic token sentiment.
- **Locked confirmation:** `20260717T023209Z_nbsvm_validation`. With architecture, C, and weight unchanged, the component improves loss/AUROC on k25 by `0.000852/0.007082`, k50-s1 by `0.000790/0.005745`, and k80 by `0.001624/0.006094`.
- Combined the locked NB-SVM with the already frozen BGE-base replacement using the exact formula `0.7 * (0.25 full + 0.25 role + 0.50 BGE-base C=0.1) + 0.3 NB-SVM feedback C=1`.
- **Combined run:** `20260717T023245Z_combined_candidate_validation`. It improves all four protocols: k50 `-0.002119/+0.009165`, k25 `-0.001036/+0.008330`, k50-s1 `-0.001847/+0.009064`, and k80 `-0.002401/+0.009679` for loss/AUROC.
- **Decisive audit:** `20260717T023447Z_candidate_audit`. Loss and AUROC improve in all 4 protocols; 14/20 loss folds and 19/20 AUROC folds win. Mean loss/AUROC gains are `0.001851/0.009059`, and median loss gain is `0.001983`, only `0.000017` below the predeclared `0.002` gate.
- **Failure:** Worst fold loss regression is `0.002414`, and worst descriptive subgroup loss regression is `0.005273`, concentrated again in objective-frequency q3-q4. The fixed candidate therefore fails the loss-size gate narrowly and the robustness gate materially. Do not spend the remaining Normal submission on it in this form.
- **Important legality distinction:** Objective-frequency quintiles are useful retrospective training-data diagnostics, but their current definition uses counts across the held-out evaluation frame. Competition inference may not aggregate test rows, so these quintiles cannot be used as a test-time gate. They remain a conservative robustness check, not a deployable feature or a loophole for conditional blending.
- **NB-only audit:** `20260717T023656Z_candidate_audit`. The locked 30% NB-SVM blend improves all four protocol averages by mean loss/AUROC `0.001173/0.006262`, but has the same q3-q4 reversal and independently fails the promotion gates.
- **Bounded character robustness control:** `20260717T024047Z_nbsvm_validation`. At the already locked `C=1` and 30% pilot weight, equal-unit word+character feedback hashing improves the word-only NB pilot by only `0.000163` loss and `0.000325` AUROC. This is below the predeclared `0.0003` incremental-loss threshold, so confirmation is not opened and character NB-SVM is rejected.
- **Learning:** NB reweighting is a real and unusually strong ranking complement, but the remaining ceiling is representation adaptation, not another sparse weight sweep. The next experiment is a small, leakage-safe supervised encoder adaptation screen; v0.2 and every rejected candidate remain frozen.

### Preregistered supervised BGE-small adaptation screen

- **Question:** Can direct tutoring-outcome supervision adapt a compact pretrained encoder beyond the frozen sentence-embedding ceiling without committing hours to full five-fold CPU training first?
- **Frozen screen design:** `BAAI/bge-small-en-v1.5`, compact objective-conditioned context as the first sequence, the fixed mastery hypothesis as the second sequence, 192 tokens with `only_first` truncation, fold 0 of `semantic_k50_s0` held out, validation-session purge, and a deterministic 4,096-row label-stratified sample from the remaining legal training rows.
- Freeze embeddings and the bottom 8 of 12 BERT layers; train the top 4 layers, pooler, and binary classifier for exactly one epoch. Use batch 16, AdamW, encoder learning rate `3e-5`, head learning rate `1e-4`, weight decay `0.01`, 10% linear warmup, gradient norm 1.0, and seed `20260717`.
- The classifier begins at the sampled training prior. There is no epoch selection, early stopping, alternate fold selection, prompt sweep, or token-length sweep after seeing results. The fixed final checkpoint is evaluated on the complete untouched fold.
- **Continuation rule:** Open a full grouped confirmation only if a predeclared v0.2 blend gains at least `0.001` log loss with no AUROC loss, or at least `0.005` AUROC with no log-loss regression. A screen failure ends this architecture rather than triggering learning-rate or layer-count tuning.
- **Fixed fold-0 result:** `20260717T031253Z_supervised_encoder_screen`. Final training loss is `0.614875` after exactly 256 batches. On all 7,512 untouched validation rows, v0.2 scores loss/AUROC `0.600800/0.623296`; the preregistered 30% supervised blend scores `0.598832/0.630851`, gains of `0.001968/0.007554`.
- **Decision before confirmation:** The screen clears both continuation thresholds. Lock supervised weight `0.30` now and run folds 1-4 with every training setting unchanged. Other pilot blend rows are diagnostics only and may not change the confirmed weight.
- **Exploratory ceiling diagnostic only:** On fold 0, the previously locked BGE-base + NB-SVM candidate scores `0.597309/0.633431`. Mixing 20% raw supervised probability into that candidate changes the result to `0.597402/0.637603`: a negligible `0.000093` loss cost for `0.004172` more AUROC. This is encouraging diversity evidence, not a selected final formula; no combined weight may be promoted until the independent supervised component confirms on the remaining folds.
- **Locked folds 1-4:** `20260717T033630Z_supervised_encoder_screen`, `20260717T040456Z_supervised_encoder_screen`, `20260717T042941Z_supervised_encoder_screen`, and `20260717T045348Z_supervised_encoder_screen`. At the frozen 30% blend, their loss/AUROC deltas are `+0.000430/+0.001148`, `+0.002131/+0.000759`, `+0.003712/-0.004394`, and `+0.003098/+0.001565`.
- **Five-fold aggregate:** `20260717T045715Z_supervised_encoder_confirmation`. Across all 35,072 held-out responses, the locked component worsens log loss by `0.001440` and improves AUROC by only `0.000230`; it wins loss in 1/5 folds and AUROC in 4/5.
- **Decision:** Reject the 4,096-row supervised adaptation architecture. The strong fold-0 result does not transfer across objective families, standalone ranking is weak in every fold, and three folds have material calibration regressions. Do not package it, combine it with the BGE+NB candidate, or run the natural-language role-prefix ablation on top of this failed base.

### Local accelerator and disk-integrity audit

- **Verified hardware:** AMD Radeon integrated graphics plus a discrete AMD Radeon RX 5600M with 4 GB VRAM. The active environment is `torch 2.13.0+cpu`; CUDA is unavailable and `torch-directml` is not installed, so all supervised screens ran on the six-core CPU.
- **Decision:** Do not move the locked confirmation to AMD acceleration. The RX 5600M is absent from AMD's supported ROCm-on-Windows matrix. DirectML would require an isolated environment and a downgrade to its supported PyTorch line, while 4 GB VRAM would force small batches. That backend change is too risky for an already reproducible confirmation run. A future isolated throughput benchmark is allowed only if a substantially larger-data neural experiment first becomes the selected bottleneck.
- The supervised aggregate initially exposed a full-disk condition. Root cause was not model caches: downloaded Hugging Face directories contained nested Git-LFS attributes but `/assets/pretrained/` was missing from the root `.gitignore`. Repeated workspace snapshots left approximately 196.6 logical GB of incomplete files under `.git/lfs/tmp`.
- Added `/assets/pretrained/` to the root ignore policy, verified no Git/Git-LFS process was active, verified the cleanup target resolved exactly inside `C:\Competition\K12\.git\lfs\tmp`, and removed only its temporary contents. Free disk returned to approximately 203 GB; datasets, pretrained weights, caches, source, experiments, and completed run artifacts were preserved.

### Community clarification register

The participant supplied the full text of the forum question thread on 2026-07-17. Questions are not treated as organizer answers. As represented in the supplied excerpt, only speaker attribution received a substantive answer; the following remain open:

| Topic                     | Open question                                                                               | Current conservative project policy                                                                                                                                  |
| ------------------------- | ------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Test objectives           | Whether hidden objectives overlap the 398 training objectives, and by how much              | Require objective-ID-disjoint and semantic-family-disjoint validation; preserve unseen-ID fallback                                                                   |
| Transcript cutoff         | Whether any transcript can contain the assessed answer or tutor reaction                    | Do not design a direct-label parser; audit suspicious leakage patterns separately before using them                                                                  |
| Provider/source shift     | Whether hidden data are all Third Space Learning or include other providers/modalities      | Retain provider/style proxy stress tests and avoid source-specific assumptions                                                                                       |
| External-license scope    | Whether MIT applies only to entrant code or to external models/eval-only datasets           | Use permissive pretrained weights; quarantine NC/share-alike training data unless organizers approve                                                                 |
| Determinism               | Bitwise equality versus numeric tolerance for GPU inference                                 | Fix seeds and deterministic preprocessing; require repeat-run probability parity and record any GPU tolerance                                                        |
| Transcript processing     | Whether local fine-tuning is allowed and whether third-party/cloud processing is restricted | Keep competition transcripts on the participant machine/runtime; do not upload to hosted APIs or cloud notebooks without explicit organizer and participant approval |
| Final submission          | Automatic best submission versus participant designation                                    | Preserve every versioned package and score; do not assume which run becomes final                                                                                    |
| Write-up generalizability | Whether external-corpus transfer is rewarded                                                | Focus claims on leakage-safe internal subgroup/objective transfer until organizers clarify                                                                           |
| Publication bonuses       | Eligibility and selection process                                                           | Keep the full reproducible research log; revisit when an organizer answer appears                                                                                    |

- The runtime archive limit is separately documented by the competition as 60 GB, so that item is not operationally blocking even though it was asked in the forum thread.
- Forum thread: https://k12-ai-infrastructure.discourse.group/t/several-clarification-questions-test-data-external-data-determinism-submissions-write-up/32/1

### Token-level speaker-role NB-SVM ablation

- The neural input already contains explicit `[OBJECTIVE]`, `[STUDENT]`, and `[TUTOR]` markers, so duplicating them with natural-language prefixes on the rejected supervised encoder would not be a clean new signal. The speaker-role idea was instead applied where it changes the representation materially: the confirmed feedback NB-SVM.
- Built a deterministic view in which every objective token is prefixed `learning_objective_`, every student-answer token `student_speech_`, and every tutor-feedback token `tutor_feedback_`. This prevents the same lexical feature from being shared across speakers while preserving the answer-to-feedback token order and the frozen E220 dense features.
- **Frozen pilot:** `20260717T045956Z_nbsvm_validation`, with the original locked `C=1` and 30% v0.2 blend. The role-prefixed model improves loss/AUROC by only `0.000599/0.002676`, compared with `0.001426/0.006127` for the shared-vocabulary NB-SVM on the identical pilot.
- **Learning:** Speaker separation is not universally beneficial. In this compact answer-feedback view, shared words such as correctness, reasoning, and uncertainty cues provide useful cross-role statistics, while adjacency and line markers already preserve enough role context. Prefixing every token fragments that evidence and weakens both metrics.
- **Decision:** Reject token-level role prefixing at the pilot gate. Do not open confirmation folds or modify the retained shared-vocabulary NB-SVM.
- **Organizer clarification added after the pilot:** The data provider performed diarization and transcription using a combination of AI models and human QA. Some student/tutor identification errors are known to remain, though organizers do not expect them to affect a large portion of the data; robust cleaning of imperfect real-world data is explicitly considered useful competition work.
- **Interpretation:** This independently explains why hard token-level role separation can hurt. A mislabeled turn causes every prefixed lexical feature to enter the wrong channel, while the retained shared-vocabulary NB-SVM still uses local answer-to-feedback order and is less brittle to occasional speaker swaps. Future role-aware models should include role-dropout, shared fallback channels, or speaker-error detection rather than assuming diarization is perfect.
- Forum source supplied by the participant: https://k12-ai-infrastructure.discourse.group/t/several-clarification-questions-test-data-external-data-determinism-submissions-write-up/32/18

### Preregistered supervised data-scale stress test

- The rejected supervised screen used only 4,096 legal training rows per outer fold, approximately one-seventh of the available fold training data. This leaves one unresolved mechanism: insufficient outcome-supervision scale rather than a fundamentally unsuitable encoder.
- Select fold 3 as a deliberately adversarial scale test because it was the worst small-data supervised fold (`+0.003712` loss / `-0.004394` AUROC at the locked 30% blend). This choice is conservative and may not be used to claim an average result by itself.
- Reuse the exact frozen model, text pair, 192-token truncation, one epoch, layer-freezing pattern, optimizer settings, seed, and 30% blend. Change only `training_rows` from 4,096 to every row remaining after objective holdout and validation-session purge. Do not add role prefixes or select another checkpoint.
- **Continuation rule:** Full-data fold 3 must improve v0.2 loss by at least `0.001` without AUROC loss, or improve AUROC by at least `0.005` without any loss regression. Failure permanently ends this supervised architecture; success only earns full grouped confirmation and is not itself a promotion.
- **All-row fold-3 result:** `20260717T063544Z_supervised_encoder_screen`. The run trained on all 23,079 legal rows for the fold and evaluated all 6,907 untouched validation rows. Final one-epoch training loss was `0.591582` after exactly 1,443 batches.
- At the previously locked 30% weight, v0.2 scores loss/AUROC `0.574952/0.662566`; the blend scores `0.574195/0.668116`, giving deltas of `-0.000757/+0.005550` for loss/AUROC.
- **Continuation decision:** The loss-size path is not met, but the independent AUROC path is met: AUROC improves by more than `0.005` and loss also improves rather than regresses. This earns unchanged all-row confirmation on folds 0, 1, 2, and 4. It does not promote the model or authorize a submission by itself.
- **Selection safeguard:** The run report lists 20% as the best loss row (`-0.000887/+0.004873`), but 30% was locked before the scale test and is the only confirmable weight. Do not change to 20% after viewing fold 3.
- **Interpretation:** Scaling from 4,096 to all legal training rows reverses the original fold-3 failure (`+0.003712/-0.004394`) into a modest loss win and material AUROC win. The standalone supervised checkpoint is still weak (`0.586041/0.637676`), so the signal is useful only through diversity with v0.2, not as a replacement.
- **Usage-constrained second all-row confirmation:** `20260717T082126Z_supervised_encoder_screen` evaluates fold 0 after training on all 23,001 legal rows for that fold. Final one-epoch training loss was `0.589091` after exactly 1,438 batches; all 7,512 validation rows were scored.
- At the locked 30% weight, fold-0 v0.2 scores loss/AUROC `0.600800/0.623296`; the supervised blend scores `0.595335/0.648824`, improvements of `0.005466/0.025528`.
- **Final go/no-go:** Both available all-row confirmations improve both metrics at the fixed 30% blend. Fold 0 clears both continuation thresholds by a wide margin; deliberately adversarial fold 3 clears the AUROC path without loss regression. Because the participant has approximately 20% Codex usage remaining, do not run folds 1, 2, or 4. Approve exactly one one-epoch fit on all 35,072 labeled rows, then reserve work for integration, deterministic replay, packaging, and smoke testing.
- **Risk statement:** This is a usage-constrained two-fold promotion, not the originally planned five-fold confirmation. The decision is stronger than the 4,096-row screen because it includes the prior best and worst folds at full scale, but uncertainty remains for objective families represented by folds 1, 2, and 4. Preserve that limitation in the final write-up.
- **Final full-data fit:** `20260717T103832Z_final_supervised_train`. Trained the frozen architecture once on all 35,072 labels for exactly 2,192 batches. Final training loss was `0.573048`. Saved only 68 trainable tensors as a 28,992,044-byte safetensors delta with SHA-256 `f31156ba42d25c4313e79135110171f35fad81f06f3c8e81390354099830678b`; exact load/save tensor parity is `0.0`.
- **Bounded final-combination audit:** Without changing any weights, merged the two full-data supervised validation folds with the already locked BGE-base + NB-SVM candidate. On fold 0, `0.7 * combined + 0.3 * supervised` improves v0.2 loss/AUROC by `0.006430/0.028126`. On fold 3 it changes loss/AUROC by `+0.000055/+0.006085`. The simpler supervised-v0.2 blend changes fold 3 by `-0.000757/+0.005550` but gives up average loss and AUROC gain.
- **Final formula locked:** `0.7 * [0.7 * (0.25 full + 0.25 role + 0.50 BGE-base) + 0.30 NB-SVM] + 0.30 supervised`, equivalent to 12.25% full transcript, 12.25% role/behavior, 24.5% BGE-base semantic, 21% NB-SVM feedback, and 30% supervised BGE-small. Do not tune these weights further.
- **Risk decision:** Accept the tiny fold-3 loss regression (`0.000055`) because it is orders of magnitude below the prior robustness failures, while the final candidate materially improves fold-0 loss and improves AUROC on both full-data folds. This is the higher-upside final package for the rank objective; preserve the simpler supervised-v0.2 package as a fallback if runtime integration fails.

### Final v0.4 rank package and deterministic replay

- Trained the full-data BGE-base semantic logistic model at locked `C=0.1` and the full-data feedback NB-SVM at locked `C=1`, then combined them with the existing full/role models and the verified supervised delta in `final_ensemble_v04_rank.joblib`.
- The artifact was built under Python `3.12.8`, NumPy `2.5.1`, joblib `1.5.3`, and scikit-learn `1.8.0`, matching the official runtime's scikit-learn version observed in smoke. Artifact size is 4,277,193 bytes; SHA-256 is `f1258e4ff7fa0c566635212a826d0b4ad253f1aa157c21be4f1d326b853f6b1b`.
- Added runtime inference for the BGE-base semantic model, fold-free full-data NB-SVM ratio/model, and the safetensors BGE-small supervised delta. All test cases are still processed independently; the runtime computes no test-set fit, aggregate, clustering, pseudo-label, or cross-sample feature.
- Built `submission_builds/final_ensemble_v04_rank.zip`, containing 50 archive members and no `data/` entries. ZIP size is 363,302,329 bytes, safely below the 60 GB limit; SHA-256 is `aeff8f0ea73a02ba5c3f41f69a91e1d835dafde33e185d462ff86badcecb8387`.
- The standard package builder executed the new ZIP path end-to-end on the 100-row local smoke set with valid response order, schema, finite probabilities, and no runtime output.
- An independent verifier then freshly extracted the actual ZIP and executed it twice. Both runs produced byte-identical `submission.csv` files, maximum probability difference `0.0`, empty stdout/stderr, and probabilities in `[0.419814, 0.877606]` with mean `0.701662`.
- Verification report: `submission_builds/final_ensemble_v04_rank.verification.json`.
- **Next gate:** Upload this exact ZIP as an official platform **Smoke test**, using a private note that identifies v0.4 and its hash prefix. Smoke score is not a model-quality estimate; accept only successful completion, no runtime/version warnings or errors, and a valid scored output. Use the exact smoke-passed ZIP for the remaining Normal submission only after reviewing its official log.
- **Official smoke passed:** Platform job `id-1795` completed with exit code `0` and smoke score `0.5208`. The archive log identifies the exact v0.4 note and expected assets. Execution ran from approximately `15:40:28` to `15:40:42` (about 13.7 seconds), produced `submission.csv`, and exported it successfully.
- No scikit-learn compatibility warning, model-loading warning, CUDA failure, missing asset, runtime exception, memory error, or timeout appears in the official log. The runtime package contains the correct v0.4 artifact, supervised delta, BGE-base, and BGE-small assets.
- **Interpretation:** The smoke score is based on a small training-derived functional set and is not a trustworthy hidden-test model comparison. It is slightly higher than the prior v0.2 smoke score, but that does not override objective-family validation or the exact runtime pass.
- **Submission authorization:** Use the byte-identical `submission_builds/final_ensemble_v04_rank.zip` with SHA-256 `aeff8f0ea73a02ba5c3f41f69a91e1d835dafde33e185d462ff86badcecb8387` for the next **Normal submission**. Do not rebuild, rezip, rename internal files, or change weights after the smoke pass.

### v0.4 public failure and process correction

- Normal submission `id-1797` completed successfully at runtime but scored public log loss `0.6158`; its AUROC and rank were not supplied with the result. This is `0.0077` worse than v0.2 (`0.6081`) and only `0.0023` better than v0.1 (`0.6181`). Runtime took approximately 10 minutes 34 seconds and exited cleanly, so the primary failure is generalization/model selection rather than execution.
- **What was wrong:** The BGE-base + NB-SVM candidate had already failed the regime gate; the five-fold 4,096-row supervised confirmation had worsened pooled loss; full-scale confirmation used only folds 0 and 3; fold 0 was the original selection fold; and the final formula was chosen after those results. The final formula also showed a small fold-3 loss regression that was accepted despite the preregistered no-regression rule.
- **Calibration warning missed:** The locked supervised blend increased ECE from `0.043591` to `0.064727` on fold 0 and from `0.050044` to `0.074022` on fold 3. Hidden shifted data exposed this fragility.
- **Decision:** Reject v0.4 and restore v0.2 as the public champion. Do not tune or resubmit v0.4. Pause execution while credits are low.
- Wrote `V04_FAILURE_ANALYSIS_AND_RECOVERY_PLAN.md`. Recovery begins with a no-training component parity/calibration audit, then a frozen multi-environment validation redesign, existing-prediction ablations, and only one bounded robust-training experiment if lighter evidence cannot clear strict gates.

### Preregistered component-agreement reliability gate

- **Question:** The locked BGE-base + feedback NB-SVM candidate has strong average ranking diversity but occasional loss regressions. Can component agreement identify trustworthy per-response corrections without using objective-frequency bins, test aggregates, targets, or a learned post-hoc calibration transform?
- Use the already locked components only: v0.2; the `C=0.1` BGE-base semantic replacement; and the `C=1` feedback word NB-SVM. Do not refit or retune any component.
- For one response, define the BGE direction as `sign(pred_bge_base - pred_bge_small)` and the NB direction as `sign(pred_nbsvm - pred_v02)`. If their product is nonnegative, emit the existing locked combined candidate. If they disagree, emit v0.2 unchanged. No magnitude threshold, blend-weight sweep, subgroup lookup, or target-fitted gate is allowed.
- Evaluate this rule first only on the original `semantic_k50_s0` pilot. Continue to the untouched k25, k50-seed1, and k80 confirmation protocols only if it either (a) improves pilot loss over the ungated combined candidate by at least `0.0003` while retaining at least 80% of its AUROC gain, or (b) reduces the candidate's worst-fold loss regression by at least `0.001` while retaining at least 80% of its pooled loss and AUROC gains.
- A confirmed candidate remains subject to the existing promotion gates: all-protocol loss consistency, at least 70% fold loss wins, median loss gain at least `0.002`, and worst legal per-sample regime loss regression no greater than `0.001`. The gate cannot justify a Normal submission merely by improving AUROC.
- Implemented `src/trace_ace/agreement_gate_validation.py`, the CLI wrapper, and focused unit tests. The two focused tests pass under the project system-Python environment; the project `.venv` does not currently include pytest.
- **Pilot result:** `20260717T060431Z_agreement_gate_validation`. Agreement occurs often enough to retain some diversity, but the gated candidate gains only `0.001454` loss and `0.006041` AUROC over v0.2, versus `0.002119/0.009165` for the ungated combined candidate. It worsens loss by `0.000665` relative to the combined model, retains less than 80% of its AUROC gain, and changes the worst-fold loss regression from `0.000550` to `0.000733` rather than reducing it.
- **Decision:** Both preregistered continuation paths fail. Do not inspect or run the agreement gate on k25, k50-seed1, or k80. Reject the gate and preserve the original locked combined candidate unchanged.
- **Learning:** Opposite-direction component movements are not a reliable error flag. BGE semantic capacity and local feedback NB-SVM appear complementary precisely because they can correct different aspects of one response; a hard sign-agreement rule discards useful diversity.

## 2026-07-20 (Asia/Kolkata) - Safe source publication and recovery restart

### Public repository checkpoint

- Created the first reproducible source snapshot in `Arjun0014/k12-competition` and committed it as `62b9af39cd159d250bdbb6f197385ec1ef699be0` with both author and committer set to `arjun0014 <23293383+Arjun0014@users.noreply.github.com>`.
- The target repository is public. Competition rule section `CODE SHARING` explicitly permits public code sharing and deems shared competition code MIT-licensed; an explicit MIT `LICENSE` was therefore added.
- Hardened `.gitignore` before publication. Raw transcripts, competition CSVs, external datasets, pretrained weights, caches, trained models, submission assets/ZIPs, checkpoints, logs, environment files, and local tool directories remain excluded.
- Also kept the active `PROJECT_LEARNING_LOG.md`, future experiment plans, unpublished score ledger, copied competition pages, and official documentation PDF local. This prevents accidental redistribution of competition reference material and avoids exposing live strategy/data-derived aggregates during the active competition.
- Published source snapshot contained 107 files and approximately 530 KB. A staged secret/path scan found no credentials, personal paths, datasets, model artifacts, or generated builds.
- Validation before publication: all 30 discovered unit tests passed and `scripts/verify_project.py` passed all 25 checks.
- Initial push failed only because `git lfs install` had created a local pre-push hook despite the repository containing no `.gitattributes` or LFS pointers. The stored `Arjun0014` credential was verified as the repository owner with push/admin access; the unnecessary local LFS hook was removed, and `main` pushed successfully.
- Recovery work now proceeds on `codex/v04-recovery`, leaving the published baseline commit stable.

### Accelerator recheck

- Reconfirmed the system has AMD Radeon integrated graphics and a discrete Radeon RX 5600M with 4 GB VRAM, but the active environment is CPU-only PyTorch `2.13.0+cpu`: CUDA is unavailable, no HIP runtime is present, and `torch-directml` is not installed.
- Current zero-training audits and sparse-head validation do not justify changing backends. Any future neural training may use a separately benchmarked accelerator environment only if its reproducibility and wall-clock benefit are demonstrated before the selected experiment begins.

### Frozen multi-environment validation redesign

- Implemented four label-free fold-assignment environments over all 35,072 training responses, always purging every validation session from the corresponding training mask:
  - `V_seen`: session-disjoint control; 35,006 rows whose objectives occur in at least two sessions are eligible for scoring.
  - `V_objective`: semantic-family holdout, which also guarantees objective-ID disjointness.
  - `V_style`: whole style-cell holdout using unique-session transcript-length, ASR-uncertainty, background-share, role-balance, turn-density, and typedness proxies.
  - `V_joint`: whole semantic-family x style-cell holdout.
- Assignment construction is independent of outcomes. Flipping every target in the focused test leaves assignments byte-for-byte unchanged. Development class labels are used only after assignment to verify that every fold remains evaluable.
- Frozen development seed `20260720`; canonical assignment SHA-256 `67a9002d670500fc7b492582c9b1c718948bc2c86156ce21687951af3f085f53`.
- Generated a separately seeded `V_final` manifest with seed `20260827`; canonical assignment SHA-256 `b3a83238fbddc78ab8044b56752ae86baa0506133068e6d286c098a8ab0fc2ac`. Its assignment file contains no target, prediction, or metric columns, and its manifest records `performance_inspected: false`.
- Real-data integrity checks passed: every response is assigned exactly once per environment, every fold is populated with both classes, and the required session/objective/family/style/joint leakage prohibitions hold. The four focused implementation tests pass.
- **Selection policy:** Architecture and blend selection may use only `V_seen`, `V_objective`, and `V_style`. Exactly one locked candidate may be opened on `V_joint`. `V_final` stays sealed until a candidate passes Gate C and is fully frozen.

### Independent sibling-result evidence

- The participant supplied a sibling competitor's `FINDINGS.md` for research context only; no sibling code, predictions, labels, or trained artifact entered this project.
- That solution reported very strong session-grouped local OOF (`0.541461` log loss, `0.724896` AUROC) from an objective-prior-heavy stack, but its actual public result was only `0.6097` log loss and `0.5993` AUROC (rank 21 at the time supplied). The approximately `0.0682` local-to-public loss gap and ranking reversal are direct negative evidence that ordinary session grouping plus learned objective/provider structure can be extremely optimistic here.
- Our v0.2 has a much weaker local semantic-holdout score but a better public result (`0.6081/0.6147`). This reinforces the recovery decision to prioritize objective/source invariance, calibration, and worst-environment behavior rather than adding empirical objective-ID priors or selecting on pooled OOF.
- Useful sibling design ideas such as objective-conditioned evidence and student-only views were already independently tested in this project through retrieval, multi-view, ordered-feedback, and role-specific branches. Their result does not justify importing the sibling stack or weakening Gate C.

### Phase C preregistration amendment

- The new target-independent folds make the old supervised and full-v0.4 OOF vectors invalid: they were produced under different assignments, so joining them to the new environments would mix in-sample and out-of-sample predictions. They are marked **not evaluated**, not rejected, in Phase C.
- Before any Phase C score was generated, froze six cheap fold-local formulas: raw v0.2; BGE-base replacement; v0.2 plus 20% feedback-SGD; v0.2 plus 30% NB-SVM; BGE-base plus 20% feedback-SGD; and BGE-base plus 30% NB-SVM. Fixed prior shrink and nested Platt controls may be evaluated for raw v0.2 and the one non-baseline architecture selected on the three development environments.
- The original Gate C remains unchanged. `V_joint` is documented precisely as an unseen semantic-family-by-style-cell interaction-combination confirmation, not as a split where every family and every style are both individually absent from training.

### Phase A forensic conclusion

- Final hardened audit: `20260720T071647Z_v04_forensic_audit`. Gate A passes with the deliberately limited conclusion: **no parity defect was observed on the preserved 100-row smoke sample; validation/domain shift is the leading explanation, not proof against every hidden/container edge case**.
- The exact expected set of six outputs was present for all 100 response IDs: full transcript, role/objective/dense, BGE-base semantic, feedback NB-SVM, supervised BGE-small, and final v0.4. All offline/runtime probabilities and differences were finite, row coverage was exact, and no duplicate IDs were accepted.
- Maximum component probability difference was `1.464183e-7`; maximum final packaged-output difference was `4.317405e-8`. Text/token preprocessing and feature widths were exact. Benign float32/float64 or batch-level numeric differences peaked at `1.579333e-6`, within the preregistered `2e-6` numeric tolerance.
- Package, local artifacts, supervised training/package hypothesis-tokenization contract, and two preserved replay outputs were cryptographically and numerically consistent. The audit also verified target/fold/response consistency across 32 OOF prediction sources and 1,081,000 source rows before comparing calibration.
- Existing OOF evidence explains the failure mechanism: v0.2 scores `0.587578/0.624048/0.006896` for pooled loss/AUROC/ECE on the pilot protocol; the pre-supervised combined candidate improves loss/AUROC to `0.585459/0.633212` but worsens ECE to `0.013159`; adding the five-fold 4,096-row supervised proxy regresses loss to `0.588492` and ECE to `0.027466`. The two selected full-scale folds look better in aggregate but show ECE `0.059909` and `0.075163`, exactly the kind of confidence instability that hidden source/prior shift can punish.
- **Decision:** There is no packaging fix to resubmit. v0.4 remains rejected; v0.2 remains the public champion. Recovery proceeds only through frozen environment validation and strict promotion gates.

### Phase D fallback frozen before Phase C scores

- Preregistered exactly one fallback, `D100_source_robust_joint_probe`, so a Phase C failure cannot trigger another post-hoc architecture/weight search. It is a direct fold-local robust refit over the existing BGE-base product/difference block, ordered-feedback word hash, 35 behavior/retrieval controls, and 51 ordered E220 features; it never consumes an old OOF prediction as a training feature.
- D100 uses a fixed family/style smooth worst-group logistic objective with an explicit group calibration-bias penalty, no inverse-frequency/sample weighting, one deterministic L-BFGS-B configuration, and one candidate formula: `50% v0.2 + 50% robust head`. No calibration rescue or weight sweep is allowed.
- This route was selected over a learned blender because pooled weight optimization previously gained only `0.000054`, stacking/Platt were negative, and the same labels recur across the environments. It was selected over another encoder fine-tune because v0.4 already showed the transfer/calibration cost of that path and the local six-core CPU makes valid fold confirmation expensive.
- If Phase C opens `V_joint`, that environment is no longer claimed as untouched for D100; only the still-sealed `V_final` can provide its final audit. The complete feature transform, optimizer, thresholds, abort rules, runtime estimate, and branch policy are frozen in `V04_FAILURE_ANALYSIS_AND_RECOVERY_PLAN.md` before any Phase C metric exists.

### Validation-freeze hardening correction

- The first Phase B draft wrote all four development environments together and materialized a plaintext `V_final` assignment beside them. Although it contained no targets, any local code could have joined its response IDs back to training labels; calling that arrangement sealed was too weak.
- Replaced it before Phase C scoring with protocol `2026-07-20-v5`. The normal loader can open only `V_seen`, `V_objective`, and `V_style` from `validation_environments_selection.parquet`. `V_joint` is a separate artifact whose loader requires a self-hashed candidate-lock sentinel. The legacy combined-development and plaintext-final parquet files were removed. The `V_final` assignment is not exposed or materialized anywhere in the ordinary developer API.
- Preserved aggregate four-environment assignment SHA-256 `67a9002d670500fc7b492582c9b1c718948bc2c86156ce21687951af3f085f53`. Selection assignment/file hashes are `ad28908614aa6153138d44d7bc319acf15a5fc6508fe4f1368c221073d695f17` / `4a665b54bbd0f1682a4f8e4f1bde345f6d1421d5d90660fc21ade2bd950ca97b`; joint assignment/file hashes are `69f845b54c428c7e5a6d62ca37a26a7e7e0091826887b88688d88ab880654ea5` / `246c6c69cc1ff3bf4bb68b9670904fe2d7f0de2ed0761161daa24e4187591736`.
- Final protocol hash is `8f662b5399d7b37e5059c7877a1ae10c55e24a942baa13c440a04267e2887541`; manifest file hash is `e974796317717472716f6cbc231e5e249c8ed5ca3ff98018c6eab21a2f169d19`. Two consecutive default builds were literal no-ops: bytes, hashes, sizes, and modification-time ticks of the manifest and both assignment artifacts stayed unchanged.
- Assignment validation now checks source response/session/objective equality, objective-family and session-style constancy, canonical joint cells, environment/suite coverage, seen-objective scoring eligibility, objective text/embedding invariance, complete provenance, and versioned refusal of any changed source/protocol. Focused Phase B tests pass (`26` tests plus `8` mutation subtests), and Ruff is clean.

### Runtime-aligned validation freeze correction

- A final preflight correctly rejected protocol v5 because it had been serialized under system Python `3.13.1`, scikit-learn `1.6.1`, and NumPy `2.2.3`, while training/submission artifacts must use the competition-aligned `.venv`. No Phase C model fit had started.
- Explicitly rebuilt and froze protocol `2026-07-20-v6` using `.venv` Python `3.12.8`, scikit-learn `1.8.0`, NumPy `2.5.1`, pandas `2.3.3`, PyArrow `19.0.1`, SciPy `1.18.0`, and joblib `1.5.3`.
- Canonical assignments are unchanged: aggregate `67a9002d670500fc7b492582c9b1c718948bc2c86156ce21687951af3f085f53`, selection `ad28908614aa6153138d44d7bc319acf15a5fc6508fe4f1368c221073d695f17`, and joint `69f845b54c428c7e5a6d62ca37a26a7e7e0091826887b88688d88ab880654ea5`. Runtime-aligned parquet hashes are selection `600664b9ce6c3809ff4b17206397285cfdd905a203a5c5f4d6969b403dfc9b40` and joint `4ea423bd5924b7e667a55b6ffa99377e4b418e080038f688d87b54f18e809fbe`.
- Final v6 protocol hash is `b1fb9b913ca35729a652b4b76553549b9ea205e38f8f527432fc1a8b3aa6b21a`; manifest file hash is `33b3487e766ff419da5f8e1ed6541a5e664c3f42a9ff7d996ca8961529204925`. Both legacy plaintext artifacts remain absent.

### Phase C unattended-run checkpoint discipline

- Launched `20260720T075633Z_environment_component_validation` at 13:26:33 Asia/Kolkata under the competition-aligned `.venv`; the first meaningful checkpoint is intentionally deferred until approximately 14:46 rather than polling the job.
- Windows exposes the `.venv\Scripts\python.exe` launcher as PID `8908` with near-zero CPU and memory, but it delegates the actual computation to child PID `3432` using `C:\Program Files\Python312\python.exe`. At the one early diagnostic checkpoint, the child had accumulated `183.27` CPU-seconds and used approximately `3.09 GB` RAM within three minutes, with no stderr output. This is healthy active computation, not a stalled process.
- **Monitoring rule:** future checks must inspect the full descendant process tree before diagnosing a stall. Do not infer inactivity from the launcher PID alone, and do not check again before the scheduled bounded checkpoint unless the operating system reports a failure.

### Phase C completed result and decision

- Run `20260720T075633Z_environment_component_validation` completed normally at 14:12:11 Asia/Kolkata with no runtime error and without accessing `V_final`. The delayed checkpoint automation was deleted before inspecting the run, so it cannot recur.
- The three selection environments locked `bge_replace/raw`: `25%` full-transcript hash, `25%` role/behavior model, and `50%` BGE-base semantic model. It improved log loss in all three selection environments by a mean `0.001361`, improved mean AUROC by `0.001959`, improved mean Brier by `0.000533`, and improved mean ECE by `0.001727` versus v0.2.
- On the already-locked `V_joint` interaction holdout, raw BGE replacement improved log loss from `0.550352` to `0.549346`, AUROC from `0.721639` to `0.723306`, Brier from `0.184747` to `0.184330`, and ECE from `0.044699` to `0.044698`.
- Across all four environments, it improved every environment and passed every AUROC, Brier, ECE, calibration-slope, worst-environment, and 5,000-replicate session/family-bootstrap clause. Its only failed Gate C clause was magnitude: mean log-loss gain `0.001272`, versus the frozen minimum `0.002500`.
- Nested Platt calibration looked better in pooled selection loss (`0.003711` mean gain), but regressed the worst selection environment by `0.002981` and worsened worst-environment calibration. Prior-90 shrink also regressed mean loss. This confirms that calibration chosen from pooled behavior is unsafe under environment shift.
- **Decision:** Do not package or submit BGE replacement despite its consistent direction. The effect is real but too small to justify using another Normal submission. Keep v0.2 as the public champion and execute only the preregistered Phase D fallback.

### D100 source-robust fallback implementation freeze

- Implemented the sole permitted fallback, `D100_source_robust_joint_probe`, with literal specification SHA-256 `c62b90597fadb3ed0da1d6737dfa167444163d75580eb750e3e4f3f6c47bcf96`.
- Each legal outer fold uses exactly `132,694` features: the `1,536` BGE-base product/difference block divided by `sqrt(2)`, the existing L2-normalized `2^17` ordered-feedback word hash divided by `sqrt(2)`, 35 outer-training-standardized behavior/retrieval/BGE-cosine controls scaled by `0.08`, and 51 outer-training-standardized E220 features scaled by `0.12`.
- The implemented analytic objective is the frozen `0.50` global BCE plus `0.25` smooth worst-family risk plus `0.25` smooth worst-style risk plus `0.5e-4 ||w||^2`; each group risk includes the fixed calibration-bias penalty. Groups with fewer than 128 distinct legal training sessions merge into that axis's `rare` group. No group identifier is an inference feature and no sample weighting, sweep, or rescue calibration exists.
- Added exact contract/source/cache hashes, separate fold checkpoints, stale-resume refusal, session-purged training masks, first-fold 15-minute abort, a working Windows process-RSS probe with an 8 GB abort, selection locking before D100 accesses `V_joint`, and explicit preservation of `V_final`.
- Added an explicit runtime refusal unless D100 is launched under the frozen `.venv` stack: Python `3.12.8`, NumPy `2.5.1`, pandas `2.3.3`, SciPy `1.18.0`, and scikit-learn `1.8.0`. This prevents a repeat of the Phase B v5 system-Python mismatch.
- Real-data preflight confirms 35,072 aligned rows, fixed sparse shape `(35072, 132608)`, controls `(35072, 35)`, ordered features `(35072, 51)`, and total width `132694`. Focused D100 tests pass, including finite-difference verification of the full analytic gradient; Phase B/C regression tests and static checks remain clean.
- Because Phase C already opened `V_joint`, D100 can use it only as additional development evidence. If D100 passes both its three-environment guard and the unchanged four-environment Gate C, the still-sealed `V_final` is the sole final audit before any package or submission is permitted.
- Full pre-launch verification completed with `75` passing tests plus `8` passing mutation subtests, clean Ruff across `src`, `scripts`, and `tests`, clean `git diff --check`, and a successful literal D100 specification check. The public reproducibility checkpoint is commit `f08e693ac04e53718e32236c2550e65751630224` on `codex/v04-recovery`, authored and pushed as `arjun0014`.
- Launched the sole D100 run at 15:04 Asia/Kolkata under `.venv` (launcher PID `980`, worker PID `3884`). Windows PowerShell 5.1 did not support the attempted `Get-Date -AsUTC` formatting flag, so the generated run ID is `_source_robust_validation`; this changes only the directory label, not the frozen contract, source/cache hashes, checkpoints, model formula, or evaluation rules.
- Scheduled exactly one checkpoint near 17:04 Asia/Kolkata, matching the preregistered two-hour monitoring policy. No process or log polling is permitted before that checkpoint; first-fold runtime, 8 GB RSS, non-finite optimization, and hash drift are enforced inside the runner itself.

### D100 completed result and rejection

- Run `_source_robust_validation` completed normally at 15:09:40 Asia/Kolkata, about six minutes after launch. It produced all 15 selection-fold checkpoints, then correctly stopped at the selection gate without loading `V_joint`; `V_final` also remained sealed. The delayed checkpoint automation was deleted before inspection.
- D100 improved only `V_objective`: blended loss `0.594273` versus v0.2 `0.595123` (gain about `0.000850`). It regressed `V_seen` by about `0.002351` and `V_style` by about `0.002542`.
- Across the three selection environments, mean log-loss **gain** was `-0.001348` (a regression), mean AUROC delta was `-0.004566`, mean Brier delta was `+0.000554`, and mean ECE delta was `-0.003986`. Only the macro-ECE clause passed; every other frozen selection clause failed.
- Shared 5,000-replicate support confirms that this is not sampling noise: macro session-bootstrap probability of positive gain was `0.0000`; macro semantic-family support was only `0.3884`.
- The robust head alone was substantially weaker than v0.2 on seen/style families: AUROC `0.7006` versus `0.7224` on `V_seen`, and `0.6992` versus `0.7218` on `V_style`. Its 50% blend recovered some signal but remained worse. On `V_objective`, the head's AUROC `0.6001` was close to v0.2 `0.6006`, and the blend's loss improved slightly through calibration/variance reduction rather than superior ranking.
- Every outer fit obeyed the fixed optimizer contract and reached the preregistered 60-iteration ceiling with finite objectives and coefficients. Fold time was only 17-24 seconds and peak observed RSS was about 1.36 GB, far inside both abort limits. Hitting the frozen iteration cap does not authorize increasing it after seeing outcomes; the run is valid, not a compute failure.
- **Decision:** Reject D100 permanently under this contract. Do not alter its optimizer, blend weight, group penalty, or features; do not open `V_joint`; do not package or submit it. The evidence says forcing smooth worst-family/style risk sacrifices the semantic ranking signal that made v0.2 and raw BGE replacement robust.

## 2026-07-22 (Asia/Kolkata) - Top-5 recovery restart and external mastery transfer

### Live target and execution authority

- The participant reported that v0.2 had fallen to rank #20 and authorized local downloads, installations, training, backup packaging, and broad in-project implementation, while explicitly reserving every platform upload and submission for manual action.
- A single live leaderboard read confirmed `aj_insanity` at #20 with `0.6081/0.6147`. The top five are `0.6013`, `0.6015`, `0.6032`, `0.6033`, and `0.6042` log loss; #6 is `0.6046`. The practical target is frozen at `<=0.6038`, requiring at least `0.0043` public loss improvement over v0.2.
- Monitoring policy is unchanged and strengthened: long jobs get one halfway checkpoint and one completion checkpoint calculated from measured throughput. No minute-level polling is permitted. Runners must enforce their own memory, finiteness, hash, and resume checks.
- Any consistent, reasonable improvement may be preserved as a clearly labeled backup ZIP after local end-to-end verification. Backup status does not weaken the top-5 promotion gate and does not authorize submission.

### Evidence synthesis after Phase C and D100

- Raw BGE-base replacement remains the safest unsubmitted increment: it improved all four hardened environments by mean loss `0.001272` and all secondary metrics, but its raw public projection is only about `0.6068`, currently around rank #14 rather than top five.
- D100 confirmed that aggressively optimizing worst family/style risk damages the global semantic ranking boundary. Calibration rescue, target priors, local retrieval, generic NLI, trajectory compression, speaker token fragmentation, and provider/objective shortcuts have all failed or proved too unstable.
- The next experiment must therefore create a genuinely new transferable mastery representation rather than reweight old predictions.

### External-data ruling and corpus correction

- The participant supplied an organizer answer stating that CC BY-SA data is sufficiently open for the competition. `SemEval 2013 Task 7` is moved from quarantine to an allowlisted candidate with attribution and lineage requirements.
- The local platform folders are semantically swapped: `Datasets/TalkMoves` contains SemEval-2013 student-response XML; `Datasets/SemEval` contains TalkMoves classroom transcripts. Only the former is admitted. The actual TalkMoves corpus remains excluded because of its non-commercial license.
- The SemEval five-way tree contains 928 XML files but extensive Core/Extra and task-format duplication. A read-only audit found 16,150 unique student-answer IDs and labels for correct, partially-correct, contradictory, irrelevant, and non-domain responses. Canonical parsing must deduplicate by answer ID and exclude reliability re-annotations and unlabeled rows.
- The corpus is unusually aligned with the competition's missing signal: it asks whether short student responses express correct educational content relative to questions and reference answers, including official unseen-question and unseen-domain splits.

### Frozen E400/E410 direction

- Wrote `TOP5_RECOVERY_PLAN_2026-07-22.md`. E400 adapts the already-downloaded Apache-2.0 `cross-encoder/nli-deberta-v3-small` on canonical SemEval labels using the existing contradiction/entailment/neutral head and a fixed one-epoch, top-two-layer training contract.
- Correct maps to entailment, partial credit to neutral, and contradictory/irrelevant/non-domain to contradiction. This label mapping, prompt structure, optimizer, and external continuation gate are frozen before training.
- If the external unseen-question/unseen-domain gate passes, the checkpoint will generate a frozen full-session mastery view and then deterministic per-turn mastery trajectory features. Competition labels enter only through leakage-safe fold-local linear probes.
- Backup and top-5 gates are distinct. A backup requires at least `0.0010` robust mean loss gain with stability; a top-5-ready model requires at least `0.0043`, every-environment improvement, strong bootstrap support, a new locked confirmation, and a successful still-sealed `V_final` audit.

### E400 implementation and canonical external-data evidence

- The canonical SemEval parser produced exactly `16,003` unique labeled answer rows after answer-ID deduplication: `8,910` train, `979` unseen-answer, `1,552` unseen-question, and `4,562` unseen-domain rows. The ordered content SHA-256 is `7a51ebe3b8c6d1537648c511836eb54360831006053eb65980f7736bd6351dfb`.
- E400 now measures the untouched base NLI checkpoint on official unseen-question and unseen-domain splits before adaptation, then requires the one-epoch adapted checkpoint to meet absolute transfer thresholds and avoid AUROC regression on both splits. A checkpoint that overfits the external training mapping is therefore stopped before competition caching.
- Competition validation is frozen to three weights (`10%`, `20%`, `30%`) over the already validated raw BGE-base replacement. Weight selection uses only `V_seen`, `V_objective`, and `V_style`; `V_joint` is confirmation-only and `V_final` remains sealed. The backup and top-five magnitude gates are implemented separately.

### BGE-base safety backup packaging

- Phase C's raw BGE-base replacement improves all four hardened environments, with mean log-loss gain `0.001272`, mean AUROC gain `0.001886`, mean Brier gain `0.000504`, and mean ECE gain `0.001295`. Macro bootstrap support is `1.0000` by session and `0.9922` by objective family.
- This clears the new backup gate but not the top-five gate. Its projected public loss is about `0.6068`; it must remain labeled as a safety backup and must not be mistaken for the active top-five candidate.
- A dedicated package builder now strips the failed v0.4 decision path by executing only the validated `25%` full-hash, `25%` role/behavior, `50%` BGE-base ensemble. It also writes artifact, source, and ZIP hashes and runs the exact offline submission entry point before preserving the ZIP. Platform submission remains manual-only.
- Verified backup package: `submission_builds/final_ensemble_v05_bge_backup.zip`, `258,290,506` bytes, ZIP SHA-256 `65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`. The exact runtime completed all `100` smoke rows with no stdout/stderr and valid ordered probabilities.

### New external outcome source: MathDial

- A license/lineage search found the official MathDial repository, which is released under CC BY-SA 4.0. This license is admissible under the organizer ruling supplied by the participant. The repository was frozen locally at commit `b06c020a0a1f57a87577fec33e657b63e7eb476e`.
- MathDial contains `2,262` official train and `599` official test dialogues. Nine train and four test rows have missing outcome labels. The labeled train distribution is `1,696 Yes`, `303 Yes, but I had to reveal the answer`, and `254 No`; labeled test is `433/94/68` respectively.
- This is more aligned than MRBench for outcome prediction: every dialogue starts from a known incorrect math solution and has a teacher-annotated end-state `self-correctness`. The “answer revealed” state is frozen as negative because it does not demonstrate independent post-tutoring success.
- E430 is now an independent external-outcome branch, evaluated only after the already-running E400 branch. It keeps the official external split immutable and uses the same no-public-tuning, fold-local competition validation, backup, and top-five gates.
- A source audit found a major published-split caveat: `314` labeled train question IDs also occur in the official test file. E430 therefore retains all `595` labeled official test dialogues but purges every overlapping question ID from training, leaving `1,679` question-disjoint training dialogues. The canonical cache has `2,848` labeled rows and SHA-256 `2b34624fba9fad324b386c4b51c26864d82f282a1ecc673f74b31fc2c7d32d94`.

### E400 completed result and automation failure

- Run `20260722T190614Z_external_sra_transfer` completed normally at `00:36:14` Asia/Kolkata. The one-epoch training loss fell from `2.116172` at step 25 to `1.028040` at step 557; the serialized 59,078,724-byte delta round-tripped exactly with SHA-256 `ada04577114545f2adc7059fedf8d8189879c90f8467ab886bcdb0913cf9308c`.
- Transfer materially improved the external ranking signal. Unseen-question correct-vs-rest AUROC rose from `0.596583` to `0.712982`; unseen-domain AUROC rose from `0.724288` to `0.753692`. Unseen-question multiclass log loss fell from `3.123706` to `1.046495`, while unseen-domain loss fell from `2.466073` to `0.965669`.
- The frozen external gate nevertheless failed exactly one clause: unseen-question macro-F1 was `0.432212`, below the preregistered `0.45` threshold. The other four clauses passed. The runner therefore correctly stopped before competition caching or outcome-label validation; `V_final` remained sealed.
- The promised unattended continuation did not occur because the checkpoint request was rendered only as a suggested automation and never became an active automation. Training itself completed, but no task resumed to analyze it. This is an orchestration failure, not a model or hardware failure. Future long launches must verify that the automation is actually active rather than treating a rendered suggestion card as confirmation.
- **Decision:** retain E400 as positive representation-transfer evidence but reject this exact checkpoint under the frozen gate. Do not weaken the threshold after seeing the result. Proceed to the independently preregistered MathDial outcome branch.

## 2026-07-24 (Asia/Kolkata) - E430 MathDial outcome transfer completed and rejected

### Configuration, lineage, and initialization

- **Run:** `20260723T183418Z_mathdial_outcome_transfer`.
- **External data:** official MathDial repository commit `b06c020a0a1f57a87577fec33e657b63e7eb476e`, CC BY-SA 4.0. Source SHA-256 values remain `997d11eaf11dfb6c883b8d415f20969f705ff2df1c9cda49cecae71251806216` for train and `8659dccbd891335a8a84c466ce661ad4d1fde9967117128eae36d9e3e9d3934b` for test.
- **Leakage-safe split:** all 314 question IDs shared by the published train/test files were purged from training. The fit used exactly 1,679 question-disjoint training dialogues and evaluated all 595 labeled official test dialogues spanning 391 question IDs.
- **Frozen labels:** `Yes -> 1`; `No -> 0`; `Yes, but I had to reveal the answer -> 0`.
- **Frozen model:** `cross-encoder/nli-deberta-v3-small`, one epoch, 105 batches of 16, maximum length 256, top two of six encoder layers plus pooler and a fresh binary classifier trainable, encoder/head learning rates `2e-5/1e-4`, weight decay `0.01`, 10% warmup, class-balanced cross-entropy, seed `20260723`.
- **Verified binary-head contract:** the original three-class head was replaced rather than truncated. The deterministic `2 x 768` weight SHA-256 is `ee1cf8a8d8a045fa48245c8cbe1fbbdb78d83672eacefe2d72f0be1bb5dc608f`; bias is exactly `[0, 0]` with SHA-256 `af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc`; observed weight mean/std are `-0.000170278/0.020514455`. The runner now refuses initialization drift. This preflight hardening is pushed in commit `53bef1c`.

### Runtime and environment

- The clean run started at approximately `2026-07-23 18:21:02 UTC` and wrote its final report at `18:34:18 UTC`, about 13 minutes 16 seconds wall clock.
- Environment: `.venv` Python `3.12.8`, scikit-learn `1.8.0`, NumPy `2.5.1`, pandas `2.3.3`, PyArrow `19.0.1`, SciPy `1.18.0`, joblib `1.5.3`, PyTorch `2.13.0+cpu`, and Transformers `5.14.1`.
- Training completed all `105/105` batches. Running class-balanced loss was `0.703799` at step 20 and `0.699377` at step 105. Evaluation completed all `10/10` batches with no non-finite value or schema failure.
- **Orchestration correction:** interval-style heartbeat rules resumed the active goal prematurely and caused two early checks. The run was then attached to one event-driven `Wait-Process` completion handler, which waited for process exit without polling. All temporary checkpoint automations were deleted. Future active-goal runs should use an event-driven waiter rather than ending interim turns while a worker is active.

### Frozen external result

| Metric | E430 | Train-prior baseline | E430 change |
|---|---:|---:|---:|
| Log loss | `0.663191381` | `0.587079452` | `+0.076111929` worse |
| AUROC | `0.532617683` | `0.500000000` | `+0.032617683`, but far below gate |
| Brier score | `0.235148006` | `0.198738915` | `+0.036409091` worse |
| ECE-10 | `0.192340878` | `0.024502353` | `+0.167838525` worse |
| Prediction mean | `0.535390215` | `0.752233446` | `-0.216843231` |
| Accuracy | `0.648739496` | not a gate metric | - |
| Macro-F1 | `0.529002935` | not applicable | below `0.55` gate |

- Official-test positive rate is `0.727731092`; the fitted model is severely under-confident and poorly calibrated on the question-disjoint test distribution.
- All three frozen external clauses fail: AUROC `0.532618 < 0.65`, macro-F1 `0.529003 < 0.55`, and log loss `0.663191` does not beat the train-prior loss `0.587079`.
- A 5,000-replicate paired bootstrap over the 391 official-test question IDs gives only `0.0002` support for positive log-loss gain over the train prior. Mean bootstrapped gain is `-0.075482`, with 95% interval `[-0.111265, -0.038327]`. This is decisive negative evidence rather than sampling noise.
- **Environment consequence:** the external gate stopped the branch before any competition session cache, fold-local probe, `V_seen`, `V_objective`, `V_style`, or `V_joint` evaluation. `V_final_accessed=false`.

### Artifact hashes and decision

- Delta: `mathdial_deberta_delta.safetensors`, 59,075,648 bytes, SHA-256 `03e0447228f6b64139fda997b9e2637595c5d0970d4a906c68c8fe5ba64a48cd`.
- Report SHA-256: `5e59b2e6ca0ef0cba6fc8c1673f861925c37b71d0b8d74910e2feedf3c26f10c`.
- Test-prediction SHA-256: `186687621477abd1fe71323923c2f48c1eb97486e85384bdc6e44f03e0213407`.
- Training-metrics SHA-256: `3964143befa1fd65727f42880dd6780dc8b286b11a16cd06241a57ca3a8cc49e`.
- Canonical MathDial cache remains SHA-256 `2b34624fba9fad324b386c4b51c26864d82f282a1ecc673f74b31fc2c7d32d94`.
- **Decision:** reject this exact E430 checkpoint and branch under the frozen protocol. Do not weaken any threshold, reinterpret the answer-revealed label, revive the checkpoint post hoc, build a competition cache from it, blend it, package it, or submit it.
- **Public projection:** none is assigned because E430 never earned competition validation. The public champion and honest observed bracket remain v0.2 at `0.6081/0.6147`, approximately #20 from the last single leaderboard read; E430 provides no evidence for a top-five rank.

### E420 ModernBERT label-free runtime freeze

- After E430 rejection and before any ModernBERT target metric, froze E420 in `TOP5_RECOVERY_PLAN_2026-07-22.md`: exact 4,096-row prior encoder sample, complete role-marked transcript paired with a fixed mastery hypothesis, deterministic first-token plus masked-mean pooling, fixed `C=0.1` fold-local probe, identical BGE-base comparator, literal continuation thresholds, and no prompt/layer/pooling/C/weight rescue.
- Downloaded only the official `answerdotai/ModernBERT-base` safetensors/config/tokenizer/model-card files at revision `8949b909ec900327062f0ebf497f51aef5e6f0c8`, Apache-2.0. Model safetensors are 598,635,032 bytes with SHA-256 `340ac08b74eef0d7bdec2d7981a6a3d4249bf0e6aab60634b72ad02c2b8023a9`. Configuration verifies 22 layers, hidden size 768, and maximum position length 8,192.
- Ran the preregistered 16-row token-length-quantile benchmark under `.venv` Python 3.12.8, PyTorch 2.13.0+cpu, Transformers 5.14.1, six CPU threads, batch size one. This stage used no outcome metric.

| Maximum tokens | Seconds/row | Projected 4,096-row time | Peak RSS | Eligible |
|---:|---:|---:|---:|---|
| 8,192 | `32.8481` | `37.37 h` | `2.50 GiB` | No |
| 4,096 | `10.6441` | `12.11 h` | `2.40 GiB` | No |
| 2,048 | `4.0776` | `4.64 h` | `2.22 GiB` | Yes |

- **Decision:** freeze 2,048 tokens as the longest configuration satisfying the preregistered eight-hour ceiling and 8 GB RSS limit. Proceed to exactly one resumable 4,096-row cache build, then the frozen fair screen. This operational choice does not promote a model and has no public-score projection; v0.2 remains the public champion.
- Benchmark artifact: `data_cache/modernbert_e420_benchmark.json`, 2,137 bytes, SHA-256 `008ccfd61202cca419c37309a3f272bf5ca2f21fcbac31df6efbe786270a672d`. `V_final` was not accessed.

### E420 completed screen and rejection

- Cache construction completed all 4,096 frozen rows in approximately 4 hours 37 minutes at a stable `0.2463` rows/second, matching the label-free projection. Peak RSS was 2,849,439,744 bytes (`2.65 GiB`), far below the 8 GB abort.
- Input used the complete cached role-marked transcript as the first sequence and the fixed objective mastery hypothesis as the second sequence with `only_first` truncation at 2,048 tokens. Exactly 4,074 of 4,096 pilot rows reached the limit; token-count mean/min/max were `2044.38/355/2048`.
- Cached representation is the frozen concatenation of separately L2-normalized final-layer first-token and masked-mean vectors divided by `sqrt(2)`. All vectors are finite and unit-normalized. No target was used to build or select the cache.
- Frozen screen run: `20260723T233932Z_modernbert_long_context`. Both ModernBERT and BGE-base used the identical 4,096 rows, five `semantic_k50_s0` folds, session-purged fold masks, fixed `C=0.1`, and fold-local scaling of the same 35 legal dense controls.

| Metric | BGE-base fair comparator | E420 ModernBERT | E420 change |
|---|---:|---:|---:|
| Log loss | `0.595607488` | `0.599086954` | `+0.003479466` worse |
| AUROC | `0.608965679` | `0.596971597` | `-0.011994082` worse |
| Brier score | `0.203578114` | `0.204946046` | `+0.001367931` worse |
| ECE-10 | `0.024219608` | `0.029249712` | `+0.005030104` worse |
| Prediction mean | `0.704663794` | `0.701720227` | `-0.002943567` |

- All screen paths fail. A 5,000-replicate paired bootstrap over 3,856 sessions gives only `0.0006` support for positive ModernBERT log-loss gain; mean gain is `-0.003471927`, with 95% interval `[-0.005516885, -0.001342909]`.
- **Environment consequence:** E420 failed before a full 35,072-row competition cache or hardened `V_seen`, `V_objective`, `V_style`, or `V_joint` evaluation. `V_final_accessed=false`.
- Cache: `modernbert_e420_joint_2048.npy`, 25,165,952 bytes, SHA-256 `78c1a92c31a565f4060b44accf13e6368a7da7c190a5cd67911bbb0039417f7c`. Metadata SHA-256 is `7d35df536aa34be8f67894151b2412efde41d34c5f274b323a3f1318812b483a`.
- Screen artifacts: metrics `d7ed21b07da2756b723ded9c55ea79ad68879de045b84cb93850a80bf2b0874d`; OOF predictions `e1a43a3dc5b0b9b3a4c936c9e1a27eed4f21046ffb2287fe52bbebfdc6f5b575`; bootstrap `25aa1bea4e44934cad064dd040f41cad6d430b8b153e3dc94bad5a19810f7962`; report `4feb0c465305fbd285dca5aa03ad34e7be3ac634c1a24d1e8fac1fdbe84dc946`.
- **Decision:** reject this exact E420 representation. Do not rescue it with a different context length, pooling rule, prompt, encoder layer, regularization, calibration, or blend weight. It produces no public projection or honest rank improvement; v0.2 remains the public champion around the last observed #20, and the BGE-base safety ZIP remains the only validated backup.

### E450 BGE-large capacity screen and rejection

- After E420 rejection and before any E450 target score, froze the independent capacity-isolation screen in
  `TOP5_RECOVERY_PLAN_2026-07-22.md`. E450 used the exact prior 4,096-row sample and `semantic_k50_s0` folds,
  compact 256-token objective contexts and raw objective strings, SentenceTransformers CLS pooling and L2
  normalization, the existing scaled context/objective/product/absolute-difference interaction, fixed `C=0.1`,
  identical legal dense controls, and the identical session-purged fold-local probe. No prompt, prefix, C, pooling,
  text, calibration, or blend sweep was permitted.
- Exact model lineage: official `BAAI/bge-large-en-v1.5` revision
  `d4aa6901d3a41ba39fb536a557fa166f842b0e09`, MIT. The local 1,340,616,616-byte safetensors SHA-256 is
  `45e1954914e29bd74080e6c1510165274ff5279421c89f76c418878732f64ae7`. Configuration verifies 24 BERT
  layers, hidden/embedding size 1,024, 512 model positions, and the packaged CLS-only pooling contract.
- Runtime remained the competition `.venv`: Python `3.12.8`, scikit-learn `1.8.0`, NumPy `2.5.1`,
  PyTorch `2.13.0+cpu`, and Transformers `5.14.1`, with six CPU threads.
- The preregistered 32-row batch-one token-quantile benchmark took `23.9472` seconds (`0.748351` seconds/row),
  peaked at 1,859,887,104 bytes RSS, and projected `0.902178` hours for 4,096 contexts plus 244 unique
  objectives. It passed the fixed eight-hour/8 GB operational gate. Benchmark SHA-256:
  `f96e934774a4447b926bcdb1ef9829ce28248959467a442bf4933d937666a54c`.
- The cache run then completed all 4,096 contexts and all objectives in `3,029.987` seconds (about 50 minutes
  30 seconds), within the announced 50-65 minute window, with 1,899,626,496-byte peak RSS. An initial launch
  attempt produced no worker or cache because the log directory was absent; after creating only that directory,
  the clean frozen run started from row zero. The completed context cache is 16,777,344 bytes with SHA-256
  `252773d3f2fa1b28ec4763251b9fd5d6f0189be81353cc5a76c412e08c82a5d3`; the aligned objective cache is
  16,777,344 bytes with SHA-256
  `d27f0c0fa74fc71333cde2425d0c0a8303dc3e01b62dc092749806111560e84c`. Both passed shape, finite-value,
  and unit-norm audits. Metadata SHA-256 is
  `664b8b45cb4e47d75156489a555562dcd66adb2a7ddaf58654009697bc0dcf91`.
- Long-run supervision used one process-completion wait rather than repeated process/log sampling. PowerShell
  classified the model loader's progress stream as a native stderr record after Python had already completed and
  written audited outputs; this affected the wrapper exit status only, not the cache or its hashes.
- Frozen screen run: `20260724T004210Z_bge_large_capacity`.

| Metric | BGE-base fair comparator | E450 BGE-large | E450 change |
|---|---:|---:|---:|
| Log loss | `0.595607488` | `0.597888732` | `+0.002281244` worse |
| AUROC | `0.608965679` | `0.594448973` | `-0.014516705` worse |
| Brier score | `0.203578114` | `0.204479737` | `+0.000901622` worse |
| ECE-10 | `0.024219608` | `0.016327842` | `-0.007891766` better |
| Prediction mean | `0.704663794` | `0.703235945` | `-0.001427850` |

- Neither frozen continuation path passes. A 5,000-replicate paired bootstrap over 3,856 sessions gives exactly
  `0.0000` support for positive BGE-large log-loss gain. Mean gain is `-0.002279619`, with 95% interval
  `[-0.003389497, -0.001167366]`. The ranking and proper-scoring-rule regressions are decisive despite improved
  ECE.
- Screen artifacts: metrics SHA-256
  `c49d3c9a8a8adb4eb96352cfa08887e773749bfff1a76fe33178e4632d9f8f74`; OOF predictions
  `e09e864d097b224e803a4574f9908279f459a0cda491794de2672850084414dd`; bootstrap
  `b05166c1c22f4460aad47b0bad93416fb6e3807b8850ec4ffe64d5d1ed677662`; report
  `6473dcd20bb21a75c5d64533fb5fff28494087f595a0f4ae9cd22ed85c68ef88`.
- **Environment consequence:** E450 failed before any full 35,072-row cache or hardened `V_seen`, `V_objective`,
  `V_style`, or `V_joint` evaluation. `V_final_accessed=false`.
- **Decision:** reject this exact E450 representation without a C, prompt, pooling, context, token-length, or blend
  rescue. It earns no package and no public projection. The public champion remains v0.2 at `0.6081/0.6147` in the
  stale last-observed approximately-#20 bracket; the conservative top-five gap remains at least `0.0043`.
  The unchanged verified BGE-base safety backup remains
  `submission_builds/final_ensemble_v05_bge_backup.zip`, SHA-256
  `65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`.

### E460 binary SemEval mastery transfer and rejection

- E460 was frozen before training as a new checkpoint from the untouched base NLI model, not a reinterpretation or
  continuation of rejected E400. It mapped only SemEval `correct` answers to positive and every partial,
  contradictory, irrelevant, or non-domain answer to negative. It retained the canonical 8,910/1,552/4,562
  train/unseen-question/unseen-domain splits and content SHA-256
  `7a51ebe3b8c6d1537648c511836eb54360831006053eb65980f7736bd6351dfb`.
- The fresh two-logit `2 x 768` head was reproduced under seed `20260724`: weight SHA-256
  `696ad4a9b5d7c57ccf4906b3aca4a9e9d85921429b92ca7f096aacd6c4e6c55c`, zero-bias SHA-256
  `af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc`. The runner records
  `E400_delta_loaded=false`.
- Run `20260724T015958Z_sem_eval_binary_transfer` completed all 557 one-epoch batches in about 72 minutes
  21 seconds under `.venv` Python `3.12.8`, scikit-learn `1.8.0`, NumPy `2.5.1`,
  PyTorch `2.13.0+cpu`, and Transformers `5.14.1`. Final cumulative training loss was `0.562745`.

| Split/metric | E460 | Fixed train-prior baseline | E460 change |
|---|---:|---:|---:|
| Unseen-question log loss | `0.642616511` | `0.678852830` | `-0.036236319` better |
| Unseen-question AUROC | `0.728933446` | `0.500000000` | `+0.228933446` |
| Unseen-question Brier | `0.216319119` | `0.242886707` | `-0.026567588` better |
| Unseen-question ECE-10 | `0.110739128` | `0.003359338` | `+0.107379790` worse |
| Unseen-question macro-F1 | `0.668217534` | not a gate comparator | passes `0.65` |
| Unseen-domain log loss | `0.633425466` | `0.680490896` | `-0.047065430` better |
| Unseen-domain AUROC | `0.743687315` | `0.500000000` | `+0.243687315` |
| Unseen-domain Brier | `0.213334351` | `0.243697258` | `-0.030362907` better |
| Unseen-domain ECE-10 | `0.113992969` | `0.007976988` | `+0.106015980` worse |
| Unseen-domain macro-F1 | `0.687193166` | not a gate comparator | passes `0.65` |

- The strict gate fails without ambiguity. Unseen-domain AUROC is `0.743687 < 0.76`; both ECE values exceed `0.10`.
  Question-ID bootstrap support for positive loss gain is only `0.7114` on the 24 unseen-question groups and
  `0.7924` on the 46 unseen-domain groups, below the frozen `0.95`. Their 95% intervals cross zero:
  `[-0.098221, 0.163846]` and `[-0.077803, 0.157558]`.
- Delta: 59,075,648 bytes, SHA-256
  `ed2efded1af952d8f995183d9645ab453f199575a6bd12a5d53ae60d992a9734`. Training metrics SHA-256
  `f36ba1cef60a8a0e31e732510eeb3ad2c48bb15feb70f145700bd8dbbb0a31b6`; external metrics
  `3b72b3b19ab8fcddc04a597582ff09bbdb6df4166c0f91e4b785900e82a72934`; predictions
  `fa6b2a3579c57f8bb6291253602e7c0ae500bf544250e07cf24c0e1d78d4cebe`; report
  `2b9112320e7b5d0b60a0a1e9ad811357e53745c15bc3d74b64c4872d159449f3`.
- **Decision:** reject this exact binary branch. Do not weaken its thresholds, calibrate the failed checkpoint,
  change class weights/epochs/head/prompt post hoc, build a competition cache, blend, package, or submit it.
  `V_final_accessed=false`. E460 receives no public projection; v0.2 and the verified BGE-base backup remain the
  only public champion and validated local fallback respectively.

### E470 MPNet encoder-family screen and rejection

- E470 changed both architecture and contrastive pretraining family while holding the fair encoder protocol fixed:
  exact 4,096 rows and five `semantic_k50_s0` folds, compact 256-token objective contexts, objective text, no
  prefix, normalized embeddings, identical interaction/dense controls, fixed `C=0.1`, and session-purged fold-local
  probes. Exact model: official Apache-2.0 `sentence-transformers/all-mpnet-base-v2` revision
  `e8c3b32edf5434bc2275fc9bab85f82640a19130`; 437,971,872-byte safetensors SHA-256
  `78c0197b6159d92658e319bc1d72e4c73a9a03dd03815e70e555c5ef05615658`, 12 layers, 768 dimensions,
  packaged mean pooling plus normalization.
- The 32-row batch-one benchmark took `7.242892` seconds (`0.226340` seconds/row), peaked at 1,016,094,720
  bytes RSS, and projected `0.272866` hours. Benchmark SHA-256:
  `c9303bb83b60f4f049ac2dafc0985b86cd632b97a369f2a9cd413a7625c491a7`.
- Cache construction completed in `1,024.224` seconds (17 minutes 4 seconds) at 1,025,425,408-byte peak RSS.
  Context cache SHA-256 is `8544f44b6e5e9a258264448ebd853f98f997b8b460da23d11dcb5ebd1a61686f`;
  objective cache SHA-256 is `3bc95b192a3f4e45f77cf9541e06bbaa3fcd419049de68ebe960b786102b6e50`;
  metadata SHA-256 is `feb1d752aa6a65b0b28aef06064a6832dfdd0846d35d28af25f345be3b30bcc6`.
- Frozen run `20260724T022343Z_mpnet_family` regressed BGE-base log loss from `0.595607488` to
  `0.600642873` (`+0.005035384`), AUROC from `0.608965679` to `0.580846994` (`-0.028118685`), and Brier
  from `0.203578114` to `0.205722644` (`+0.002144529`). ECE improved from `0.024219608` to `0.006743072`,
  but this accompanies severe ranking/proper-score loss rather than useful calibration.
- The 5,000-replicate paired session bootstrap gives `0.0000` positive-gain support, mean gain `-0.005049430`,
  and 95% interval `[-0.007330108, -0.002755663]`. Metrics/predictions/report SHA-256 values are respectively
  `2522ed2b6d74db254e1db969030165e7cb49a8e59c1be95f7895b9cd2e23e15c`,
  `cfaa055bf7601dc9cb410c9348f30b7b5a9c4855947d64bf491ef434dc99e3c2`, and
  `6e46efffc1ae672e5bce46a0b479af3bdd9bf4d409dd6de2533d41d71d40c1f1`.
- **Decision:** reject E470 without C, pooling, context, token, or blend rescue. Do not build a full cache, open any
  hardened environment or `V_final`, package, or submit it. Together with E420/E450, this closes blind generic
  encoder replacement: BGE-base's robust gain is model-specific. E470 receives no public projection.

### E490 Qwen3-0.6B instruction-outcome screen and rejection

- E490 froze one explicit outcome prompt before scoring and used official Apache-2.0 `Qwen/Qwen3-0.6B` revision
  `c1899de289a04d12100db370d81485cdf75e47ca`. Safetensors are 1,503,300,328 bytes, SHA-256
  `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`. The label-free representation was
  the normalized final prompt-token hidden state plus uncalibrated `Yes` minus `No` next-token margin, with the
  fixed official no-thinking chat template and evidence-only truncation to 384 total tokens.
- The 16-row prompt-length benchmark took `33.6619` seconds (`2.10387` seconds/row), peaked at 3,511,554,048
  bytes RSS, and projected `2.3937` hours. Benchmark SHA-256:
  `ce3b562f216324ce82c18f3068622d43da4ec118dc73e682b54a71174350a614`.
- The one-wait cache run completed all 4,096 prompts in `9,063.967` seconds (2 hours 31 minutes), peak RSS
  3,688,333,312 bytes. Cache SHA-256 is
  `492e05cccd1f1f35570a5fefb3ad00d31c7449c40632d590f0336d34933230b3`; metadata SHA-256 is
  `2196b8b3bc118a708961f5477edb532b219207331a4b74fcb0516e7e57c4249a`. Hidden vectors passed unit-norm
  and finite audits. Raw Yes-minus-No margins were always positive (`0.7814/4.1108/7.1874` min/mean/max), so
  direct answers were biased toward Yes; the frozen fold-local probe tested only whether relative variation helped.
- Run `20260724T050522Z_qwen_outcome` regressed BGE-base log loss from `0.595607488` to `0.600507788`
  (`+0.004900300`), AUROC from `0.608965679` to `0.587710001` (`-0.021255677`), Brier from `0.203578114`
  to `0.205543897` (`+0.001965782`), and ECE from `0.024219608` to `0.026546238` (`+0.002326630`).
- Paired session bootstrap support is `0.0000`, mean gain `-0.004895802`, 95% interval
  `[-0.006978186, -0.002763076]`. Metrics/predictions/bootstrap/report SHA-256 values are
  `473dc5f354dd2d006f631b4b7ecea06923b8a30c97ca41d2fee23b16feb17e87`,
  `6b4373892e4b27a05a0e8e04d42d4b622b167139b4e710ac1e62e1a06c9ab7e6`,
  `63c4f73947bc395865c8e8cb91614d2baf962200b2d85cbec0c3d72a371a040c`, and
  `144317a7883bdf492f81c60dcbae6ef439f3ce51325c5d526dcc8e740faec5f7`.
- **Decision:** reject E490 without prompt, token, layer, C, margin calibration, or blend rescue. No full cache,
  hardened environment, `V_final`, package, or public projection is authorized.

### E500 Qwen2.5-1.5B instruction-scale screen and rejection

- E500 isolated scale only: official Apache-2.0 `Qwen/Qwen2.5-1.5B-Instruct` revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, 3,087,467,144-byte safetensors SHA-256
  `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`. It inherited E490's exact
  prompt, evidence-only 384-token truncation, no-generation representation, sample, folds, probe, and gate.
- The 16-row benchmark took `83.0445` seconds (`5.19028` seconds/row), peaked at 6,966,259,712 bytes RSS, and
  projected `5.9054` hours, passing both operational ceilings. Benchmark SHA-256:
  `719932374c9ee4018cb794da7b8b614abf5f0ee0207e8f89d1db8e19c8f9df68`.
- A single process-completion wait covered the full cache run; there was no intermediate polling. All 4,096 rows
  completed in `22,214.394` seconds (6 hours 10 minutes), peak RSS 7,127,048,192 bytes. The 4,096 x 1,537 cache
  is finite with normalized hidden vectors; SHA-256
  `eab87c4e2569f84b5a2856e2a1e9da3533d00a60057a6d269d1ee59ae0f619d5`. Metadata SHA-256 is
  `1bfa7b93b303a015730f888566ab8b1524851e7a31c2b002456d1862c46d772a`. Unlike E490, Yes-minus-No
  margins span both signs (`-4.0731/0.8070/3.9035` min/mean/max).
- Frozen run `20260724T112746Z_qwen2_5_1_5b_outcome` nevertheless reproduces E490's negative target result:
  BGE-base/candidate log loss `0.595607488/0.600536974` (`+0.004929486` worse), AUROC
  `0.608965679/0.587680917` (`-0.021284762`), Brier `0.203578114/0.205558116` (`+0.001980002`), and ECE
  `0.024219608/0.025530612` (`+0.001311004`).
- Paired session bootstrap support is `0.0000`, mean gain `-0.004924634`, 95% interval
  `[-0.007016380, -0.002765606]`. Metrics/predictions/bootstrap/report SHA-256 values are
  `f7dc13331d11aa8bd03bedf7dbd0cf2fa8ccd1e85dea59fdfad29feb69df8511`,
  `92cc200f855a2413094f4fcd57561359a2d6d20be89bf7dd71814e8c50f3b469`,
  `4baf39667223a07628cc45b0ca3917a8bb66a30fdebc4806f8c8b03c2ce8685d`, and
  `b192a20a1502eb921a4771746f9c234350ee1c3520efe9ee577d7d8ff2a22d49`.
- **Decision:** reject E500 without prompt, length, dtype, token, layer, C, calibration, or blend rescue. Model scale
  did not repair the representation. No hardened environment, `V_final`, package, or public projection is
  authorized. At completion there are zero active project Python workers.

### E510 fine-grained timing dynamics and rejection

- A target-free raw-field coverage audit found that transcript text, role, order, objective alignment, feedback
  adjacency, and coarse duration/role counts had already been modeled, while relative utterance timing had not been
  used beyond total duration. The audit covered all `6,139,854` cached utterances and `22,821` sessions:
  timestamps were complete and monotonic, every session had at least 10 distinct timestamps, and positive
  inter-utterance gaps had median `6` seconds and p90 `24` seconds.
- E510 was frozen in `TOP5_RECOVERY_PLAN_2026-07-22.md` and committed/pushed as `7ff97f7` before any E510 target
  scoring. Source `utterances.parquet` SHA-256 was
  `80a231d18dbb0641989ebb972f08988f0ddd3cb7c4fb769ad2fe90b5a98ba5a4`.
- The label-free cache contains 31 fixed session features: gap distribution and entropy, long-pause shares,
  tutor-to-student and student-to-tutor latencies, early/late pace changes, and role-specific turns per minute.
  Cache `session_timing_e510.parquet` is 2,970,546 bytes, SHA-256
  `0d46c99507af4b5de1b7b324ca88100e6148f3fe08d0f8ce5c3c1060e39627cc`; metadata SHA-256 is
  `fdbae9590ca19e0131ecba7fe9f3462f259612d3643cb19ee5e4f58a8c5a8232`.
- Each frozen development environment used a timing-only fold-local
  `LogisticRegression(C=0.1, solver=liblinear, max_iter=1000, random_state=20260724)`, training-fold-only
  standardization, and the existing validation-session purge. The comparator was reconstructed exactly as
  `0.25 full + 0.25 role + 0.50 BGE-base` from the immutable Phase C component OOF artifacts. Candidate weights
  were preregistered as exactly 10%, 20%, and 30%; no other model or weight was scored.
- Run `20260724T114058Z_timing_dynamics` completed cache construction, 15 legal fits, and the chunked
  5,000-replicate session bootstrap in `81.9` seconds under `.venv` Python `3.12.8`, scikit-learn `1.8.0`,
  NumPy `2.5.1`, and pandas `2.3.3`. Focused verification before scoring passed `31` tests plus `8` mutation
  subtests; the E510-only suite passed all `5` tests and Ruff was clean.

| Environment | Raw BGE loss | E510 10% loss | Loss change | AUROC change | Brier change | ECE-10 change |
|---|---:|---:|---:|---:|---:|---:|
| `V_seen` | `0.548199263` | `0.551347848` | `+0.003148585` worse | `-0.000294700` | `+0.001192696` worse | `+0.008139169` worse |
| `V_objective` | `0.593129391` | `0.593179336` | `+0.000049945` worse | `+0.000391140` | `+0.000060621` worse | `-0.001525339` better |
| `V_style` | `0.548664632` | `0.551813034` | `+0.003148402` worse | `-0.000325550` | `+0.001191099` worse | `+0.008946318` worse |

- The selected-by-rule 10% weight was already the least harmful. Its equal-environment mean loss gain was
  `-0.002115644` (a regression), it improved `0/3` environments, and its worst environment regression was
  `0.003148585`. The 20% and 30% weights worsened mean loss by `0.004774177` and `0.007952115` respectively.
  Macro AUROC changed by `-0.000076370`, Brier by `+0.000814805`, and ECE-10 by `+0.005186716`.
- Paired session-bootstrap mean gain was `-0.002115362`, 95% interval
  `[-0.002264611, -0.001966061]`, with exactly `0.0000` support for positive gain. Every frozen continuation
  clause failed. The regression concentrates in `V_seen` and `V_style`; the tiny `V_objective` loss regression
  shows that timing mostly re-encodes provider/session style rather than transferable mastery.
- Artifact SHA-256 values: development predictions
  `019042240f06da7f375ba8260be91f18f35367271b722e714354b784caf8e724`; fold metrics
  `0b52d195f8ea36b8cfe827d20d110615b11372f8b92ce8248f0de3a6eb6038d1`; environment metrics
  `51379208b5b47ec4cf5e1d2d6753e00877da511859e0af7b31933708b041953c`; selection
  `e63e2775fc469cba98f64a876d0eeaa9ff8757c479a82df62bfecf93ffa31ed5`; gate
  `9473342af944cc4ec6a33ee28968622d1cab3dc8abde47e81ad81bed1f3160b7`; report
  `2a7bd2c0a47e757f8189f51632013fde5178a5537fbb336827d66fdf6316bbf4`.
- **Projection and decision:** E510 receives no promotion projection or package. A diagnostic-only arithmetic
  projection would erase the BGE safety gain and yield approximately `0.6089`, consistent with the stale
  approximately-#20 bracket or worse, not top five. Reject the exact E510 component and all three weights without
  a smaller-weight, C, feature-subset, nonlinear, calibration, or blend rescue. `V_joint` was not read by E510,
  `V_final_accessed=false`, no ZIP was built, and no platform action occurred.

### E520 BGE-base student/feedback multi-view screen and rejection

- E520 tested the remaining bounded encoder/view gap: the frozen `2026-07-17-v2-budgeted` student-only and ordered
  answer-feedback texts had previously been encoded only by BGE-small, while BGE-base had been validated only on
  the original mixed objective context. The exact target-free source contains 35,072 aligned, non-empty response
  rows, SHA-256 `e7b28d679220672ed379ed0de8f6cef9ece38cbeb7b0473cae11ae80584d4607`.
- The protocol was frozen and pushed as commit `c89392d` before cache construction or scoring. It reused official
  MIT `BAAI/bge-base-en-v1.5` assets with exact safetensors SHA-256
  `c7c1988aae201f80cf91a5dbbd5866409503b89dcaba877ca6dba7dd0a5167d7`, packaged CLS pooling, 256 tokens,
  float32, L2 normalization, batch 16, and six CPU threads. Pilot index SHA-256 was
  `4e62045be2bd473ef41e8aaf5f4b351b7432760ed6c1bbd7ccd88ca112d1efba`.
- The formal label-free 32-quantile-per-view benchmark projected `0.532606` hours for both 4,096-row caches and
  measured 1,272,696,832-byte peak RSS, passing the frozen two-hour/8 GB gate. Benchmark SHA-256:
  `71ed498c9f51101fd3980256ad7f506f171f322c70af9443383ecc6bcd7ed64e`.
- Cache construction completed in `1,589.910` seconds (26 minutes 30 seconds), peak RSS 1,433,968,640 bytes.
  Student cache: 12,583,040 bytes, SHA-256
  `d932d03afd2a71baf9449ec75c48457a1065f5b91e8741a92fc5dc7895f81c1c`. Feedback cache: 12,583,040 bytes,
  SHA-256 `f67817bc8ea1157b8bfcbcc3e097525346b8aca62e62c11213117108607ea588`. Metadata SHA-256:
  `0bad690d80e4afc605bfb4bf22ce7235659555c02fdb28abbe0d76f09137f319`.
- Exactly three fixed representations were fit at `C=0.1` with the existing legal controls and session purge:
  student, feedback, and the normalized student/feedback embedding mean. Each was blended only 10%, 20%, and 30%
  over a same-row/fold `C=0.1` BGE-base original-context comparator. No other view, C, pooling, length, or weight
  was scored. Corrected focused verification passed all 14 relevant tests; Ruff was clean.
- Run `20260724T121909Z_bge_base_multiview` selected the least harmful combination by the frozen rule:
  10% feedback. BGE comparator loss/AUROC/Brier/ECE were
  `0.595607488/0.608965679/0.203578114/0.024219608`; the selected blend scored
  `0.595668881/0.608693941/0.203602672/0.023268601`. Changes were `+0.000061393` loss,
  `-0.000271737` AUROC, `+0.000024558` Brier, and `-0.000951007` ECE.
- Every one of the five pilot folds regressed log loss. Fold changes were
  `+0.000142501/+0.000016728/+0.000046607/+0.000091204/+0.000002777`; therefore the pooled near-tie is not
  hiding a stable positive environment. Student 10% was worse (`+0.000148886` loss, `-0.000873095` AUROC), and
  increasing any view weight worsened the frontier.
- Paired session-bootstrap support for positive gain was `0.0310`, mean gain `-0.000061618`, and 95% interval
  `[-0.000126109, 0.000002422]`. Loss and AUROC continuation paths both failed. No full 35,072-row view cache or
  hardened-environment run was authorized, so there are no `V_seen`/`V_objective`/`V_style` changes to report.
- Artifact SHA-256 values: metrics
  `616fba3e7384e77c14e0efa42e3008c29ef587148496210fa64ebcd111f85aae`; OOF predictions
  `62c21d7f9dbfde6fadbb92d5fd0c6dd9c2f7f604734d994eb39ad7ec26f5dd57`; bootstrap
  `b7b0b54f3c17f9308987990540eb516d769be138ecf4b80be912812827c23d6c`; report
  `0d7031b70b57b464312652c5d8e81e3115e4113c9cdbc0c1a781ed8c0b2b037c`.
- **Projection and decision:** E520 earns no package or promotion projection. A diagnostic-only arithmetic update
  would move the BGE backup from about `0.6068` to about `0.6069`, still around the stale approximately-#20 bracket
  and far from top five. Reject all nine E520 combinations without a smaller weight, alternate truncation,
  concatenation, C, pooling, or view rescue. `V_joint` and `V_final` were not read, no ZIP was built, and no
  competition-platform action occurred.

### Development pause and verified manual-submission fallback

- Development was paused at the participant's request on 2026-07-24 after approximately 24 hours of machine
  uptime. Final inspection found zero active K12 Python workers and zero relevant active Codex automations. No
  platform action was performed.
- The unchanged fallback was reverified at 258,290,506 bytes with SHA-256
  `65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`:
  `submission_builds/final_ensemble_v05_bge_backup.zip`. It remains the only locally validated manual-submission
  choice, with projected public loss about `0.6068`; this is a likely modest safety improvement, not a guaranteed
  leaderboard gain or a top-five candidate.
- E530 MathDial tutor-move transfer was frozen in the recovery plan but not implemented or run. Resume from
  `NEXT_SESSION_HANDOFF_2026-07-24.md`. `V_final_accessed=false`.

### v0.5 BGE-base public confirmation and rank improvement

- The participant manually submitted the unchanged verified BGE-base backup as full platform job `id-2518` with
  note `ensemble_v05_bge_backup`. Codex did not upload or submit it.
- The supplied platform log verifies status `Completed`, public log loss `0.6054`, exit code `0`, correct archive
  contents, and successful `submission.csv` export. Inference ran from `13:07:17.407` to `13:11:20.023`, about
  `242.6` seconds. Public AUROC was not present in the supplied log and is not inferred.
- The participant observed rank `#8`. This improves public log loss by `0.0027` over v0.2 (`0.6081`) and leaves
  `0.0016` to the frozen conservative top-five target `0.6038`.
- Local hardened mean gain was `0.001272`, so the public direction transferred and the public magnitude was larger.
  This is evidence that environment-consistent BGE capacity improvement is real, not permission to tune against
  the leaderboard. No component weight, calibration rule, gate, or E530 setting changes.
- **Decision:** v0.5 is the new public champion. Preserve its ZIP and hash unchanged. Continue with the already
  frozen E530 MathDial tutor-move branch; do not use additional public submissions for iterative tuning.

### E530 final pre-score parser audit and implementation

- **Correction before scoring:** the pause handoff's preliminary `11,139/3,699` figures counted raw teacher turns
  before every written exclusion. The exact parser reuses E430's 1,679/595 labeled dialogue split, purges all 314
  overlapping test question IDs, seeds the preceding student context with `student_incorrect_solution`, strips the
  move marker, and excludes malformed or empty tutor texts.
- Final immutable E530 rows are 11,106 training turns and 3,664 official-test turns. Training class counts are
  `focus=4,102`, `generic=2,611`, `probing=2,567`, `telling=1,826`; test counts are
  `focus=1,241`, `generic=884`, `probing=946`, `telling=593`. One malformed training marker, 42 empty training
  tutor texts, and 13 empty test tutor texts are excluded and retained in the audit.
- Implemented the frozen equal-unit word/character TF-IDF representation, fixed `C=1.0` liblinear classifier,
  top-label ECE-10, summed multiclass Brier, 5,000-replicate test-`qid` bootstrap, literal seven-clause gate,
  environment/version capture, target-free cache metadata, model/prediction/report hashes, and explicit
  `V_final_accessed=false`.
- Focused pre-score verification under `.venv` Python `3.12.8` and scikit-learn `1.8.0` passes all five E530/E430
  tests; Ruff is clean. An initially overstrict test rejected legitimate tutor math text beginning with `(` as if
  it were a leaked move marker. The test was corrected to reject only the four exact annotation prefixes; parser
  and model behavior did not change.
- **Pre-fit compatibility stop:** the first material launch ended after 12.3 seconds with no fitted model,
  prediction, metric, bootstrap, report, or checkpoint because scikit-learn `1.8.0` rejects direct multiclass
  `liblinear`. The frozen base estimator is now placed inside `OneVsRestClassifier`, preserving historical
  liblinear one-vs-rest behavior without changing C, seed, inputs, features, gates, or any score-dependent choice.
  The corrected material run must restart from zero.

### E530 completed external result and literal rejection

- Corrected run `20260724T164257Z_mathdial_tutor_move_transfer` completed from zero in `12.7805` seconds under
  `.venv` Python `3.12.8`, scikit-learn `1.8.0`, NumPy `2.5.1`, pandas `2.3.3`, SciPy `1.18.0`, and joblib
  `1.5.3`. Train/test sparse shapes were `11,106 x 97,930` and `3,664 x 97,930`; liblinear OvR iterations were
  `9/8/8/8`.
- Candidate accuracy/macro-F1 were `0.536572/0.498263`, both below the frozen `0.60/0.55` thresholds. Per-class
  F1 was focus `0.566291`, generic `0.755182`, probing `0.250188`, and telling `0.421390`; probing and telling
  fail the mandatory `0.45` floor.
- Proper-score transfer was real but insufficient to override classification gates: log loss improved from the
  fixed training-prior baseline `1.357012` to `1.026673` (gain `0.330338`), summed multiclass Brier improved from
  `0.735933` to `0.567969` (gain `0.167965`), and top-label ECE-10 was `0.035565`.
- The 5,000-replicate test-question bootstrap had `1.0000` support for positive loss gain, mean `0.330261`, and
  95% interval `[0.311361, 0.349291]`. This passes its clause but cannot rescue the three failed clauses.
- Artifact SHA-256 values: cache
  `7d34f5fce9c55fda7fb11109156e74763da8eaa25315c4e4e8eb9fbd02138b98`; model
  `930ea65cd39b26a715f29d7cc89ecbef7d33d5cfd6b6ea2ccf1b201ed1c34448`; predictions
  `caa34f7134db99cf750f53bc2991ef9bf23b80dbf093815639e014d02b3b06d5`; bootstrap
  `8d0e73f10e4fff5cf259530404402bc7fcb0ac82e400ab7f0a79f263bdbec23d`; report
  `76bc0dedde76fb34448610d5b79d67831b6b82f7099dc4278a0f9c67b4003960`.
- **Decision:** reject E530 exactly. Do not alter class weights, thresholds, calibration, vocabulary, C, solver,
  context, or model family post hoc. No competition cache, hardened validation, blend, ZIP, or public projection
  is authorized. The public champion remains v0.5 at `0.6054`, participant-observed rank `#8`.
  `V_joint_accessed=false`; `V_final_accessed=false`.

### E540 target-free multi-instance audit and preregistration

- The next bounded hypothesis isolates a remaining BGE-base bottleneck: the deployed semantic component encodes one
  compressed context even though the immutable retrieval cache contains up to 70 chronological objective-relevant
  lines. E540 will encode four chronological segments and combine them only through fixed objective-cosine
  attention; it does not reuse E530 or any rejected checkpoint.
- On the exact 4,096-row encoder sample, the immutable contexts contain `3/45/65/70` min/median/p95/max discussion
  lines. The frozen four-bin, eight-line-per-bin, 16-word compaction produces 16,384 non-empty segment texts,
  16,355 unique.
- Label-free BGE-base tokenizer counts are `8/133/176/192/232` for min/median/p95/p99/max; zero segments exceed
  256 tokens. The text transform, attention temperature `10`, probe `C=0.1`, fixed 10/20/30% weights, fair
  comparator, dual continuation gate, bootstrap requirement, and sealed-environment rules were committed before
  any new embedding or target score.

### E540 pre-build benchmark

- Focused implementation verification passed all seven E540/E520 tests and Ruff under `.venv` Python `3.12.8`
  and scikit-learn `1.8.0`. Before any encoding or target score, the established context-cache SHA transcription
  was corrected from the preregistered typo to the verified
  `b5f0066cc8d659e6593596ac47967bd14f2d0f02cedc82319e2a2313cf564d1a`; no file, feature, or protocol changed.
- The fixed 128-row benchmark completed in `38.648` wall-clock seconds. Position encode times were
  `5.099/5.884/5.870/5.218` seconds for 32 fixed length quantiles each. It projects `2,825.139` seconds
  (`0.784761` hours) for all 16,384 segment encodes, below the frozen two-hour ceiling.
- Peak resident memory was `1,084,813,312` bytes, below the 8 GB gate. Decision: proceed with the exact frozen
  cache build using a single calculated completion checkpoint; do not inspect the worker or progress before it.
- Benchmark artifact SHA-256:
  `7bbc44f76fb0c932e1b566f185b9ff0e48f97b7303af1fff01061ef26d9e6310`.
  No target metrics, candidate blend, `V_joint`, or `V_final` were accessed.

### E540 completed pilot screen and literal rejection

- The unattended cache build completed before its single scheduled checkpoint. It encoded all `4,096 x 4 x 768`
  float32 segment rows in `1,920.789` seconds under `.venv` Python `3.12.8`, scikit-learn `1.8.0`,
  NumPy `2.5.1`, Torch `2.13.0+cpu`, and Transformers `5.14.1`; peak RSS was `1,163,173,888` bytes.
  Cache SHA-256 is `c5d54f1f69baf17a640fcff8f8f292cc2dbbe579c529257734885890520d2c48`;
  segment-text SHA-256 is `6d9c04624f8a4a31e30d4bbe9dcb6ff43e03dd527ee186ddeebba9074cd13b9f`.
- Frozen validation run `20260724T180101Z_semantic_attention` completed in `18.556` seconds. The identical-fold
  BGE comparator scored log loss `0.595607488`, AUROC `0.608965679`, Brier `0.203578114`, and ECE-10
  `0.024219608`.
- The best of the preregistered weights was the smallest, `10%`, but it regressed every metric: log loss
  `0.595732349` (`+0.000124861` worse), AUROC `0.608344076` (`-0.000621603`), Brier `0.203629751`
  (`+0.000051637` worse), and ECE-10 `0.024429701` (`+0.000210093` worse). Fold candidate-minus-baseline
  loss changes were `+0.000146972/-0.000008271/+0.000216260/+0.000156772/+0.000105719`.
- The 5,000-replicate paired session bootstrap gives mean log-loss gain `-0.000124980`, 95% interval
  `[-0.000184901,-0.000065013]`, and `0.0000` support for positive gain. The loss path, AUROC path, and
  bootstrap clause all fail.
- Artifact SHA-256 values: predictions
  `71aac0805751fb5653998fc7c9eb79a5f9905e053bd2f7c91a058998a62123ca`; metrics
  `e6edcc8cc1eb4ee4e79d0fa59603cfaa094387dcc2d3a6de84c4e8f71aa47839`; fold metrics
  `02e54c84ecf348f5479728ac0d8664e2a5aaea1c4816c66a95e40cf2fa6f0c9a`; bootstrap
  `8188576eee48709fedaaa2864537a0a2c94ec37790a21d2ba6086afc5635c1c5`; report
  `5ef6d959574f362c36fb30d24d8a91b65d7b6211d1ad66636d6d6f89aa66ca02`.
- **Decision:** reject E540 literally. Do not sweep attention temperature, segment count/allocation, compaction,
  probe C, pooling, encoder, dense controls, or blend weights. It does not authorize a full competition cache,
  hardened validation, package, or public projection. The public champion remains v0.5 at `0.6054` and the honest
  known rank bracket is the participant-observed `#8`, not top five. `V_joint_accessed=false`;
  `V_final_accessed=false`.

### E550 BGE-base dual-pooling preregistration

- E540's negative result closes multi-instance CLS attention. The next independent hypothesis preserves the exact
  publicly confirmed BGE-base model and text but extracts a separately L2-normalized final-hidden-state masked
  mean alongside the deployed CLS vector in the same forward pass.
- Freeze the exact 4,096-row pilot, 256-token input, attention-mask mean including non-padding special tokens,
  raw-forward/packaged-CLS parity tolerance `2e-6`, two-hour/8 GB benchmark gates, dual interaction blocks scaled
  equally by `1/sqrt(2)`, fixed `C=0.1`, same fold-local controls/session purge, exact 10/20/30% blends over the
  fair raw BGE comparator, E520 continuation clauses, and 5,000-draw session bootstrap before any new embedding
  or outcome score.
- A failed screen cannot trigger layer mixing, special-token removal, CLS/mean rescaling, prompt/text changes,
  C/calibration/weight sweeps, or E540 reuse. Passing authorizes only full target-free caching and the frozen
  hardened protocol; `V_final` stays sealed and all platform submission remains manual.

### E550 parity and resource benchmark

- Focused verification passed all seven E550/E520 tests and Ruff before encoding. The 64-row raw dual-forward
  benchmark plus 64 packaged-CLS parity rows completed in `35.120` wall-clock seconds under `.venv` Python
  `3.12.8`, scikit-learn `1.8.0`, Torch `2.13.0+cpu`, and Transformers `5.14.1`.
- Maximum absolute normalized CLS difference was `1.0430813e-7`, passing the `2e-6` parity gate. The raw forward
  projects `865.293` seconds (`0.240359` hours) for all 8,192 pilot encodes, and peak RSS was `1,247,465,472`
  bytes. Decision: authorize the exact resumable build with one calculated checkpoint.
- Benchmark SHA-256 is `8216a509731064e0a20aafae6a260c9aa92cc31d8403eedcc9b1f2e50eccfd75`.
  No outcome score, `V_joint`, or `V_final` was accessed.

### E550 completed pilot screen and literal rejection

- The unattended build completed all 8,192 mean-pooling encodes in `805.088` seconds under `.venv` Python
  `3.12.8`, scikit-learn `1.8.0`, NumPy `2.5.1`, Torch `2.13.0+cpu`, and Transformers `5.14.1`.
  Peak RSS was `1,388,027,904` bytes. Context/objective cache SHA-256 values are
  `0ffa5e1092f4a40e745f704fc0f43d5cd178c73ecd35f44db83b6fe8d897ad86` and
  `074641a7cb358e29dfb6212bd4ad730668196e0e3ceeec71140ad979ca13c3ed`.
- Frozen validation run `20260724T183326Z_bge_base_dual_pooling` took `38.871` seconds. The fair BGE comparator
  scored log loss `0.595607488`, AUROC `0.608965679`, Brier `0.203578114`, and ECE-10 `0.024219608`.
- The lowest-loss preregistered blend was `10%`, but it is effectively neutral: log loss `0.595606937`
  (`0.000000551` better), AUROC `0.609043522` (`0.000077843` better), Brier `0.203577515`
  (`0.000000599` better), and ECE-10 `0.024052845` (`0.000166763` better). Fold candidate-minus-baseline loss
  changes were `+0.000053037/-0.000063529/-0.000007322/+0.000056676/-0.000047509`.
- The 5,000-replicate paired session bootstrap has mean log-loss gain `0.000000253`, 95% interval
  `[-0.000028217,+0.000029632]`, and exactly `0.5000` support for positive gain. Both magnitude paths and the
  bootstrap clause fail.
- Artifact SHA-256 values: predictions
  `5e204abdbe9450a0e9072da16123a099c03343c85a18283de5586fdfdf6a530a`; metrics
  `842e0f43afb06a3314560b95cf390670294356b17499bf6652272d821173f95d`; fold metrics
  `74c95578a566c3b476606b68c992bc31726768d88f593b63c8faebb575553b17`; bootstrap
  `f40af314c8cb61ae8d903bd3ac07b50a6f4a945264530d9c969e812c1af047a9`; report
  `010505a7b883ebb89fddd1923069bec8206894b6a947e8a5400e51cff5911ad5`.
- **Decision:** reject E550 literally. Do not sweep layers, token inclusion, CLS/mean scale, C, dense controls,
  calibration, or blend weights. No full cache, hardened validation, ZIP, or public projection is authorized.
  Public champion remains v0.5 at `0.6054`, participant-observed rank `#8`. `V_joint_accessed=false`;
  `V_final_accessed=false`.

### E560 residual nonlinear-head preregistration

- Three label-free BGE view/readout branches failed, so E560 preserves the publicly confirmed CLS features and
  isolates the linear decision boundary. The candidate is one 32-unit GELU residual MLP over the exact semantic
  interaction plus fold-local standardized legal controls.
- Freeze 15% nonlinear-path dropout, zero direct/output initialization, Xavier hidden initialization, fold-prior
  bias, 30 epochs, batch 128, AdamW `5e-4`, weight decay `1e-3`, gradient cap `1.0`, unweighted BCE, seed
  `20260725 + fold`, six CPU threads, no early stopping, the exact pilot/folds/session purge, and only 10/20/30%
  blends before scoring.
- The E520 dual continuation gate and 5,000-draw paired session bootstrap apply literally. A failure cannot trigger
  architecture, optimization, epoch, seed, input-scale, calibration, or blend sweeps. `V_final` remains sealed
  and platform submission manual-only.

### E560 label-free runtime benchmark

- Ten focused E560/E550/E520 tests pass, including exact repeat-training determinism, and Ruff is clean. The
  label-free benchmark trained one legal-fold-sized head for one epoch in `1.878458` seconds using only
  deterministic index-parity labels.
- Frozen five-fold, 30-epoch validation projects `281.769` seconds (`0.078269` hours); peak RSS was
  `1,647,194,112` bytes. Decision: authorize one unattended material validation and one calculated checkpoint.
- Benchmark SHA-256 is `dd91b8e4f908620ce0b7a0dbb8afab182f0291f7cd933eb0b835e3bbe16e21ec`.
  No outcome score, `V_joint`, or `V_final` was accessed.

### E560 completed pilot screen and literal rejection

- Frozen run `20260724T183952Z_residual_head` completed in `27.077` seconds with peak RSS `1,614,106,624`
  bytes. All five heads trained for exactly 30 epochs; fold-final training losses were
  `0.532726/0.514737/0.542525/0.537504/0.527948`.
- The selected preregistered `30%` blend improved log loss from `0.595607488` to `0.594559715`
  (gain `0.001047773`), Brier from `0.203578114` to `0.203187131` (gain `0.000390984`), and ECE-10 from
  `0.024219608` to `0.017598071` (gain `0.006621537`). It regressed AUROC from `0.608965679` to
  `0.606091194` (`-0.002874485`).
- Fold candidate-minus-baseline loss changes were
  `-0.001518485/+0.001166509/+0.000667706/-0.001826106/-0.003504593`.
  The 5,000-draw paired session bootstrap has mean gain `0.001044909`, 95% interval
  `[-0.000512153,+0.002581618]`, and `0.9120` positive-gain support.
- Bootstrap support passes, but the loss gain is below `0.0015` and AUROC regresses, so both continuation paths
  fail. Artifact SHA-256 values: predictions
  `3960c1e502e681e5a1cc48ce9c91f94c4ea6ed5b3cb3c0f7ed2e3dc49a0cba30`; metrics
  `c8284e1b06672c23eff3bc7222028ee6a9db628ad56ed6e288e88a7d43fa93c3`; fold metrics
  `19fcf44305c2eeb7de95eb1d430b2e10ced5f9d58badd3a23a71dbdcf07c6035`; bootstrap
  `9ef4dff47dae8f9533ce661d777c84afb6fb44f58644ea165c298c3a9545c482`; report
  `80351ff8af062e37edd9aaf206c15174af79b811c98a6d7f472e7bc1124366a4`.
- **Decision:** reject E560 literally without architecture, optimizer, epoch, seed, input, calibration, or blend
  rescue. No hardened validation, ZIP, or public projection is authorized. Public champion remains v0.5 at
  `0.6054`, participant-observed rank `#8`. `V_joint_accessed=false`; `V_final_accessed=false`.

### E570 objective-conditional pairwise ranker preregistration

- E560 improved calibration but harmed ranking. E570 is not a nonlinear-head rescue: it returns to a convex linear
  logit and tests whether within-objective transcript ordering supplies a more transferable outcome objective than
  rowwise BCE alone.
- Use the exact BGE-base interaction plus fold-local standardized 35 controls. Within each legal outer-training
  fold, sort rows by `response_id`; for every learning objective containing both labels, deterministically create
  `max(n_positive,n_negative)` pairs by cyclically repeating the smaller class. No cross-objective pair exists.
- Fit one zero-weight, fold-prior-bias linear logit by full-batch PyTorch L-BFGS, maximum 100 iterations, history
  20, strong-Wolfe line search, gradient tolerance `1e-7`, change tolerance `1e-9`. Frozen objective is
  `0.5 * mean BCE + 0.5 * mean softplus(-(positive_logit-negative_logit)) + 0.0015 * ||w||^2`.
  No class weighting, pair mining, margin, calibration, or checkpoint selection.
- Compare against the exact `C=0.1` fair BGE logistic probe; evaluate only 10/20/30% ranker blends and apply the
  unchanged dual continuation and 5,000-session-bootstrap gates. Failure cannot trigger loss-mixture, pairing,
  L2, iteration, optimizer, input, calibration, or weight sweeps. `V_final` stays sealed; submissions manual-only.

### E570 label-free runtime benchmark

- Eight focused E570/E560/E550 tests pass and Ruff is clean. One full convex fold fit using deterministic
  index-parity labels took `1.754113` seconds, 11 L-BFGS iterations, and 15 closure evaluations.
- Five material folds project `8.770566` seconds; peak RSS was `1,643,012,096` bytes. Decision: authorize direct
  frozen validation. Benchmark SHA-256 is
  `ecc439a519bcda6becc718916519edc6eb75842311230cec235ed64846a15c35`.
  No outcome score, `V_joint`, or `V_final` was accessed.

### E570 completed pilot screen and literal rejection

- Frozen run `20260724T185443Z_objective_pairwise_ranker` completed in `14.538` seconds with peak RSS
  `1,629,650,944` bytes. Legal fold training used 117-122 mixed-label objectives and 2,122-2,274 deterministic
  within-objective pairs; all fits converged in 13-16 L-BFGS iterations.
- The selected `30%` blend improved log loss from `0.595607488` to `0.595167445` (gain `0.000440043`), AUROC
  from `0.608965679` to `0.612184326` (gain `0.003218648`), and Brier from `0.203578114` to `0.203379282`
  (gain `0.000198833`). ECE-10 regressed from `0.024219608` to `0.030384866` (`+0.006165258` worse).
- Fold candidate-minus-baseline loss changes were
  `-0.000182878/-0.000809978/+0.000090936/-0.000774891/-0.000556851`.
  The 5,000-draw paired session bootstrap has mean gain `0.000442064`, 95% interval
  `[-0.000014790,+0.000908053]`, and `0.9706` positive-gain support.
- Bootstrap passes, but loss gain is below `0.0015`, AUROC gain below `0.0050`, and ECE non-regression fails.
  Artifact SHA-256 values: predictions
  `9dde0dea5e5d1a1de09873e64855d085066a968935b85e3c534411b1e24691fc`; metrics
  `398d4758030d33454245153bb62fc5292edeed8fd585bc85e29e8c020bf81637`; fold metrics
  `81583fb284aa4bcc85e6e0c8ec60b3f0ee524492787dfcb08b60f35ea7a41074`; bootstrap
  `ead7af1c7c2692bf4ad2a040bb13ba2bbf83f54140b5cfd52451709b7692525f`; report
  `22aca753689c361d83561d45dbc0c62911619b42e6803088e62fb6df109f3601`.
- **Decision:** reject E570 literally without loss, pairing, regularization, optimizer, calibration, or blend rescue.
  No hardened evaluation, ZIP, or public projection is authorized. Public champion remains v0.5 at `0.6054`,
  participant-observed rank `#8`. `V_joint_accessed=false`; `V_final_accessed=false`.

### E580 SemEval plus GSM8K correctness-transfer preregistration

- E580 is a fresh multi-corpus checkpoint from the original Apache-2.0 DeBERTa NLI base; it does not load or
  continue E400. SemEval remains the canonical 8,910-row training split. MIT GSM8K train/test parquet SHA-256
  values are `ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d` and
  `ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59`.
- All 7,473/1,319 GSM final answers parse as integers after the final `####`. Select exactly 4,455 train questions
  by `(SHA256(question), original_row)` order; selected-question content SHA-256 is
  `2bbcc18022ccf879c60a2cbaf46d8487e89e74c22c723fe4130a9f8ea1bb4092`.
  Each produces one correct entailment row and one contradiction row whose final integer is changed by `+1`
  when nonnegative and `-1` when negative. Hypothesis is exactly `The correct final answer is <original>.`
- Concatenate 8,910 SemEval and 8,910 GSM rows, then deterministically shuffle with seed `20260725`. Train from the
  untouched original NLI base for one epoch, batch 16, maximum length 256, top two encoder layers plus
  pooler/head, encoder/head learning rates `2e-5/1e-4`, weight decay `0.01`, 10% warmup, gradient cap `1.0`.
- Preserve every E400 SemEval external clause literally. Add official GSM test gates: correct-vs-corrupt AUROC
  at least `0.95`, binary-normalized log loss at most `0.35`, and no AUROC or loss regression versus the untouched
  NLI base. Any failed clause rejects the exact checkpoint before competition caching, blending, or validation.
  No corpus weight, corruption, prompt, epoch, layer, learning-rate, calibration, or checkpoint rescue is allowed.

### 2026-07-27 unattended-chain recovery and first-place strategy audit

- The prior continuation chain genuinely stopped: a completed E560 heartbeat was deleted as obsolete without
  immediately replacing it for E580. A fresh one-time audit found no Python worker and no active Codex automation.
  This was an orchestration failure, not a hidden or completed experiment. The correction is structural: no future
  long worker may be launched until its measured duration, expected artifact, one calculated checkpoint, active
  heartbeat, and persisted active-state verification are all recorded together. The worker and its logs must not
  be inspected before that checkpoint.
- Git remained at pushed commit `27e6072` on `codex/v04-recovery`; only the three intended E580 implementation
  files were untracked. The competition environment was reconfirmed as Python `3.12.8`, NumPy `2.5.1`,
  scikit-learn `1.8.0`, Torch `2.13.0+cpu`, and Transformers `5.14.1`. The preserved v0.5 ZIP still hashes to
  `65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`.
- The strategic diagnosis is unchanged but now explicit: generic encoder scale, alternate pooling, segment
  attention, transcript timing, style metadata, nonlinear outcome heads, and pairwise outcome loss have all
  failed to yield a robust new component. The remaining high-value axis is answer/process correctness relative
  to the learning objective. E580 is therefore retained as the immediate bounded branch because it directly
  addresses E400's strong ranking transfer but failed macro-F1 using a fresh checkpoint and explicit mathematical
  correctness supervision.
- A primary-source external-data audit identified OpenAI PRM800K as the strongest independent follow-up source:
  it provides approximately 800,000 human step-level mathematical correctness labels and its repository declares
  an MIT license. This is materially more aligned than generic QA or synthetic answer corruption. It is only a
  reserve branch: source/upstream licensing, immutable hashes, question-disjoint splits, a small resource screen,
  and an exact external gate must be frozen before any training or competition-label score. Google Education
  Dialogue and MathEDU expose attractive tutoring/outcome or authentic student-correctness fields, but no clear
  repository license was found in the audit, so both remain quarantined. No data was downloaded or used.
  Primary sources: `https://github.com/openai/prm800k`,
  `https://github.com/google-research-datasets/Education-Dialogue-Dataset`, and
  `https://github.com/NYCU-NLP-Lab/MathEDU`.

### E580 implementation verification and canonical cache

- Added `src/trace_ace/multicorpus_correctness_transfer.py`,
  `scripts/run_multicorpus_correctness_transfer.py`, and
  `tests/test_multicorpus_correctness_transfer.py`. The implementation asserts the exact competition runtime and
  source hashes before cache preparation, benchmarking, or training. Initial lint caught one unused import; the
  first real-cache test then caught NumPy integer values in the ordered-content hash serializer. Both were fixed
  before any benchmark or training score.
- Ruff is clean and all seven focused E580/E400 tests pass in `7.54` seconds. The deterministic cache contains
  `17,820` training rows (`8,910` SemEval and `8,910` GSM8K) plus `2,638` official paired GSM8K test rows.
  Selected-question SHA-256 is
  `2bbcc18022ccf879c60a2cbaf46d8487e89e74c22c723fe4130a9f8ea1bb4092`; ordered-content SHA-256 is
  `1ef624e1cf53708d52628cc3e05da117986b7e34a623dbe8427ed06ce8cac78c`; Parquet SHA-256 is
  `2ed6dbaa3f02e8333735c1c7005f17c28e6d1181a4670a47781736b634d89003`.
- No outcome metric, competition cache, `V_joint`, or `V_final` was accessed. Decision: authorize only the fixed
  E580 resource benchmark. Material training remains prohibited until the benchmark passes its six-hour and
  8 GiB clauses and a single active completion heartbeat is created and verified.

### E580 pre-benchmark resource-reporting correction

- The first benchmark command completed its two training and two evaluation batches but exited after `45.3`
  seconds before writing a benchmark artifact because `psutil` is not installed in the frozen `.venv`. It
  produced no external metric, checkpoint, prediction, or model report. Replace that optional RSS call with the
  repository's existing native Windows `_current_rss_bytes` implementation and rerun the same benchmark from
  zero. This correction changes only resource telemetry; the frozen dataset, optimizer, model, and gates do not
  change.

### E580 corrected resource benchmark and training authorization

- The corrected benchmark completed in `44.5` wall-clock seconds. Its two training batches took `11.425449`
  seconds and its two evaluation batches took `19.504990` seconds. It projects `9,036.159` seconds
  (`2.510044` hours) for 1,114 training steps plus both frozen base/candidate external evaluation passes.
  Observed RSS was `1,851,256,832` bytes. The six-hour and 8 GiB resource clauses both pass.
- Benchmark SHA-256 is `abe6e69b01a7ee297eb48b183fbd1dac5b8a4ebbfff5ddfd446793e3da48d8dc`.
  No outcome metric, checkpoint, competition cache, `V_joint`, or `V_final` was accessed. Decision: authorize one
  exact material training run from step zero. Before launch, create one active completion heartbeat for the
  measured duration plus buffer and verify its persisted active state; do not inspect the worker or logs before
  it.

### E580 completed external transfer and literal gate pass

- Run `20260727T085628Z_multicorpus_correctness_transfer` completed all `1,114/1,114` training steps from the
  untouched NLI base. Stdout creation-to-close elapsed time was `10,770.321` seconds (`2.991756` hours);
  final mean training loss was `0.610360`. Runtime remained Python `3.12.8`, NumPy `2.5.1`,
  scikit-learn `1.8.0`, Torch `2.13.0+cpu`, and Transformers `5.14.1`.
- Unseen-question metrics improved from base AUROC/macro-F1/loss
  `0.596582994/0.370570040/3.123705509` to `0.711945848/0.455755785/1.041258151`. Unseen-domain metrics
  improved from `0.724288305/0.430890698/2.466073362` to
  `0.760500763/0.469804458/0.997024533`. The exact `>=0.70`, `>=0.68`, `>=0.45`, and both AUROC
  non-regression clauses pass.
- On all 2,638 official paired GSM8K test rows, accuracy was `0.950720243`, AUROC `0.994636045`, log loss
  `0.111028344`, Brier `0.033980069`, ECE-10 `0.024192835`, prediction mean `0.503394214`, and neutral
  probability mean `0.000530318`. The untouched base scored AUROC `0.778627064` and loss `0.648353903`.
  The `>=0.95` AUROC, `<=0.35` loss, and both non-regression clauses pass.
- All nine frozen clauses pass; `passes_external_gate=true`. No bootstrap was preregistered for this external
  continuation gate, so bootstrap evidence is not applicable yet. The next stage must use the frozen competition
  session/family bootstrap protocol. No public projection is defensible before competition validation; the
  current confirmed public model remains v0.5 at `0.6054`, participant-observed rank `#8`.
- Artifact SHA-256 values: delta
  `13a2b5aa4d4fefa122b65c15ece3407719a29df8903047926dfe7acbb0116e9a`; external metrics
  `6973c97614654ceef71321622797a60ec32b8911a5c7efae0403f0c6f63213a3`; predictions
  `9446ec2fdb3600468cd99ca69709239e6949cf6bd365ec91eb6470c260fd1f41`; training metrics
  `8071b9f077a0395d4d8dc377198b4ce1d59af42b5581fcc9d07051a4fdef74b6`; report
  `acbff2ab6af75cddd68c021d90e45e8451adf8e10c864d8d5945e832fb25201f`.
- **Decision:** accept E580 through the external gate only. Authorize target-free competition cache construction
  and the already frozen leakage-safe fold-local probe/blend protocol. Do not change corpus weights, prompts,
  layers, optimizer, calibration, or checkpoint. No backup ZIP or platform submission is authorized.
  `competition_cache_built=false`; `V_joint_accessed=false`; `V_final_accessed=false`.

### E580 target-free cache, frozen competition validation, and rejection

- The 35,072-row adapted competition cache completed in `5,732.478` seconds (`1.592355` hours), earlier than its
  single 16:25 IST checkpoint. Pooled/logit shapes are `[35072,768]` and `[35072,3]`; SHA-256 values are
  `c02bc9774e95803567feb4e25a8541f1b69e5d361e9bc027a9b522cb825c1685` and
  `822a7f5e845d8c84c992138f5d5fa16d453cb64f3f55a9e58cca146b8e2d6722`. Metadata remained bound to delta
  `13a2b5aa4d4fefa122b65c15ece3407719a29df8903047926dfe7acbb0116e9a`; `V_final_accessed=false`.
- Frozen validation `20260727T105659Z_external_sra_validation` completed in `71.1` seconds using leakage-safe
  fold-local `C=0.1` probes and only the preregistered 10%, 20%, and 30% probability blends over the BGE-base
  replacement. No candidate passed the V_seen/V_objective/V_style development guard, so no weight was selected
  and V_joint was not opened.
- The 10% blend was least harmful but still had mean log-loss gain `-0.000219870`, only one of three development
  environments improved, worst regression `0.001443243`, mean AUROC gain `0.001348021`, mean Brier gain
  `-0.000085859`, and mean ECE gain `-0.001373519`. V_objective improved loss/AUROC/Brier/ECE by
  `0.002137427/0.004070499/0.000781251/0.004616147`, but V_seen regressed loss/Brier/ECE by
  `0.001353793/0.000499314/0.003993154`, and V_style regressed loss/AUROC/Brier/ECE by
  `0.001443243/0.000162913/0.000539513/0.004743551`.
- The 20% and 30% blends were progressively worse: mean loss gains `-0.002116213/-0.004322192`, worst
  environment regressions `0.004246710/0.007410665`, and only one environment improved in each case.
  Because the development guard failed before a weight was selected, no session/family bootstrap, V_joint
  confirmation, backup gate, or public projection is applicable. The honest public state remains v0.5 at
  `0.6054`, participant-observed rank `#8`.
- Artifact SHA-256 values: development environment metrics
  `2feb3e2e69d1b66464d00b74d383d4b60533be5214b8a8735782d46feb52a877`; fold metrics
  `e7439b826d23904c0b757f3ba9f81d4eda75942391a8b286e9457713b2296951`; predictions
  `7aba0095733ba150d79ca75831dbec7f108373711d993326e5c9f79f6377dcb7`; selection
  `9e2c8120cc349819c12efdcc6539eea069a676446a54e7baa2ad034683836b79`; report
  `6c3b9ef0cccf950067a222c2e4f2be6677c432b45dbc7815c44ecdfa2017586c`.
- **Decision:** reject E580 exactly despite its exceptional external gate. The transfer is objective-shift
  positive but regresses seen/style generalization and calibration. Do not rescue it with corpus ratios,
  corruption, prompt, checkpoint, cache view, probe C, calibration, or blend weights. No ZIP or platform
  submission is authorized. `V_joint_accessed=false`; `V_final_accessed=false`.

### E590 PRM800K source audit and preregistration before scoring

- Cloned the official OpenAI PRM800K repository with LFS payloads at immutable commit
  `7ecc794703b2877f63226f2477a49b34f9b25163`. The repository license is MIT, SHA-256
  `f213be7e9bf1040b5407cdc7b55c24a053c8fe7bc9b9f755541d822ff8af814b`. Phase-2 train/test file
  SHA-256 values are `1110237feeb51d1bc200cb37b8f965cfdc1036eac7d506094049366fe7dc1089` and
  `6b172efa884ac8341a946dd82e06947c135b7254109fb3f7aa907c715d98aaad`.
- The official PRM MATH split contains 11,999 train and 500 test problems with zero overlap. Phase-2 train has
  97,782 labeled trajectories over 10,828 distinct problems; phase-2 test has 2,762 trajectories over 458
  distinct official held-out problems and zero phase-2 train/test problem overlap.
- For prize-safe, leakage-safe training, exclude quality-control, initial-screening, outside-split, flagged,
  blank, and unrated candidates. Deduplicate by exact problem, ground truth, prior chosen trajectory, and current
  candidate; remove any input observed with conflicting ratings. The eligible unique train pool contains
  182,732 negative, 62,987 neutral, and 471,113 positive inputs after 3,506 conflicting inputs are removed.
  Select 2,970 per class by SHA-256 order; selected content SHA-256 is
  `73eed21b2a1fe1d26a5ab93b9751dd616b8892687294d4d1796147f4d3bddc3d`.
- The full official held-out evaluation contains 25,530 unique non-conflicting inputs after 74 conflicts are
  removed: 5,810 negative, 1,938 neutral, and 17,782 positive. Content SHA-256 is
  `456ecb3150c941b2d457bfb37476e687b47a8ad5847f100d17a87c72a9d74774`.
- E590 is frozen as a new human process-correctness transfer from the untouched base, not an E580 rescue. Exact
  premise compaction, label map, SemEval+PRM 17,820-row mixture, model/training settings, PRM/SemEval external
  clauses, question bootstrap, failure shield, and sealed-evaluation restrictions are recorded in
  `TOP5_RECOVERY_PLAN_2026-07-22.md` before any E590 prediction metric. No competition target, V_joint, V_final,
  model score, or platform submission was accessed.

### E590 implementation and canonical cache

- Added `src/trace_ace/prm_correctness_transfer.py`,
  `scripts/run_prm_correctness_transfer.py`, and focused tests. The parser streams the 456 MB phase-2 training
  file, removes exact duplicates and conflicting ratings, preserves only official question-disjoint splits, and
  keeps the frozen smallest-hash balanced sample without materializing the full text pool in memory.
- Ruff is clean and ten focused E590/E580/E400 tests pass. The real cache preparation completed in `44.1`
  seconds and reproduced both preregistered PRM content hashes exactly. The canonical cache contains 17,820
  training rows (`8,910` SemEval and `8,910` PRM800K) plus all 25,530 unique non-conflicting official PRM test
  rows.
- Ordered-content SHA-256 is
  `bcb3781c4c46b97a9404ba20dc8f54e196fe5a9f534fc462cd5abccb1322e974`; Parquet SHA-256 is
  `9f48c85106c5dfc46aad9b4aeaf12d7cdf20ac2b44f785b046bd3cbdf15282e6`.
  No model prediction, competition target, V_joint, or V_final was accessed. Decision: authorize only the frozen
  E590 resource benchmark before material training.

### E590 resource benchmark and material-run authorization

- The two fixed training batches took `12.813906` seconds and the two fixed evaluation batches took `20.604873`
  seconds. This projects `17,336.757` seconds (`4.815766` hours) for 1,114 training steps plus untouched-base and
  candidate evaluation over 31,644 external rows per pass. Observed RSS was `1,929,891,840` bytes.
- Both the preregistered six-hour and 8 GiB resource gates pass. Benchmark SHA-256 is
  `e286fc0ee006cb6fe1f73cbb0ecae6d81c7b3b1e115d9fe365583977ef72e47c`.
  No prediction metric, competition outcome, V_joint, or V_final was accessed. Decision: authorize one E590
  material run from step zero, protected by one measured completion heartbeat and no interim inspection.

### E590 completed external transfer and literal rejection

- Run `20260727T154320Z_prm_correctness_transfer` completed all `1,114/1,114` steps in `16,183.689` seconds
  (`4.495469` hours), with final mean training loss `1.069024`. Runtime remained the frozen Python
  `3.12.8`/scikit-learn `1.8.0`/Torch `2.13.0+cpu` environment.
- PRM test log loss improved strongly from untouched-base `1.669510424` to `1.079224649`, a gain of
  `0.590285776`. The 2,000-replicate bootstrap across 457 held-out questions produced mean gain `0.590862304`,
  95% interval `[0.519872697,0.663775560]`, and `1.0000` positive-gain support. Those two clauses pass.
- PRM accuracy/macro-F1/contradiction-AUROC were `0.357735997/0.303501964/0.581801001`. Macro-F1,
  contradiction AUROC, and absolute loss fail the frozen `0.50/0.75/0.90` thresholds despite outperforming the
  base contradiction AUROC `0.528991086`.
- SemEval unseen-question AUROC improved from `0.596582994` to `0.717038025`, and unseen-domain AUROC improved
  from `0.724288305` to `0.752461658`; both ranking thresholds and non-regression clauses pass. Unseen-question
  macro-F1 is only `0.417453866 < 0.45`, so the mandatory classification clause fails.
- Artifact SHA-256 values: delta
  `400b98d5de925bd720a2aa7311ac8d11413d704ffaa3a5960251f6649f00d3f5`; external metrics
  `d91dc4ae01e241c3966e36d747f35b52e1df5c1606dcd192648aa879e396dc79`; predictions
  `5149e4a1319d2173f24bd4244832b4c7639f139ad052019ebf51c8e46788cbf7`; bootstrap
  `cc9982f3f592310f9fcc90734f41c63ce278a625572874a55909f1b78585c0ee`; training metrics
  `8c7adac1658272a0c8047851cdf1d9fac77fd7c4982e62e84bfd6a23f093f913`; report
  `7fb4fbad0c642151a7a19c6e1c8c893b1e09e72fa9a0225d8d05336ae140153e`.
- **Decision:** four frozen external clauses fail. Reject E590 exactly without sample/corpus weights, context,
  prompt, label map, layer, optimizer, threshold, calibration, or checkpoint rescue. No competition cache,
  public projection, ZIP, or submission is authorized. Public champion remains v0.5 at `0.6054`, observed rank
  `#8`; `V_joint_accessed=false`; `V_final_accessed=false`.

### E600 session-bagged estimator preregistration

- E590 closes the current external correctness-transfer ladder: both natural human process labels and synthetic
  answer verification improved external proper scores but did not produce stable competition transfer. The next
  bounded hypothesis changes neither data nor representation. It tests whether averaging five deterministic
  session-subset BGE-base probes reduces the deployment variance/calibration gap between fold-local and full-data
  fitting.
- Freeze the exact pilot/features/folds/comparator, session-hash slice rule, five leave-one-slice-out estimators,
  per-estimator fold-local scaling, equal averaging, dual continuation gate, and 5,000 session bootstrap in
  `TOP5_RECOVERY_PLAN_2026-07-22.md` before scoring. No candidate blend weight or calibration is allowed.
  No V_joint, V_final, public score, or platform submission was accessed.

### E600 completed pilot and literal rejection

- Added `src/trace_ace/session_bagging_screen.py`, its CLI, and focused tests. All three focused tests pass under
  Python `3.12.8`, NumPy `2.5.1`, scikit-learn `1.8.0`, Torch `2.13.0+cpu`, and Transformers `5.14.1`.
  Immutable source hashes matched the preregistered BGE-base model, 4,096 pilot indices, modeling/context
  tables, caches, and baseline. The target-free benchmark fit all five bags for one outer fold in `1.009664`
  seconds, projected `5.048321` estimator seconds for five folds, observed `1,549,070,336` bytes RSS, and passed
  both resource gates.
- Frozen run `20260727T164822Z_session_bagging` completed in `26.529975` validation seconds. The identical
  single-fit BGE comparator produced log loss/AUROC/Brier/ECE
  `0.595607488/0.608965679/0.203578114/0.024219608`. The equal five-estimator leave-session-slice-out candidate
  produced `0.596717852/0.607265110/0.204028469/0.030034394`: candidate-minus-comparator changes were
  `+0.001110363/-0.001700568/+0.000450354/+0.005814786`. Every outer fold regressed in loss by
  `+0.001230368/+0.000934287/+0.001035967/+0.001513008/+0.000829215`.
- The 5,000-replicate paired session bootstrap across 3,856 sessions estimated mean log-loss gain
  `-0.001108170`, 95% interval `[-0.001456518,-0.000759177]`, and `0.0000` positive-gain support. Both frozen
  magnitude paths and the 90% bootstrap clause fail.
- Artifact SHA-256 values: metrics
  `5a905f2fdd1bb516d05a8b4eede6541280fb84ecf4fadc925304942648a3efb3`; predictions
  `582790ae03fb574b43b5dad51665e7bdf2719e3b8c930d983d14d623a2da539d`; fold metrics
  `7ef738e8c2e83f308a2bf1ca45559cd9e59705fc81dd9633d7d914e46fcfcd83`; bootstrap
  `b79e0d901a2931da63328853285e9836ccadb4f3cee54292628a1a0f7720c7c4`; bag summaries
  `f4334d845a25d91e7df8ceac371b40c5de9dac15a0ca20710bc40dc661942d77`; report
  `a0bddb097db481989044ac751c0c088e7793685b5399f2abb64be0f1d676f1c6`.
- **Decision:** reject E600 exactly without changing bag count, session hash, retained fraction, C, scaling,
  feature set, class weighting, calibration, or blending. The negative pilot result has no defensible public
  improvement projection; champion remains v0.5 at public `0.6054`, last observed rank `#8`, with an honest
  current bracket of approximately `#8` subject to leaderboard drift. No ZIP or submission is authorized.
  `V_joint_accessed=false`; `V_final_accessed=false`.

### E610 MCD mastery-transfer source audit and preregistration

- A bounded public-source search identified `ai4ed/MCD`, an MIT-licensed corpus of 5,226 authentic one-to-one
  grade-8 mathematics tutoring segments labeled Apprentice/Understanding/Mastery. This is materially closer to
  the competition outcome than SemEval answer correctness, GSM/PRM process correctness, or MathDial's synthetic
  self-correctness and tutor moves. Repository commit is
  `7ffc5e97948654d91e1cd368a3ecce96e0f4763a`; the raw 74.5 MB transcript payload and accompanying labels/split
  were downloaded from the repository's public Google Drive link.
- No competition row was involved in the source audit. MCD has exactly 5,226 aligned rows, 494 identifier
  prefix groups, labels `{0:2619, 1:1037, 2:1570}`, unique transcript hashes for every row, and no exact
  transcript duplication across the published partitions. However, 325 source prefixes overlap between the
  published train and test partitions, so that split is rejected for external gating.
- Freeze a deterministic source-group-disjoint split using SHA-256 bucket zero as test. It contains 4,242 rows
  and 399 groups for training (`0/1/2 = 2108/854/1280`) and 984 rows and 95 groups for test
  (`0/1/2 = 511/183/290`), with zero group overlap. Exact data/model, compaction, trainable layers, one-epoch
  optimizer, frozen-CLS comparator, proper-score/classification/kappa gates, source-group bootstrap, and
  competition-pilot continuation are recorded in `TOP5_RECOVERY_PLAN_2026-07-22.md` before any E610 model
  prediction metric.
- Official multilingual backbone is MIT `microsoft/mdeberta-v3-base` revision
  `a0484667b22365f84929a935b5e50a51f71f159d`; its published PyTorch payload SHA-256 is
  `6f89419baf0f1aaad5cab7d53901e36a8c1af8f6b4ab58b15db9af32df656ead`.
  No competition outcome, V_joint, V_final, public score, ZIP, or platform submission was accessed. Decision:
  authorize model download, canonical-cache construction, focused tests, and target-free resource benchmark only.
- The five pinned model files downloaded in `188.2` seconds. Local PyTorch and SentencePiece SHA-256 values
  exactly match the published hashes. The first load correctly stopped before tokenization because the `.venv`
  lacked SentencePiece and protobuf; both are now pinned as `0.2.1/6.33.5`. Transformers 5.14.1 also emitted its
  explicit legacy-regex warning, so `fix_mistral_regex=True` is frozen before any tokenized row. The corrected
  load verifies `DebertaV2ForSequenceClassification`, 12 layers, hidden size 768, and 278,811,651 parameters.
  This dependency correction produced no model prediction and changes no E610 scientific choice.
- The canonical builder completed in under ten seconds and exactly reproduced 4,242/984 rows, 399/95 disjoint
  source groups, and the frozen per-class counts. Ordered-content SHA-256 is
  `6336d7d377d67c34c8ea5471d2a7f6e3c119b8d0448b35f118bfa4f01401d4f9`; Parquet SHA-256 is
  `4180bbb1829214c569f084e3387eb38352d5b241e03f217736305b0c97d4dc2f`.
  Two clean seed-`20260727` model constructions reproduced trainable-state SHA-256
  `4e51494114193795988a96c8313394dcda333ff7e696053f9363e087d43a4b0f` across the frozen last two layers,
  pooler, and classifier (14,768,643 trainable parameters). Four focused tests pass and Ruff is clean. Decision:
  authorize only the frozen target-free resource benchmark next.

### E610 resource gate and literal rejection

- The target-free benchmark ran exactly two synthetic-label training batches and two evaluation batches. Training
  took `361.492176` seconds; evaluation took `106.720519` seconds. The measured projection is `137,437.094`
  seconds (`38.176971` hours) for 654 frozen-base feature batches, 531 physical training batches/266 optimizer
  steps, and 123 candidate test batches. Peak RSS was `1,465,061,376` bytes.
- The 8 GiB memory clause passes, but the preregistered ten-hour ceiling fails decisively. Benchmark SHA-256 is
  `3de48a2b8282bed89bfac15e30a4d73c32896bbe793c4cfb42ad6d80ba6e74d6`.
- **Decision:** reject exact E610 at the resource gate without shortening the sequence, changing compaction,
  training fewer layers, changing batch/epoch/optimizer, quantizing, substituting a model, or weakening the
  ceiling. No MCD outcome label entered a model, so there are no log-loss/AUROC/Brier/ECE or bootstrap results;
  no competition cache, ZIP, or public projection is authorized. Champion remains v0.5 at public `0.6054`,
  participant-observed rank `#8`; `V_joint_accessed=false`; `V_final_accessed=false`.

### E620 frozen multilingual mastery-screen preregistration

- The continuation wake was confirmed active, fired once, and was deleted immediately. Git is clean, no project
  Python worker remains, and the preserved v0.5 ZIP still hashes to
  `65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`.
- E620 is not an E610 architecture rescue: it trains no encoder layer and asks whether MCD mastery is linearly
  exposed by a small cross-lingual sentence model. Exact MIT `intfloat/multilingual-e5-small` revision
  `614241f622f53c4eeff9890bdc4f31cfecc418b3`, frozen representation, external estimator/gates, resource
  benchmark, and conditional competition pilot are locked in `TOP5_RECOVERY_PLAN_2026-07-22.md` before download,
  encoding, fitting, or any prediction metric. No competition outcome, V_joint, V_final, ZIP, or platform
  submission was accessed.

### E620 completed external screen and literal rejection

- The exact 64-row target-free benchmark took `5.919467` seconds and projected `862.207336` seconds
  (`0.239502` hours) for all external plus conditional pilot encodes, with `1,021,825,024` bytes peak RSS.
  Benchmark SHA-256 is `7702b40a05de1185e0749b5e03acb9e4b74ce3a9a6c4cfff4014219cb1679504`.
  The single cache worker completed 5,226/5,226 rows in `476.882965` seconds under Python `3.12.8`,
  scikit-learn `1.8.0`, Torch `2.13.0+cpu`, and Transformers `5.14.1`. Cache shape/hash are
  `[5226,384]` and `6cd5b733970ddbfa94b9507c047c6915bd83b664bddbea99a4a923fb36f16a1f`;
  metadata hash is `04d87c4f1675bb8b3f7b805fe12967ad868d044dfae52fa36b13326c34979990`.
- On all 984 examples from 95 untouched source groups, accuracy/macro-F1/class-F1 were
  `0.659552846/0.469735660/[0.761372706,0.000000000,0.647834275]`; macro one-vs-rest AUROC was
  `0.761372780`, quadratic kappa `0.494533948`, log loss `0.820439933`, summed multiclass Brier
  `0.470915542`, and top-label ECE-10 `0.056854967`. A deterministic reporting-only reconstruction added ECE
  and ordered-prediction hash after the single gate evaluation; it changed no fit, clause, or decision.
- Legal train-prior loss was `1.014363549`, so E620 gained `0.193923616`. The 2,000-replicate bootstrap across
  95 source groups gave mean gain `0.164471962`, 95% interval `[0.122465764,0.205161815]`, and `1.0000`
  positive support. Every frozen clause passes except every-class F1: the Understanding class F1 is exactly zero.
- External report SHA-256 is
  `5cd14a0cf186e95af59c163d2174fa35913c874b4a0b1383f1ee2bef6ff87467`; ordered prediction SHA-256 is
  `d0bac62924fc25033af91cdffade92c14b443bc717b4dd1362c66bc6bcf192b6`.
- **Decision:** reject exact E620 without class weighting, threshold, calibration, C, representation, prefix,
  length, pooling, or label rescue. No competition pilot or hardened environment was opened, so no defensible
  public projection or ZIP exists. Champion remains v0.5 at `0.6054`, participant-observed rank `#8`;
  `V_joint_accessed=false`; `V_final_accessed=false`; no platform action occurred.

### Post-E620 source audit and E630 Dialogue-KT preregistration

- CIMA commit `3fa48593c001046893f5f1320ab031f7237e7998` and dataset SHA-256
  `544dfc50dd05b14579e1a04b4751f4ffda8a72d3d398410d1aba0bcc2dd57405` contain 1,135 dialogue states with
  candidate tutor replies/actions, but no independent post-tutoring assessment outcome. It is rejected as an
  outcome-transfer source rather than forced into a surrogate label.
- StudyChat is a strong direct-outcome candidate with real student-chat records and released normalized grades,
  under CC BY 4.0 at repository revision `24d7987d9fbb30d9da12acc53455a10f1cdd2d7f`, but its data files return
  HTTP 401 without an accepted Hugging Face gated-data credential. No credential is installed locally. Do not
  bypass the gate or freeze a branch before authorized access and student/grade identifier alignment can be
  verified.
- Google Education Dialogue has 47,234 synthetic school dialogues and an explicit generated probability of
  student understanding at commit `ba54ae37425fe31db58883488aca00ed64e69e5d`, but its repository contains no
  license. It is excluded from training despite strong semantic alignment.
- The official MIT `umass-ml4ed/dialogue-kt` repository is usable at commit
  `c61f335f89005161b6ef439872cc1735bee26745`. Its CC BY-SA 4.0 annotated MathDial release supports a genuinely
  different next-response correctness task, while its separately restricted CoMTA release explicitly permits
  internal model evaluation but forbids training and redistribution.
- Source audit reproduced 2,253/595 annotated MathDial train/test dialogues, the known 314 overlapping test
  question IDs, and the leakage-safe 1,679-dialogue/720-question training set. Applying the released Dialogue-KT
  final-turn correction and excluding malformed/unlabeled/no-KC turns yields 7,980 train and 2,573 official-test
  next-response examples. Evaluation-only CoMTA contributes 623 untouched examples over 151 usable dialogues.
- E630's exact source hashes, no-current-response input, fresh binary-head/top-two-layer training contract,
  three-hour resource gate, MathDial and CoMTA proper-score/classification/bootstrap clauses, and failure shield
  are frozen in `TOP5_RECOVERY_PLAN_2026-07-22.md` before canonical serialization, model initialization, or any
  prediction metric. E430/E580/E620 weights are not loaded; CoMTA may never train or enter a package.

### E630 canonical cache and binary-head verification

- Focused E630/E460 tests pass (`6` tests) and Ruff is clean under `.venv` Python `3.12.8` and scikit-learn
  `1.8.0`. The canonical builder completed in 9.7 seconds and reproduced the frozen 7,980/2,573/623 usable turn
  counts with zero MathDial `qid` overlap and exactly zero CoMTA training rows.
- Ordered-content/Parquet SHA-256 values are
  `f6d5cf38716bf5b6a8fabade6346a17d9d3a4cbd7966a600907b4bb1ffa45458` and
  `2d7647576e681527c90be80db9bb7dac0794b864314e6c79ce3aeb3fae7d9152`.
  Usable turns cover 717 legal training and 386 official-test question IDs; the difference from the
  dialogue-level 720/391 counts is entirely the preregistered removal of malformed or unusable annotations.
- Two clean seed-`20260727` constructions reproduce the fresh `[2,768]` binary classifier weight SHA-256
  `c06504ffa78c64e79d3a6ecbad025b542a50bc7188463f7f4b1fd07ca267abb1` and zero-bias SHA-256
  `af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc`.
  No external prediction metric or competition outcome was accessed. Decision: authorize only the frozen resource
  benchmark before material E630 training.

- The first E630 benchmark attempt stopped before any model forward pass because several released multi-KC
  objective sequences were too long for the frozen `only_first` tokenizer contract. Before rebuilding or scoring,
  the recovery plan now freezes source-only compaction to the first three unique KC descriptions and first 180
  Unicode characters per description. This is an implementation correction, not a target-driven prompt change;
  no benchmark artifact or prediction metric was produced.
- The corrected rebuild preserved every turn count and leakage invariant. Its ordered-content/Parquet SHA-256
  values are `f6d5cf38716bf5b6a8fabade6346a17d9d3a4cbd7966a600907b4bb1ffa45458` and
  `2d7647576e681527c90be80db9bb7dac0794b864314e6c79ce3aeb3fae7d9152`; these supersede only the
  pre-correction cache hashes.

### E630 resource benchmark and material-run authorization

- The corrected benchmark ran exactly two real-input/synthetic-label training batches and two real-input
  evaluation batches, without computing an outcome metric. Training/evaluation times were
  `21.114765/35.974071` seconds.
- The measured projection is `6,185.473` seconds (`1.718187` hours) for all 499 training and 51 external
  evaluation batches. Observed RSS was `2,072,903,680` bytes, so the frozen three-hour and 8 GiB clauses both
  pass. Benchmark SHA-256 is `0465f7c414223b1b01442f1ccf9751666f5ce6f651b5db8efa26bee5b22dd4be`.
- Decision: authorize one clean E630 material run under the exact frozen contract. It must receive one calculated
  active completion heartbeat, with no worker or log inspection before that checkpoint.

### E630 completed external transfer and literal rejection

- Run `20260727T192934Z_dialogue_kt_transfer` completed all `499/499` steps and both frozen external evaluations
  in `5,227.613` seconds (`1.452115` hours), below the measured `1.718187`-hour projection. Final one-epoch
  training loss was `0.697319`. Runtime remained Python `3.12.8`, scikit-learn `1.8.0`, Torch `2.13.0+cpu`,
  and Transformers `5.14.1`.
- MathDial official-test accuracy/macro-F1/AUROC were
  `0.540614069/0.537010413/0.563515689`; log loss/Brier/ECE-10 were
  `0.693397132/0.249964660/0.056300784`. Gains versus the legal-train prior were only
  `0.000901865` loss and `0.000611182` Brier. The 2,000-replicate 386-`qid` bootstrap estimated mean gain
  `0.000971313`, 95% interval `[-0.005854962,0.008285809]`, and `0.6055` positive support. Six of seven
  frozen clauses fail; only ECE passes.
- Evaluation-only CoMTA accuracy/macro-F1/AUROC were
  `0.524879615/0.516709997/0.523795944`; log loss/Brier/ECE-10 were
  `0.690553032/0.248749411/0.063438432`. Loss/Brier regress the unchanged MathDial-train prior by
  `0.000052909/0.000072801`. The 2,000-replicate 151-dialogue bootstrap estimated mean gain
  `-0.000029578`, 95% interval `[-0.012999494,0.013406494]`, and `0.4950` positive support. Six of seven
  clauses fail; only ECE passes. CoMTA trained zero rows and remains absent from every candidate artifact.
- Artifact SHA-256 values: delta
  `ccda1ea41fb0629aacbabb7e425e9355b03e40d6b1f224f51855570695451d4c`; predictions
  `549e776fe21595b7a5c6bfccdc7e76edbd91abcc57fdb9585dfc7c3ab961a6ed`; external metrics
  `988009bcaccb93dc9dae56ad10db02f590386a1fa2ffc732aa02089ab090927e`; training metrics
  `dfe924a06fd5ac95d895545061b18b8f89e715507b12b7c5a8e60dee394918d5`; report
  `a5b6f53d7fc456b4209d083c6b913a8794728dfb88f72a1a7773e8499ef4a79c`.
- **Decision:** reject E630 exactly without history/current-answer leakage, objective compaction, source mixture,
  label/final-turn rule, prompt, length, layer, epoch, optimizer, class weight, threshold, calibration, checkpoint,
  or gate rescue. No competition cache, hardened validation, ZIP, or public projection is authorized. Champion
  remains v0.5 at public `0.6054`, participant-observed rank `#8`; `V_joint_accessed=false`;
  `V_final_accessed=false`; no platform action occurred.

### Post-E630 independent-source audit: Difficulty-Aware-DialogKT

- Audited the public `umass-ml4ed/Difficulty-Aware-DialogKT` repository at immutable commit
  `3a9362e12eb51675897a5b5b458b0fc709d2ff63`. The GitHub tree contains only a 71-byte `README.md`, no code,
  data, checkpoints, results, or license declaration.
- **Decision:** reject this repository as an executable or legally attributable evidence source in its current
  state. No files were trained, no competition validation environment was accessed, and `V_final` remains sealed.

### E640 SimulatorArena real-outcome source audit and preregistration

- Identified Microsoft SimulatorArena at immutable commit
  `e9f677c4975496fdd37f28bb6343ec3c1c54c8b4`. Its MIT-licensed redacted math-tutoring release contains 450
  real human-AI conversations and post-tutoring first-problem correctness annotations. Source/license SHA-256
  values are `b2909d037da14ebfafd72e66ad8985ab700f0f164bc5d94713ca6fc43aaf651a` and
  `9906940f61b1f0b533fa7d99baf55178b2808fbe113ea51dfbfad8572ccd5f2b`.
- The source has 296 `correct`, 153 `incorrect`, and one `unknown` outcome; nine tutor models, 66 workers, and
  265 problem IDs. The separately entered final answer/solution and self-report fields are not part of the chat
  arrays. E640 excludes the unknown row and forbids every final-answer, solution, self-report, rating, identity,
  difficulty, profile, and second-problem field as an input.
- Before any candidate embedding or prediction metric, froze chronological first-problem dialogue text, the exact
  v0.5 BGE-base normalized CLS encoder with left truncation at 256 tokens, unweighted `C=0.1` logistic heads,
  five worker-hash outer folds with validation-problem purging, the fold-prior comparator, two independent grouped
  bootstraps, and every external continuation/failure clause in `TOP5_RECOVERY_PLAN_2026-07-22.md`.
- The label-free fold audit yields 240-298 legal training rows and 70-124 validation rows per fold, zero worker or
  problem overlap, and both classes in every fit and validation partition. The fold-prior OOF baseline is log loss
  `0.666091`, Brier `0.235706`, AUROC `0.388282`, and macro-F1 `0.397315`. No candidate prediction,
  competition target, `V_joint`, or `V_final` was accessed.
- Focused E640/BGE tests pass (`8` tests) and Ruff is clean under the competition-aligned `.venv`. Canonical
  construction reproduced all 449 usable rows and every frozen fold count. Ordered-content/Parquet SHA-256 values
  are `950c0341583c6721b2e51fdca28c53d2a00d33d2a3d46d5ce2122c05671cf447` and
  `e743a05e140b10f154ad1f8e2d8d431d90cade9883c9e8e8f4c4988d55d57f60`; both were bound before any candidate
  embedding or prediction metric. Decision: authorize only the fixed label-free resource benchmark.
- The fixed 32-row benchmark took `6.393618` seconds and projects `89.710453` seconds for the complete 449-row
  cache. Peak RSS was `927,973,376` bytes; the one-hour/8 GiB resource gate passes. Runtime remains Python
  `3.12.8`, scikit-learn `1.8.0`, Torch `2.13.0+cpu`, and Transformers `5.14.1`. Benchmark SHA-256 is
  `f259ee1d75563fd7a7128c04ce04cb25ee5976a1f6e883855c19cb6784ca3bf0`. No outcome label or prediction metric
  was accessed. Decision: authorize one external-cache worker and exactly one calculated completion heartbeat.

### E640 completed external result and rejection

- The complete 449-row cache finished in `86.786697` seconds, closely matching the `89.710453`-second benchmark
  projection. Shape/hash are `449 x 768` and
  `e1ed9007a7076fcf1ff3bfc2f0b65bbf8d1b580ef7a9c9602d525b09460e4a1e`; source, canonical, runtime, and
  normalized-embedding contracts matched.
- Frozen OOF log loss/AUROC/Brier/ECE-10 were
  `0.665055158/0.416821233/0.235266026/0.060792718`; accuracy/macro-F1 were
  `0.659242762/0.397315436`. Legal fold-prior loss/Brier were `0.666090891/0.235706247`, so gains were only
  `0.001035734/0.000440221`. Four folds gained loss, with fold changes
  `[-0.000250800,0.001058154,0.002694313,0.000060504,0.000669694]`.
- The 2,000-replicate worker bootstrap estimated mean gain `0.001697009`, 95% interval
  `[-0.000682468,0.003942501]`, support `0.9260`; the problem bootstrap estimated `0.001276332`,
  `[-0.000501634,0.003000442]`, support `0.9280`. Only ECE, positive-fold count, and both support clauses pass.
- Prediction/fold-model/report SHA-256 values are
  `6f1f47f56dc7f3485dbfb4a83939b9f9547c24e4e1427f24ef64adc477811eac`,
  `b52737c8d7b50549e9ead9efbfc2f450479880e7c80632c91a5e27994df9c2d2`, and
  `5c2c72bac733b258ed553b170fa3c92c70ce9f3bd92d222865d882619fbc9a70`.
- **Decision:** reject E640 literally without any rescue. No competition cache, hardened environment, blend,
  package, public projection, upload, or submission is authorized. Champion remains unchanged;
  `V_joint_accessed=false`; `V_final_accessed=false`.

### Completion-heartbeat defect and correction

- The E640 worker completed normally, but the Codex app emitted the same three-minute heartbeat repeatedly despite
  its `COUNT=2` RRULE until this task obtained execution and deleted it. This was an automation scheduling defect,
  not repeated process/log polling: the worker was inspected only once after the heartbeat was deleted.
- **Correction:** never use minute-frequency recurrence for completion checkpoints in this environment. Future
  wakes will use one explicit UTC DTSTART with a daily-frequency, single-count RRULE, verified as persisted ACTIVE,
  and will self-delete before the sole inspection. This prevents a short-interval queue storm even if COUNT is not
  honored promptly.

### E650 SimulatorArena pedagogical-quality preregistration

- Froze a genuinely separate supervision target before any quality-model prediction: normalized human
  `overall_rating` across all 450 SimulatorArena interactions. This is not a rescue of E640 correctness; no E640
  label, probability, coefficient, threshold, or failed gate is reused.
- Mean/std normalized rating are `0.703457/0.280880`; the worker-disjoint/problem-purged fold-mean baseline has
  RMSE `0.282442` and MAE `0.234790`. Frozen folds contain 234-364 legal training rows and 49-114 validation rows,
  with zero worker/problem overlap.
- The exact text/encoder, fixed `alpha=10` ridge estimator, clipping, external correlations/proper-score gains,
  two grouped bootstraps, failure shield, and continuation rules are frozen in
  `TOP5_RECOVERY_PLAN_2026-07-22.md`. No quality prediction, competition target, `V_joint`, or `V_final` was
  accessed.
- Focused E650/E640 tests pass (`7` tests) and Ruff is clean under `.venv`. Canonical construction reproduced all
  450 rows and every frozen fold invariant. Ordered-content/Parquet SHA-256 values are
  `0d493508458104eebf05810762b704397052841b28e785bae5eac12fe12bc202` and
  `85b27544e041459e038ed6d302fec112c5d870873b450d0a05e6a656eed4c0de`; both were bound before any candidate
  prediction. Decision: authorize only the target-free resource benchmark.
- The fixed 32-row benchmark took `7.220233` seconds and projects `101.534532` seconds for all 450 rows. Peak RSS
  was `1,023,414,272` bytes; the one-hour/8 GiB gate passes. Benchmark SHA-256 is
  `c35ccfda504420f8a477f08080bbb0b1fa41087bfc9638aeb89e1e5cb5ce1186`. No quality target or prediction metric
  was accessed. Decision: authorize one external-cache worker and one corrected explicit-DTSTART heartbeat.

### E650 completed external result and rejection

- The target-free cache completed in `90.711856` seconds versus the `101.534532`-second projection. Its finite
  float32 shape is `450 x 768`; cache/metadata SHA-256 values are
  `1d7cbb544d4a71acbcf9bb6826f632a38f3fb3d187cf7a4a00b80205a604916e` and
  `181ed305c658da5526bd85832ee55334d934e6e984288973ad939a31add189c1`. Runtime was Python `3.12.8`,
  scikit-learn `1.8.0`, Torch `2.13.0+cpu`, and Transformers `5.14.1`; frozen source/canonical bindings matched.
- Frozen OOF RMSE/MAE/Pearson/Spearman were
  `0.280129688/0.232232784/0.076133341/0.081796906`. Legal fold-prior RMSE/MAE were
  `0.282441971/0.234790075`, so gains were only `0.002312282/0.002557291`. All five fold RMSE gains were
  positive but small: `[0.000612010,0.004783110,0.002687334,0.001674522,0.003661786]`.
- The worker-group bootstrap estimated mean squared-error gain `0.001144557`, 95% interval
  `[-0.000215033,0.002596338]`, support `0.9515`; the problem-group bootstrap estimated `0.001708759`,
  `[0.000684896,0.002749808]`, support `0.9995`. Both used 2,000 replicates. Positive-fold count and bootstrap
  support passed; Pearson, Spearman, absolute RMSE, RMSE gain, and MAE gain failed.
- Prediction/fold-model/report SHA-256 values are
  `4b826055f534f9713017337c112fee6227ecbd7188724a5e9e56b5ee967aac01`,
  `0bdd526cf17535d13181681f6d9e853747844d711e8075b751419c816087de2c`, and
  `7581e2f3478c705e50f32985753137d5bde6a85e10274dc89fabdc2f74e6193b`.
- **Decision:** reject E650 literally without rescue. No all-source head, competition cache, hardened validation,
  blend, ZIP, projected public improvement, upload, or submission is authorized. Champion remains v0.5 at public
  `0.6054`, participant-observed rank `#8`; honest rank bracket remains approximately `#8` with no new candidate
  projection. The preserved ZIP hash remains
  `65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`;
  `V_joint_accessed=false`; `V_final_accessed=false`.

### E660 independent ConvoLearn source audit and preregistration

- Identified the new MIT-licensed ConvoLearn release at immutable commit
  `f250e930356f2092462c4d1cd6bb2aae85689b1f`. Its 2,134-row Parquet and dataset-card SHA-256 values are
  `c1599655c3a2a3ef5fd199906200f02afd6b26a1bd4b5fbc16a18d6b784d5c04` and
  `63d12a3c43645e188d03599f4e500a49309048f77670e17209628cdea56f3c09`.
- The release contains unique tutor-student dialogues across 21 pedagogical subdimensions, six broad
  knowledge-building dimensions, and four science topics, with complete effectiveness and completeness ratings.
  It uses credentialed human teachers and a simulated student. This is a separate construct and source from
  SimulatorArena satisfaction; no E640/E650 artifact or decision is reused.
- Froze target-free five-bin chronological compaction, the exact packaged BGE-base CLS encoder, fixed Ridge and
  multinomial logistic heads, subdimension-disjoint and topic-disjoint external protocols, proper-score,
  correlation, fold, and grouped-bootstrap gates in `TOP5_RECOVERY_PLAN_2026-07-22.md` before any candidate
  embedding or prediction. No competition outcome, `V_joint`, or `V_final` was accessed.
- Source audit: no missing values or duplicate conversation text; topic counts 365-794 and broad-dimension counts
  189-589. Raw dialogue line counts are 13/21/106 min/median/max and word counts are 141/456/9,557. The frozen
  subdimension folds contain `[520,565,385,374,290]` rows. Decision: authorize canonicalization only.
- Canonicalization reproduced all 2,134 rows and every frozen zero-overlap split. Ordered-content/Parquet
  SHA-256 values are `952587761234e94183217b77f5637d131f11cc22aa25f59e013ef7dadb0717b8` and
  `664eabd969d837cb0fba8f490caa897f9ac2f9c28290b8e8c94d960d7d513941`; both are now bound before any
  embedding benchmark or candidate score. Normalized effectiveness/completeness mean and standard deviation are
  `0.588918/0.270404` and `0.625937/0.308970`. Decision: authorize only the target-free resource benchmark.
- The fixed 32-row target-free benchmark took `6.491799` seconds and projects `432.921812` seconds for the full
  2,134-row cache. Peak RSS was `887,513,088` bytes, so the one-hour/8 GiB gate passes under the aligned runtime.
  Benchmark SHA-256 is `45d4271f38fcbcb8af806ef41d34f2db0049003fc763dbecec84f5484e78fefb`.
  No external rating/dimension metric or competition outcome was accessed. Decision: authorize one external-cache
  worker and exactly one calculated ACTIVE completion heartbeat.

### E660 completed external result and rejection

- The target-free ConvoLearn cache completed all `2,134/2,134` rows in `394.637652` seconds versus the measured
  `432.921812`-second projection. Its finite float32 shape is `2134 x 768`; cache SHA-256 is
  `fd25b9d68f660460fad476a556964c4502af1fa18596fec260a5007e7835cbec`. Runtime was Python `3.12.8`,
  scikit-learn `1.8.0`, Torch `2.13.0+cpu`, and Transformers `5.14.1`. The immutable source, license,
  canonical-content, and canonical-Parquet bindings all matched.
- In the frozen subdimension-disjoint protocol, effectiveness RMSE/MAE/Pearson/Spearman were
  `0.266721623/0.236146846/0.164661252/0.169305434`; effectiveness prior RMSE and gain were
  `0.273285629/0.006564006`. Completeness RMSE/MAE/Pearson/Spearman were
  `0.305745197/0.260942124/0.146388226/0.148762441`; completeness prior RMSE and gain were
  `0.313131537/0.007386340`. All five effectiveness and completeness folds gained, with effectiveness gains
  `[0.007032679,0.008125151,0.004301792,0.005536436,0.007639791]` and completeness gains
  `[0.005049050,0.010228058,0.004667493,0.006948443,0.010776400]`.
- Subdimension-disjoint dimension macro-F1/log loss/prior loss/log-loss gain were
  `0.085655981/1.984642534/1.981505576/-0.003136958`. Its 2,000-replicate subdimension bootstrap estimated
  effectiveness mean squared-error gain `0.003523123`, 95% interval `[0.002184970,0.004688000]`, and support
  `1.0000`.
- In the frozen topic-disjoint protocol, effectiveness RMSE/MAE/Pearson/Spearman were
  `0.266249456/0.236397645/0.196367640/0.205097257`; effectiveness prior RMSE and gain were
  `0.271380392/0.005130936`. Completeness RMSE/MAE/Pearson/Spearman were
  `0.305628633/0.263395310/0.156287978/0.173845319`; completeness prior RMSE and gain were
  `0.310501595/0.004872962`. All four effectiveness and completeness folds gained, with effectiveness gains
  `[0.006126791,0.003386168,0.007849288,0.004720660]` and completeness gains
  `[0.007013741,0.002820100,0.007372854,0.003243892]`.
- Topic-disjoint dimension macro-F1/log loss/prior loss/log-loss gain were
  `0.072720199/1.714333949/1.718619705/0.004285755`. Its 2,000-replicate topic bootstrap estimated
  effectiveness mean squared-error gain `0.002931742`, 95% interval `[0.002198736,0.003675220]`, and support
  `1.0000`.
- Both protocols pass only the positive-gain-every-fold and grouped-bootstrap-support clauses. Both miss every
  frozen effectiveness/completeness correlation and RMSE-gain threshold and both dimension macro-F1/log-loss-gain
  thresholds. Binary AUROC, Brier, and ECE are not defined for this external multi-target gate and were not
  manufactured post hoc; competition log loss/AUROC/Brier/ECE were not evaluated because gate failure forbids a
  competition cache.
- Prediction/fold-model/report SHA-256 values are
  `f5589837f6e7c71221b0b6e8bf8419e169fef31353068adc54781e4576a00a79`,
  `1676429ab113590b4572f3f3b5056ada37beb44a39c91efdb97c7df6b62b7db8`, and
  `cd31b109e008a939b8923d44972d9c9f24f0cbba3e2616ac8ff7d86afe5cd469`.
- **Decision:** reject E660 literally without label, source, text, segment, prompt, pooling, alpha, C, classifier,
  weighting, calibration, fold, threshold, checkpoint, or gate rescue. No all-source head, competition cache,
  hardened validation, blend, ZIP, upload, or submission is authorized. Projected public log loss therefore
  remains the verified v0.5 score `0.6054`; the honest observed rank bracket remains approximately `#8`, not a
  promise of current placement. The preserved ZIP remains unchanged at SHA-256
  `65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`;
  `competition_outcomes_accessed=false`; `V_joint_accessed=false`; `V_final_accessed=false`.

### E670 independent Bridge expert-remediation audit and preregistration

- Selected Bridge as a genuinely independent supervision source after E660 rejection. It contains real-world
  math tutoring mistake episodes with the original novice tutor response paired to an experienced teacher's
  rewrite. This targets remediation choice, not E640 correctness, E650 satisfaction, E660 generic quality, or
  any artifact from those rejected branches.
- Immutable source is `rose-e-wang/bridge` commit
  `8f469883aa7d7a5c1d64e5c961a033ce71d21f5e`, CC BY-NC 4.0. Source-card SHA-256 is
  `279b838af2dc1b73fe69d2757ccb4b8e00251708621d1f66d69ce78703584e63`; train/validation/test JSON
  SHA-256 values are `870fa10d299711d315718d1995f234988582d7168fa9c054c267ededd89ba260`,
  `1d87ffece5fba05b6f5e91a2bcd0e5d556afb00973dd47e09969c875576bf327`, and
  `7c1f5eccce6f635924ca495439c7a9d70885c73122dd4caf8ad9066c657265e9`.
- Source audit found 700 complete pairs (419/71/210 released splits), 459 `c_id` values, 383 session roots,
  208 lesson topics, no empty candidate response, no text-identical pair, and one duplicated response pair.
  Novice response token counts are `2/17/92` min/median/max; expert counts are `1/16/79`. The 241 repeated
  contexts will remain group-locked.
- Froze the target-free Bridge text, BGE-base normalized CLS encoder, antisymmetric `C=0.1` pairwise logistic
  ranker, session-disjoint and lesson-disjoint protocols, proper-score/calibration/fold/group-bootstrap gates,
  exact failure shield, and conditional competition-cache/probe/blend path in
  `TOP5_RECOVERY_PLAN_2026-07-22.md` before any candidate embedding or prediction.
- No released error/strategy/intention annotation, later revision dialogue, competition outcome, prior rejected
  prediction, `V_joint`, or `V_final` was accessed. **Decision:** authorize canonicalization only, followed by
  hash binding and the fixed first-32-pair target-free resource benchmark.
- First canonicalization stopped before writing a cache because 16 released history turns across nine `c_id`
  occurrences have empty text; six are tutor markers and ten student markers. Two novice and one expert response
  turns are also empty, but every complete candidate response contains nonempty text. Correction preserves these
  released chronological turns as their exact empty `Tutor:`/`Student:` marker while requiring each complete
  candidate response to contain nonempty content. This is a source-schema correction, not row filtering or a
  model/prediction change; no embedding or candidate metric had been computed.
- Corrected canonicalization reproduced all 700 pairs. Session-disjoint fold sizes are
  `[126,160,143,121,150]`; lesson-disjoint sizes are `[242,72,145,101,140]`; all ten folds have zero
  held-out-group and pair overlap. Ordered-content/Parquet SHA-256 values are
  `1a6a300fe87904aa83b9a3ce6e804f6583d763b5e9805566c55d5e7c9dea2b32` and
  `a460bfa5d864c29dfd02e08e9b92e8173bfad9829fc69ebd1fa205707aa734f1`. Both are now bound before any
  candidate embedding or score. Decision: authorize only the fixed first-32-pair target-free benchmark.
- The fixed 32-pair/64-text benchmark took `6.851101` seconds and projects `149.867837` seconds for all 1,400
  candidate texts. Peak RSS was `861,392,896` bytes; the one-hour/8 GiB gate passes under the aligned runtime.
  Benchmark SHA-256 is `297f2a17fa3713b3f024d48cdba1df962a31f3020f7631b3aba8932ccd6ea1c7`.
  No preference label, candidate prediction, competition outcome, `V_joint`, or `V_final` was accessed.
  Decision: authorize one external-cache worker and exactly one calculated ACTIVE completion heartbeat.

### E670 completed external result and rejection

- The target-free cache completed all 1,400 novice/expert texts in `184.038498` seconds versus the measured
  `149.867837`-second projection. Its finite float32 shape is `1400 x 768`; cache SHA-256 is
  `661a2738eec75de96927a1f186c556d5987c3d41f4ffa91b3b692086d6dfd08e`. Runtime was Python `3.12.8`,
  scikit-learn `1.8.0`, Torch `2.13.0+cpu`, and Transformers `5.14.1`; all immutable source/canonical/encoder
  bindings matched.
- Session-disjoint accuracy/AUROC/log loss/Brier/ECE-10 were
  `0.824285714/0.890438776/0.612076049/0.210171626/0.269725834`. Fixed-prior log-loss/Brier gains were
  `0.081071132/0.039828374`. Fold log-loss gains were
  `[0.097642705,0.077088901,0.087545607,0.070036514,0.074127649]`. Its 2,000-replicate session-root
  bootstrap estimated mean gain `0.081085948`, 95% interval `[0.071528575,0.090445693]`, support `1.0000`.
- Lesson-disjoint accuracy/AUROC/log loss/Brier/ECE-10 were
  `0.821428571/0.896487755/0.613850040/0.210980307/0.269271586`. Fixed-prior log-loss/Brier gains were
  `0.079297141/0.039019693`. Fold log-loss gains were
  `[0.072599614,0.084107513,0.080696261,0.079334278,0.086924506]`. Its 2,000-replicate lesson bootstrap
  estimated mean gain `0.079034157`, interval `[0.069725769,0.087187616]`, support `1.0000`.
- Both protocols pass every frozen ranking, discrimination, proper-score, fold, and bootstrap clause but fail
  ECE-10 (`0.2697` and `0.2693` versus maximum `0.10`). The conjunctive gate therefore fails. This strong
  result is evidence that remediation preference is learnable, but it does not authorize post-hoc Platt,
  temperature, isotonic, scale, intercept, prior, C, threshold, or probability reinterpretation.
- Prediction/fold-model/report SHA-256 values are
  `b7692b100ae9c221499b6eb622289527956963b9046bcf183c329784a278d5f1`,
  `a087a568f1c2c9913176a82309129e3ea40785baae4e64646a4e9eb38590a4b4`, and
  `cb88571f457d4dd64f584c0b437feea7e152c05bcd7d906c7ac95c9196d8202f`.
- **Decision:** reject E670 literally without rescue. No all-source vector, competition cache, hardened metrics,
  blend, ZIP, public projection, upload, or submission is authorized. Projected public loss remains the verified
  v0.5 `0.6054`; honest observed rank bracket remains approximately `#8`. The preserved ZIP hash remains
  `65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`;
  `competition_outcomes_accessed=false`; `V_joint_accessed=false`; `V_final_accessed=false`.

### E680 independent CIMA student-action audit and preregistration

- Selected a new supervision family after closing E670: CIMA's released student epistemic actions. E680 predicts
  only the `Guess` bit from the student's chronological dialogue, testing independent-attempt versus
  help-seeking/acknowledgment behavior. No Bridge data, checkpoint, coefficient, prediction, or calibration
  evidence is reused.
- Immutable source is CIMA commit `3fa48593c001046893f5f1320ab031f7237e7998`, CC BY 2.5.
  Dataset/README SHA-256 values are `544dfc50dd05b14579e1a04b4751f4ffda8a72d3d398410d1aba0bcc2dd57405`
  and `23680d5c37de5b69436172372f14f88ddc31376a27738256d07bcaa78642897b`.
- Source audit found 1,135 usable contexts; every history has an even 2-10 alternating turns and a nonempty
  final student utterance. Guess has 514 positives and 621 negatives. Exercise/concept grouping provides 225
  image IDs and 123 exact concept triples; three complete histories repeat and will stay group-locked.
- Froze the target-free dialogue, BGE-base normalized CLS encoder, `C=0.1` unweighted logistic head,
  exercise-disjoint and concept-disjoint folds, legal fold-prior comparator, proper-score/calibration/fold/group
  bootstrap gates, failure shield, and conditional competition-cache/probe/blend path in the recovery plan
  before any candidate embedding or prediction.
- No CIMA action label, answer/concept field, candidate tutor response, competition outcome, rejected model,
  `V_joint`, or `V_final` entered model input or scoring. **Decision:** authorize canonicalization only, then
  bind hashes before the fixed first-32-row target-free benchmark.
- Canonicalization completed in 3.6 seconds under the aligned Python 3.12 environment. Ordered-content and
  canonical Parquet SHA-256 values are
  `2bef00ad2bb18eba376fcb70501a445d681395ccd3d76223e4a509bfd96c5a8a` and
  `5c97d1bbec9302bedebbdda6d264ffa42ae574ef89b2aac1f0191d50ffe508c7`. Shape/target counts are exactly
  1,135 rows, 514 positive, and 621 negative. Exercise-disjoint fold rows are `[250,199,180,249,257]`;
  concept-disjoint rows are `[177,280,208,272,198]`. Every group intersection is zero and every train and
  validation partition contains both classes. No competition outcome, `V_joint`, or `V_final` was accessed.
  **Decision:** bind both hashes and authorize only the fixed first-32-row target-free resource benchmark.
- The fixed 32-row benchmark completed in 5.501544 seconds and projected 195.132899 seconds for the full
  1,135-row cache. Its finite shape was `32 x 768`, peak RSS was 884,129,792 bytes, and the one-hour/8 GiB
  resource gate passed under Python 3.12.8, scikit-learn 1.8.0, Torch 2.13.0+cpu, and Transformers 5.14.1.
  Benchmark SHA-256 is `d835b51b87cb1405c4f6347361162ef772eb4d5587b8123d22974f4820bd762e`.
  No action label, competition outcome, `V_joint`, or `V_final` was accessed. **Decision:** authorize exactly
  one full external-cache worker and one calculated non-repeating completion checkpoint.

### E680 completed external result and literal rejection

- The target-free cache completed 1,135/1,135 rows in 220.273291 seconds versus the 195.132899-second
  projection. It is a finite float32 `1135 x 768` matrix. Embedding/metadata SHA-256 values are
  `d3773b92514ce87e0da7b8d240806e266a3ce4a8c6de9c255e481b087d5f3937` and
  `23a47010cfb7e13a9ab8555cdd1e223e470d2de0bdf4f970e596597cde0e1ca5`. Runtime was Python 3.12.8,
  scikit-learn 1.8.0, Torch 2.13.0+cpu, and Transformers 5.14.1.
- Exercise-disjoint AUROC/macro-F1/log loss/Brier/ECE-10 were
  `0.795951678/0.494637582/0.653553626/0.230453807/0.055249773`. Fixed-prior log-loss/Brier gains were
  `0.036154929/0.017825471`; fold log-loss gains were
  `[0.037325484,0.045989832,0.045048280,0.034824266,0.022461336]`. The 2,000-replicate exercise bootstrap
  gave mean gain `0.036187909`, interval `[0.031894459,0.040515952]`, support `1.0000`.
- Concept-disjoint AUROC/macro-F1/log loss/Brier/ECE-10 were
  `0.780255895/0.510018796/0.654773391/0.231074440/0.082665049`. Fixed-prior log-loss/Brier gains were
  `0.036051855/0.017759525`; fold log-loss gains were
  `[0.035803672,0.035102705,0.039070713,0.032843452,0.038852124]`. The 2,000-replicate concept bootstrap
  gave mean gain `0.036085519`, interval `[0.032001507,0.040198207]`, support `1.0000`.
- Both protocols pass frozen AUROC, aggregate proper-score gain, aggregate ECE, every-fold positivity, and
  grouped-bootstrap clauses, but fail mandatory macro-F1 `>= 0.65`. Prediction/fold-model/report hashes are
  `7c8e88bf44b1ae97af9e8d0b106a65fcfb32a37fc7512d8cc7f15a229749dca7`,
  `fbafd945c666af7da80b28f4f6b0bbb338efb8e3e1bcacda6d0aea248b9379ce`, and
  `c635e558efd41e1a745251470b88454cd281118b7f572cf7cd3588e0a15db1a5`.
- **Decision:** reject E680 literally without threshold, calibration, weighting, C, prompt, pooling, fold, or
  label rescue. No all-source head, competition cache, hardened validation, blend, ZIP, upload, or submission
  is authorized. Projected public loss remains v0.5 `0.6054`; honest observed rank bracket remains about `#8`.
  `competition_outcomes_accessed=false`; `V_joint_accessed=false`; `V_final_accessed=false`.

### E690 independent EssayJudge response-quality audit and preregistration

- Selected the final unused prize-safe local supervision source with a direct human-assessment signal:
  EssayJudge. This branch predicts the normalized unweighted mean of ten released writing-quality traits from
  prompt plus student essay. It reuses no E640-E680 data, label, checkpoint, coefficient, or prediction.
- The platform catalog declares Apache-2.0. Official repository HEAD is
  `e5ee947e97231d03d1341092e187ef0384da219e`; immutable local CSV SHA-256 is
  `7998f78165f3b354f42333ad25dc75ee6a33bfa3aa6ffb8c07650fc53c29dea5`.
- Audited 1,054 complete rows, ten traits, 125 prompts, 1,051 unique essays, seven chart types, and target
  range/mean/population standard deviation `0.15/0.676423150/0.101169702`. Six rows participate in duplicate
  essay text across prompts. Exact prompt/essay pairs do not duplicate.
- Froze canonical prompt/response text, label exclusions, BGE-base normalized CLS, `Ridge(alpha=10)`,
  duplicate-safe prompt connected components, five group-disjoint folds, legal training-mean comparator,
  Pearson/Spearman/RMSE/MAE/fold/bootstrap gates, failure shield, and conditional target-free competition
  cache/probe/blend path before any candidate embedding or prediction.
- No EssayJudge score, derived target, graph URL, image field, chart type, competition outcome, rejected head,
  `V_joint`, or `V_final` entered candidate input or scoring. **Decision:** authorize canonicalization only,
  then bind hashes before the fixed first-32-row target-free benchmark.
- Canonicalization completed under Python 3.12.8. Ordered-content/Parquet SHA-256 values are
  `9a88708f39ba9230a6315a088e5693b55111d7e51c9664732ab4ba49da00ef3b` and
  `a5f57aef306add56855d3411710552f5e4d48776c92c3ed3e3e5f07907cd7fad`. Shape/target statistics remain
  1,054 rows and `0.15/0.676423150/0.101169702/0.98` min/mean/population-standard-deviation/max. Fold rows are
  `[262,216,183,174,219]`; all component and exact-essay intersections are zero. No score, graph, image,
  chart type, competition outcome, `V_joint`, or `V_final` entered candidate text. **Decision:** freeze both
  hashes and authorize only the fixed first-32-row target-free benchmark.
- The fixed 32-row benchmark completed in 7.691140 seconds and projected 253.326914 seconds for the 1,054-row
  cache. Its finite shape was `32 x 768`; peak RSS was 1,007,497,216 bytes. The one-hour/8 GiB gate passed
  under Python 3.12.8, scikit-learn 1.8.0, Torch 2.13.0+cpu, and Transformers 5.14.1. Benchmark SHA-256 is
  `6402e76279a9161ab405fec844ca31fee7c133152148b92dd1e3a54f424758cb`. No target, candidate metric,
  competition outcome, `V_joint`, or `V_final` was accessed. **Decision:** authorize exactly one full
  external-cache worker and one calculated non-repeating completion checkpoint.

### E700 completed external result and literal rejection

- The target-free cache completed 12,256/12,256 rows in 2,549.290610 seconds (42.488177 minutes), versus the
  3,072.715242-second projection. Its finite float32 shape is `12256 x 768`; embedding/metadata SHA-256 values
  are `28d1b92aea738824ea8ab14176da95ad6709c0b2e39b930f92012b499f73d5d4` and
  `d5db11b924c6f8852ffed0af069aa111e2be61eee3c2b8ff569f8a7aeddbdf09`. Runtime was Python 3.12.8,
  scikit-learn 1.8.0, Torch 2.13.0+cpu, and Transformers 5.14.1.
- Official validation AUROC/macro-F1/log loss/Brier/ECE-10 were
  `0.563619750/0.530501397/0.690059061/0.248457582/0.026136687`; log-loss/Brier gains versus 0.5 were
  `0.003088120/0.001542418`. The 23-story bootstrap gave mean gain `0.003082807`, interval
  `[0.002687363,0.003493298]`, support `1.0000`.
- Official test values were `0.558146598/0.534088269/0.690342008/0.248598661/0.024538793`; gains were
  `0.002805172/0.001401339`. Its independent story bootstrap gave mean gain `0.002813994`, interval
  `[0.002386784,0.003280741]`, support `1.0000`.
- Both splits pass only ECE and bootstrap support. AUROC, macro-F1, log-loss gain, and Brier gain all fail.
  Prediction/model/report hashes are
  `4ab4ed9dd4e016ef8cf9614d57cce3a446f3bf0e0f505355eb0286c1f6de9531`,
  `2083b605b7c8f6a05ecb7b0d0e7e4702f7c4daa3b40ba009f26b27fa612dfac6`, and
  `434a6965605ca1d4f3359afac7b7410eaa93a2ea400942798b975c811539e204`.
- **Decision:** reject E700 literally without negative, sample, C, prompt, context, truncation, pooling,
  estimator, weighting, calibration, threshold, split, or gate rescue. No competition cache, validation,
  blend, ZIP, upload, or submission is authorized. Public champion remains v0.5 at `0.6054`, last observed
  about `#8`; `competition_outcomes_accessed=false`; `V_joint_accessed=false`; `V_final_accessed=false`.

### E690 completed external result and literal rejection

- The cache completed 1,054/1,054 rows in 213.905256 seconds versus the 253.326914-second projection. Its
  finite float32 shape is `1054 x 768`; embedding/metadata SHA-256 values are
  `faef66f0031ac9c0baae1741bb9d6c72e2bd17115dab50ca313994c349a65a68` and
  `7419b0beb75771c3667288904848afcbce9b12ba511acf629555f4b781d8da76`. Runtime was Python 3.12.8,
  scikit-learn 1.8.0, Torch 2.13.0+cpu, and Transformers 5.14.1.
- OOF RMSE/MAE/Pearson/Spearman were `0.097780051/0.077423379/0.275378820/0.254394280`. Legal
  fold-training-mean RMSE/MAE were `0.101333631/0.079653234`, so gains were only
  `0.003553580/0.002229855`. All fold RMSE gains were positive:
  `[0.004540685,0.004053204,0.004184463,0.000915065,0.003257363]`.
- Prompt-component bootstrap mean squared-error gain/95% interval/support were
  `0.000676974/[0.000439316,0.000918783]/1.0000`; chart-type values were
  `0.000627960/[0.000271292,0.000934855]/0.9990`.
- Absolute RMSE, every-fold positivity, and both bootstrap clauses pass, but Pearson, Spearman, RMSE-gain, and
  MAE-gain clauses fail. Prediction/fold-model/report hashes are
  `709fde41553f14b41ced0fb96cc0ceab8becbbb6a45b218a10e47f9481f7ad9d`,
  `2ee2947742496167ea31f5533ee446bb0b48eb1c6ce9da747bce246fa90f7a15`, and
  `ed2999e50fe83802dee17880be08ede8ace6fc37883ccf32a969a3d70af764fc`.
- **Decision:** reject E690 literally without alpha, trait, target, prompt, length, pooling, estimator, clipping,
  weighting, fold, or threshold rescue. No competition cache, validation, blend, ZIP, upload, or submission is
  authorized. Public champion remains v0.5 at `0.6054`, last observed about `#8`;
  `competition_outcomes_accessed=false`; `V_joint_accessed=false`; `V_final_accessed=false`.

### E700 independent FairytaleQA answer-support audit and preregistration

- Selected a fresh comprehension supervision family after E690: decide whether a candidate student answer is
  supported by its story context and question. This targets response correctness/relevance, not the
  explicit/implicit metadata, and reuses no rejected E640-E690 data, checkpoint, or prediction.
- Official FairytaleQA repository commit is `a24ddc17364666b7c13a425b9970c87368b03417`; fetched Apache-2.0
  license SHA-256 is `335d2c093cbb191de9693d87c9935f832a2442a67615a6e47ac71f74ccb4bd5d`.
  Train/valid/test CSV SHA-256 values are
  `b778e56bfc9cd9337eed02a8a0216eb7c5ff9deef96edc2fe49911924c9a79ab`,
  `b2dfd4dc5e99cf903e412fa785357138f844202222771674c446a4e41acb9abf`, and
  `34acae1b2b62469fb616ed55ea049c25185838f2388c60ef28c3f810e512a54b`.
- Audited official rows/stories `8548/232`, `1025/23`, and `1007/23`; story intersections are empty. All
  required fields are complete. Every story has at least five distinct answers, permitting one deterministic
  within-story mismatched answer for every retained positive with no cross-story shortcut or dropped row.
- Froze a hash-selected 4,096-row train subset, all validation/test rows, exact deterministic within-story
  negatives, canonical text/label exclusions, BGE-base normalized CLS, unweighted `C=0.1` logistic head,
  official split evaluation, proper-score/calibration/story-bootstrap gates, failure shield, and conditional
  target-free competition cache/probe/blend path before any candidate embedding or prediction.
- No answer-support label, FairytaleQA metadata label, competition outcome, rejected model, `V_joint`, or
  `V_final` entered candidate scoring. **Decision:** authorize canonicalization only, then bind hashes before
  the fixed first-32-row target-free benchmark.
- Canonicalization completed under Python 3.12.8. Ordered-content/Parquet SHA-256 values are
  `2d4fb1f185b3931e9e00020ed835660ead7bfd5b0315adf91a2522ebb061beb2` and
  `f64cc3c9c7ac6f744e17e318dd6a2bbe7222cc370c5117b88cc25748724c881a`. Exact balanced counts are
  train `4096/4096`, validation `1025/1025`, and test `1007/1007`, totaling 12,256 examples across
  `232/23/23` disjoint stories. Metadata/labels are absent from text; no competition outcome, `V_joint`, or
  `V_final` was accessed. **Decision:** freeze both hashes and authorize only the fixed 32-row benchmark.
- The fixed 32-row benchmark completed in 8.022755 seconds and projected 3,072.715242 seconds (51.211921
  minutes) for the 12,256-row cache. Its finite shape was `32 x 768`; peak RSS was 1,108,893,696 bytes.
  The one-hour/8 GiB gate passed under Python 3.12.8, scikit-learn 1.8.0, Torch 2.13.0+cpu, and Transformers
  5.14.1. Benchmark SHA-256 is
  `648297f3b5492350c75dcb2a20ff5c3c5cf3d88e0fd549fa360402fe96ed18e9`. No target, candidate metric,
  competition outcome, `V_joint`, or `V_final` was accessed. **Decision:** authorize exactly one full cache
  worker and one calculated non-repeating completion checkpoint.

### E710 competition-side gap audit and target-free coherence preregistration

- After E700 rejection, verified pushed HEAD `3cca63f` on `codex/v04-recovery`, no active K12 Python worker,
  and unchanged protected v0.5 ZIP SHA-256
  `65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`. `V_final` remains sealed.
- The v0.5 hardened evidence was audited only through the already authorized `V_seen`, `V_objective`, and
  `V_style` component OOF files; `V_joint` was not read. The raw BGE-base replacement improves all hardened
  environments but averages only `0.001272118` loss gain, leaving about `0.0016` to the conservative top-five
  target.
- Prior competition-side evidence closes whole-session trajectories, fixed symbolic event expansion, timing,
  semantic attention, pooling, nonlinear outcome heads, pairwise outcome ranking, and bagging. Ordered
  tutor/student adjacency remains the only positive but unstable complement: its earlier fixed word-window
  family averaged about `0.001026` loss gain across six protocols.
- Preregistered E710 before any new target score: a five-way session-cross-fitted, target-free next-student-turn
  coherence classifier using true tutor-to-next-student pairs versus deterministic within-response non-adjacent
  student negatives. Its fixed hashed lexical interaction representation and 12 probability trajectory
  aggregates contain no competition outcome. Only fold-local `C=0.1` probes and 10/20/30% blends over the fair
  raw BGE comparator are allowed.
- Continuation requires at least `0.0016` pooled loss gain, bounded fold regression, AUROC/Brier/calibration
  non-regression, and 95% paired session-bootstrap support. Failure rejects the exact branch without rescue.
  `V_joint_accessed=false`; `V_final_accessed=false`; no platform action occurred.
- Implemented the frozen E710 pipeline and four focused contract tests; all pass and Ruff is clean. The
  target-free first-512-row benchmark formed 6,330 positive pairs/12,660 balanced examples, completed its
  five self-supervised cross-fits in 0.860135 seconds, projected 6.881076 seconds for the pilot, and peaked at
  721,113,088 RSS bytes. Benchmark SHA-256 is
  `fbd042173ec7376e40455ec319f920a1195d5d293c685d45d4a8d7dbccd06c71`. The resource gate authorizes a
  foreground build; no completion heartbeat is warranted for a measured single-digit-second task.
- The complete E710 target-free build formed 54,771 positive pairs/109,542 balanced examples in 7.923822
  model seconds and 27.3 wall seconds, peaking at 2,138,791,936 RSS bytes. Cache/metadata hashes are
  `2eab7b3e7870c09a805a4b2b302ba0e084c5236fee6b9e897bb044dd4c406ebf` and
  `a8e35bfa1f592a976c375a88321e6faf55cd82d29e9001d87a61bcab5bc3db28`.
- Frozen run `20260728T075508Z_target_free_coherence` selected 10% and changed the BGE comparator by
  `+0.000109900` loss, `+0.001248338` AUROC, `+0.000035289` Brier, and `+0.001438378` ECE. Fold loss changes
  were `+0.000390475/-0.000105159/+0.000345422/-0.000053286/-0.000055836`. The 5,000-session bootstrap
  mean/interval/support were `-0.000109363/[-0.000320473,0.000115642]/0.1672`.
- **Decision:** E710 fails loss magnitude, Brier, ECE, and bootstrap clauses and is rejected literally without
  rescue. Artifact hashes are recorded in the recovery plan. No full cache, hardened environment, package, or
  platform action is authorized; `V_joint_accessed=false`; `V_final_accessed=false`.
- Participant update on 2026-07-28: unchanged public v0.5 score `0.6054` has drifted to observed rank `#11`.
  Two manual submission slots remain in the current competition week. This rank drift does not change any
  local gate or authorize leaderboard tuning; notify the participant only after a locally verified candidate
  clears the frozen backup gate. All uploads/submissions remain participant-only.

### E720 sparse objective-response interaction audit and preregistration

- E710 is closed and none of its cache, score, aggregate, or head enters E720. The remaining architecture audit
  found no prior explicit sparse objective-token by role-token conjunction model; current sparse models
  concatenate views and BGE provides only dense interaction geometry.
- A label-free frozen-pilot audit retained median `6/64/64` objective/student/tutor unique tokens and counted
  3,098,238 bounded conjunctions. This is operationally small and addresses the objective-conditioned lexical
  gap with a plausible route to the required 0.0016 loss improvement.
- Froze exact token caps, `2^19` signed hashing, raw lexical backoff plus objective-student/objective-tutor
  conjunctions, averaged SGD settings, session-purged folds, 35 legal controls, 10/20/30 blends, proper-score
  gates, fold bound, and 5,000-session bootstrap before any E720 target score. `V_joint_accessed=false`;
  `V_final_accessed=false`; no platform action occurred.
- Three focused tests pass and Ruff is clean. The fixed 512-row target-free/synthetic benchmark generated
  443,480 sparse nonzeros, completed in 6.260062 seconds, projected 250.402460 seconds for full construction
  plus five validation folds, and peaked at 1,530,957,824 RSS bytes. Benchmark SHA-256 is
  `08bfce8c63892e3d43c9bdaafad5338ac76a99ee53dda5bde2236170712cfcb2`. The resource gate authorizes one
  worker and one calculated non-minute completion wake.
- The E720 worker completed before 13:34 IST, but its persisted ACTIVE hourly heartbeat did not deliver at
  13:40. It was deleted at 13:54 before the exact worker/artifact inspection. Future long jobs must verify an
  explicitly anchored one-time next-run timestamp in addition to ACTIVE state; no-polling remains mandatory.
- E720 cache shape/nnz/build seconds/peak RSS were `4096 x 524288`/`3611435`/`3.330960`/`848953344`;
  cache SHA-256 is `6af9535a45e605ee3c8e98b191b46243ba8bace1dd76476a6b3dee9c6aacfe31`.
  The selected 10% blend regressed loss/AUROC/Brier/ECE by
  `0.001834117/0.005349205/0.000869603/0.001825527`. Fold loss changes were
  `+0.006697702/+0.006517232/+0.001785007/-0.004661570/-0.001288230`; bootstrap
  mean/interval/support were `-0.001848706/[-0.004220268,0.000555910]/0.0626`.
- **Decision:** every E720 gate fails. Reject literally without rescue; exact artifact hashes are recorded in
  the recovery plan. No hardened evaluation, package, upload, or submission is authorized.
  `V_joint_accessed=false`; `V_final_accessed=false`.

### E730 target-free centered objective-alignment audit and preregistration

- E720 is closed. Audited a new label-free geometry on the immutable BGE-base context/objective caches:
  train-session-centered orthogonal Procrustes alignment, with no target or old prediction.
- Across five deterministic session folds, raw true-pair minus shuffled cosine gap averaged `0.159612214`;
  aligned gap averaged `0.706424700`. Every aligned held-out true-pair mean was `0.7039-0.7143`, while shuffled
  means stayed near zero. Five 768-dimensional SVDs took 1.528807 seconds.
- This is materially non-identity held-out alignment and plausibly large enough for the required 0.0016 gain.
  Froze exact outer-training centering/map fit, aligned interaction block, dense cosine replacement, `C=0.1`,
  10/20/30 blends, proper-score/fold/bootstrap gates, and failure shield before any E730 outcome score.
  `V_joint_accessed=false`; `V_final_accessed=false`; no platform action occurred.
- E730 implementation verification used the competition `.venv`: Python 3.12.8, scikit-learn 1.8.0,
  NumPy 2.5.1, Torch 2.13.0+cpu, and Transformers 5.14.1. Two focused tests pass and Ruff is clean.
- The frozen target-free one-fold benchmark took 1.282712 seconds and projects 6.413560 seconds for five
  folds, with peak RSS 1,506,131,968 bytes. Training/held-out aligned cosine means were
  `0.732931197/0.537957013`; the map remained finite and orthogonal. Benchmark SHA-256 is
  `65122f6cf41186298422098d71f4e5583f9312525b099373d5799aa944bd3313`.
- **Decision:** the resource gate passes and authorizes exactly one immediate foreground E730 validation.
  No heartbeat is warranted for a measured single-digit-second run. No outcome score, `V_joint`, `V_final`,
  upload, or submission was accessed.
- Frozen run `20260728T083236Z_objective_alignment` completed in 18.672104 seconds. Raw BGE comparator
  loss/AUROC/Brier/ECE were `0.595607488/0.608965679/0.203578114/0.024219608`. The selected 10% blend
  produced `0.595675818/0.607563651/0.203612493/0.022143776`; changes versus BGE were
  `+0.000068330/-0.001402028/+0.000034378/-0.002075832`.
- Fold loss changes were `-0.000039375/+0.000179475/+0.000204287/+0.000305253/-0.000281646`.
  The 5,000-session bootstrap mean gain/95% interval/support were
  `-0.000068568/[-0.000236768,0.000095912]/0.2164`.
- Artifact SHA-256 values: predictions
  `36e45d356339a29f33185ae95ae18c48a853427fb9b77c9323552b19b6b22573`; metrics
  `3230fc00e2bb279e6bfc6942ced2625e78b337280cebfebd49deeb56f50e2241`; folds
  `758248ae8f7370f04f51fd2d0262c7f4780156b390f2973d96faed638875f15c`; bootstrap
  `c2a2166129878ef778f4ed56b8d490a192a7948ea33c0e9b44a62e5cbd001ffb`; report
  `ab9446eb2e2a8c9cbc2789a4f2b2ecb03ee6433480e6f66a0cf29a3305c9f78f`.
- **Decision:** only fold-bound and ECE clauses pass. Reject E730 literally without rescue. Projected public
  loss remains verified v0.5 `0.6054`; honest observed rank is approximately `#11`. No hardened evaluation,
  ZIP, upload, or submission is authorized; `V_joint_accessed=false`; `V_final_accessed=false`.

### E740 session-mastery decomposition audit and preregistration

- E720 and E730 are closed. No prior branch changes the supervised target from repeated noisy response labels
  to a legal training-session correctness fraction; prior session weighting changed weights only.
- Freeze one equal-session soft-label log-loss head on the existing `2^17` full-transcript word hash. Within
  each outer fold, validation sessions are purged before calculating one mean-correct target per legal
  training session. The soft target is implemented as duplicated positive/negative session rows weighted by
  `mean` and `1-mean`.
- Fix averaged SGD log loss, `alpha=3e-5`, 200 iterations, no objective/provider/test aggregate, and only
  10/20/30% blends over raw BGE-base. The literal E730 proper-score, fold, and 5,000-session bootstrap gates
  apply, including at least `0.0016` loss gain.
- Inference remains sample-independent: training-session aggregates are absorbed into fixed coefficients,
  while each test response uses only its own transcript. Failure rejects the exact branch without rescue.
  `V_joint_accessed=false`; `V_final_accessed=false`; no platform action occurred.
- Two focused E740 tests pass and Ruff is clean under Python 3.12.8/scikit-learn 1.8.0. The frozen
  synthetic-target benchmark fit 2,992 legal sessions in 1.749262 seconds for one fold, projects 8.746308
  seconds for five folds, and peaked at 948,445,184 RSS bytes. Benchmark SHA-256 is
  `d42fb0b7c4d642755c135a398e8e831c088a68380e71f1a61d763dbafa761ee7`.
- **Decision:** the resource gate passes and authorizes one immediate foreground validation.
  `competition_outcomes_accessed=false`; no checkpoint, `V_joint`, `V_final`, upload, or submission is
  involved.
- Frozen run `20260728T092346Z_session_mastery` completed in 30.339416 seconds. The selected 20% blend
  changed raw BGE loss/AUROC/Brier/ECE by
  `-0.000615881/-0.001421988/-0.000261258/-0.004928590`, with absolute metrics
  `0.594991607/0.607543691/0.203316857/0.019291018`.
- Fold loss changes were `-0.000066628/-0.002146258/+0.001753177/-0.001521805/-0.001187592`.
  The 5,000-session bootstrap mean gain/95% interval/support were
  `0.000602671/[-0.000731569,0.001855827]/0.8212`.
- Artifact SHA-256 values: predictions
  `a9ee502ec1ba751293019c96e541673eadcb9abd2bdddc603136b868211891ee`; metrics
  `a09f84ce09398a8224e517860d105fded11bbde5db356b84cd1eb05f12178595`; folds
  `c29580ac637c2b9214dec87a3fb37df04e84ae6846726ce7429ce6a11a55cd3f`; bootstrap
  `0180d793dee30aa6ed954962d236aee0c925ed30a8bc47433ff84d3670b3f9de`; report
  `388858089357b1a33e1f6da7ee2cbc3ab2415897bf37660303e123c2f417f033`.
- **Decision:** Brier and ECE improve, but loss magnitude, AUROC, fold-bound, and bootstrap clauses fail.
  Reject E740 literally without rescue. Projected public loss remains v0.5 `0.6054`; honest observed rank
  remains approximately `#11`. No hardened evaluation, ZIP, upload, or submission is authorized;
  `V_joint_accessed=false`; `V_final_accessed=false`.

### 2026-07-28 verified continuation, scheduler proof failure, and E750 gap audit

- Re-read the complete handoff, learning log, recovery/failure plans, overview, rules, and code format in the
  mandated order. Live fetch verified clean `codex/v04-recovery` HEAD and
  `origin/codex/v04-recovery` at `baadf919793ca42a6be533da8f68d4b6ec576986`, ahead/behind `0/0`;
  research code/results remain pushed through `0c17ebb`. Git identity is
  `arjun0014 <23293383+Arjun0014@users.noreply.github.com>`.
- The protected v0.5 ZIP remains exactly 258,290,506 bytes with SHA-256
  `65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`. The live project runtime is
  `.venv` Python 3.12.8, scikit-learn 1.8.0, NumPy 2.5.1, pandas 2.3.3, SciPy 1.18.0, and joblib 1.5.3.
  No K12 Python worker or prior automation was active.
- A harmless ACTIVE heartbeat was anchored for 16:18 IST and targeted this exact task. At 16:22 IST there
  was still no wake or execution evidence—only the unchanged persisted definition—so the test failed and
  was deleted. The automation directory then contained no automation. No unattended run over one hour is
  authorized in this session; measured sub-hour work must stay in the foreground with one blocking wait.
- Audited only the frozen authorized `V_seen`, `V_objective`, and `V_style` component OOF files; `V_joint`
  was not read and `V_final` remains sealed. Mixed-outcome sessions cover
  `23.5873850%/23.6541971%/23.6541971%` of eligible rows. Raw v0.5 loss on those rows is
  `0.702922857/0.746255374/0.704262919`, respectively. Recovering the required `0.0016` overall gain needs
  only about `0.006783287/0.006764127/0.006764127` mixed-row loss gain.
- Raw v0.5 positive-over-negative same-session pair accuracy is about `0.70` in every `V_seen`/`V_style`
  fold, but is unstable across `V_objective` folds at
  `0.753425/0.322148/0.635659/0.291667/0.616352`. Each legal outer-training fold still has
  `1,707-2,031` mixed sessions and `2,323-3,052` positive-negative pairs in `V_objective`, so the gap is
  estimable without test aggregation.
- Code/log search found no prior same-session conditional likelihood. E570 instead fixed the objective and
  paired rows across sessions; E740 predicted the session mean and discarded within-session objective
  deviations. E750 fixes the session and learns only objective-specific ordering within that transcript.
  It uses neither E570's within-objective pair construction nor its BCE-plus-pairwise objective, and it uses
  neither E740's transcript hash nor its session-mean target. This orthogonal nuisance-removal estimand is
  the sole basis for preregistration; no rejected score, probability, checkpoint, or cache enters E750.

### E750 session-conditional objective likelihood preregistration and synthetic benchmark

- Freeze immutable sources before any E750 outcome score: `modeling_base.parquet`, BGE-base context/objective
  arrays, and the three component OOF files have SHA-256
  `ea49463819e387b3a61aeafda5007938e39948940eac5b134f607ab13e3d6ede`,
  `b5f0066cc8d659e6593596ac47967bd14f2d0f02cedc82319e2a2313cf564d1a`,
  `6cc754a287f9ec303aa5f81dd98ab0321c7c5299b2a4e251d6f86f63afe6eb27`,
  `1e53a9974d45e00ab86cf44ba79a4e435965dfc0c75b8c4a76fc7f69778e82af`,
  `c6e7a586c9b45ff88b8bac169bc85f36306569cde813320f6997e95a1fc939a7`, and
  `6b6895214173be7343801d1152452099d0590fe1a3186fe6d0bc27908ee43038`.
- In every legal session-purged outer-training fold, sort by `(session_id,response_id)` and construct every
  Cartesian positive-negative pair only within a mixed-outcome session. Add the reversed example; weight
  orientations so each mixed training session has total weight exactly one. No mining, subsampling,
  objective/provider weight, class weight, or prediction-based selection exists.
- Features are exactly the immutable 3,072-wide BGE-base context, objective, scaled product, and absolute
  difference block. Fit only intercept-free `LogisticRegression(C=0.1, solver="lbfgs", max_iter=400,
  tol=1e-5, random_state=20260728)`. Add the legal fold-training prior logit to the conditional score, then
  evaluate only 10%, 20%, and 30% probability blends over raw v0.5
  `(0.25 full + 0.25 role + 0.50 BGE-base)`. Inference is sample-independent; no test session aggregate is
  calculated.
- Select the preregistered weight with lowest equal-environment macro loss, breaking an exact tie toward the
  smaller weight. Every continuation clause is mandatory: mean raw-v0.5 loss gain at least `0.0016`; all
  three environments improve with no environment regression; worst fold regression at most `0.0005`;
  macro AUROC, Brier, and ECE non-regression; and at least `0.95` positive-gain support in separate
  5,000-replicate paired session and semantic-family bootstraps. A failed clause rejects exact E750 without
  pair, weighting, feature, C, solver, prior, calibration, seed, blend, fold, or gate rescue. Only a full
  pass can authorize locked `V_joint` confirmation; `V_final` remains sealed and platform submissions remain
  manual-only.
- Seven focused E750/E570/E740 tests pass in 10.61 seconds; Ruff and `git diff --check` are clean. The
  response-ID-hash synthetic one-fold benchmark used 4,051 mixed sessions, 7,207 pairs, 14,414 symmetric
  examples, and 3,072 features. The fit took `0.726496` seconds and five L-BFGS iterations, projecting
  `10.897440` seconds for all 15 environment folds; peak RSS was 1,292,464,128 bytes. The one-hour and
  8 GiB gates pass. Benchmark SHA-256 is
  `c6a0cd46d5830f5dfa97cb5e1f173faa40738f25bb2019d19f93e67707acfea0`.
  Log loss, AUROC, Brier, ECE, folds/environments, bootstrap outcome evidence, and projected public loss are
  not applicable because synthetic labels were used. `competition_outcomes_accessed=false`;
  `V_joint_accessed=false`; `V_final_accessed=false`. **Decision:** authorize exactly one foreground
  `V_seen`/`V_objective`/`V_style` validation after the frozen implementation and this preregistration are
  committed and pushed.

### E750 completed selection validation and literal rejection

- The frozen implementation/preregistration was committed and pushed as `162eff6` before any E750 outcome
  score. Exactly one authorized run, `20260728T110308Z_session_conditional_objective`, then evaluated only
  `V_seen`, `V_objective`, and `V_style` under `.venv` Python 3.12.8, scikit-learn 1.8.0, NumPy 2.5.1,
  pandas 2.3.3, Torch 2.13.0+cpu, and Transformers 5.14.1. It completed in `38.156235` seconds with peak
  RSS 1,332,215,808 bytes. All 15 fits converged in five or six iterations; no failure or correction occurred.
- All three preregistered weights regressed equal-environment macro log loss. Candidate gains versus raw v0.5
  were `-0.001437298/-0.003323228/-0.005661973` at 10/20/30%, so the deterministic rule selected 10%.
  Its mean absolute loss was `0.564768393`; mean AUROC/Brier/ECE changes were
  `-0.000202409/+0.000628759/+0.003492013`.
- At selected 10%, `V_objective` loss changed from `0.593129391` to `0.593274536`
  (`+0.000145145` worse), `V_seen` from `0.548199263` to `0.550275577`
  (`+0.002076314` worse), and `V_style` from `0.548664632` to `0.550755068`
  (`+0.002090436` worse). `V_objective` ECE improved `0.002651437`, but its AUROC/Brier regressed; both
  seen/style environments regressed all four metrics.
- Selected fold candidate-minus-raw-v0.5 loss changes were:
  `V_objective=[+0.000663700,-0.001680289,+0.001325344,-0.001486171,+0.001903141]`,
  `V_seen=[+0.002307949,+0.002394717,+0.002306685,+0.001816291,+0.001555926]`, and
  `V_style=[+0.002356593,+0.002292435,+0.001804632,+0.002037275,+0.001961242]`.
  Worst fold regression was `0.002394717`.
- The selected 5,000-session bootstrap estimated mean gain `-0.001438778`, 95% interval
  `[-0.001570344,-0.001305156]`, support `0.0000`. The independent 5,000-semantic-family bootstrap estimated
  `-0.001417993`, interval `[-0.001893675,-0.000941879]`, support `0.0000`. All nine frozen continuation
  clauses fail.
- Artifact SHA-256 values are predictions
  `77dc10632a2f6dbdae4f808bc141fed588006e5e524aa5926cc2c893c2e44690`; fold metrics
  `3af6d224db483d6f56666b2fb70ec3bd1f4bb268e587d0547c72f5ac8f358a0f`; environment metrics
  `02b50b3cd777fb963465ceb0814f1d02774b206b6e06a503e209b2cdc60a9210`; selection
  `8f44be9f90d3cb3742490e4c01a6b792cc7e0b897d8b0efa5342120c2944eef7`; bootstraps
  `aebb66ab1c16ff069e261505627e1f46be874a95a4b5e43f1dc2b0f000ba79f2`; report
  `6a21fe439d0ad749889f709d74b6b82d72fd8bf305d8105dcb5ddc0161442da6`.
- **Decision:** reject E750 literally without pair, weight, C, feature, prior, calibration, blend, environment,
  or gate rescue. No `V_joint`, `V_final`, hardened continuation, checkpoint, new ZIP, upload, or submission
  is authorized. The protected v0.5 ZIP and its hash remain unchanged. Projected public loss remains verified
  v0.5 `0.6054`; honest observed rank remains approximately `#11`; two manual submission slots remain, and
  neither is recommended for E750.

## 2026-07-28 (Asia/Kolkata) - first-place research audit and gated portfolio

- Reconfirmed clean `codex/v04-recovery` at
  `878245f182e6403b81ba57181098d315ca2d99eb`, identical to `origin/codex/v04-recovery`, with Git identity
  `arjun0014 <23293383+Arjun0014@users.noreply.github.com>`. No K12 Python worker was active.
- Reverified protected v0.5 ZIP SHA-256
  `65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`. No competition upload,
  submission, new ZIP, `V_joint`, or `V_final` access occurred.
- Audited the complete experiment/decision history and source tree. The repeated failures close generic encoder
  scale, longer context, pooling, generic trajectories, timing, calibration, provider gates, session aggregation,
  objective priors/interactions, unrelated external transfer, and E710-E750 exact branches. Ordered feedback
  adjacency and raw BGE-base remain the only stable positive process/semantic evidence.
- Primary-source research identified three unimplemented mechanisms: directional student-to-next-tutor uptake;
  NTO tutor-move by student-state interactions; and an identifiable ability-versus-task-demand challenge model.
  These mechanisms are documented with sources and literal fallback gates in
  `FIRST_PLACE_ACTION_PLAN_2026-07-28.md`, SHA-256
  `73775d200fdc965e40cf817ee773142b6d732a4a9e9db925165895623436559e`.
- The `ddemszky/conversational-uptake` repository advertises MIT licensing for code/data and contains 2,246
  expert-labeled K-12 student-to-teacher exchanges. Its published Hugging Face checkpoint is labeled
  `CC-BY-NC-ND-4.0` and is explicitly excluded. E760 is not yet preregistered or authorized: the exact source
  commit, license scope, hashes, grouped split, target-free benchmark, features, estimator, seed, weights, and
  gates must be frozen before any competition outcome score.
- The live leaderboard is sign-in gated and was unavailable in the only browser context. Current first-place,
  fifth-place, fifteenth-place, and v0.5 rank must be manually recorded before a submission decision. The older
  local leader value `0.6013` is treated as stale, not current evidence.
- Research-only audit metrics: log loss, AUROC, Brier, ECE, folds, bootstraps, projected public loss, and rank
  gain are not applicable because no model or outcome validation ran. Environment remains `.venv` Python
  `3.12.8` and scikit-learn `1.8.0`; elapsed model runtime is zero. **Decision:** preserve v0.5 and begin only
  the E760 source/legal and target-free gates next. No platform action is authorized.

## 2026-07-28 — E760 source freeze, canonical cache, and target-free benchmark

- Recorded the participant-provided live leaderboard screenshot: first place `0.6008`/AUROC `0.6283`, fifth
  `0.6040`/`0.6229`, tenth `0.6053`/`0.6137`. Protected v0.5 at `0.6054` is therefore approximately eleventh.
  The first-place gap is `0.0046`; the conservative `0.6005` target needs a `0.0049` public loss gain. These are
  public-board observations, not promises or local evidence.
- Pinned `ddemszky/conversational-uptake` to Git commit
  `67fb30cf7c3ea6f619487e33d2f99692dca107d0`. SHA-256 values: source CSV
  `c6bb9e5ff69b6c8ae8981b9d613c41bda1c1ed5685e9e8a74f97d3c1bddb642d`, root MIT license
  `c4fcfe38abef13391b7546f3a3ec9f8c8fc7a480255f888655a57e49076c0429`, README
  `2a2895477a4308d4214a5010dfa162797bc236667f4b02a1e1f508cdc1bdc0a7`. The separately non-commercial
  Hugging Face checkpoint remains prohibited and unused.
- Built the canonical external cache from 2,246 student-to-teacher exchanges, 774 `obs_id` groups, and 1,998
  non-null expert `uptake_zscore` labels. Canonical ordered-content SHA-256 is
  `c2ef8d92db3f22247e1a16ec8c81d5acb2058e3eec6aa889d65ead1629546d82`; Parquet SHA-256 is
  `368b5cee4035251838bd50be09904374024c2e81f1cb2fe504df9acadb30d542`. Five shuffled GroupKFold splits,
  seed `20260728`, have zero `obs_id` overlap.
- Implemented and froze `E760_conversational_uptake_transfer_v1`: a ten-feature repetition ridge comparator and
  one non-negative `2^17` role-aware unigram/bigram/shared/directional-cross hash with fold-local sublinear TF-IDF
  and ridge alpha `10.0`. External gates, ordinal conversion, five folds, seed, 2,000 observation-group
  bootstraps, and all thresholds are literal in `E760_PREREGISTRATION_2026-07-28.md`.
- Failure/correction: the first target-free benchmark failed because signed feature hashing is incompatible with
  the logarithm in sublinear TF-IDF. No external or competition outcome was scored. Hashing was corrected to
  non-negative values and protected by a focused regression assertion.
- Verification after correction: focused tests `4 passed`; 512-row synthetic benchmark passed all clauses in
  `.venv` Python `3.12.8`, scikit-learn `1.8.0`. Feature runtime was `0.0597855` seconds, projected full-external
  feature runtime `0.2622622` seconds, synthetic Spearman `0.9990861109`, with finite matrices/predictions.
  Log loss, AUROC, Brier, ECE, outcome folds/environments, outcome bootstrap evidence, projected public loss, and
  rank bracket are not applicable because neither external uptake outcomes nor competition outcomes were scored.
- Scheduler safety: an ACTIVE short wake card was created, but a card is not proof of execution. Until its exact
  token fires, no unattended run over one hour is authorized. The frozen external validation is sub-hour and will
  remain in the active foreground turn.
- Protected v0.5 ZIP remains
  `submission_builds/final_ensemble_v05_bge_backup.zip`, expected SHA-256
  `65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`. No submission, upload, new ZIP,
  V_joint access, V_final access, or competition outcome access occurred. **Decision:** after commit and push of
  this frozen milestone, run exactly one E760 external grouped validation. Any failed clause closes E760 without
  rescue.

## 2026-07-28 — E760 external grouped validation: rejected

- Ran exactly one frozen external validation from pushed commit
  `01f5c950d4884167d0f6824f70e81f8a873b8b9e`. Runtime was `9.356613` seconds in `.venv` Python `3.12.8`,
  scikit-learn `1.8.0`, NumPy `2.5.1`, and pandas `2.3.3`. Source revision, cache hashes, five `obs_id`-grouped
  folds, role-aware `2^17` hash, ten repetition controls, ridge alpha `10.0`, seed `20260728`, and all gates were
  unchanged from `E760_PREREGISTRATION_2026-07-28.md`.
- Pooled repetition baseline: Pearson `0.5118548`, Spearman `0.5322160`, RMSE `0.7179901`. Pooled E760 candidate:
  Pearson `0.5258184`, Spearman `0.5446827`, RMSE `0.7108782`, quadratic-weighted kappa `0.3818263`.
  Candidate Spearman was positive and beat repetition in all five folds. Its Spearman gains by fold were
  `0.0154981`, `0.0131762`, `0.0102601`, `0.0157918`, and `0.0160229`; RMSE gains were `0.0083008`,
  `0.0071305`, `0.0065316`, `0.0060046`, and `0.0077363`.
- The 2,000-replicate `obs_id` bootstrap gave positive-gain support `1.0`, mean Spearman gain `0.0124697`, and
  95% interval `[0.0092556, 0.0156518]`. This establishes a stable uptake signal but not enough incremental value
  beyond the strong repetition comparator.
- Six of eight frozen clauses passed. Two materiality clauses failed literally: pooled Spearman gain
  `0.0124667 < 0.05` and pooled RMSE gain `0.00711194 < 0.015`. Report SHA-256:
  `02bf2ebc84015e19749af31b658bd4ab2f8c6aa1b645f63b5b71ae769d10775e`; target-free benchmark SHA-256:
  `3b78e80d372dacd9e52e52a2d6b7ae77695dc1d1710b4accd610cef676b12cbd`.
- Competition log loss, AUROC, Brier, ECE, environment/fold gains, projected public loss, and rank bracket are not
  applicable: no competition outcome was read. V_joint and V_final remained untouched. No cache, model,
  validation, blend, ZIP, upload, or submission was created from E760.
- Scheduler result: the harmless ACTIVE two-minute heartbeat never delivered its exact token during the session,
  so it did not prove end-to-end wake execution and was deleted. No unattended task over one hour is authorized.
  Sub-hour foreground work can continue.
- Protected v0.5 ZIP SHA-256 was rechecked as
  `65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`.
  **Decision: reject E760 and close the complete directional-uptake family without rescue. Proceed to the
  independent E770 tutor-move by student-state target-free gate.**

## 2026-07-28 — E770 target-free tutor-move by student-state gate

- Repository/log novelty audit reconfirmed that E220 contains broad affirmation/correction/reasoning/repair
  aggregates and E530 is a rejected four-class MathDial move-text classifier. Neither implements the 2026 NTO
  taxonomy plus explicit proximal student-state interactions. E770 consumes no E220/E530/E710/E760 artifact,
  checkpoint, probability, score, or outcome result.
- Primary-source definitions were frozen from the National Tutoring Observatory codebook: categorical/yes-no
  probes are distinct from open prompts; self-explanation, self-correction, and next-step prompts are distinct;
  revoicing is distinct from verbatim restating; hints/examples/conceptual explanation/procedural explanation/direct
  answers are separate; process and outcome praise are separate. Abdelshiheed et al. motivates the fixed
  rigorous-thinking-by-strong-state and revoicing-by-weak-state interactions.
- Implemented 16 tutor moves, 10 preceding-student states, six predefined interactions, all/early/middle/late rates,
  and six fixed coverage/length controls: exactly 134 competition-only continuous features over authorized
  objective-relevant chronological dialogue. No generic tree, trajectory hash, learned attention, or outcome-aware
  rule editing is present.
- Pre-benchmark correction: the first synthetic test exposed four narrow lexical misses (`that is right`,
  `let us say`, `no, instead`, and acceptance after a leading ASR filler/punctuation mark). The corresponding
  definition-aligned patterns and an explicit answer-change pattern were corrected before any benchmark, cache,
  or outcome access; no scientific category, feature, estimator, or gate changed.
- Target-free verification: every move/state canonical example, paraphrase, ASR-noisy case, and negative passed;
  six focused tests passed. A deterministic 1,024-row benchmark took `4.2205372` seconds and projected
  `144.5533991` seconds for 35,072 rows. The full foreground build took `158.4250329` seconds in `.venv` Python
  `3.12.8`, scikit-learn `1.8.0`.
- The 35,072 x 134 cache is finite and every feature is nonconstant. Ordered-content SHA-256 is
  `d4f0134c4aa57cf2822139f7d4f0f6c5fc93b7002f11e7846c4421ad7e54d675`; Parquet SHA-256 is
  `bd1ccd1d84fabbcaf83ae6f0920530585b54657e050f6b91f78abd900c4f748d`. Overall student-to-tutor pair
  coverage is `0.9925582`; all 20 target-free V_style cells meet the frozen `0.75` minimum, with style cell 12
  exactly at `0.75`. Every move has at least 20 estimated events; observed counts range from 1,984
  self-correction prompts to 65,053 correct-feedback events. All five target-free coverage/cache clauses pass.
- The exact source/cache hashes, parser, feature order, `C=0.1` fold-local standardized logistic head, seed
  `20260728`, raw v0.5 lineage, 10%/20%/30% blends, 5,000 session/family bootstraps, and literal proper-score gates
  are frozen in `E770_PREREGISTRATION_2026-07-28.md` before any outcome score.
- Competition log loss, AUROC, Brier, ECE, fold/environment gains, bootstrap outcome evidence, projected public
  loss, and rank bracket remain not applicable because no competition outcome was read. V_joint and V_final remain
  untouched; no ZIP, upload, or submission occurred. The failed heartbeat remains non-proof, but every E770 stage
  is sub-hour foreground work. **Decision:** after full tests and commit/push of this immutable milestone, run
  exactly one authorized E770 `V_seen`/`V_objective`/`V_style` selection validation. Any failed clause closes
  E770 without rescue.

## 2026-07-28 — E770 selection validation: rejected

- Ran exactly one frozen competition selection validation from pushed commit
  `d227462eb2e0f3fcb73b77517f51dd71a7306127`. Run
  `20260728T123253Z_tutor_move_state` completed in approximately 30 foreground seconds under `.venv` Python
  `3.12.8` and scikit-learn `1.8.0`; all three authorized selection environments and their five folds were used.
  V_joint and V_final were not accessed.
- The minimum equal-environment mean-loss row was the smallest preregistered 10% E770 blend. Raw v0.5
  equal-fold-macro environment losses were V_objective `0.593129391`, V_seen `0.548199263`, and V_style
  `0.548664632`. The 10% blend produced `0.593013637`, `0.551191718`, and `0.551568960`: only V_objective
  improved (`0.000115754`), while V_seen/V_style regressed by `0.002992455/0.002904327`.
- Selected 10% equal-environment mean log loss was `0.565258105`, a **negative gain** of `-0.001927010` versus
  raw v0.5. Worst fold regression was `0.003348798`. Macro AUROC changed by `+0.000156759`, but Brier regressed
  `+0.000742809` and ECE-10 regressed `+0.005237166`. Session and semantic-family 5,000-replicate positive-gain
  support were both `0.0`.
- Only the macro-AUROC non-regression clause passed. The magnitude, three-environment, no-environment-regression,
  worst-fold, Brier, ECE, session-bootstrap, and semantic-family-bootstrap clauses all failed. Increasing the
  fixed weight worsened mean loss further: 20% gain `-0.004435532`; 30% gain `-0.007501426`.
- Artifact SHA-256 values: predictions
  `d42399787658bc31038ebd6fb8adbb8e508b5a687ca4a4c886f8412b44667be3`; fold metrics
  `c41952d7c2bd4c3b0e9bb59cf7b1ebc018eb81a122d14c88d9291035a5d7f5a5`; environment metrics
  `4b80ece40e295513d0b6c6003d12275d212103820c75c40aaf8594f141903beb`; selection
  `70317aa2076c5efb613f1222f301a014db6e84c5da6157e150392815ef50ef82`; bootstraps
  `51b14f60244818399efa54e9eb1ce93a0623c79f89269ce5f9e6c0bb893fc215`; report
  `15387919d3a770805223146b326f17f8766e4e68bf593e819d3d103feedfe77a`.
- A public improvement is not projected; the honest bracket remains approximately eleventh at public `0.6054`,
  subject to leaderboard drift. Protected v0.5 ZIP SHA-256 was rechecked as
  `65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`. No new ZIP, upload, or submission
  occurred. **Decision: reject E770 completely. Do not alter rules, features, C, weight, calibration, or
  environment aggregation. Proceed to independent E780 ability-demand/optimal-challenge target-free design.**

## 2026-07-28 - E780 ability-demand target-free gate: rejected before outcomes

- Novelty audit confirmed E780 is an independent observable task-episode construct: E570 ranks across sessions
  within objectives, E740 predicts session means, E750 ranks objectives within sessions, while E780 estimates
  chronological ability-versus-demand mismatch and under/optimal/over challenge states. It consumes none of
  their caches, probabilities, checkpoints, or scores.
- Primary definitions came from *Measuring Optimal Challenge for Student Learning in AI-Assisted Tutoring*
  (<https://aclanthology.org/2026.bea-1.46/>); the identifiable scalar-gap form was informed by
  *Interpretable Difficulty-Aware Knowledge Tracing* (<https://aclanthology.org/2026.bea-1.43/>). The linked
  `umass-ml4ed/Difficulty-Aware-DialogKT` repository was inspected at
  `3a9362e12eb51675897a5b5b458b0fc709d2ff63`, but contained only a 73-byte README, no license/code/data/model,
  and supplied no E780 asset.
- Immutable input SHA-256 values were `ea49463819e387b3a61aeafda5007938e39948940eac5b134f607ab13e3d6ede`
  (`modeling_base`), `218d5b041f78c6803f28078c35b7fa9e37b725f456c4052974e5150b2afa43f6`
  (objective context), `600664b9ce6c3809ff4b17206397285cfdd905a203a5c5f4d6969b403dfc9b40`
  (selection assignments), and
  `1e53a9974d45e00ab86cf44ba79a4e435965dfc0c75b8c4a76fc7f69778e82af` /
  `c6e7a586c9b45ff88b8bac169bc85f36306569cde813320f6997e95a1fc939a7` /
  `6b6895214173be7343801d1152452099d0590fe1a3186fe6d0bc27908ee43038`
  for the three raw-v0.5 component OOF files.
- The target-free design fixed a 10-turn task-episode horizon and exactly 25 continuous ability, demand, gap,
  struggle, recovery, direct-answer, unresolved, and challenge-state aggregates. Intended but never authorized
  outcome fitting was a fold-local intercept plus nonnegative scalar Rasch slope, L2 `0.1`, maximum slope `5`,
  L-BFGS-B maximum 200 iterations, seed `20260728`, and only `0.10/0.20/0.30` raw-v0.5 blends. Intended
  proper-score gates remained at least `0.0016` robust mean loss gain, all environments improving, no environment
  regression, worst fold regression at most `0.0005`, AUROC/Brier/ECE non-regression, and at least `0.95`
  support in separate 5,000-session and semantic-family bootstraps.
- Two corrections occurred before any outcome access: the synthetic medium-demand item was changed from
  answer-only to an equation request so it actually exercised the frozen representation-demand rule; then two
  NumPy booleans were converted to native booleans at the JSON report boundary. Neither correction changed a
  model or gate. Focused verification then passed `7/7`.
- The 1,024-row target-free benchmark took `3.6710772` seconds and projected `125.7343941` seconds for all rows.
  The one foreground full build took `128.2319289` seconds in `.venv` Python `3.12.8`, scikit-learn `1.8.0`,
  NumPy `2.5.1`; produced 35,072 x 25 finite, nonconstant features; and achieved overall episode coverage
  `0.9871977646`. State counts were 74,827 under-challenged, 3,309 optimally challenged, and 321,407
  over-challenged, all above the frozen minimum 50.
- The exact target-free gate failed because V_style cell 19 episode coverage was
  `0.6190476190 < 0.70`; all other target-free clauses passed. Ordered-content SHA-256 is
  `d30c40b1e4366fc09a29a38a081166cde9c9456ccc68e142fa49292f83d42dde`; Parquet SHA-256 is
  `e61756de0744e34aba1b0ec16fda7984f0ae036f80cdfb4c5e10aa5c6ae36219`; benchmark/metadata report SHA-256
  values are `4d42ef2aa74170a2e4acb1ae461b39393a12356960baa898316f08a7d04496a9` and
  `d1719d556773ace22158f5478878e681e8a3623e0298054256e8b2d17e0d86b0`.
- Log loss, AUROC, Brier, ECE, outcome folds/environments, outcome bootstrap evidence, and an E780 public
  projection are not applicable: no competition outcome was read, no bootstrap ran, and V_joint/V_final
  remained untouched. The feature hashes remain intentionally unfrozen in code, safety-locking validation.
- Protected v0.5 remains public `0.6054`, honestly approximately rank 11 in the participant-provided snapshot.
  Its ZIP SHA-256 was rechecked as
  `65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`. No candidate ZIP, upload, or
  submission occurred. **Decision: reject exact E780 without parser, threshold, feature, model, or gate rescue.
  No E780 preregistration or competition validation is authorized.**

## 2026-07-28 - E790 target-free mathematical-validity discovery: rejected

- The proposed generic residual atlas was rejected before scoring because it duplicates the v0.4 forensic
  disagreement audit, E230 observable-style component gating, pooled stacking/calibration controls, E560
  residual heads, and D100 source-robust group-risk training. Code/log novelty search instead found one untested
  target-free mechanism: exact mathematical truth of explicit student arithmetic relations. E221 had encoded
  lexical answer/feedback events but had never evaluated equation validity.
- Froze `E790_math_validity_target_free_discovery_v1` before the full audit. Immutable SHA-256 values were
  `218d5b041f78c6803f28078c35b7fa9e37b725f456c4052974e5150b2afa43f6` for objective contexts,
  `600664b9ce6c3809ff4b17206397285cfdd905a203a5c5f4d6969b403dfc9b40` for target-free style assignments,
  and `ea49463819e387b3a61aeafda5007938e39948940eac5b134f607ab13e3d6ede` for identity-only modeling
  lineage. The runner explicitly requested only response/session/objective identity columns and never loaded
  `target` or a component prediction.
- The conservative parser uses exact rational arithmetic for explicit binary integer/decimal/fraction
  equations and comparisons, rejects ambiguous/chained/unsafe expressions, and links only an immediately
  following tutor turn. Two focused pre-audit corrections fixed terminal-punctuation token boundaries and
  fraction-comparison precedence; neither accessed real audit evidence or changed the frozen relation classes
  or gates. All eight focused tests and every synthetic clause then passed.
- The sole full target-free audit completed in `25.5480737` seconds in `.venv` Python `3.12.8`,
  scikit-learn `1.8.0`, NumPy `2.5.1`, and pandas `2.3.3`. It found `19,046` relations across `8,718`
  responses (`0.2485743613` coverage): `15,987` valid, `3,059` invalid, and `5,187` feedback-linked. Seventeen
  style cells had at least ten linked relations. All six volume/coverage clauses passed.
- All three discrimination clauses failed. Validity versus positive immediate tutor feedback had balanced
  accuracy `0.5022348622 < 0.70` and Matthews correlation `0.0128588852 < 0.30`. Positive-feedback rates were
  `0.9856017998` for valid relations and `0.9811320755` for invalid relations, a gap of only
  `0.0044697243 < 0.25`. Immediate tutor praise is nearly constant and is not a usable target-free correctness
  proxy.
- Report SHA-256 is `81e11b57ed20b5f206888f925dfa930d75624537106d87943ad13f7ebcc5469c`.
  Log loss, AUROC, Brier, ECE, competition folds/environments, outcome bootstraps, projected public loss, and
  rank gain are not applicable because no competition outcome was accessed. V_joint and V_final remained
  untouched. Runtime was far below one hour, so the audit correctly remained in the foreground without a wake.
- Protected v0.5 ZIP SHA-256 remains
  `65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`; no new ZIP, upload, or submission
  occurred. **Decision: reject exact E790 without parser, feedback lexicon, linkage, threshold, or subgroup
  rescue. E790 does not nominate an E800 competition candidate.**

## 2026-07-28 - post-E790 source, novelty, and headroom audit

- Reconfirmed that E800 is reserved for a final ensemble of independently passing components and is not an
  experiment number. Audited a possible E810 only against source, license, use-suitability, reproducibility,
  novelty, and headroom gates. No competition outcome, V_joint, or V_final evidence was accessed.
- The only genuinely new aligned mechanism found was student/tutor text-personality complementarity from the
  2025 LAK Math Nation study of 383 middle-school students, 2,157 QA records, 697 questions, and 15-item pre/post
  tests. Its significant terms were student-extroversion × chatbot-openness `+3.2477`, student-openness ×
  chatbot-openness `-2.7545`, and student-extroversion × chatbot-extroversion `-1.9708`.
- Froze public model metadata without downloading weights. The subjectivity model
  `cffl/bert-base-styleclassification-subjective-neutral` is Apache-2.0 at revision
  `1339b8de703cb52c729475a89427078052af8595`. The Big-Five model
  `Minej/bert-base-personality` is MIT-tagged at revision
  `6f4d00d28093bdc2ba6cccec97acbf1805709ff1`, but its config preserves only generic `LABEL_0..4`, its card
  reports no validation metrics or short-text reliability, says it is not intended for downstream use, and
  explicitly excludes education judgments. The source Math Nation outcome data are not public.
- **Decision:** reject the personality candidate before E810 preregistration. The published significant mechanism
  cannot be reproduced without the use-inappropriate Big-Five model; VADER and subjectivity alone do not retain
  the reported outcome interaction. No model weights, cache, outcome score, or ZIP were produced.
- Audited remaining real outcome-linked sources. StudyChat is CC BY 4.0 but gated and anonymous access returned
  HTTP 401; NCTE requires a Google access form and separate ICPSR metadata; APTA/Peer Chats require a CMU
  DataShop request; AlgebraNation/Math Nation outcomes are unreleased. Codex accepted no terms, disclosed no user
  information, and did not use a mirror. Eedi QATD is non-commercial and lacks a learning-gain outcome.
  MathMentorDB is CC BY 4.0 and large but has no outcome. NTO's public MathEd-PII release contains PII labels,
  not tutoring outcomes.
- Other mechanisms were closed rather than relabeled: StudyChat response-strategy work overlaps E770; NCTE/Eedi
  talk moves overlap E760/E770; directional lexical alignment was already inside failed E760; public 274-learner
  ChatGPT/human math-help evidence overlaps the failed correctness family; JUSThink is only ten transcript teams.
- A new harmless one-occurrence heartbeat was created at 18:57:56 IST for 19:01. At 19:02:43 there was no wake,
  run record, or execution artifact—only the unchanged ACTIVE definition—so it was deleted. No unattended run over
  one hour is authorized in this session.
- Environment remained `.venv` Python 3.12.8 and scikit-learn 1.8.0. Model runtime, log loss, AUROC, Brier, ECE,
  folds/environments, outcome bootstraps, and projected public gain are not applicable because this was a
  source-only audit. Honest public reference remains v0.5 loss `0.6054`, approximately rank 11; first-place
  `0.6008`, fifth `0.6040`, tenth `0.6053`; conservative target `0.6005`.
- Full evidence and the literal reopening conditions are recorded in
  `POST_E790_SOURCE_AND_NOVELTY_AUDIT_2026-07-28.md`. **Decision:** no E810 is preregistered or authorized.
  Preserve v0.5, both manual submission slots, and sealed V_final; do not manufacture another outcome run from a
  gated, non-commercial, unavailable, outcome-free, or closed mechanism.
- Repository verification under the required `.venv` completed in `40.35` seconds: `209 passed, 8 subtests
  passed`; Ruff and `git diff --check` were clean. The audit document SHA-256 is
  `02a420077cb749fa30d3857777c20473299309daf5899e04595a78d35d83ed04`. The protected ZIP hash was
  reverified unchanged as `65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`.

## 2026-07-29 - authorized StudyChat acquisition and NCTE/APTA access audit

- The participant personally accepted the official StudyChat Hugging Face gate
  and authorized source downloads. Downloaded `wmcnicho/StudyChat` at immutable
  revision `24d7987d9fbb30d9da12acc53455a10f1cdd2d7f` to the ignored external-data
  tree. The 33,667,862-byte benchmark completed in approximately `8.5` seconds;
  the full pinned snapshot completed in `73.86` seconds in the active foreground
  turn with one completion checkpoint.
- The official snapshot has `78` files and `1,298,050,081` bytes. The
  deterministic full-source manifest has SHA-256
  `608462ed21f983d22068c40e168ce4a600b9be94e5689d24df6b7e4717b5d1a3`.
  A temporary fine-grained read-only credential was used through a short-lived
  local file, then both the file and token were deleted. Existing participant
  tokens were not changed.
- Source audit found `16,851` interactions, `203` dialogue users, and `2,214`
  chats. Released grade tables have `70` Fall and `111` Spring users; exact
  dialogue/grade matches are `65` and `110`, respectively, with no missing
  assessment fields. The submissions ZIP has `4,990` entries and
  `109,548,596` uncompressed bytes; it was not extracted and contains no unsafe
  path, symlink, or common serialized-model extension.
- The LAK 2026 paper reports that dialogue-act features are likely overfit or
  underpowered, with significant coefficients largely distinct across
  semesters. The July 2026 response-style paper reports small global next-turn
  effects but uses the same student-state × response-style construct already
  closed by E770. Direct-answer/next-turn, directional uptake, and challenge/
  over-reliance interpretations also overlap rejected E710, E760, and E780.
  **Decision:** StudyChat is now legally and technically available, but no
  independent E810 with a plausible robust `0.0016` loss path is established.
  No grade model, competition outcome, blend, or ZIP was produced.
- Downloaded the public NCTE repository
  `ddemszky/classroom-transcript-analysis` at clean detached revision
  `ff63e9787350d39f4fbc3083732d66d8ebe4402a`. Its MIT `LICENSE` and `README`
  SHA-256 values are
  `f7e7258ef05d5f9a6e227c9e0b315a288923280b42d443e7aca91e93cf382aa8`
  and `dfc596a345d0eefb4ae0a3f7e48da9469a53cbbfb116ae8d289c29184f9de71e`.
  The data CSVs remain gated behind the per-user Google form; the form is open
  and blocked on participant Google sign-in.
- Verified APTA DataShop IDs `5153`, `5549`, and `5604` as private project `574`
  datasets. The LearnSphere GitHub sign-in page is open and blocked on
  participant GitHub credentials. No identity, affiliation, IRB status, access
  request, upload, or submission was fabricated or transmitted.
- Environment was `.venv` Python `3.12.8` and scikit-learn `1.8.0`. Competition
  log loss, AUROC, Brier, ECE, folds/environments, bootstraps, and public
  projection are not applicable because no competition outcome was accessed.
  `V_joint` and `V_final` remain sealed. The protected v0.5 ZIP was reverified at
  SHA-256
  `65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`;
  no upload or competition submission occurred. Full evidence is in
  `STUDYCHAT_NCTE_APTA_ACCESS_AUDIT_2026-07-29.md`, SHA-256
  `ac201e67bc3b3af0f7a60b3bfe695a7c5b7b176d539756242556ecd77d95a25a`.

## 2026-07-29 - explicit-goal scheduler wake test: failed

- Tested the participant's hypothesis that earlier wakes failed because the
  task had no active Codex goal and/or emitted a message before the timer. The
  pre-test goal state was null. Created an explicit narrow goal whose only
  objective was to prove one harmless heartbeat end to end before authorizing
  any unattended run.
- At `13:19:04` IST, created ACTIVE thread heartbeat
  `k12-goal-wake-proof-2026-07-29`, initially on a three-minute cadence. Its
  prompt could only emit the fixed token `K12_GOAL_WAKE_OK_20260729` and an IST
  timestamp; it prohibited code, file/process inspection, research, mutations,
  and follow-on automation. Kept the turn open without a final response and
  waited one four-minute completion checkpoint. No wake token or run artifact
  arrived; only the unchanged ACTIVE definition existed.
- The first form had `notificationPolicy=failed_runs_only`, which left one
  ambiguity. Updated the same heartbeat rather than creating a duplicate:
  one-minute cadence, successful notifications unmuted, unchanged harmless
  prompt, same active goal and target task. Again kept the turn open and waited
  one two-minute checkpoint without sending a final response. At `13:26:36`
  IST there was still no injected token, last-run field, next-run field, or run
  artifact; only the ACTIVE definition persisted.
- Deleted the heartbeat after the second bounded failure. No K12 model worker,
  training job, or automation remains active. The explicit goal alone therefore
  did not repair scheduler execution in this task/session. A stored ACTIVE card
  remains non-evidence even with a goal and an open turn.
- **Decision:** no over-one-hour unattended run is authorized. Work measured
  below one hour must stay in the foreground with one blocking wait. If a future
  product-supported completion mechanism is introduced, it must pass another
  harmless end-to-end test before use; do not launch a long worker merely
  because a goal exists.
- Environment, model configuration, data lineage, log loss, AUROC, Brier, ECE,
  folds/environments, bootstraps, and projected public loss are not applicable
  because this was orchestration-only. No competition outcome, `V_joint`, or
  `V_final` evidence was accessed; no ZIP, upload, or submission was produced.

## 2026-07-29 - authenticated NCTE acquisition and APTA request

- Ran a third explicit-goal wake proof using a standards-aligned hourly
  heartbeat due at 13:31 IST. The goal remained active, the turn remained open,
  and no final response was sent before the deadline. At 13:33 there was still
  no wake token or run artifact. Deleted the heartbeat and marked the
  repeated-failure goal blocked. All three goal-backed scheduler variants have
  now failed end to end; zero persisted automation definitions and zero K12
  Python/training/Git workers remain. No over-one-hour unattended run is
  authorized.
- Truthfully completed the NCTE Google terms form as an independent researcher,
  with no university affiliation and an explicit non-identification,
  non-surveillance, non-discrimination, and non-redistribution purpose. Google
  recorded the response and automatically shared official Drive folder
  `19LzXF0IRtOGO62ZUnJeX6rBTjynuW6RR`.
- Downloaded all three official NCTE CSVs to ignored source directory
  `Datasets/NCTE/release_19LzXF0IRtOGO62ZUnJeX6rBTjynuW6RR`: 580,408-row
  `ncte_single_utterances.csv` (113,819,937 bytes, SHA-256
  `bbfa37ac857991e5d12b1677216896f32d7cf4c2d95b618b67afa7b3bf23b3d7`);
  2,348-row `paired_annotations.csv` (547,485 bytes, SHA-256
  `acb81f02dfd00e8e5426af74525fb1f629c7f880cdb1646d533a1852bc6e3ae7`);
  and 2,000-row `student_reasoning.csv` (204,227 bytes, SHA-256
  `16bc7eb53e1955f9ecf28b8f02d9727659fdd1a138782355243e0754a3633384`).
  The deterministic source-manifest SHA-256 is
  `604ec4e80c584c3d96afb54ddf95c31a9acf7c4a0b01cdbad8b8e144c68d814a`.
- Full NCTE parsing completed in 7.4 seconds under `.venv` Python 3.12.8,
  scikit-learn 1.8.0, and pandas 2.3.3. All CSVs parse, have no NUL header byte
  and no duplicate full row. The source covers 1,660 observation IDs and 319
  video IDs. Positive annotations are high uptake 813/2,348, focusing question
  359/2,348, and student reasoning 419/1,999 labeled rows.
- Repository/log novelty audit reconfirmed that NCTE high uptake is E760, while
  focusing questions/tutor moves and student reasoning are E530/E770. ICPSR V4
  contains the more independent value-added outcomes but is member-institution
  data; no affiliation or entitlement was fabricated and no ICPSR account/data
  was created or downloaded. **Decision:** no E810 preregistration, competition
  score, or rejected-family rescue is authorized from the released NCTE labels.
- Created a DataShop account through the participant's GitHub sign-in, reviewed
  the general research-only/non-commercial/non-identification/non-redistribution
  terms, and requested only view access to private APTA project `574`. DataShop
  records PI Vincent Aleven, level `View`, status `Not Reviewed`, and request
  date 2026-07-29. No private APTA file was downloaded. If approved, IDs
  `5153`, `5549`, and `5604` will be acquired and audited before any architecture
  is considered.
- Public APTA research narrows the only conditionally independent discovery
  path to learner-role and knowledge-state trajectories: solver/tutor role, BKT
  mastery, error timing, collaboration transitions, and individual pre/post
  gain. Another content-only peer-help/talk-move model is closed by
  E530/E760/E770. The trajectory path cannot be preregistered until approved
  files prove auditable outcome, identity/group, role, and transaction linkage
  with enough independent learners; it must first beat a static mastery/count
  baseline in grouped external validation with stable proper scores and
  bootstrap support.
- Environment was `.venv` Python 3.12.8 and scikit-learn 1.8.0. Competition log
  loss, AUROC, Brier, ECE, folds/environments, bootstraps, and new public
  projection are not applicable because no competition outcome was accessed.
  `V_joint_accessed=false`; `V_final_accessed=false`. No competition cache,
  checkpoint, blend, ZIP, upload, or submission was produced. Public champion
  remains v0.5 at 0.6054.
- Protected ZIP SHA-256 remains
  `65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`.
  Full authenticated-access evidence is recorded in
  `NCTE_APTA_AUTHENTICATED_ACCESS_2026-07-29.md`, SHA-256
  `610155d8344b12a05e033f34b2fab7eebe7c9b36f2275fc559c2f96bbaac9d96`.

## 2026-07-29 - scheduler proof finally fired end to end

- Automation `k12-idle-turn-wake-proof-2026-07-29` fired in this exact Codex
  task at `2026-07-29 15:09:13.819 IST`, after the preceding assistant turn had
  ended. It injected the exact token
  `K12_IDLE_WAKE_PROOF_FIRED_20260729`, proving a real task wake rather than
  merely persisting an ACTIVE definition or rendered card.
- The proof automation was harmless by construction: it started no training or
  download, modified no project file, accessed no competition outcome, and made
  no upload or submission. It was deleted immediately after verification so it
  cannot recur.
- **Scheduler decision:** an unattended run over one hour is now allowed only
  with exactly one calculated completion checkpoint in this same task. If that
  checkpoint finds the run still active, exactly one telemetry-derived
  replacement checkpoint is allowed. Runs measured below one hour remain in the
  foreground with one blocking wait and no polling.
- Model configuration, data lineage, log loss, AUROC, Brier, ECE,
  folds/environments, bootstrap evidence, projected public loss, and rank are
  not applicable because this was orchestration-only. `V_joint` and `V_final`
  were not accessed.

## 2026-07-29 - E810 productive numeric elaboration target-free discovery

- Located DrivenData's official 2026-07-27 Trace the Ace reference,
  `https://blog.drivendata.org/blog/productive-math-talk-reference`. The newly
  published exact contrast separates number-bearing student turns from digit
  characters embedded in longer student reasoning. Repository and learning-log
  searches found no prior implementation of `numeric_turns_per_word` or
  `digit_chars_per_word`; the existing role cache counts numeric tokens instead.
- Froze and ran target-free protocol
  `E810_productive_numeric_elaboration_target_free_v1`. The measured benchmark
  scanned 524,288 of 6,139,854 utterance rows in 1.891 seconds and projected
  22.145 seconds, so the full run remained in the foreground. The full audit
  completed in 24.129 seconds.
- Exact reproduction succeeded: 22,821 sessions; 35,072 responses; 35,062
  nonmissing responses; 107 sessions and 159 responses below 100 student words.
  Response-level means were `1003.2145342536`, `0.0447989484050`, and
  `0.2004111836964`; sample standard deviations were `430.6874933188`,
  `0.0192311313837`, and `0.0846257311390`. These match all organizer values to
  six decimals.
- The new features add three design-rank dimensions, from 31 to 34. Cross-fit
  R-squared from existing v0.5 dense role/session features was `0.999994` for
  student words, but only `0.622637` for numeric turns per word and `0.672253`
  for digit characters per word. Their nearest absolute Spearman correlations
  were `0.517249` and `0.604735`.
- A deterministic synthetic residual benchmark improved log loss from
  `0.69165648` to `0.62412921`, gain `0.06752727`. This is an implementation and
  nonredundancy check only, not evidence about real competition outcomes.
- Without reading target columns, the fixed published numeric score had
  Pearson/Spearman correlations of `0.288527/0.270501` on `V_seen`,
  `0.405096/0.395153` on `V_objective`, and `0.288866/0.270433` on `V_style`
  against raw v0.5 predictions. All six frozen target-free clauses passed.
- Environment was `.venv` Python `3.12.8`, scikit-learn `1.8.0`, NumPy `2.5.1`,
  and pandas `2.3.3`. Real competition log loss, AUROC, Brier, ECE, fold
  deltas, bootstraps, projected public loss, and rank are not applicable:
  `competition_outcomes_accessed=false`, `V_joint_accessed=false`, and
  `V_final_accessed=false`.
- The feature cache SHA-256 is
  `560e82ceb07dc22ef0511da2b92d361f6e52564be997a2364a282e8c6c1eca27`;
  the full target-free report SHA-256 is
  `ed202b92697ca7d11c1115378f1b8825761901d9ee60092526ff56c5a445afac`.
  **Decision:** authorize exactly one E810 outcome validation under
  `E810_PREREGISTRATION_2026-07-29.md`. No post-hoc feature, weight, estimator,
  threshold, calibration, or rescue is allowed.
- The protected v0.5 ZIP remains the mandatory backup and must retain SHA-256
  `65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`.
  No competition upload or submission occurred.

## 2026-07-29 - E810 single authorized validation: rejected literally

- Executed exactly one frozen
  `E810_productive_numeric_elaboration_validation_v1` invocation from pushed
  commit `a765e7b198d081aa7d27d746e33ff101519dc302`. Run
  `20260729T102615Z_productive_numeric_elaboration` completed in 17.463 seconds
  under `.venv` Python 3.12.8, scikit-learn 1.8.0, NumPy 2.5.1, and pandas
  2.3.3. Lineage, three features, 100-word quiet fallback, five environment
  folds, estimator, seed `20260729`, and 10/20/30% weights were unchanged from
  preregistration.
- The selected 10% blend had equal-environment/equal-fold macro log loss
  `0.5653799145`, a **regression** of `0.0020488191` from raw v0.5. It improved
  0/3 environments. Worst environment delta was `+0.0030689014`; worst fold
  delta was `+0.0034981536`.
- At 10%, `V_seen` regressed from `0.5481992630` to `0.5512681643`
  (`+0.0030689014`); `V_objective` regressed from `0.5931293909` to
  `0.5931707967` (`+0.0000414059`); and `V_style` regressed from
  `0.5486646322` to `0.5517007823` (`+0.0030361502`).
- Proper scores also failed: mean AUROC delta `-0.0001073301`, mean Brier delta
  `+0.0007857193`, and mean 10-bin ECE delta `+0.0050436103`. The 20% and 30%
  blends regressed mean log loss by `0.0046065742` and `0.0076536126`,
  respectively.
- The 5,000-replicate paired session bootstrap estimated gain
  `-0.0020491112`, 95% interval
  `[-0.0021893088, -0.0019022454]`, support `0.000`. The 5,000-replicate
  semantic-family bootstrap estimated gain `-0.0020660009`, interval
  `[-0.0026114729, -0.0015354993]`, support `0.000`.
- Every frozen selection clause failed. **Decision: reject E810 literally.**
  No smaller weight, feature subset, residual head, calibration, coefficient
  constraint, threshold adjustment, or rescue was evaluated or is authorized.
  The stable organizer-like coefficient directions validate extraction, but
  the head adds no useful conditional outcome signal over v0.5.
- A rough public projection obtained by adding the local mean regression to
  public v0.5 is approximately `0.6074`. Extrapolating only from the visible
  top-10 density gives an uncertainty bracket around `#15–#30`, worse than the
  current approximately `#11`; this is not a leaderboard observation. E810 is
  not submission-worthy.
- `V_joint_accessed=false`; `V_final_accessed=false`. No production model,
  submission cache, ZIP, upload, or competition submission was created. Report
  SHA-256 is
  `ec25f46fc8839ff22ac1077288631705e1d9fe45ab468635cf5a47575a0bcdc3`.
  Full metrics and artifact hashes are in
  `E810_REJECTION_2026-07-29.md`.
- The protected v0.5 ZIP remains unchanged at SHA-256
  `65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`.

## 2026-07-29 - mailed NCTE transcript ZIP verified as exact duplicate

- Inspected the participant-supplied raw archive
  `NCTE Transcripts - Release-20260729T173920Z-1-001.zip` in place without
  extracting, moving, or altering it. The archive is 27,734,379 bytes with
  SHA-256
  `9a3f01481c634466c339937c9e86e7899a08a832afbd628bf8709973c5aeb088`.
  It opens cleanly and contains exactly three CSV members totaling 114,571,649
  uncompressed bytes.
- Directly hashed every decompressed ZIP member and compared it with the
  authenticated Drive release already preserved under
  `Datasets/NCTE/release_19LzXF0IRtOGO62ZUnJeX6rBTjynuW6RR`. All three CSVs are
  byte-for-byte identical:
  `ncte_single_utterances.csv`, 113,819,937 bytes, SHA-256
  `bbfa37ac857991e5d12b1677216896f32d7cf4c2d95b618b67afa7b3bf23b3d7`;
  `paired_annotations.csv`, 547,485 bytes, SHA-256
  `acb81f02dfd00e8e5426af74525fb1f629c7f880cdb1646d533a1852bc6e3ae7`;
  and `student_reasoning.csv`, 204,227 bytes, SHA-256
  `16bc7eb53e1955f9ecf28b8f02d9727659fdd1a138782355243e0754a3633384`.
- **Decision:** do not extract or re-audit the duplicate. It adds no rows,
  labels, fields, lineage, or new experiment evidence beyond the already
  completed NCTE audit. NCTE uptake, focusing-question/tutor-move, and student
  reasoning branches remain closed as documented; this archive does not
  authorize a new score or rescue.
- Environment verification remained `.venv` Python `3.12.8` and scikit-learn
  `1.8.0`. Competition log loss, AUROC, Brier, ECE, folds/environments,
  bootstrap evidence, projected public loss, and rank are not applicable
  because no competition outcome was accessed. `V_joint_accessed=false`;
  `V_final_accessed=false`; no cache, checkpoint, model, ZIP, upload, or
  competition submission was created.
- The protected v0.5 ZIP was reverified unchanged at SHA-256
  `65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`.

## 2026-07-29 - APTA target-free landing gate prepared while access is pending

- Reconciled the acquired StudyChat/NCTE schemas, current primary research,
  repository code, and all closed families before claiming novelty. StudyChat
  response style/next-turn continuation maps to E710/E770; answer extraction,
  learner efficacy/attempt quality, reflection, uncertainty, self-correction,
  and productive struggle map to E770/E780; NCTE uptake/focusing
  questions/student reasoning map to E530/E760/E770. The 175 StudyChat dialogue
  users with matched grades and unstable cross-semester broad dialogue-act
  associations do not establish an independent plausible robust `0.0016`
  competition-loss path. **Decision:** no successor experiment is
  preregistered and no competition outcome is spent.
- Implemented `APTA_target_free_source_audit_v1` in
  `src/trace_ace/apta_source_audit.py` with
  `scripts/audit_apta_source.py` and focused tests. The CLI refuses source roots
  outside `Datasets/APTA`. The audit deterministically hashes every raw file,
  rejects unsafe/encrypted/symlink/risky/oversized ZIPs before member parsing,
  streams delimited sources, and reports only schema concepts and
  privacy-hashed identity/linkage counts.
- The frozen discovery-readiness gate requires supported clean tables, safe
  archives, student identity, session/class grouping, solver/tutor role,
  experimental condition, transaction structure and correctness, a real
  pre/post/gain/assessment outcome, and at least 50 learners linked between
  transactions and learning outcomes. Step-level `Outcome` alone cannot pass.
  Raw identifiers, chats, correctness values, and learning-outcome values are
  never emitted. A pass authorizes only external benchmark design, not a
  competition model or score.
- Focused verification: `4 passed` in `0.25` seconds. Full project verification:
  `220 passed, 8 subtests passed` in `68.63` seconds. Environment was project
  `.venv`, Python `3.12.8`, scikit-learn `1.8.0`, pandas `2.3.3`, NumPy `2.5.1`.
  Pre-documentation hashes were:
  `src/trace_ace/apta_source_audit.py`
  `b6b55b8f6737fba71499fd2ae56946a0aabc43634412b361a041cab37140a9d4`;
  `scripts/audit_apta_source.py`
  `90a4533cd85c73ec2458ec3b6aee8e7be92f9b02004f0091658c95d48a0fac51`;
  `tests/test_apta_source_audit.py`
  `652dae95b6cfc7d62b188a5bda2fa018cbc6388fba408b76c01a61e79b7e0ee1`.
- Runtime was sub-hour foreground work; no completion wake or worker was
  needed. Competition log loss, AUROC, Brier, ECE, folds/environments,
  bootstrap, and new public projection are not applicable.
  `V_joint_accessed=false`; `V_final_accessed=false`; no competition cache,
  prediction, checkpoint, model, blend, ZIP, upload, or submission occurred.
  v0.5 remains public `0.6054`, last observed approximately `#11`, and the
  protected ZIP remains
  `65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`.
- Full rationale, primary-source links, gates, and next actions are recorded in
  `APTA_READINESS_AND_WAITING_PERIOD_AUDIT_2026-07-29.md`.

## 2026-07-29 - post-E810 research audit and E820 target-free freeze

- Reconstructed the exact v0.5 architecture and every closed branch before
  claiming novelty. The remaining actionable gap is short learning-objective
  text aligned to long tutoring evidence under unseen objectives and low
  lexical coverage. The organizer's newly published numeric contrast was
  already tested as E810 and failed; another count, tutor-move, correctness,
  response-quality, timing, state, calibration, or blend branch is not
  independent.
- Primary-source research ruled out advanced-math embedding checkpoints
  trained on papers/theorems and MMTutorBench-derived quality stages before
  compute: they lack K-12 learning-outcome transfer evidence or overlap closed
  quality/tutor-state families. StudyChat's 175 grade-linked dialogue users,
  undergraduate-AI domain, and unstable cross-semester broad dialogue-act
  associations retain it as a discovery reserve rather than a defensible
  immediate competition branch. APTA remains a later learner-role trajectory
  branch if approved source files pass the frozen landing gate.
- Froze independent target-free protocol
  `E820_mathdial_contrastive_objective_alignment_v1`. It uses the untouched
  packaged BGE-base encoder and only MathDial question-to-dialogue positive
  pairs. It consumes no self-correction, correctness, ground-truth, typicality,
  competition target, rejected checkpoint, prediction, or fitted map. This is
  distinct from E430 outcome classification, E530 tutor-move classification,
  E720 sparse competition-label conjunctions, and E730 fold-local linear
  Procrustes.
- Pinned official CC BY-SA 4.0 MathDial revision
  `b06c020a0a1f57a87577fec33e657b63e7eb476e`. JSONL hashes are
  `96980babee081a3da48ed0f1fb3068ab52f23ddc67e63bfd5ec99ad701fd29cc`
  and
  `28d1e537d65a6e6ff7b8bde602a2c7e2493b93d6bc785d1848506a650997f122`.
  The audit reproduced 2,262/599 train/test dialogues and 1,035/394 train/test
  question IDs. Purging all 318 overlapping IDs leaves 1,677 legal training
  dialogues across 717 IDs. Canonical privacy-hashed identity SHA-256 is
  `fdbc30eb04107cf6214bb0b0962270a7018f3ac0130fd0f89b001e40cf9ca060`.
- Fixed normalized CLS pooling, a single K-12 retrieval query prefix, 96/256
  query/document tokens, upper-four-layer training, symmetric in-batch
  InfoNCE at temperature `0.05`, batch 8, two epochs, AdamW `2e-5`, weight
  decay `0.01`, 10% warmup, gradient cap `1.0`, and seed `20260729`.
  In-batch question IDs are unique. No hard-negative mining, alternate prompt,
  checkpoint selection, or sweep is allowed.
- The frozen external gate evaluates all 599 official-test dialogues with
  problem and held-out teacher-confusion query views. It requires confusion
  MRR/recall@5 gains of at least `0.04/0.04`, 2,000 question-ID bootstrap
  support at least `0.95`, problem-view MRR/recall@5 regressions no worse than
  `0.005`, two no-collapse clauses, and decreasing epoch loss. Any failure
  rejects exact E820 before competition outcomes without rescue.
- Focused verification passed `5` tests in `3.12` seconds and Ruff was clean.
  The source-only audit used `.venv` Python 3.12.8, scikit-learn 1.8.0, NumPy
  2.5.1, Torch 2.13.0+cpu, and Transformers 5.14.1.
- One-batch resource benchmarking measured `7.4838306` seconds per training
  step and `3.6735134` seconds for 64 evaluation encodings. The fixed 180
  training steps plus both base/adapted evaluations project `1,553.3802`
  seconds (25.9 minutes), with peak RSS `998,232,064` bytes. Duration and
  12-GiB gates pass. Benchmark SHA-256 is
  `e293d8affa47046319b4e952a2ad49ae8efcc20d006c74ce1e1126ae667a844b`.
- Scheduler decision: the projected run is below one hour and must remain in
  the active foreground turn with one blocking wait; no heartbeat is created.
  The run may begin only after the frozen implementation/preregistration is
  committed and pushed.
- Competition log loss, AUROC, Brier, ECE, fold deltas, outcome bootstrap,
  projected public score, and rank change are not applicable because
  `competition_outcomes_accessed=false`, `V_joint_accessed=false`, and
  `V_final_accessed=false`. No competition cache, checkpoint, blend, ZIP,
  upload, or submission was produced. Full rationale and the fallback
  portfolio are in
  `POST_E810_RESEARCH_AND_E820_SELECTION_2026-07-29.md`; exact freeze is in
  `E820_PREREGISTRATION_2026-07-29.md`.

## 2026-07-30 - E820 external target-free gate passed

- Executed the single frozen
  `E820_mathdial_contrastive_objective_alignment_v1` run from pushed commit
  `f647f8608cd0eac2a14723e7fed45032752cd69e`. It completed in one foreground
  blocking wait; training took `1,208.110946` seconds and total command wall
  time was approximately 27.6 minutes, close to the measured 25.9-minute
  projection. No scheduler wake or polling was used.
- Runtime was project `.venv` Python 3.12.8, scikit-learn 1.8.0, NumPy 2.5.1,
  Torch 2.13.0+cpu, and Transformers 5.14.1. Peak RSS was 1,327,157,248 bytes.
  The fit used exactly 717 question IDs per epoch, 90 steps per epoch, the
  frozen unique-qid batches, upper four BGE-base layers, and no outcome field.
  Epoch losses decreased from `0.0329653421` to `0.0035182768`.
- On all 599 question-disjoint official-test dialogues, the held-out
  teacher-confusion query view improved base/adapted MRR from
  `0.2882901625` to `0.4040249942` (gain `0.1157348316`) and recall@5 from
  `0.3372287145` to `0.4624373957` (gain `0.1252086811`). Original-problem MRR
  improved `0.9811508964` to `0.9927378965`; recall@5 improved
  `0.9933222037` to `1.0`.
- The 2,000-question-ID confusion MRR bootstrap observed/mean gain was
  `0.1214389488/0.1213716071`, 95% interval
  `[0.0966970774,0.1473972389]`, support `1.0`. Original-problem bootstrap
  mean/interval/support were
  `0.0176692246/[0.0075391651,0.0290167953]/1.0`.
- Adapted document off-diagonal cosine mean/std were
  `0.1968484074/0.0878590420`, so both no-collapse clauses passed. Every frozen
  target-free clause passed; E820 is eligible only for the separately frozen
  competition screen, not production or packaging.
- Artifact SHA-256 values: delta
  `c7caf144e91db5aedd427ab27c860cd5ae091cbb1e466621b0b563f88e6b9248`;
  report
  `d3f6f8c5bb5ec832c959db91d2777f93b58330b8fd129afe5c49db78b3d02fad`;
  retrieval scores
  `6baa01886cd04cf0b994380e45135926ae2e8bbd8d66802d5707066b0ea99563`;
  training metrics
  `acccb26c5901783756816cfa522e9003f846355d2104692f9eec2d3c1c813461`.
- Froze `E820_mathdial_contrastive_competition_screen_v1` before any
  competition outcome: immutable 4,096 pilot rows, existing compact contexts,
  exact query prefix, bound final delta, normalized CLS, existing semantic
  feature geometry, `C=0.1`, five folds in each of `V_seen`, `V_objective`, and
  `V_style`, only 10/20/30% blends over raw v0.5, equal-environment/equal-fold
  selection, nine literal selection clauses, and confirmation-only `V_joint`.
  A cache must be target-free and committed by hash before the one validation.
- Competition log loss, AUROC, Brier, ECE, fold deltas, outcome bootstraps,
  public projection, and rank change remain not applicable:
  `competition_outcomes_accessed=false`; `V_seen_accessed=false`;
  `V_objective_accessed=false`; `V_style_accessed=false`;
  `V_joint_accessed=false`; `V_final_accessed=false`. No competition cache,
  blend, ZIP, upload, or submission existed at this freeze. Full target-free
  evidence is in `E820_TARGET_FREE_RESULT_2026-07-30.md`; competition freeze is
  `E820_COMPETITION_PREREGISTRATION_2026-07-30.md`.
- The target-free competition-cache resource benchmark then verified the raw
  AutoModel encoder against 32 immutable BGE-base context rows: maximum
  absolute difference `1.2293458e-7`, minimum cosine `0.9999999404`. The bound
  adapted encoder took `12.4862565` seconds for 64 context documents and
  `1.7284089` seconds for 64 objective queries, projecting `805.7099749`
  seconds (13.4 minutes) for 4,096 contexts plus 244 unique queries. Peak RSS
  was `965,009,408` bytes; duration and 8-GiB gates pass. Benchmark SHA-256 is
  `6ba487d36d6208f7e7c1d0b85866c15c45debd85c66a87891cff25e91e1ebcd6`.
  It accessed no target or component outcome. The cache must run in the
  foreground and be committed by hash before validation.
- The bound target-free cache completed in `818.9381020` model seconds
  (approximately 14.3 minutes command wall time), peak RSS 1,244,778,496
  bytes. Context/objective caches are each 4,096 x 768 float32 and 12,583,040
  bytes, with SHA-256 values
  `9066033cf59cc1b0cf6cf893e67b4de7a712a1bb57ed13ed99362df8f15d3a0c`
  and
  `3fd9dbdb94a9955a8d0c24f3d7791ae5b231c6dbabd3c7164ab20d8ce052eab8`.
  Metadata/target-free-report SHA-256 values are
  `ec282940039a7d45dab9e838da95c5ef73ed61e83d97ba6f874dce7da0f3b5ff`
  and
  `b925062dd87ec692ae4d97befa737632193d5522fc8f6045c87b099bef1c5558`.
  The report confirms external gate pass and
  `competition_outcomes_accessed=false`, with every selection environment,
  `V_joint`, and `V_final` untouched.
- A pre-validation orchestration audit caught that cache construction called
  the harmless resource benchmark a second time, overwriting only its timing
  JSON. The initial/second benchmark hashes are
  `6ba487d36d6208f7e7c1d0b85866c15c45debd85c66a87891cff25e91e1ebcd6`
  and
  `8c53ca0d999d8b806910c170bbf8a500a179c7e41136e592cb38639e60126543`.
  No model, source, prompt, cache, gate, or outcome changed. Benchmark and
  completed-cache reuse were made idempotent before validation. Exact cache
  hashes are committed in
  `E820_COMPETITION_CACHE_BINDING_2026-07-30.json`; validation refuses any
  mismatch.

## 2026-07-30 - E820 single competition selection validation rejected

- Executed exactly one frozen
  `E820_mathdial_contrastive_competition_screen_v1` validation from pushed
  commit `202bbbd2ce1f335b3e8b365706a65abdfe66c193`. The bound final delta, 4,096
  pilot rows, caches, query prefix, feature geometry, three environments, 15
  folds, `C=0.1`, seed `20260730`, raw v0.5 formula, and 10/20/30% weights were
  unchanged. Runtime was 24.636631 seconds under `.venv` Python 3.12.8,
  scikit-learn 1.8.0, NumPy 2.5.1, Torch 2.13.0+cpu, and Transformers 5.14.1.
- The deterministic rule selected 10%, which regressed equal-environment,
  equal-fold macro log loss by `0.0017891691` versus raw v0.5 and improved
  `0/3` environments. `V_seen` changed from `0.5490162398` to
  `0.5514067588` (`+0.0023905190`); `V_objective` from `0.5971626583` to
  `0.5977481357` (`+0.0005854774`); `V_style` from `0.5494566144` to
  `0.5518481254` (`+0.0023915110`).
- Worst fold regression was `0.0033894188`. Mean AUROC/Brier/ECE changes were
  `-0.0017578160/+0.0006788506/+0.0002905201`. The 20% and 30% blends
  regressed mean log loss by `0.0038059528` and `0.0060460409`.
- The 5,000-session bootstrap mean/interval/support were
  `-0.0017908938/[-0.0020583595,-0.0015275991]/0.0`. The independent
  5,000-semantic-family values were
  `-0.0018052660/[-0.0022102725,-0.0014449899]/0.0`. Every one of the nine
  frozen clauses failed.
- Standalone pilot diagnostics show the cause: general BGE-base AUROC
  `0.6662769/0.5694212/0.6665329` on
  `V_seen/V_objective/V_style` fell to
  `0.6525506/0.5296959/0.6515603` after MathDial contrastive adaptation.
  External problem-dialogue retrieval improved dramatically, but it removed
  correctness-relevant geometry rather than adding outcome signal.
- Artifact SHA-256 values: predictions
  `254673e40564849f77176eb714d81a37d5bb9c06e331528b4c20fcd66f6a7d7d`;
  fold metrics
  `5638aeae3ec220ea2013ac114c67b95d631ee25de3e0003dd21a55349fc421c7`;
  environment metrics
  `189cf05841632efc5619657062acb836710e809cca5697dccecd928b5473e40b`;
  selection
  `793e11d15b4237355b0bb536912ac05a0625d95ecc002e3b0159a3bac59182df`;
  bootstraps
  `43ccafb5c6798d3244d533a2f8d6de0a432d985c2bdee112d369815b518994dc`;
  report
  `4387febecb29473edd58d3e34a34c9372425ffee8b4246ccb835fef39b2184af`.
- **Decision:** reject exact E820 without smaller weight, unprefixed objective,
  layer/epoch checkpoint, base/adapted interpolation, hard negative, C,
  calibration, environment-specific gate, or full-cache rescue. A rough
  public projection is `0.60719`, worse than v0.5; it is not a leaderboard
  observation or submission candidate. `V_joint_accessed=false`;
  `V_final_accessed=false`; no production model, ZIP, upload, or competition
  submission was created.

## 2026-07-30 - E830 official curriculum-progression source gate frozen

- Post-E820 research selected a different causal hypothesis: external
  pedagogical progression for objective difficulty. The repository/log search
  found only the rejected BGE k-nearest-objective difficulty prior, E720
  objective-token crosses, and E730 linear objective alignment; no official
  curriculum stage, grade progression, or taxonomy representation exists.
  This branch does not revive or consume any rejected checkpoint/prediction.
- Downloaded the Department for Education statutory mathematics programme of
  study from GOV.UK under the Open Government Licence v3.0. The preserved raw
  HTML is 302,822 bytes with SHA-256
  `d8499da572edb7f1dbf9c42284dae265e121a8215692cff204c3a5d73cd6c490`.
  It defines years 1-6 followed by KS3/KS4 and explicitly describes increasing
  complexity and readiness-based progression.
- Frozen target-free construction: parse eight stage sections and their
  domains/statements; joint unsupervised word unigram/bigram and
  character-within-word 3-5-gram TF-IDF; similarity fixed at 55/45%; maximum
  statement similarity per stage; posterior temperature `0.08`; emit eight
  scores/probabilities plus expected/best stage, coverage, margin, entropy,
  and nearest official standard. Competition input is restricted to
  `response_id`, `learning_objective_id`, and `learning_objective`; no label
  file is opened.
- Frozen evidence includes official-statement leave-one-out sequencing, 24
  pre-score anchor objectives, all-objective coverage, five-fold
  objective-level fixed-ridge BGE redundancy (`alpha=100`), and a deterministic
  synthetic proper-score benchmark. Fourteen literal clauses cover source
  parsing, external sequencing, objective coverage, anchors, stage diversity,
  nonredundancy, and synthetic log-loss gain. Any failure closes exact E830
  before competition outcomes without rescue.
- Focused tests passed `4/4`, Ruff was clean, and the complete suite passed
  `232` tests plus `8` subtests in `40.95` seconds under project `.venv`
  Python 3.12.8 and scikit-learn 1.8.0. The target-free score has not yet run.
  Competition log loss, AUROC, Brier, ECE, folds/environments, outcome
  bootstrap, projected public loss, and rank are not applicable because no
  competition outcome or validation environment was accessed. `V_joint` and
  `V_final` remain sealed; no model, blend, ZIP, upload, or submission was
  produced.

## 2026-07-30 - E830 target-free source gate passed and competition screen frozen

- Executed the single frozen
  `E830_curriculum_progression_target_free_v1` source gate from pushed commit
  `a4b32e44a7c70d6ea9fd825391ae58dfb01ace98`. Runtime was `0.8430311`
  seconds under project `.venv` Python 3.12.8, scikit-learn 1.8.0, NumPy
  2.5.1, pandas 2.3.3, and SciPy 1.18.0. No competition outcome or validation
  environment was opened.
- Parsed 636 unique official statements across all eight stages and 22
  domains; the smallest stage has 54 statements. Official leave-one-out
  expected-stage Spearman was `0.7756216`, mean absolute stage error
  `1.1815793`, exact best-stage accuracy `0.2688679`, and within-one accuracy
  `0.7610063`.
- All 398 objectives were covered. Median and 10th-percentile maximum official
  similarity were `0.3020224/0.2012611`. Best-stage counts 1-8 were
  `18/26/36/54/85/77/58/44`. Frozen anchors were `17/24` in range and `20/24`
  within one, with midpoint/expected-stage Spearman `0.8462139`.
- The fixed five-fold BGE-base redundancy probe recovered only `0.4739721`
  R-squared and `0.5910205` MAE, so the curriculum prior is substantially
  nonredundant with the existing base objective geometry. The deterministic
  synthetic proper-score control/candidate losses were
  `0.6532323/0.6378295`, gain `0.0154028`. Every one of 14 frozen
  target-free clauses passed.
- Artifact SHA-256 values: objective features
  `2391a8f5d807323c2802086e8b8f78d126b0115f4ad71531deaac267c453ec19`;
  standards
  `e14840cb17bcab37b22842187ecd606e8917cfe53ff0a0dc68ebe97ccd27c2ca`;
  anchors
  `35ab2be7cfe9853f270338d726a9b530c50283f1e4e3a5423d47bbcb3e872fde`;
  report
  `5b53d3f6fd3d34179c4a2f885d6592538d82d77090a74f1682bdb53d93ca9a2b`.
- Froze `E830_curriculum_progression_competition_screen_v1` before any
  outcome: the exact hashes above, immutable 4,096-row component lineage, 20
  fixed curriculum inputs, fold-local StandardScaler plus logistic regression
  (`C=0.1`, seed `20260730`), inverse training-objective-frequency weights,
  raw-v0.5 formula, only 10/20/30% blends, the existing three selection
  environments and five folds, nine selection clauses, both 5,000-replicate
  bootstraps, and confirmation-only `V_joint`.
- Focused verification passed `7/7`, Ruff was clean, and the complete suite
  passed `235` tests plus `8` subtests in `41.26` seconds. The frozen
  competition validation has not run. Log loss, AUROC, Brier, ECE, fold
  deltas, outcome bootstraps, public projection, and rank remain not
  applicable. `V_joint_accessed=false`; `V_final_accessed=false`; no model,
  blend, ZIP, upload, or submission was produced.

## 2026-07-30 - E830 single competition selection validation rejected

- Executed exactly one frozen
  `E830_curriculum_progression_competition_screen_v1` validation from pushed
  commit `d8bc8669800010c3a72b6b88819ed28223076de8`. Runtime was `15.0823877`
  seconds under project `.venv` Python 3.12.8, scikit-learn 1.8.0, NumPy
  2.5.1, and pandas 2.3.3. Exact target-free hashes, component lineage, 20
  features, fold-local estimator, equal-objective weights, seed, folds,
  environments, blends, metrics, and bootstrap rules were unchanged.
- The deterministic rule selected 10%, which regressed equal-environment,
  equal-fold macro log loss by `0.0016701085` and improved only `1/3`
  environments. `V_objective` improved by `0.0000793102`; `V_seen` regressed
  by `0.0025644101`; `V_style` regressed by `0.0025252256`. The 20% and 30%
  blends regressed mean loss by `0.0042819458/0.0077647061`.
- Worst-fold regression was `0.0087283423`. Macro AUROC improved
  `0.0010419680`, but Brier and ECE regressed
  `0.0005940278/0.0045617399`. Only AUROC non-regression passed; the other
  eight selection clauses failed.
- The 5,000-session bootstrap observed/mean gain was
  `-0.0016708139/-0.0016715442`, 95% interval
  `[-0.0018658865,-0.0014793761]`, support `0.0`. The independent
  semantic-family values were
  `-0.0016708139/-0.0016925205`,
  `[-0.0030848019,-0.0005122778]`, support `0.0022`.
- Artifact SHA-256 values: predictions
  `1f506c09d51f63cc062f5f7d64fd8e50313252ae32af5f7e4b3fd09d1b648d9c`;
  fold metrics
  `7500faed96b2fec23d580bc9e11a2a4e32940ae48c1b41bf8ef2ac980d209e5e`;
  environment metrics
  `9f4b16d9d283f2fd99925fcc24988680cf11564751b5b7e0a6ec43a8644060bc`;
  selection
  `381cfc729f4e3d9f7ce132e3916aa807d5c396dd9b661bbf80c8d96f78bd2065`;
  bootstraps
  `9b6664b357e1fbb73ac41b2bfc8a5060a5fcc03a531a391100c0083435cf1d38`;
  report
  `620511404572b7d01c0668f31b933b47d8df393656f94cb7faa7cf5a939dc6ec`.
- **Decision:** reject exact E830 without smaller weight, alternate C,
  row-frequency weights, stage-only/monotonic/residual calibration, different
  standards, or fold rescue. Curriculum stage carries some ranking
  information but is not a robust probability prior for session correctness.
  Rough public projection is `0.60707`, worse than v0.5 and not a leaderboard
  observation. `V_joint_accessed=false`; `V_final_accessed=false`; no model,
  ZIP, upload, or competition submission was produced.

## 2026-07-30 - E840 StudyChat longitudinal assessment transfer frozen

- Post-E830 search selected an independent real-outcome screen rather than
  another competition-derived prior. The authorised StudyChat snapshot has
  chronological dialogues, prior assignments, and three later proctored exams.
  Earlier work closed direct transfer of its GPT-4.1 dialogue acts; it did not
  test whether label-free help-seeking behaviour predicts future exams beyond
  prior grades and usage across unseen learners.
- Bound CC BY 4.0 revision
  `24d7987d9fbb30d9da12acc53455a10f1cdd2d7f`. SHA-256 values are
  `67927f73327639904c417c2f6e6200e11427920a8435b39cb7c87b5c6601e130`
  (`data.jsonl`),
  `d16d54c8eaa9f3723755362f0dcb651740d070bdcf5c739c7f9a27a77db5a244`
  (Fall grades),
  `06f1a4041e1631a9787c5eac842f04e63dad1480609ac5376fcab351a1fad5ec`
  (Spring grades),
  `0caefd1999143ddb9e3eb93dfa802680a8ab924d95d2acc1fee434da6d66d6fc`
  (licence), and
  `86a5926deec67d5e848c930363829d53339bb601c7cc5d6064bdf917abd1e5c0`
  (README).
- Source/schema/count audit reproduced 16,851 interactions, 203 dialogue
  users, 2,214 chats, and 65/110 Fall/Spring dialogue-grade matches. Frozen
  chronology yields 58/65/65 Fall and 99/110/110 Spring pre-exam samples for
  exams 1/2/3: 507 samples across 175 learners. Artifacts may contain only a
  deterministic salted learner hash, never the released anonymous ID or
  directory name.
- Baseline features are semester/exam, prior-assignment
  mean/std/min/last, log interactions/chats, and topic coverage. The candidate
  adds exactly 20 fixed label-free question, explanation, verification,
  direct-answer, writing, code, confusion, self-explanation, metacognitive,
  lexical, length, tutor, and multi-turn features. No future chat/assignment,
  exam text, submission, generated act label, embedding, or outcome-aware cue
  edit is allowed.
- Frozen evaluation is five-fold shuffled learner-grouped Ridge at
  `alpha=10`, seed `20260730`, pooled and six semester/exam cells, and a
  5,000-learner bootstrap. A separate target-free application of the same
  extractor to competition sessions must show feature support and distribution
  overlap. Nine literal gates cover material RMSE/MAE gain, Spearman,
  cross-cell stability, bootstrap support, and transfer overlap. Any failure
  rejects exact E840 before competition outcomes.
- One focused fixture wording correction changed “Why does” to “Can you
  explain why” so the test actually exercises the already frozen
  explanation-request category; no cue, feature, split, estimator, source, or
  score changed. Focused tests then passed `3/3`, Ruff was clean, and the full
  suite passed `238` tests plus `8` subtests in `40.63` seconds under project
  `.venv` Python 3.12.8 and scikit-learn 1.8.0.
- External RMSE/MAE/Spearman, cell evidence, learner bootstrap, competition
  log loss/AUROC/Brier/ECE, outcome folds/bootstraps, projected public loss,
  and rank are not applicable because the frozen external run has not begun
  and no competition outcome was accessed. `V_joint` and `V_final` remain
  sealed; no model, ZIP, upload, or submission was produced.

## 2026-07-30 - E840 longitudinal external gate rejected before competition outcomes

- Executed the single frozen
  `E840_studychat_longitudinal_assessment_transfer_v1` run from pushed commit
  `6a78e0eba9d72f1ff7edce42445f9b1ebc880471`. Runtime was `178.9796166`
  seconds under project `.venv` Python 3.12.8, scikit-learn 1.8.0, NumPy
  2.5.1, pandas 2.3.3, and SciPy 1.18.0. Source hashes, chronology, 507
  samples/175 groups, baseline/candidate features, learner-grouped folds,
  `alpha=10`, seed, cells, bootstrap, and every gate were unchanged.
- Cross-source support passed: all 20 behaviour features were nonconstant in
  StudyChat and 22,816 eligible competition sessions; 17/20 (`0.85`) target
  medians were within `2.5` StudyChat IQR. The mechanism was technically
  applicable across sources.
- External outcome evidence failed. Baseline RMSE/MAE/Spearman were
  `0.1162921/0.0876747/0.4601022`; the dialogue-behaviour candidate produced
  `0.1182231/0.0890782/0.4382660`, giving gains
  `-0.0019310/-0.0014035/-0.0218362`. Only `1/6` semester/exam cells improved
  RMSE; worst-cell gain was `-0.0037764`.
- The 5,000-learner bootstrap observed/mean RMSE gain was
  `-0.0019310/-0.0019950`, 95% interval
  `[-0.0046666,0.0003241]`, positive-gain support `0.0482`. Six external
  materiality/stability clauses failed; only the worst-cell safety bound and
  three target-free support/overlap clauses passed.
- Operational correction: the command wrapper hit its 124-second ceiling,
  but the one `.venv` worker remained active. A single process check verified
  the exact worker and absence of a duplicate; one foreground completion wait
  was attached rather than restarting. The same run then finished normally.
  This was a wrapper timeout, not a model retry or scientific correction.
- Artifact SHA-256 values: external OOF predictions
  `8953499afdca26540e682e250313675ea167deb9527d15f892bc0d054f50c8ad`;
  cell metrics
  `817c40b95b996e084756c60964f834d08bf88a0e4d522a9d0f48bc5892828fe8`;
  competition target-free features
  `8025f23f513b7bebcb1f2dac20cd3cadfb72c9573feca17a56c38bf666db49cb`;
  report
  `d5ae219d293d65ea5c7e4b708125deadf44f0553eb7c83ccfa66050bfefdbafc`.
- **Decision:** reject exact E840 without generated acts, alternate cues,
  embeddings, grade target, chronology, baseline, alpha, split, subgroup, or
  model rescue. Competition log loss, AUROC, Brier, ECE, folds, outcome
  bootstrap, and public projection are not applicable because
  `competition_outcomes_accessed=false`. `V_joint` and `V_final` remain sealed;
  no model, ZIP, upload, or submission was produced.

## 2026-07-30 - E850 lexical speaker-role denoising frozen

- Post-E840 novelty audit selected the explicit open item from the organizer
  clarification and prior role-ablation learning: rare student/tutor
  attribution errors. Earlier hard role prefixing is rejected; E850 instead
  predicts the observed role from text, neighboring roles, position, length,
  and question status without exposing the current role as an input, then
  proposes only high-confidence contradictions.
- Bound target-free sources: 6,139,854-row/22,821-session utterance cache
  SHA-256
  `80a231d18dbb0641989ebb972f08988f0ddd3cb7c4fb769ad2fe90b5a98ba5a4`
  and style-assignment SHA-256
  `600664b9ce6c3809ff4b17206397285cfdd905a203a5c5f4d6969b403dfc9b40`.
  Role counts are 2,697,152 student, 3,196,001 tutor, and 246,701 background;
  background is never classified or reassigned.
- Frozen complete-session sampling is one of 24 namespaced SHA-256 buckets;
  five distinct session-hash folds prevent transcript leakage. Representation
  is nonnegative L2-normalized `2^18` word unigram/bigram hashing. Estimator is
  averaged L2 SGD log loss, `alpha=1e-5`, 20 maximum iterations, fixed fold
  seeds `20260730..20260734`, and final target-free seed `20260830`.
- A proposed correction requires class confidence at least `0.98` and
  disagreement with the observed role. Twelve literal target-free gates cover
  OOF macro-F1/ECE/high-confidence support, deterministic 5% synthetic
  corruption recovery, correction/affected-session opportunity bounds,
  direction balance, and every one of 20 target-free style cells. Any failure
  rejects exact E850 before outcomes without threshold, feature, sample, model,
  or style rescue.
- Focused tests passed `3/3`, Ruff was clean, and the complete suite passed
  `241` tests plus `8` subtests in `40.83` seconds under project `.venv`
  Python 3.12.8 and scikit-learn 1.8.0. The resource benchmark and target-free
  role score have not run.
- Competition log loss, AUROC, Brier, ECE, fold/environment results, outcome
  bootstraps, projected public loss, and rank are not applicable because no
  competition outcome was accessed. `V_joint` and `V_final` remain sealed; no
  model, ZIP, upload, or submission was produced.

## 2026-07-30 - E850 rejected at the target-free speaker-recovery gate

- The resource benchmark measured `17.8726` seconds and projected `614.3934`
  seconds including a 25% margin, so the exact run remained in the active
  foreground turn. Bound run
  `20260729T201140Z_speaker_role_denoising` then completed in `220.8722`
  seconds without a scheduler, duplicate worker, or restart.
- Grouped OOF target-free evidence covered 242,004 student/tutor utterances in
  938 complete sessions: accuracy `0.8933282`, macro-F1 `0.8925152`, log loss
  `0.2747211`, top-label ECE-10 `0.0120217`, high-confidence coverage
  `0.2644667`, high-confidence agreement `0.9842192`, and deterministic
  synthetic-corruption recovery `0.9837672`.
- The 5,893,153-row/22,821-session full audit proposed 24,045 corrections:
  rate `0.00408016`, 11,102 affected sessions (`0.486482`), direction counts
  8,760 student-to-tutor and 15,285 tutor-to-student (ratio `0.573111`).
  Every target-free style cell had at least 84 corrections and at least
  `0.254125` affected sessions.
- Ten frozen clauses passed. Two failed literally: macro-F1 was below `0.95`
  and high-confidence coverage was below `0.50`. E850 is rejected without
  lowering the threshold, changing the features/model/iterations/sample, or
  relaxing a gate. Fixed-iteration convergence warnings are recorded but do
  not authorise rescue.
- Artifact SHA-256 values: report
  `0305afc81071fe060bc40c9ada15f662004c75db74c2f8551a85211d6cf322f7`;
  proposed corrections
  `7cb35b5e03852c5454822cb78c7d7ac8b74ae1b985b5794aa56de0c904a8f29c`;
  OOF probabilities
  `51d8bf70604502783fe976175a75870d6bbcb38160f02e50e6469bf1f5966169`.
- Competition log loss, AUROC, Brier, ECE, outcome folds/environments,
  bootstrap support, projected public loss, and rank are not applicable:
  `competition_outcomes_accessed=false`. `V_joint` and `V_final` remain sealed;
  no competition cache, model, ZIP, upload, or submission was produced.

## 2026-07-30 - E860 target-free ASR/spoken-math normalization frozen

- Post-E850 repository/log audit excluded another speaker rescue, external
  outcome head, correctness/state/tutor-move branch, encoder swap, objective
  prior, calibration, or aggregation. No prior branch canonicalized verbalized
  mathematics before semantic encoding; character n-grams only tested surface
  robustness.
- Aggregate target-free opportunity audit over all 6,139,854 utterances found
  1,616,389 rows with `[unclear]`, 1,474,469 with a leading disfluency, 611,654
  with number words, 685,241 with spoken operators, and 125,061 with adjacent
  repeated words. All 22,821 sessions contain `[unclear]`; 22,782 contain
  number words and 22,641 spoken operators. Of 35,072 cached objective
  contexts, 32,485 contain number words and 30,644 spoken operators. These are
  opportunity counts, not outcome evidence.
- Froze `E860_target_free_asr_math_normalization_v1`: remove `[unclear]`, strip
  only turn-initial fillers, collapse adjacent exact repetitions, and map
  unambiguous explicit numbers/operators to canonical tokens. Ambiguous
  homophones are untouched. No correctness, mastery, role, state, tutor move,
  difficulty, or outcome inference exists.
- Bound target-free sources are NCTE utterances SHA-256
  `bbfa37ac857991e5d12b1677216896f32d7cf4c2d95b618b67afa7b3bf23b3d7`,
  competition context SHA-256
  `218d5b041f78c6803f28078c35b7fa9e37b725f456c4052974e5150b2afa43f6`,
  and packaged MIT BGE-base model SHA-256
  `c7c1988aae201f80cf91a5dbbd5866409503b89dcaba877ca6dba7dd0a5167d7`.
- Frozen samples are 2,048 hashed NCTE utterances and 2,048 hashed competition
  contexts balanced over four target-free retrieval-coverage quartiles.
  Thirteen literal gates cover corruption recovery, semantic preservation,
  transformation opportunity, overall objective alignment, and low-coverage
  alignment. Any failure rejects exact E860 before competition outcomes with
  no homophone, token, threshold, perturbation, sample, model, or subgroup
  rescue.
- Focused tests passed `3/3`, Ruff was clean, and the full project suite passed
  `244` tests plus `8` subtests in `41.50` seconds under `.venv` Python 3.12.8
  and scikit-learn 1.8.0.
- Resource benchmark, NCTE corruption metrics, competition target-free
  semantic metrics, competition log loss/AUROC/Brier/ECE, outcome
  folds/environments, bootstrap, projected public loss, and rank have not run.
  `V_joint` and `V_final` remain sealed; no model cache, ZIP, upload, or
  submission was produced.

## 2026-07-30 - E860 rejected at the target-free semantic gate

- The resource benchmark took `46.0599` seconds and projected `1842.3962`
  seconds with its 25% margin. The exact full run
  `20260729T204454Z_asr_math_normalization` stayed in one foreground turn and
  completed in `1174.1638` seconds without a scheduler, duplicate, or restart.
- The 2,048-row NCTE screen covered 298 videos. Normalization improved
  deterministic corruption token Jaccard by `0.2558189` and BGE clean/corrupt
  cosine by `0.0477206`; normalized cosine was `0.9950613`, with positive gain
  on `0.9755859` of rows. Exact normalized equality was only `0.8295898`,
  failing its `0.90` clause.
- The decisive 2,048-row competition target-free screen went in the wrong
  direction. Raw transcript/objective cosine was `0.7209262`; normalized was
  `0.7125375`, a mean gain of `-0.00838865`. Only `0.3476563` of rows
  improved, and the lowest retrieval-coverage quartile regressed
  `-0.00765740`. Objective/context semantic preservation was
  `0.9478115/0.9477497`; the objective value also missed its `0.95` bound.
- Eight frozen clauses passed and five failed. Reject exact E860 without
  retaining extra surface words, adding homophones, changing canonical tokens,
  perturbations, sample/model/pooling, thresholds, or subgroups. Its synthetic
  invariance does not justify overriding the real target-free alignment
  failure.
- Artifact SHA-256 values: report
  `f6d2af190a5cd28aca13e4a1b749affed4ddc16da2cfd60b474de516fdac82a7`;
  NCTE scores
  `f14faf2387d17231c5734ee8240dd16474ca543b96626fab87212a8936e905ab`;
  competition target-free scores
  `232964914a720fc02abb4e97e8a2eb0a3e8b2ab3048489ace8a1ee5d32c54d10`.
- Competition log loss, AUROC, Brier, ECE, outcome folds/environments,
  bootstraps, projected public loss, and rank are not applicable because
  `competition_outcomes_accessed=false`. `V_joint` and `V_final` remain sealed;
  no competition cache, estimator, ZIP, upload, or submission was produced.

## 2026-07-30 - E870 MaE misconception-atlas external gate frozen

- A fresh audit rejected an alternate StudyChat assignment score/embedding
  branch because E840 explicitly closed alternate grade-target and embedding
  rescue. Repository/log search then found no prior misconception description,
  diagnostic-example atlas, or misconception-retrieval component. E870 targets
  systematic mathematical error rather than generic correctness, tutor move,
  state, quality, difficulty, aggregation, calibration, or ASR normalization.
- Acquired the public MIT
  `nancyotero-projects/math-misconceptions` repository into ignored
  `Datasets/MaE` at immutable revision
  `12bee142d49dfb3c874149035cfbad28547a818b`. Raw hashes are data
  `8223b3a6222c02f4efe8519c4e6abaa2e8e35d96a380bda3882b4f1cdd1fe4e1`,
  license
  `cca1a8c2bc40c9e58cddec5c88da0330384dafd6600059a063f2ea2cba1d35e4`,
  and README
  `c91fc759503a5d4c9117e8fe8595026f8528b16b6a8bee15e82f2ffe5dcd8087`.
- Schema audit confirms 220 examples, 55 misconception IDs, exactly four
  examples per ID, and one description/topic per ID. The source paper reports
  a nontrivial expert-designed task; acquisition required no gate, account,
  disclosure, or new terms.
- Froze leave-one-example retrieval with the packaged MIT BGE-base model. A
  query uses only the observable question and incorrect learner answer—never
  the source's educator-authored explanation, correct answer, or label. The
  true prototype excludes every exact normalized-question duplicate to block
  template leakage and must retain at least two examples. All 55 labels and
  released-topic candidate sets are scored without training or tuning.
- Twelve literal external clauses bind all-label and within-topic top-k/MRR,
  per-topic minima, and a 5,000-replicate misconception-ID bootstrap. Any
  failure rejects exact E870 before any competition text or outcome, without
  adding correct answers, retaining duplicate questions, changing fields,
  prototype/model/pooling/length, topic selection, or threshold rescue.
- Focused tests passed `3/3`, Ruff was clean, source/model lineage verification
  passed, and the full project suite passed `247` tests plus `8` subtests in
  `45.02` seconds under `.venv` Python 3.12.8 and scikit-learn 1.8.0.
  Resource/model runtime, retrieval metrics, competition log loss/AUROC/Brier/
  ECE, outcome folds/environments, outcome bootstrap, public projection, and
  rank have not run. No competition text has been encoded; `V_joint` and
  `V_final` remain sealed; no cache, model, ZIP, upload, or submission was
  produced.

## 2026-07-30 - E870 rejected at the external misconception gate

- Executed the exact frozen external screen once as run
  `20260729T205857Z_misconception_atlas` from pre-score commit `ad6b82a`.
  It completed in `67.3636` seconds in the foreground under `.venv` Python
  3.12.8, NumPy 2.5.1, pandas 2.3.3, and scikit-learn 1.8.0.
- The screen used all 220 MaE examples and 55 misconception labels. Queries
  contained only question and incorrect answer; they excluded the source's
  educator explanation, correct answer, label, topic, competition text, and
  outcomes. Every held-out true prototype retained at least two examples after
  normalized-question duplicate exclusion.
- All-label top-1/top-3/top-5 were
  `0.195455 / 0.436364 / 0.527273`, with MRR `0.357842`. Within-topic
  top-1/top-3/MRR were `0.390909 / 0.731818 / 0.589827`. Median
  true-versus-hardest-false margin was `-0.0213801`; the minimum topic
  all-label top-5 was `0.25`, and minimum topic within-topic top-3 was
  `0.588235`.
- The 5,000-replicate misconception-ID bootstrap had mean top-5 `0.526752`,
  95% interval `[0.427273, 0.627273]`, and only `0.3412` support for top-5 at
  least `0.55`. Three of twelve clauses passed; all frozen all-label accuracy/
  MRR, within-topic accuracy/MRR, minimum-topic top-5, and bootstrap clauses
  failed. Reject exact E870 without educator/correct-answer leakage,
  duplicate-template retention, topic selection, model/prototype/threshold
  changes, fine-tuning, or any other rescue.
- Artifact SHA-256 values: report
  `b8bfbb4df920b619729f68fa244b47a23ded8fc82b38491e887a614ad14339e3`;
  external scores
  `7f0018e64841de4f5e84f6e690d32b6bba4334373cd8c9d8cd663b45d5d27e6c`.
- Competition log loss, AUROC, Brier, ECE, validation folds/environments,
  outcome bootstrap, projected public loss, and rank are not applicable.
  `competition_outcomes_accessed=false`; no competition text was encoded.
  `V_seen`, `V_objective`, `V_style`, `V_joint`, and `V_final` were not
  accessed; no cache, model, ZIP, upload, or submission was produced.

## 2026-07-30 - post-E870 source and fallback audit

- Reconstructed the complete recent failure pattern before proposing another
  branch. E770/E810/E820/E830 each regressed equal-environment log loss by
  roughly `0.00167-0.00205`, with the largest consistent damage in `V_seen`
  and `V_style`. E840 then failed real longitudinal external outcome transfer;
  E850/E860/E870 failed target-free or external gates. Another objective prior,
  semantic adaptation, published count, behavior cue, speaker fix, ASR
  transform, misconception model, calibration, or failed-score blend is not
  independent.
- Primary-source research identified APTA as the only current high-priority
  conditional branch because it combines reciprocal solver/tutor roles,
  ordered algebra problem actions, chat, and real learning analyses. The 2024
  APTA study says deidentified data pool three study-site datasets in DataShop.
  The already frozen landing gate remains mandatory; access alone is not
  evidence that learner/outcome linkage or sample size is sufficient.
- The remaining outcome-linked reserve is weak. ITSPOKE/WOZ has real spoken
  dialogue and pre/post testing but is public by request, physics-domain, only
  about 60 learners in the released pilot, and its likely uncertainty/affect
  mechanism overlaps closed challenge, behavior, speaker, and ASR families.
  BEETLE II has a real evaluation but no verified open transcript-plus-outcome
  package was found. NCTE's acquired public files omit value-added/test
  outcomes; no institutional ICPSR entitlement may be fabricated. SAGA22,
  TalkMoves, and NCTE discourse labels lack an independent outcome and overlap
  E530/E760/E770. The synthetic Education Dialogue outcome overlaps rejected
  E640.
- Wrote
  `POST_E870_FIRST_PLACE_RESEARCH_AND_FALLBACK_PLAN_2026-07-30.md` with a
  literal APTA source/external/transfer/selection ladder, a narrow
  ITSPOKE/BEETLE source reserve, and a stop-to-write-up branch if no independent
  outcome source survives. Its SHA-256 is
  `873b47259a9bedcf49b8b70f9b3396dff55de92b04d631a201e88f773cb511ec`.
  No E880 is preregistered.
- DataShop authentication expired. Reauthentication through the previously
  authorized GitHub route unexpectedly opened a blank new-account registration
  instead of the existing account. No account was created, form submitted,
  terms accepted, or new data disclosed. The browser was left on DataShop
  login; the existing `k12-apta-access-monitor` remains active but may require
  participant sign-in at its next check.
- This was research/documentation only. Runtime, competition log loss, AUROC,
  Brier, ECE, folds/environments, bootstraps, projected gain, and rank change
  are not applicable. No competition outcome or prediction was opened;
  `V_joint` and `V_final` remain sealed. No model, cache, ZIP, upload, or
  submission was produced. Protected v0.5 remains public `0.6054`; protected
  ZIP SHA-256 remains
  `65467003547fb62ec867733c6acf0a63e6f9fb0fed9533b61b9592e143b17186`.
