# Email Overload Organizer

An NLP project that scores email **importance**, assigns a **priority tier** (High / Medium / Low), and extracts **named entities** when messages look important.

The current implementation uses:

- A **SentenceTransformer** (`all-MiniLM-L6-v2`) + **stacking classifier** for importance prediction, combined with **rule-based filters and soft boosts** (marketing noise, negated urgency, keyword cues) in `src/classification.py`
- **Optional vendored weights:** if `models/all-MiniLM-L6-v2/` exists, the classifier loads it with `local_files_only=True`; otherwise it uses the Hugging Face hub and respects **`HF_HUB_OFFLINE=1`** for air-gapped runs
- **spaCy NER** — loads the first available English model in order **`en_core_web_sm` → `en_core_web_md` → `en_core_web_lg`** (smaller models first for memory safety; parser excluded; statistical NER + optional `entity_ruler` patterns). The repo’s `requirements.txt` pins **`en_core_web_sm`** via wheel for Streamlit Cloud; locally you can `python -m spacy download en_core_web_md` or `en_core_web_lg` for higher quality
- NER runs on **important** emails only — not important mail skips NER for speed
- A single **`EmailAnalyzer`** pipeline in `src/pipeline.py` that maps classifier output + NER signals to priority, summaries, deadlines, and suggested actions
- **`format_for_classifier`** in `src/email_format.py` so training and inference both use `Subject: …` / `Body: …` text

> **Hardware note:** Development environment is an Intel Core Ultra 7 155H (16 cores, no NVIDIA GPU, 32 GB RAM). CUDA is not available; all training and inference runs on CPU. The notebook SBERT cell is tuned accordingly — see [Notebook Workflow](#notebook-workflow).

## Project Goals

- Reduce email overload by prioritizing important emails
- Surface key entities (people, organizations, dates, times, money, places, events, facilities) from important messages
- Keep a notebook-first workflow for experimentation and a script-based workflow for retraining and inference

## Repository Structure

```text
email-overload-organizer/
  app/
    app.py
  data/
    raw/
      cleaned_enron_emails.json
      threaded_emails.json
    processed/
      emails_with_labels.csv
  models/
    all-MiniLM-L6-v2/          # optional vendored SBERT snapshot (config + weights)
    embedding_stacking_model.pkl
  notebooks/
    01_load_and_explore.ipynb
    02_preprocessing.ipynb
    03_classification.ipynb
    04_stacking classifier.ipynb
    05_ner.ipynb
  scripts/
    train_embedding_model.py
    evaluate_dataset_accuracy.py
    stress_test_enron_style_2.py
    stress_test_pdf_emails.py
    eval_emails_16_30.py        # optional extra stress subset
    test_model.py               # optional quick sanity check
  src/
    classification.py
    ner.py
    pipeline.py
    email_format.py
    supplemental_training_emails.py
    email_analyzer_demo.py
    test_pipeline.py
  requirements.txt
```

## Data and Features

Input data (raw):

- `data/raw/cleaned_enron_emails.json`
- `data/raw/threaded_emails.json`

Processed dataset:

- `data/processed/emails_with_labels.csv`

Important columns used in modeling:

- `Body`, `Subject`
- `clean_body`, `clean_subject`
- `label` (binary importance target)
- Extra engineered fields in notebook workflow: `email_length`, `num_recipients`, `sender_domain`, etc.

## Model Details

**Classifier (`EmailClassifier`):**

- Embeddings: `SentenceTransformer('all-MiniLM-L6-v2')` — from `models/all-MiniLM-L6-v2` when present, else from the hub (with `HF_HUB_OFFLINE` respected when not using a local folder)
- Embeddings are **L2-normalised** (`normalize_embeddings=True`) so cosine similarity equals dot product — no extra normalisation step needed downstream
- Base learners in the stack: Logistic Regression, Random Forest, Linear SVM (`LinearSVC`)
- Meta learner: Logistic Regression (via `StackingClassifier`)
- **Post-model logic:** probability threshold (`IMPORTANCE_PROBA_THRESHOLD`), promotional / newsletter pattern checks, negated-urgency patterns, and light **soft boosts** on top of stacking probabilities for short, request-like phrasing

> **Ensemble note:** The stacking ensemble uses `StackingClassifier` (sklearn). The base learners are LR, RF, and LinearSVC.

**Training script (`scripts/train_embedding_model.py`):**

- Builds training text with `format_for_classifier(clean_subject, clean_body)` so it matches the app
- Optionally **appends** duplicated hand-labeled rows from `src/supplemental_training_emails.py` (PDF benchmark cases)
- Uses balanced sampling (up to 10,000 rows per class when available), then `train_test_split(..., test_size=0.2, stratify=y)`

Saved artifacts:

- `models/embedding_stacking_model.pkl`

Latest script-based evaluation (from prior `src`/`scripts` runs, no notebook recalculation):

- Combined dataset size: `20,424` rows (Enron sample + supplemental custom rows)
- Held-out test split accuracy (`20%` split): `80.12%`
- Overall full-dataset accuracy (all `20,424` rows): `89.21%`

Interpretation:

- Use `80.12%` as the primary generalization metric (held-out data).
- `89.21%` is a full-dataset aggregate and includes training rows, so it is not a pure unseen-test score.

To **recompute metrics** for the currently saved `embedding_stacking_model.pkl` (same sampling + supplemental logic as training), run `python -m scripts.evaluate_dataset_accuracy` from the project root. That script loads SBERT with `local_files_only=True` — ensure the model name is cached or use a vendored snapshot under `models/`.

## NER Details

NER component in `src/ner.py`:

- Loads the first available English model in order: **`en_core_web_sm` → `en_core_web_md` → `en_core_web_lg`** (parser excluded; statistical NER + optional `entity_ruler` patterns)
- Runs on **important** emails only; for not important, the pipeline returns empty entities
- Deep cleaning: strips quoted replies, forwarded headers, and common signatures before extraction
- Validation: filters extreme lengths, numeric-only noise, and weak single-token hits
- Display buckets include: **People involved**, **Organizations**, **Dates**, **Times**, **Money amounts**, **Places**, **Events**, **Buildings and facilities**

## Application & End-to-End Pipeline

**Streamlit app (`app/app.py`):**

- Subject + body inputs combined with `format_for_classifier` before analysis
- **`@st.cache_resource`** on analyzer construction so SBERT/spaCy are not reloaded on every rerun
- Shows importance (yes/no), **priority** (High / Medium / Low), and **confidence** (stacking probability, adjusted score, or rule path)
- **Deadline** bar when a date can be parsed (uses `dateparser` if installed)
- **Entity chips**, tabs for Summary / **Signals** (decision path, scores, keyword cues) / raw **Entities** JSON, and a **Suggested action** callout with an inferred action tag (Reply, Schedule, etc.)

**Core pipeline (`src/pipeline.py`):**

1. Classify with `EmailClassifier.predict_with_meta` → important vs not important
2. If **not important** → priority **Low**, NER skipped
3. If **important** → run NER → **High** if dates/times/money/events are present, else **Medium**
4. Return a dict including: `importance`, `classification` (priority tier), `entities`, `summary`, `explanation`, `key_details` (deadline, event, org, requirement, required action), `suggested_action`, `signals`, model confidence fields, and `priority_tier_source`

## Setup

### 1. Create and activate virtual environment (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

From the project root (recommended — matches Streamlit Cloud and pins core libraries + `en_core_web_sm`):

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For local Jupyter notebooks (plots and kernels), add:

```powershell
pip install jupyter matplotlib seaborn
```

For a larger spaCy model (optional, if not satisfied with `sm`):

```powershell
python -m spacy download en_core_web_md
```

or:

```powershell
python -m spacy download en_core_web_lg
```

### 3. Run the web UI

From the project root:

```powershell
streamlit run app/app.py
```

## Run Inference Test

From the project root:

```powershell
python -m src.test_pipeline
```

This runs sample emails through the end-to-end analyzer and prints predicted importance, priority, and entities.

## Dataset accuracy report (optional)

From the project root:

```powershell
python -m scripts.evaluate_dataset_accuracy
```

Prints train / test / full-dataset accuracy for the saved stacking model using the same Enron-balanced + supplemental pipeline as `train_embedding_model.py`. Requires a cached or vendored `all-MiniLM-L6-v2` because the script uses `local_files_only=True`.

## Stress tests (optional)

From the project root:

```powershell
python scripts/stress_test_enron_style_2.py
python scripts/stress_test_pdf_emails.py
```

These compare classifier output to hand labels from benchmark PDFs.

For additional ad-hoc evaluation:

```powershell
python scripts/eval_emails_16_30.py
```

## Retrain the stacking model

Requires `data/processed/emails_with_labels.csv`:

```powershell
python scripts/train_embedding_model.py
```

## Notebook Workflow

Recommended order:

1. `notebooks/01_load_and_explore.ipynb`
2. `notebooks/02_preprocessing.ipynb`
3. `notebooks/03_classification.ipynb` — SBERT embedding + StackingClassifier training
4. `notebooks/04_stacking classifier.ipynb` — evaluation & confusion matrix
5. `notebooks/05_ner.ipynb`

### SBERT Embedding Cell (CPU-optimised)

The embedding cell in `03_classification.ipynb` is tuned for **CPU-only machines** (no NVIDIA GPU):

| Setting | Value | Reason |
|---|---|---|
| `torch.set_num_threads` | `os.cpu_count()` (16 on Ultra 7 155H) | Uses all P-cores + E-cores |
| `batch_size` (CPU) | `128` | Safe with 32 GB RAM; ~2× faster than default 64 |
| `batch_size` (CUDA) | `256` | Used automatically if a CUDA GPU is detected |
| `normalize_embeddings` | `True` | L2-normalises output; cosine similarity = dot product |
| Cache | `data/processed/X_train_embed_minilm.npy` | Skips re-encoding on subsequent runs |

The cell prints `[INFO] No CUDA GPU found — running on CPU with N threads` at startup so you know which path is active.

> **Classifier architecture separation:** The ensemble imports (`StackingClassifier`, `LogisticRegression`, `LinearSVC`, `RandomForestClassifier`) belong **only in the notebook training cells**. `src/classification.py` is inference-only — it loads the trained `.pkl` and does not import sklearn ensemble classes.

## Common Issues and Fixes

### `ModuleNotFoundError: No module named 'src'`

Cause: Running scripts from inside `src/` with package-style imports.

Fix: Run from the project root using:

```powershell
python -m src.test_pipeline
```

### `NameError: name 'tfidf' is not defined`

Cause: In older notebook flows, a save cell ran before training cells.

Fix: Run notebook cells in order (or **Run All**) before saving model artifacts.

### `OSError: [E050] Can't find model 'en_core_web_md'` (or other spaCy models)

Cause: spaCy model not installed in the active environment.

Fix: `pip install -r requirements.txt` installs `en_core_web_sm` via the wheel in `requirements.txt`, or install another model explicitly:

```powershell
python -m spacy download en_core_web_sm
```

### `OSError` from `SentenceTransformer` / offline load

Cause: No local folder at `models/all-MiniLM-L6-v2`, and weights are not cached, or `HF_HUB_OFFLINE` is set without a local snapshot.

Fix: Run once with network access so Hugging Face caches the model, add a vendored copy under `models/all-MiniLM-L6-v2`, or unset `HF_HUB_OFFLINE` for development.

## Current Notes

- `src/email_analyzer_demo.py` contains a small analyzer demo snippet rather than preprocessing utilities.
- `data/` and common tabular/JSON data extensions are gitignored — place datasets locally per the structure above.
- `scripts/test_model.py` and `scripts/eval_emails_16_30.py` are optional sanity/eval helpers and not required for train/app flow.
- Development hardware has **no NVIDIA GPU** (Intel Arc integrated only). `torch.cuda.is_available()` returns `False`; all training runs on CPU. CUDA paths in the code are kept for portability (e.g., Google Colab, cloud VMs).

## Future Improvements

- Convert `src/` into a formal package (`src/__init__.py`, CLI entrypoint)
- Add unit tests for classifier, NER extractor, and end-to-end pipeline
- Support additional entities (e.g., phone, email) with custom rules
- If a CUDA GPU becomes available: set `SBERT_BATCH_SIZE=256` and verify `torch.cuda.is_available()` returns `True` before retraining
