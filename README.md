# Email Overload Organizer

An NLP project that classifies emails as important or not important and extracts named entities from important emails.

The current implementation uses:
- A TF-IDF + stacking classifier for importance prediction
- A spaCy NER pipeline (`en_core_web_sm`) for entity extraction
- A combined inference pipeline in `src/pipeline.py`

## Project Goals

- Reduce email overload by prioritizing important emails
- Surface key entities (person, organization, dates, time, money, location) from important messages
- Keep a notebook-first workflow for experimentation and a script-based workflow for inference

## Repository Structure

```text
email-overload-organizer/
  data/
    raw/
      cleaned_enron_emails.json
      threaded_emails.json
    processed/
      emails_with_labels.csv
  models/
    tfidf_vectorizer.pkl
    stacking_model.pkl
  notebooks/
    01_load_and_explore.ipynb
    02_preprocessing.ipynb
    03_classification.ipynb
    04_stacking classifier.ipynb
    05_ner.ipynb
  src/
    classification.py
    ner.py
    pipeline.py
    preprocessing.py
    test_pipeline.py
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

Classifier:
- Vectorizer: `TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english", min_df=5)`
- Base learners in stack:
  - Logistic Regression
  - Multinomial Naive Bayes
  - Linear SVM (`LinearSVC`)
- Meta learner: Logistic Regression
- Train/validation split: `test_size=0.2`, `random_state=42`, `stratify=y`

Saved artifacts:
- `models/tfidf_vectorizer.pkl`
- `models/stacking_model.pkl`

Notebook result currently recorded in `04_stacking classifier.ipynb`:
- Accuracy: `0.9891833318820809`
- F1: `0.9856261249678581`

## NER Details

NER component in `src/ner.py`:
- Model: `en_core_web_sm`
- Active task: entity extraction from important emails
- Extracted labels: `PERSON`, `ORG`, `DATE`, `TIME`, `MONEY`, `GPE`
- Duplicate entities are removed in output

## End-to-End Pipeline

`EmailAnalyzer` in `src/pipeline.py`:
1. Classify input email text with `EmailClassifier`
2. If prediction is important (`1`), run NER extraction
3. Return a dictionary:
   - `importance`: `"Important"` or `"Not Important"`
   - `entities`: extracted entity dict or `None`

## Setup

### 1. Create and activate virtual environment (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install pandas scikit-learn joblib spacy matplotlib seaborn jupyter
python -m spacy download en_core_web_sm
```

## Run Inference Test

From the project root:

```powershell
python -m src.test_pipeline
```

This runs a few sample emails through the end-to-end analyzer and prints predicted importance + entities.

## Notebook Workflow

Recommended order:
1. `notebooks/01_load_and_explore.ipynb`
2. `notebooks/02_preprocessing.ipynb`
3. `notebooks/03_classification.ipynb`
4. `notebooks/04_stacking classifier.ipynb`
5. `notebooks/05_ner.ipynb`

## Common Issues and Fixes

### `ModuleNotFoundError: No module named 'src'`
Cause:
- Running scripts from inside `src/` with package-style imports.

Fix:
- Run from project root using:
```powershell
python -m src.test_pipeline
```

### `NameError: name 'tfidf' is not defined`
Cause:
- Save cell executed before the TF-IDF/training cells in notebook.

Fix:
- Run notebook cells in order (or `Run All`) before saving model artifacts.

### `OSError: [E050] Can't find model 'en_core_web_sm'`
Cause:
- spaCy model not installed in active environment.

Fix:
```powershell
python -m spacy download en_core_web_sm
```

## Current Notes

- `src/preprocessing.py` currently contains a small analyzer demo snippet rather than preprocessing utilities.
- `data/` is gitignored by current `.gitignore`.

## Future Improvements

- Add `requirements.txt` or `pyproject.toml` for reproducible environment setup
- Convert `src/` into a formal package (`src/__init__.py`, CLI entrypoint)
- Add unit tests for classifier, NER extractor, and end-to-end pipeline
- Add model calibration/threshold tuning to reduce false positives
- Support additional entities (e.g., `PHONE`, `EMAIL`) with custom rules
