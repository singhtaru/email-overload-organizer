"""
Report classifier accuracy on:
  - Enron sample (10k + 10k) + supplemental PDF emails (same pipeline as training)
  - Train split (80%) and test split (20%) with random_state=42
  - Full combined set (apparent / resubstitution accuracy)

Run from project root:
  python -m scripts.evaluate_dataset_accuracy
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.email_format import format_for_classifier
from src.supplemental_training_emails import SUPPLEMENTAL_TEXTS_LABELS


def load_combined_texts_and_labels():
    data_path = _ROOT / "data" / "processed" / "emails_with_labels.csv"
    df = pd.read_csv(data_path)
    df["clean_subject"] = df["clean_subject"].fillna("")
    df["clean_body"] = df["clean_body"].fillna("")
    df["text"] = df.apply(
        lambda r: format_for_classifier(str(r["clean_subject"]), str(r["clean_body"])),
        axis=1,
    )
    target_count = 10000
    class0_pool = df[df["label"] == 0]
    class1_pool = df[df["label"] == 1]
    df_class0 = class0_pool.sample(n=min(target_count, len(class0_pool)), random_state=42)
    df_class1 = class1_pool.sample(n=min(target_count, len(class1_pool)), random_state=42)
    df_subset = pd.concat([df_class0, df_class1]).sample(frac=1, random_state=42).reset_index(drop=True)

    sup_texts = [t for t, _ in SUPPLEMENTAL_TEXTS_LABELS]
    sup_y = np.array([y for _, y in SUPPLEMENTAL_TEXTS_LABELS], dtype=int)

    X_text = df_subset["text"].tolist() + sup_texts
    y = np.concatenate([df_subset["label"].values, sup_y])
    n_enron = len(df_subset)
    n_sup = len(sup_texts)
    return X_text, y, n_enron, n_sup


def main():
    print("Loading combined dataset (Enron balanced sample + supplemental)...")
    X_text, y, n_enron, n_sup = load_combined_texts_and_labels()
    n_total = len(y)
    print(f"  Enron (sampled): {n_enron}")
    print(f"  Supplemental (PDF benchmarks × duplicates): {n_sup}")
    print(f"  Total rows: {n_total}")

    print("Loading SentenceTransformer + saved stacking model...")
    model_st = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
    clf = joblib.load(_ROOT / "models" / "embedding_stacking_model.pkl")

    print("Encoding (this may take several minutes)...")
    X = model_st.encode(X_text, show_progress_bar=True, batch_size=64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    full_acc = clf.score(X, y)

    print()
    print("=== Current saved model: embedding_stacking_model.pkl ===")
    print(f"  Train accuracy (80% split, in-sample): {train_acc:.4f}")
    print(f"  Test accuracy  (20% holdout):           {test_acc:.4f}")
    print(f"  Full dataset accuracy (all {n_total} rows): {full_acc:.4f}")
    print()
    print(
        "Note: 'Full dataset' mixes the 80% the model was fit on and the 20% holdout, "
        "so it is between train and test accuracy."
    )


if __name__ == "__main__":
    main()
