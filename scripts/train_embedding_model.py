import sys
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.email_format import format_for_classifier
from src.supplemental_training_emails import SUPPLEMENTAL_TEXTS_LABELS


def main():
    print("Loading data...")
    base_dir = _ROOT
    data_path = base_dir / "data" / "processed" / "emails_with_labels.csv"

    df = pd.read_csv(data_path)

    # Fill NAs — structured Subject:/Body: matches inference & app
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
    print(f"Appending {len(sup_texts)} supplemental labeled rows (PDF benchmarks, duplicated).")

    X_text = df_subset["text"].tolist() + sup_texts
    y = np.concatenate([df_subset["label"].values, sup_y])

    print(f"Loading SentenceTransformer... (using {len(X_text)} emails)")
    model_st = SentenceTransformer("all-MiniLM-L6-v2")

    print("Encoding texts... (This might take a couple of minutes)")
    X_embed = model_st.encode(X_text, show_progress_bar=True, batch_size=64)

    X_train, X_test, y_train, y_test = train_test_split(
        X_embed, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Stacking Classifier on semantic embeddings...")
    lr = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    svm = LinearSVC(random_state=42, dual="auto")

    stacking_model = StackingClassifier(
        estimators=[
            ("lr", lr),
            ("rf", rf),
            ("svm", svm),
        ],
        final_estimator=LogisticRegression(),
        cv=5,
        n_jobs=-1,
    )

    stacking_model.fit(X_train, y_train)

    acc = stacking_model.score(X_test, y_test)
    print(f"Model trained! Test Accuracy: {acc:.4f}")

    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)

    save_path = models_dir / "embedding_stacking_model.pkl"
    joblib.dump(stacking_model, save_path)
    print(f"Model saved successfully to {save_path}")


if __name__ == "__main__":
    main()
