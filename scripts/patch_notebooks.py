import json
import os
from pathlib import Path

def patch_notebook(file_path):
    print(f"Checking {file_path}...")
    if not os.path.exists(file_path):
        print("  File not found.")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb.get("cells", []):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])

            # 1. Imports
            source = source.replace(
                "from sklearn.feature_extraction.text import TfidfVectorizer",
                "from sentence_transformers import SentenceTransformer"
            )
            source = source.replace(
                "from sklearn.naive_bayes import MultinomialNB",
                "from sklearn.ensemble import RandomForestClassifier"
            )

            # 2. Extractors
            if "tfidf = TfidfVectorizer" in source:
                # We replace the whole block dynamically
                import re
                source = re.sub(
                    r"tfidf = TfidfVectorizer\(.*?\)",
                    "model_st = SentenceTransformer('all-MiniLM-L6-v2')",
                    source,
                    flags=re.DOTALL
                )
                source = source.replace(
                    "X_train_tfidf = tfidf.fit_transform(X_train)",
                    "X_train_tfidf = model_st.encode(X_train.tolist(), show_progress_bar=True, batch_size=64)"
                )
                source = source.replace(
                    "X_test_tfidf = tfidf.transform(X_test)",
                    "X_test_tfidf = model_st.encode(X_test.tolist(), show_progress_bar=True, batch_size=64)"
                )

            # 3. Models
            source = source.replace("nb = MultinomialNB()", "nb = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)")
            source = source.replace("MultinomialNB()", "RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)")

            # Re-split by newline but preserving them
            lines = [line + ("\n" if i < len(source.split("\n"))-1 else "") for i, line in enumerate(source.split("\n"))]
            # Strip trailing empty string if produced by split on trailing newline
            if lines and lines[-1] == "":
                lines.pop()

            cell["source"] = lines

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    
    print(f"  Successfully patched {file_path}.")

def main():
    base_dir = Path(__file__).resolve().parent.parent
    patch_notebook(base_dir / "notebooks" / "03_classification.ipynb")
    patch_notebook(base_dir / "notebooks" / "04_stacking classifier.ipynb")

if __name__ == "__main__":
    main()
