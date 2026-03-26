import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import os

def main():
    print("Loading data...")
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "processed" / "emails_with_labels.csv"
    
    df = pd.read_csv(data_path)
    
    # Fill NAs
    df['clean_subject'] = df['clean_subject'].fillna('')
    df['clean_body'] = df['clean_body'].fillna('')
    df['text'] = df['clean_subject'] + ' ' + df['clean_body']
    
    # We want a balanced subset to speed up embedding computation
    # 10k class 0, 10k class 1 (or max available if less)
    target_count = 10000
    class0_pool = df[df['label'] == 0]
    class1_pool = df[df['label'] == 1]
    
    df_class0 = class0_pool.sample(n=min(target_count, len(class0_pool)), random_state=42)
    df_class1 = class1_pool.sample(n=min(target_count, len(class1_pool)), random_state=42)
    
    df_subset = pd.concat([df_class0, df_class1]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    X_text = df_subset['text'].tolist()
    y = df_subset['label'].values
    
    print(f"Loading SentenceTransformer... (using {len(X_text)} emails)")
    # Loads the tiny, fast all-MiniLM-L6-v2 model automatically
    model_st = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Encoding texts... (This might take a couple of minutes)")
    X_embed = model_st.encode(X_text, show_progress_bar=True, batch_size=64)
    
    X_train, X_test, y_train, y_test = train_test_split(X_embed, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training Stacking Classifier on Semantic Dependencies...")
    lr = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    svm = LinearSVC(random_state=42, dual='auto')
    
    stacking_model = StackingClassifier(
        estimators=[
            ('lr', lr),
            ('rf', rf),
            ('svm', svm)
        ],
        final_estimator=LogisticRegression(),
        cv=5,
        n_jobs=-1
    )
    
    stacking_model.fit(X_train, y_train)
    
    acc = stacking_model.score(X_test, y_test)
    print(f"Model trained! Test Accuracy: {acc:.4f}")
    
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    save_path = models_dir / "embedding_stacking_model.pkl"
    joblib.dump(stacking_model, save_path)
    print(f"Model saved successfully to {save_path}")

if __name__ == '__main__':
    main()
