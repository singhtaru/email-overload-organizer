import pandas as pd
from pathlib import Path
from src.pipeline import EmailAnalyzer

base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "data" / "processed"

# Load dataset
df = pd.read_csv(data_dir / "emails_with_labels.csv")

# Create input text
df["text"] = df["clean_subject"].fillna("") + " " + df["clean_body"].fillna("")

analyzer = EmailAnalyzer()

# Run on sample (change to full later)
results = []

for text in df["text"].head(1000):
    results.append(analyzer.analyze(text))

# Convert results
results_df = pd.DataFrame(results)
df_sample = df.head(1000).reset_index(drop=True)

final_df = pd.concat([df_sample, results_df], axis=1)

# Save output
final_df.to_csv(data_dir / "predictions.csv", index=False)

print(final_df[["text", "importance"]].head())
