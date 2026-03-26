import json
import os
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parent.parent
    paths = [
        base_dir / 'notebooks' / '03_classification.ipynb',
        base_dir / 'notebooks' / '04_stacking classifier.ipynb'
    ]

    for p in paths:
        if not p.exists():
            continue
            
        with open(p, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        for cell in nb.get('cells', []):
            if cell['cell_type'] == 'code':
                source = "".join(cell['source'])
                
                bad_string1 = "model_st = SentenceTransformer('all-MiniLM-L6-v2'),\n    stop_words=\"english\",\n    min_df=5\n)\n"
                bad_string2 = "model_st = SentenceTransformer('all-MiniLM-L6-v2'),\n    ngram_range=(1,2),\n    stop_words=\"english\",\n    min_df=5\n)\n"
                bad_string3 = "model_st = SentenceTransformer('all-MiniLM-L6-v2'),\n    max_features=5000,\n    ngram_range=(1,2),\n    stop_words=\"english\",\n    min_df=5\n)\n"
                
                if bad_string1 in source:
                    source = source.replace(bad_string1, "model_st = SentenceTransformer('all-MiniLM-L6-v2')\n")
                if bad_string2 in source:
                    source = source.replace(bad_string2, "model_st = SentenceTransformer('all-MiniLM-L6-v2')\n")
                if bad_string3 in source:
                    source = source.replace(bad_string3, "model_st = SentenceTransformer('all-MiniLM-L6-v2')\n")

                lines = []
                parts = source.split('\n')
                for i, p_part in enumerate(parts):
                    if i < len(parts) - 1:
                        lines.append(p_part + '\n')
                    elif p_part:
                        lines.append(p_part)
                
                cell['source'] = lines
                
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
            
    print("Fixed syntax errors!")

if __name__ == '__main__':
    main()
