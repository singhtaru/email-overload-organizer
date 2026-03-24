import joblib
import re
from pathlib import Path

class EmailClassifier:
    def __init__(self):
        base_dir = Path(__file__).resolve().parent.parent
        models_dir = base_dir / "models"
        self.tfidf = joblib.load(models_dir / "tfidf_vectorizer.pkl")
        self.model = joblib.load(models_dir / "stacking_model.pkl")
        self.negated_urgency_patterns = [
            re.compile(r"\bno\s+urgent\b", re.IGNORECASE),
            re.compile(r"\bnot\s+urgent\b", re.IGNORECASE),
            re.compile(r"\bnothing\s+urgent\b", re.IGNORECASE),
            re.compile(r"\bno\s+urgent\s+updates?\b", re.IGNORECASE),
            re.compile(r"\bno\s+immediate\s+action\b", re.IGNORECASE),
            re.compile(r"\bnot\s+time[- ]?sensitive\b", re.IGNORECASE),
        ]
        self.strong_importance_patterns = [
            re.compile(r"\burgent\b", re.IGNORECASE),
            re.compile(r"\bapproval\b", re.IGNORECASE),
            re.compile(r"\bschedule(?:d)?\b", re.IGNORECASE),
            re.compile(r"\breport\b", re.IGNORECASE),
            re.compile(r"\binvoice\b", re.IGNORECASE),
            re.compile(r"\bpayment\b", re.IGNORECASE),
            re.compile(r"\bdue\b", re.IGNORECASE),
            re.compile(r"\bdeadline\b", re.IGNORECASE),
            re.compile(r"\bmeeting\b", re.IGNORECASE),
            re.compile(r"\bcontract\b", re.IGNORECASE),
            re.compile(r"\bfinance\b", re.IGNORECASE),
            re.compile(r"\bbudget\b", re.IGNORECASE),
            re.compile(r"\ballocation\b", re.IGNORECASE),
            re.compile(r"\bimportant\b", re.IGNORECASE),
            re.compile(r"\bcritical\b", re.IGNORECASE),
            re.compile(r"\basap\b", re.IGNORECASE),
            re.compile(r"\breview\b", re.IGNORECASE),
            re.compile(r"\baction\s+required\b", re.IGNORECASE),
            re.compile(r"\battention\b", re.IGNORECASE),
            re.compile(r"\bbug\b", re.IGNORECASE),
            re.compile(r"\bissue\b", re.IGNORECASE),
            re.compile(r"\bpriority\b", re.IGNORECASE),
            re.compile(r"\bupdate\b", re.IGNORECASE),
            re.compile(r"\bfyi\b", re.IGNORECASE),
        ]

    def _normalize_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _rule_override(self, text):
        has_negated_urgency = any(p.search(text) for p in self.negated_urgency_patterns)
        has_strong_importance = any(p.search(text) for p in self.strong_importance_patterns)
        if has_strong_importance:
            return 1
        if has_negated_urgency and not has_strong_importance:
            return 0
        return None

    def predict(self, text):
        normalized_text = self._normalize_text(text)
        override = self._rule_override(normalized_text)
        if override is not None:
            return override
        vec = self.tfidf.transform([normalized_text])
        
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(vec)[0][1]
            # Lower threshold since original weak labels made model overly strict
            return 1 if proba >= 0.15 else 0
            
        return self.model.predict(vec)[0]
