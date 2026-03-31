import joblib
import os
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Calibrated for stacking outputs that are often low; tune with stress tests.
IMPORTANCE_PROBA_THRESHOLD = 0.20


def _hf_hub_offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes")


class EmailClassifier:
    def __init__(self):
        base_dir = Path(__file__).resolve().parent.parent
        models_dir = base_dir / "models"
        # Vendored snapshot (optional): models/all-MiniLM-L6-v2 with config + weights
        local_sbert = models_dir / "all-MiniLM-L6-v2"
        if local_sbert.is_dir():
            sbert_id = str(local_sbert)
            local_only = True
        else:
            sbert_id = "sentence-transformers/all-MiniLM-L6-v2"
            local_only = _hf_hub_offline()
        self.sbert = SentenceTransformer(
            sbert_id,
            local_files_only=local_only,
        )
        self.model = joblib.load(models_dir / "embedding_stacking_model.pkl")

        # Promotional / newsletter noise — checked BEFORE keyword "importance" style rules.
        self.marketing_not_important_patterns = [
            re.compile(r"\bunsubscribe\b", re.IGNORECASE),
            re.compile(r"\bcontinue receiving newsletters\b", re.IGNORECASE),
            re.compile(r"\bnewsletter\b.*\bpreferences\b", re.IGNORECASE),
            re.compile(r"\bexclusive\s+\d+%\s*(off|discount)", re.IGNORECASE),
            re.compile(r"\blimited[- ]time\s+(offer|deal|offers)\b", re.IGNORECASE),
            re.compile(r"\bshop\s+now\b", re.IGNORECASE),
            re.compile(r"\bdon'?t\s+miss\s+out\b", re.IGNORECASE),
            re.compile(r"\bemployee\s+discount\s+program\b", re.IGNORECASE),
            re.compile(r"\bmembership\s+discount\b", re.IGNORECASE),
            re.compile(r"\bcomplete this survey\b", re.IGNORECASE),
            re.compile(r"\bsubscribed to newsletter\b", re.IGNORECASE),
            re.compile(r"\bnewsletter submission\b.*\boptional\b", re.IGNORECASE),
            re.compile(r"\b50%\s*discount\b", re.IGNORECASE),
        ]

        self.negated_urgency_patterns = [
            re.compile(r"\bno\s+urgent\b", re.IGNORECASE),
            re.compile(r"\bnot\s+urgent\b", re.IGNORECASE),
            re.compile(r"\bnothing\s+urgent\b", re.IGNORECASE),
            re.compile(r"\bno\s+urgent\s+updates?\b", re.IGNORECASE),
            re.compile(r"\bno\s+immediate\s+action\b", re.IGNORECASE),
            re.compile(r"\bnot\s+time[- ]?sensitive\b", re.IGNORECASE),
            re.compile(r"\bno\s+action\s+needed\b", re.IGNORECASE),
            re.compile(r"\bno\s+rush\b", re.IGNORECASE),
        ]

        # Work-urgency cues (do NOT include bare "important" — spam subjects abuse it).
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
            re.compile(r"\bcritical\b", re.IGNORECASE),
            re.compile(r"\basap\b", re.IGNORECASE),
            re.compile(r"\baction\s+required\b", re.IGNORECASE),
            re.compile(r"\battention\b", re.IGNORECASE),
            re.compile(r"\bbug\b", re.IGNORECASE),
            re.compile(r"\bissue\b", re.IGNORECASE),
            re.compile(r"\bpriority\b", re.IGNORECASE),
        ]

    def _normalize_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _is_marketing_noise(self, raw_lower):
        """Use raw text (lowered) so %, punctuation, etc. stay visible."""
        return any(p.search(raw_lower) for p in self.marketing_not_important_patterns)

    def _soft_boost_probability(self, raw_text, proba):
        """Light boosts for short / implicit work mail (no retraining)."""
        t = raw_text.lower()
        boost = 0.0
        tags = []

        if re.search(
            r"\b(today|tomorrow|tonight|eod|end of day|this morning|this afternoon|before midnight|by \d)\b",
            t,
        ):
            boost += 0.08
            tags.append("time")

        if re.search(
            r"\b(please review|please confirm|need your|need confirmation|can you|could you|let me know|before i send|before sending)\b",
            t,
        ):
            boost += 0.06
            tags.append("request")

        if re.search(r"(?m)^subject:\s*re:", raw_text, re.IGNORECASE):
            boost += 0.04
            tags.append("reply")

        if "?" in raw_text:
            boost += 0.03
            tags.append("question")

        if "fyi" in t and any(
            x in t for x in ("client", "proposal", "deadline", "tomorrow", "today", "contract", "legal")
        ):
            boost += 0.04
            tags.append("fyi_work")

        adjusted = min(0.99, proba + boost)
        return adjusted, "+".join(tags) if tags else "none", boost

    def _rule_override(self, raw_text, text_norm):
        if self._is_marketing_noise(raw_text.lower()):
            return 0, "rule:marketing_noise"

        has_negated = any(p.search(text_norm) for p in self.negated_urgency_patterns)
        has_strong = any(p.search(text_norm) for p in self.strong_importance_patterns)
        if has_strong:
            return 1, "rule:strong_importance"
        if has_negated and not has_strong:
            return 0, "rule:negated_urgency"
        return None, None

    def predict_with_meta(self, text):
        raw_text = text if isinstance(text, str) else ""
        normalized_text = self._normalize_text(raw_text)
        override, source = self._rule_override(raw_text, normalized_text)
        if override is not None:
            return {
                "label": int(override),
                "confidence": None,
                "adjusted_confidence": None,
                "source": source,
            }

        vec = self.sbert.encode([normalized_text], show_progress_bar=False)

        if hasattr(self.model, "predict_proba"):
            proba = float(self.model.predict_proba(vec)[0][1])
            adjusted, boost_tags, boost_amt = self._soft_boost_probability(raw_text, proba)
            label = 1 if adjusted >= IMPORTANCE_PROBA_THRESHOLD else 0
            return {
                "label": int(label),
                "confidence": proba,
                "adjusted_confidence": adjusted,
                "boost": boost_amt,
                "boost_tags": boost_tags,
                "source": "stacking_model:predict_proba",
            }

        pred = int(self.model.predict(vec)[0])
        return {
            "label": pred,
            "confidence": None,
            "adjusted_confidence": None,
            "source": "stacking_model:predict",
        }

    def predict(self, text):
        return self.predict_with_meta(text)["label"]

    def matched_importance_cues(self, raw_text: str, max_hits: int = 12) -> list:
        """Surface which urgency-style patterns matched (for explainability UI)."""
        if not raw_text or not isinstance(raw_text, str):
            return []
        norm = self._normalize_text(raw_text)
        seen = set()
        hits = []
        for p in self.strong_importance_patterns:
            m = p.search(norm)
            if m:
                key = m.group(0).strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    hits.append(m.group(0).strip())
            if len(hits) >= max_hits:
                break
        return hits

    def matched_negation_cues(self, raw_text: str) -> list:
        if not raw_text or not isinstance(raw_text, str):
            return []
        norm = self._normalize_text(raw_text)
        out = []
        for p in self.negated_urgency_patterns:
            m = p.search(norm)
            if m:
                out.append(m.group(0).strip())
        return out
