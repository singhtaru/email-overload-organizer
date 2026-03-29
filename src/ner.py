import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import spacy
from spacy.language import Language

try:
    import dateparser
except ImportError:
    dateparser = None  # type: ignore


def _build_nlp() -> Language:
    """Prefer larger English models; fall back if not installed. Parser excluded — not needed for NER."""
    last_err: Optional[OSError] = None
    nlp: Optional[Language] = None
    for name in ("en_core_web_lg", "en_core_web_md", "en_core_web_sm"):
        try:
            nlp = spacy.load(name, exclude=["parser"])
            break
        except OSError as e:
            last_err = e
    if nlp is None:
        raise OSError(
            "No spaCy English model found. Install one of: "
            "python -m spacy download en_core_web_lg"
        ) from last_err

    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        # Token-only patterns (no ENT_TYPE — ruler runs *before* statistical NER).
        patterns = [
            {"label": "DATE", "pattern": [{"LOWER": "due"}, {"LOWER": "by"}, {"LIKE_NUM": True}]},
            {"label": "DATE", "pattern": [{"LOWER": "due"}, {"LOWER": "on"}, {"LIKE_NUM": True}]},
            {
                "label": "MONEY",
                "pattern": [
                    {
                        "LOWER": {
                            "IN": [
                                "invoice",
                                "payment",
                                "amount",
                                "fee",
                                "cost",
                                "salary",
                                "stipend",
                                "price",
                            ]
                        }
                    },
                    {"IS_PUNCT": True, "OP": "?"},
                    {"LIKE_NUM": True},
                ],
            },
            {
                "label": "MONEY",
                "pattern": [{"LOWER": {"IN": ["usd", "eur", "gbp", "inr", "rs"]}}, {"LIKE_NUM": True}],
            },
        ]
        ruler.add_patterns(patterns)

    return nlp


class NERExtractor:
    TARGET_LABELS = {
        "PERSON": "People involved",
        "ORG": "Organizations",
        "DATE": "Dates",
        "TIME": "Times",
        "MONEY": "Money amounts",
        "GPE": "Places",
        "LOC": "Places",
        "EVENT": "Events",
        "FAC": "Buildings and facilities",
    }

    NOISE_TOKENS = frozenset(
        {
            "hi",
            "hello",
            "dear",
            "thanks",
            "thank",
            "regards",
            "sincerely",
            "best",
            "cheers",
            "morning",
            "afternoon",
            "evening",
        }
    )
    MIN_ORG_LENGTH = 3

    def __init__(self):
        self.nlp = _build_nlp()
        self.stop_entities = {
            "eligible branches",
            "eligibility criteria",
            "category",
            "ctc",
            "stipend",
            "website",
            "last date",
            "date of visit",
            "pursuing degree",
            "in ug",
            "ug",
            "x",
            "xii",
        }
        self.org_noise_contains = (
            "engineering",
            "science",
            "integrated",
            "branch",
            "criteria",
            "degree",
            "cgpa",
            "arrears",
        )

    @staticmethod
    def _normalize_entity_text(_spacy_label: str, text: str) -> str:
        t = re.sub(r"\s+", " ", text).strip()
        t = t.strip(' \t.,;:!?\'"')
        return t

    def _maybe_resolve_date(self, text: str) -> str:
        if not dateparser:
            return text
        s = text.strip()
        if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
            return s
        # Do not normalize bare "due by N" fragments or other substrings that parse poorly.
        if re.search(r"\bdue\s+by\b", s, re.IGNORECASE):
            return text
        parsed = dateparser.parse(
            s,
            settings={
                "PREFER_DATES_FROM": "future",
                "RELATIVE_BASE": datetime.now(),
            },
        )
        if parsed is None:
            return text
        try:
            return parsed.strftime("%Y-%m-%d")
        except (ValueError, OSError):
            return text

    def _is_valid_entity(self, text: str, display_label: str) -> bool:
        cleaned = text.strip()
        lowered = cleaned.lower()
        if not cleaned:
            return False
        if lowered in self.stop_entities:
            return False
        if lowered in self.NOISE_TOKENS:
            return False
        if len(cleaned) <= 1:
            return False
        if cleaned.isdigit():
            return False
        if display_label == "Organizations":
            if len(cleaned) < self.MIN_ORG_LENGTH:
                return False
            if "\n" in cleaned:
                return False
            if any(word in lowered for word in self.stop_entities):
                return False
            if any(token in lowered for token in self.org_noise_contains):
                return False
            if cleaned.isupper() and len(cleaned) <= 4:
                return False
        return True

    def _normalize_for_ner(self, text):
        # Reduce noisy email formatting so entity spans do not absorb section headers.
        text = re.sub(r"[\u2022Ø]", " ", text)
        text = re.sub(r"\n{2,}", ". ", text)
        text = re.sub(r"\n", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _extract_notice_company(self, raw_text):
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        for idx, line in enumerate(lines):
            if "name of the company" in line.lower() and idx + 1 < len(lines):
                candidate = lines[idx + 1].strip(":- ")
                if self._is_valid_entity(candidate, "Organizations"):
                    return self._normalize_entity_text("ORG", candidate)
        return None

    def extract(self, text) -> Dict[str, List[str]]:
        normalized_text = self._normalize_for_ner(text)
        doc = self.nlp(normalized_text)

        entities: Dict[str, List[str]] = {}

        for ent in doc.ents:
            display_label = self.TARGET_LABELS.get(ent.label_)
            if not display_label:
                continue

            cleaned_text = self._normalize_entity_text(ent.label_, ent.text.strip())
            if not self._is_valid_entity(cleaned_text, display_label):
                continue

            if ent.label_ == "DATE" and dateparser:
                cleaned_text = self._maybe_resolve_date(cleaned_text)

            if display_label not in entities:
                entities[display_label] = []

            if cleaned_text not in entities[display_label]:
                entities[display_label].append(cleaned_text)

        company = self._extract_notice_company(text)
        if company:
            entities.setdefault("Organizations", [])
            if company not in entities["Organizations"]:
                entities["Organizations"].insert(0, company)

        return entities

    def strong_signal_count(self, entities: Dict[str, Any]) -> int:
        """How many urgency buckets are non-empty (dates, times, money, events)."""
        keys = ("Dates", "Times", "Money amounts", "Events")
        return sum(1 for k in keys if entities.get(k))
