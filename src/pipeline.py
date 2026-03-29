from src.classification import EmailClassifier
from src.ner import NERExtractor
import re


class EmailAnalyzer:
    """
    Architecture (primary → secondary):
      1. Classifier: Important (1) / Not Important (0)
      2. If Not Important → priority Low (NER skipped for speed)
      3. If Important → run NER → High vs Medium from entity signals only
    """

    def __init__(self):
        self.classifier = EmailClassifier()
        self.ner = NERExtractor()
        self.absolute_date_pattern = re.compile(
            r"\b\d{1,2}(st|nd|rd|th)?\s+"
            r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
            r"\s+\d{2,4}\b",
            re.IGNORECASE,
        )
        self.numeric_date_pattern = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")
        self.deadline_line_pattern = re.compile(
            r"(last\s*date|deadline)\s*(for\s*\w+)?\s*[:\-]?\s*(.+)",
            re.IGNORECASE,
        )
        self.requirement_patterns = (
            re.compile(r"\bminimum\s+\d+(\.\d+)?\s*cgpa\b", re.IGNORECASE),
            re.compile(r"\b\d+(\.\d+)?\s*cgpa\b", re.IGNORECASE),
            re.compile(r"\bno\s+standing\s+arrears\b", re.IGNORECASE),
            re.compile(r"\beligible\s+branches\b", re.IGNORECASE),
        )

    def _priority_from_ner(self, entities):
        """
        When classifier says Important, map NER output to High vs Medium.
        High = time-bound, financial, or scheduled-event signals from entities.
        Medium = important but weaker NER urgency (e.g. org/person only).
        """
        if not entities:
            return "Medium", "ner:empty_defaults_medium"

        if entities.get("Dates") or entities.get("Times"):
            return "High", "ner:dates_or_times"
        if entities.get("Money amounts"):
            return "High", "ner:money"
        if entities.get("Events"):
            return "High", "ner:events"

        return "Medium", "ner:other_entities"

    def _summarize(self, text):
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return ""
        if len(lines) == 1:
            return lines[0][:200]
        return f"{lines[0]} | {lines[1][:140]}"

    def _extract_deadline(self, text, entities):
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for idx, line in enumerate(lines):
            match = self.deadline_line_pattern.search(line)
            if match:
                tail = match.group(3).strip()
                if tail.lower() in {"for", "of", "on", "by"} and idx + 1 < len(lines):
                    tail = lines[idx + 1].strip()
                if len(tail) <= 2 and idx + 1 < len(lines):
                    tail = lines[idx + 1].strip()
                date_match = self.absolute_date_pattern.search(tail) or self.numeric_date_pattern.search(tail)
                if date_match:
                    return date_match.group(0)
                if tail and len(tail) >= 4:
                    return tail[:60]

        absolute_date = self.absolute_date_pattern.search(text)
        if absolute_date:
            return absolute_date.group(0)
        numeric_date = self.numeric_date_pattern.search(text)
        if numeric_date:
            return numeric_date.group(0)

        if "Dates" in entities and entities["Dates"]:
            for value in entities["Dates"]:
                if self.absolute_date_pattern.search(value) or self.numeric_date_pattern.search(value):
                    return value
        return None

    def _extract_event_name(self, text):
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in lines:
            lowered = line.lower()
            if any(token in lowered for token in ("internship", "registration", "campus drive", "interview", "drive")):
                return line
        return None

    def _extract_organization(self, entities):
        orgs = entities.get("Organizations", [])
        return orgs[0] if orgs else None

    def _extract_required_action(self, text, priority):
        text_lower = text.lower()
        if "register" in text_lower or "registration" in text_lower:
            return "Register for this opportunity."
        if "apply" in text_lower:
            return "Submit your application."
        if "deadline" in text_lower or "last date" in text_lower:
            return "Complete the required steps before the deadline."
        if priority == "Low":
            return "No immediate action needed."
        return "Review this email and take action if needed."

    def _extract_requirement(self, text):
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for idx, line in enumerate(lines):
            if "eligibility criteria" in line.lower() and idx + 1 < len(lines):
                next_line = lines[idx + 1]
                if len(next_line) <= 100:
                    return next_line
        for pattern in self.requirement_patterns:
            match = pattern.search(text)
            if match:
                return match.group(0)
        return None

    def _build_reason(self, priority, text, importance_label):
        text_lower = text.lower()
        if importance_label == "Not Important":
            return "The classifier marked this as not important, so it is treated as low priority."
        if priority == "High":
            if "deadline" in text_lower or "last date" in text_lower:
                return "The classifier marked this as important, and named entities suggest a deadline or time-sensitive detail."
            return "The classifier marked this as important, and named entities suggest higher urgency (dates, money, or events)."
        if priority == "Medium":
            return "The classifier marked this as important; named entities did not show strong time or money pressure, so priority is medium."
        return "This email looks informational and does not appear urgent."

    def _build_signals(self, text, pred_meta, priority_tier_source):
        """Structured lines for UI (classifier path, scores, keyword cues)."""
        rows = []
        src = pred_meta.get("source")
        if src:
            rows.append(("Decision path", str(src)))

        if pred_meta.get("confidence") is not None:
            p = float(pred_meta["confidence"])
            rows.append(("Stacking P(important)", f"{p:.1%}"))
        if pred_meta.get("adjusted_confidence") is not None:
            adj = float(pred_meta["adjusted_confidence"])
            rows.append(("After soft boosts", f"{adj:.1%}"))
        if pred_meta.get("boost") is not None and pred_meta.get("boost", 0) > 0:
            rows.append(("Boost amount", f"+{float(pred_meta['boost']):.3f}"))
        bt = pred_meta.get("boost_tags")
        if bt and bt != "none":
            rows.append(("Boost tags", str(bt)))

        if self.classifier._is_marketing_noise(text.lower()):
            rows.append(("Noise filter", "Promotional / newsletter cues present (rules prefer not important)"))

        negs = self.classifier.matched_negation_cues(text)
        for n in negs[:5]:
            rows.append(("Negated urgency", f"“{n}”"))

        cues = self.classifier.matched_importance_cues(text)
        for c in cues[:8]:
            rows.append(("Urgency cue", f"“{c}”"))

        if priority_tier_source:
            rows.append(("Priority tier source", str(priority_tier_source)))

        return rows

    def analyze(self, text):
        text_lower = text.lower()

        # -------- 1) Classifier: Important / Not Important (PRIMARY) --------
        pred_meta = self.classifier.predict_with_meta(text)
        pred = pred_meta["label"]
        importance = "Important" if pred == 1 else "Not Important"

        priority_tier_source = None
        if pred == 0:
            priority = "Low"
            entities = {}
            priority_tier_source = "skipped_ner_not_important"
        else:
            # -------- 2) NER → High / Medium only when Important --------
            entities = self.ner.extract(text)
            priority, priority_tier_source = self._priority_from_ner(entities)

        ner_signal_score = self.ner.strong_signal_count(entities) if entities else 0

        # -------- SUMMARY --------
        summary = self._summarize(text)
        deadline = self._extract_deadline(text, entities)
        event_name = self._extract_event_name(text)
        organization = self._extract_organization(entities)
        requirement = self._extract_requirement(text)
        required_action = self._extract_required_action(text, priority)
        reason = self._build_reason(priority, text, importance)

        suggested_action = required_action
        if priority == "High" and deadline:
            suggested_action = f"{required_action} Try to finish this before {deadline}."

        return {
            "importance": importance,
            "classification": priority,
            "entities": entities if entities else {},
            "summary": summary.strip(),
            "explanation": reason,
            "key_details": {
                "Deadline": deadline,
                "Event name": event_name,
                "Organization": organization,
                "Requirement": requirement,
                "Required action": required_action,
            },
            "suggested_action": suggested_action,
            "classifier_source": pred_meta.get("source"),
            "model_confidence": pred_meta.get("confidence"),
            "adjusted_confidence": pred_meta.get("adjusted_confidence"),
            "priority_tier_source": priority_tier_source,
            "ner_signal_score": ner_signal_score,
            "signals": self._build_signals(text, pred_meta, priority_tier_source),
        }
