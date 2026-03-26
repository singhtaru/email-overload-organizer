from src.classification import EmailClassifier
from src.ner import NERExtractor


class EmailAnalyzer:
    def __init__(self):
        self.classifier = EmailClassifier()
        self.ner = NERExtractor()

    def analyze(self, text):
        text_lower = text.lower()

        # -------- ML prediction --------
        pred = self.classifier.predict(text)

        if pred == 1:
            priority = "Medium"
        else:
            priority = "Low"

        # -------- Rule-based override --------
        if any(word in text_lower for word in ["urgent", "asap", "immediately"]):
            priority = "High"
        elif any(word in text_lower for word in ["deadline", "tomorrow", "today"]):
            priority = "High"

        # -------- NER --------
        entities = self.ner.extract(text)

        # -------- SUMMARY --------
        sentences = text.split(".")
        summary = sentences[0] if sentences else text

        if len(summary) > 150:
            summary = summary[:150] + "..."

        return {
            "classification": priority,
            "entities": entities if entities else {},
            "summary": summary.strip()
        }