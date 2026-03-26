from src.classification import EmailClassifier
from src.ner import NERExtractor


class EmailAnalyzer:
    def __init__(self):
        self.classifier = EmailClassifier()
        self.ner = NERExtractor()

    def analyze(self, text):
        pred = self.classifier.predict(text)

        if pred == 1:
            return {
                "importance": "Important",
                "entities": self.ner.extract_with_display_names(text)
            }

        return {
            "importance": "Not Important",
            "entities": None
        }