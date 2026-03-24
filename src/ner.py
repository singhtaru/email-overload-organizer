import spacy

class NERExtractor:
    def __init__(self):
        self.nlp = spacy.load(
            "en_core_web_sm",
            disable=["parser", "tagger", "lemmatizer", "attribute_ruler"],
        )

    def extract(self, text):
        doc = self.nlp(text)

        entities = {
            "PERSON": [],
            "ORG": [],
            "DATE": [],
            "TIME": [],
            "MONEY": [],
            "GPE": []
        }

        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)

        return {k: list(set(v)) for k, v in entities.items()}
