import spacy


class NERExtractor:
    # Entity labels to extract, with display-friendly names
    TARGET_LABELS = {
        "PERSON": "People",
        "ORG": "Organizations",
        "DATE": "Dates",
        "TIME": "Times",
        "MONEY": "Amounts",
        "GPE": "Locations",
        "EVENT": "Events",
        "FAC": "Facilities",
    }

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def extract(self, text):
        doc = self.nlp(text)

        entities = {}

        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []

            if ent.text not in entities[ent.label_]:
                entities[ent.label_].append(ent.text)

        return entities