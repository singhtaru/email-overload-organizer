import spacy


class NERExtractor:
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