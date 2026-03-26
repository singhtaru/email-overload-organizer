import spacy
import re

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
        # Use md for better accuracy — install with:
        # python -m spacy download en_core_web_md
        # Fallback note: swap to en_core_web_sm if md is unavailable
        self.nlp = spacy.load(
            "en_core_web_md",
            disable=["parser", "tagger", "lemmatizer"],
            # attribute_ruler kept ON — helps with entity tokenization edge cases
        )

    def _clean_email_text(self, text: str) -> str:
        """
        Strip quoted reply chains, forwarded headers, and email signatures
        before running NER. These are the biggest sources of phantom entities.
        """
        # Remove quoted reply lines (lines starting with >)
        text = re.sub(r'^>.*$', '', text, flags=re.MULTILINE)

        # Remove forwarded message headers
        text = re.sub(
            r'-{3,}.*?(forwarded|original message).*?-{3,}',
            '',
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

        # Remove common signature patterns
        text = re.sub(
            r'\n(regards|thanks|sincerely|best|cheers|warm regards)'
            r'[\s\S]{0,300}$',
            '',
            text,
            flags=re.IGNORECASE
        )

        # Collapse excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _is_valid_entity(self, text: str, label: str) -> bool:
        """
        Filter out garbage entities — single chars, pure numbers tagged
        as ORG/PERSON, extremely long strings that are likely noise.
        """
        text = text.strip()

        # Too short or too long
        if len(text) < 2 or len(text) > 100:
            return False

        # Pure numeric strings shouldn't be PERSON or ORG
        if label in ("PERSON", "ORG", "EVENT") and text.replace(" ", "").isnumeric():
            return False

        # Single-word all-lowercase entities are usually not real named entities
        if label == "PERSON" and text.islower() and " " not in text:
            return False

        return True

    def extract(self, text: str) -> dict:
        """
        Extract named entities from email text.
        Returns only non-empty entity categories.
        """
        cleaned = self._clean_email_text(text)
        doc = self.nlp(cleaned)

        entities = {label: set() for label in self.TARGET_LABELS}

        for ent in doc.ents:
            if ent.label_ in entities:
                cleaned_ent = ent.text.strip()
                if self._is_valid_entity(cleaned_ent, ent.label_):
                    entities[ent.label_].add(cleaned_ent)

        # Return only non-empty categories, with lists (not sets) for JSON safety
        return {
            label: sorted(list(values))
            for label, values in entities.items()
            if values
        }

    def extract_with_display_names(self, text: str) -> dict:
        """
        Same as extract() but uses human-readable keys.
        Useful for demo output / frontend display.
        """
        raw = self.extract(text)
        return {
            self.TARGET_LABELS[label]: values
            for label, values in raw.items()
        }