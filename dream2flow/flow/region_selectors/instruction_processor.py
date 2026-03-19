from typing import Optional


class InstructionProcessor:
    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        try:
            import spacy
        except ImportError as exc:
            raise ImportError("spaCy is required for GroundingDINO instruction processing.") from exc

        try:
            self._nlp = spacy.load(model_name)
        except OSError:
            spacy.cli.download(model_name)
            self._nlp = spacy.load(model_name)

        self._action_verbs = {
            "open", "close", "push", "pull", "grasp", "grab", "pick", "move", "lift",
            "place", "put", "set", "position", "rotate", "turn", "twist", "press",
            "touch", "hold", "release", "drop", "throw", "catch", "hit", "strike",
            "reach", "get", "take", "bring", "carry", "slide", "roll", "spin",
        }

    def extract_object_noun(self, instruction: str) -> str:
        if not instruction or not instruction.strip():
            return instruction

        instruction = instruction.split(".")[0].strip()
        doc = self._nlp(instruction.strip())

        object_noun = self._extract_direct_object(doc)
        if object_noun:
            return object_noun

        object_noun = self._extract_object_after_verb(doc)
        if object_noun:
            return object_noun

        object_noun = self._extract_main_subject(doc)
        if object_noun:
            return object_noun

        object_noun = self._extract_first_noun_phrase(doc)
        if object_noun:
            return object_noun

        return self._extract_fallback_phrase(doc)

    def _extract_direct_object(self, doc) -> Optional[str]:
        for token in doc:
            if token.dep_ == "dobj" and token.pos_ in ["NOUN", "PROPN"]:
                noun_phrase = self._get_noun_phrase(doc, token)
                if noun_phrase:
                    return noun_phrase
        return None

    def _extract_object_after_verb(self, doc) -> Optional[str]:
        for token in doc:
            if token.pos_ == "VERB" and token.lemma_.lower() in self._action_verbs:
                for child in token.rights:
                    if child.pos_ in ["NOUN", "PROPN", "ADJ"]:
                        noun_phrase = self._get_noun_phrase(doc, child)
                        if noun_phrase:
                            return noun_phrase
        return None

    def _extract_main_subject(self, doc) -> Optional[str]:
        for token in doc:
            if token.dep_ in ["nsubj", "nsubjpass"] and token.pos_ in ["NOUN", "PROPN"]:
                noun_phrase = self._get_noun_phrase(doc, token)
                if noun_phrase:
                    return noun_phrase
        return None

    def _extract_first_noun_phrase(self, doc) -> Optional[str]:
        for chunk in doc.noun_chunks:
            cleaned = self._clean_noun_phrase(chunk.text)
            if cleaned:
                return cleaned
        return None

    def _extract_fallback_phrase(self, doc) -> str:
        meaningful_words = []
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN", "ADJ"] and not token.is_stop and len(meaningful_words) < 3:
                meaningful_words.append(token.text)
        if meaningful_words:
            return " ".join(meaningful_words)
        words = [token.text for token in doc if not token.is_space]
        return " ".join(words[:3]) if words else doc.text

    def _get_noun_phrase(self, doc, token) -> Optional[str]:
        root = token
        while root.head.pos_ in ["NOUN", "PROPN", "ADJ"] and root.head.dep_ in ["compound", "amod"]:
            root = root.head

        phrase_tokens = []
        for candidate in doc:
            if (candidate.head == root or candidate == root) and candidate.pos_ in ["NOUN", "PROPN", "ADJ", "DET"]:
                phrase_tokens.append(candidate)

        if phrase_tokens:
            phrase_tokens.sort(key=lambda item: item.i)
            phrase_text = " ".join([item.text for item in phrase_tokens])
            return self._clean_noun_phrase(phrase_text)
        return None

    def _clean_noun_phrase(self, phrase: str) -> str:
        phrase = phrase.lower().strip()
        phrase = phrase.replace("the ", "").replace("a ", "").replace("an ", "")
        stop_words = {
            "one", "hand", "finger", "arm", "leg", "foot", "eye", "ear", "nose", "mouth",
            "is", "are", "was", "were", "will", "can", "could", "should", "would",
            "this", "that", "these", "those", "it", "them", "they",
        }
        filtered_words = [word for word in phrase.split() if word.lower() not in stop_words]
        cleaned = " ".join(filtered_words)
        return cleaned.strip(".,;:!? ")

    def process_for_grounding_dino(self, instruction: str) -> str:
        object_noun = self.extract_object_noun(instruction)
        if not object_noun.endswith("."):
            object_noun += "."
        return object_noun
