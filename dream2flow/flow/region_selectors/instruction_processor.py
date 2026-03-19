class InstructionProcessor:
    def process_for_grounding_dino(self, object_name: str) -> str:
        object_name = object_name.strip()
        if not object_name:
            raise ValueError("object_name must be a non-empty string.")
        if not object_name.endswith("."):
            object_name += "."
        return object_name
