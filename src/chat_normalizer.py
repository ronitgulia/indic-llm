# Indic LLM - Chat Input Normalization
from src.transliterate import ScriptType, detect_script, roman_to_devanagari


def normalize_chat_input(text: str) -> str:
    if detect_script(text) == ScriptType.LATIN:
        return roman_to_devanagari(text)
    return text
