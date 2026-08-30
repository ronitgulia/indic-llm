# Indic LLM - Chat Input Normalization
from src.transliterate import detect_script, roman_to_devanagari, ScriptType

def normalize_chat_input(text: str) -> str:
    if detect_script(text) == ScriptType.LATIN:
        return roman_to_devanagari(text)
    return text
