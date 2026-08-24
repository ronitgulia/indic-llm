# Indic LLM — Safety, Content Moderation & Guardrails

## 1. Overview
The Indic LLM guardrail pipeline provides defense-in-depth across input sanitization and output moderation:
- **PII Redaction:** Indian Aadhaar (Verhoeff-validated), PAN card, phone numbers (+91), and emails.
- **Toxicity Filtering:** Devanagari and Hinglish abusive lexicon matching with continuous score computation.
- **Prompt Injection Defense:** Multi-lingual pattern detection for system prompt override attempts.
