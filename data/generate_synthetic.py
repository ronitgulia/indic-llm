"""Indic Synthetic Data Generation & Instruction Evolution Engine.

Implements Evol-Instruct methodology adapted for low-resource Indic languages,
generating diverse, culturally grounded, and syntactically rich instruction pairs.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("indic_llm.synthetic")


@dataclass
class SyntheticSample:
    """Represents a generated synthetic instruction-response pair with metadata."""
    instruction: str
    response: str
    language: str
    mutation_type: str
    quality_score: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InstructionMutator:
    """Mutates seed instructions using Indic-specific evolution strategies."""

    MUTATION_TYPES = [
        "concretize",
        "deepen_constraints",
        "increase_reasoning_steps",
        "multilingual_cross_lingual",
    ]

    CULTURAL_ANCHORS: dict[str, list[str]] = {
        "hi": ["भारतीय इतिहास", "योग और आयुर्वेद", "मौसमी कृषि और मानसून", "शास्त्रीय संगीत और नृत्य", "पंचायती राज व्यवस्था"],
        "bn": ["রবীন্দ্রসঙ্গীত ও সাহিত্য", "বাংলার উৎসব ও সংস্কৃতি", "সুন্দরবন ও বাস্তুতন্ত্র", "ঐতিহাসিক নদীমাতৃক ভূগোল", "বিজ্ঞান ও সমাজ সংস্কার"],
        "ta": ["சங்க இலக்கியம் மற்றும் வரலாறு", "பாரம்பரிய மருத்துவம்", "தமிழ் கட்டிடக்கலை மற்றும் சிற்பங்கள்", "விவசாயம் மற்றும் நீர்ப்பாசனம்", "கிராமப்புற நிர்வாகம்"],
        "te": ["తెలుగు సాహిత్యం మరియు పద్యాలు", "ఆంధ్ర చరిత్ర మరియు సంస్కృతి", "వ్యవసాయ పద్ధతులు మరియు నీటి నిర్వహణ", "జానపద కళలు మరియు పండుగలు", "ఆధునిక శాస్త్ర సాంకేతిక పరిజ్ఞానం"],
        "gu": ["ગુજરાતી સાહિત્ય અને ઇતિહાસ", "વેપાર અને વાણિજ્ય નીતિઓ", "સૌરાષ્ટ્રની લોકસંસ્કૃતિ", "સહકારી મંડળીઓ અને કૃષિ", "પર્યાવરણ સંરક્ષણ"],
        "mr": ["महाराष्ट्राचा इतिहास आणि दुर्ग", "संत साहित्य आणि वारकरी परंपरा", "शेतकरी जीवन आणि जलसंधारण", "मराठी नाट्य आणि चित्रपट", "स्थानिक स्वराज्य संस्था"],
    }

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def mutate(self, instruction: str, mutation_type: str, language: str = "hi") -> str:
        """Mutate an instruction based on the requested mutation type and language."""
        if mutation_type == "concretize":
            return self._concretize(instruction, language)
        elif mutation_type == "deepen_constraints":
            return self._deepen_constraints(instruction, language)
        elif mutation_type == "increase_reasoning_steps":
            return self._increase_reasoning(instruction, language)
        elif mutation_type == "multilingual_cross_lingual":
            return self._cross_lingual(instruction, language)
        return instruction

    def _concretize(self, instruction: str, language: str) -> str:
        anchors = self.CULTURAL_ANCHORS.get(language, self.CULTURAL_ANCHORS["hi"])
        anchor = self.rng.choice(anchors)
        if language == "hi":
            return f"{instruction} इसे {anchor} के विशेष संदर्भ में एक वास्तविक उदाहरण देकर स्पष्ट करें।"
        elif language == "bn":
            return f"{instruction} এটিকে {anchor}-এর বিশেষ প্রেক্ষাপটে বাস্তব উদাহরণ দিয়ে ব্যাখ্যা করুন।"
        elif language == "ta":
            return f"{instruction} இதை {anchor} என்ற பின்னணியில் நடைமுறை உதாரணத்துடன் விளக்குக."
        elif language == "te":
            return f"{instruction} దీనిని {anchor} సందర్భంలో వాస్తవ ఉదాహరణతో వివరించండి."
        return f"{instruction} (संदर्भ: {anchor})"

    def _deepen_constraints(self, instruction: str, language: str) -> str:
        if language == "hi":
            return f"{instruction} उत्तर में कम से कम तीन मुख्य बिंदु शामिल करें, तकनीकी शब्दों की व्याख्या दें और निष्कर्ष संक्षेप में प्रस्तुत करें।"
        elif language == "bn":
            return f"{instruction} উত্তরে অন্তত তিনটি মূল বিষয় অন্তর্ভুক্ত করুন, পারিভাষিক শব্দের অর্থ দিন এবং সংক্ষেপে উপসংহার টানুন।"
        elif language == "ta":
            return f"{instruction} பதிலில் குறைந்தது மூன்று முக்கிய குறிப்புகள், கலைச்சொற்கள் விளக்கம் மற்றும் சுருக்கமான முடிவு இருக்க வேண்டும்."
        elif language == "te":
            return f"{instruction} సమాధానంలో కనీసం మూడు ముఖ్య అంశాలు, సాంకేతిక పదాల వివరణ మరియు సంక్షిప్త ముగింపు ఉండాలి."
        return f"{instruction} (3 मुख्य बिंदुओं और निष्कर्ष के साथ स्पष्ट करें)"

    def _increase_reasoning(self, instruction: str, language: str) -> str:
        if language == "hi":
            return f"चरण-दर-चरण तार्किक विश्लेषण प्रस्तुत करें: {instruction} प्रत्येक चरण का कारण स्पष्ट करें।"
        elif language == "bn":
            return f"ধাপে ধাপে যৌক্তিক বিশ্লেষণ প্রদান করুন: {instruction} প্রতিটি ধাপের কারণ ব্যাখ্যা করুন।"
        elif language == "ta":
            return f"படிப்படியான தர்க்கரீதியான பகுப்பாய்வை வழங்குக: {instruction} ஒவ்வொரு படியின் காரணத்தையும் விளக்குக."
        elif language == "te":
            return f"దశలవారీగా తార్కిక విశ్లేషణను అందించండి: {instruction} ప్రతి దశ వెనుక గల కారణాన్ని వివరించండి."
        return f"तार्किक विश्लेषण: {instruction}"

    def _cross_lingual(self, instruction: str, language: str) -> str:
        if language == "hi":
            return f"{instruction} कृपया उत्तर हिंदी में दें तथा प्रमुख पारिभाषिक शब्दों के अंग्रेजी समानार्थी भी कोष्ठक में लिखें।"
        elif language == "bn":
            return f"{instruction} অনুগ্রহ করে বাংলায় উত্তর দিন এবং মূল পারিভাষিক শব্দের ইংরেজি প্রতিশব্দ বন্ধনীতে লিখুন।"
        elif language == "ta":
            return f"{instruction} தயவுசெய்து தமிழில் பதிலளித்து, முக்கிய கலைச்சொற்களுக்கு அடைப்புக்குறிக்குள் ஆங்கில சொற்களையும் தருக."
        elif language == "te":
            return f"{instruction} దయచేసి తెలుగులో సమాధానమిస్తూ, కీలక సాంకేతిక పదాలకు బ్రాకెట్లలో ఆంగ్ల సమానార్థకాలను చేర్చండి."
        return f"{instruction} (द्विभाषी पारिभाषिक शब्दावली सहित)"
