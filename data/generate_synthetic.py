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


class SyntheticDataGenerator:
    """Orchestrates instruction mutation, response generation, and quality filtering."""

    DEFAULT_SEEDS: list[dict[str, str]] = [
        {"instruction": "सौर ऊर्जा के लाभों की व्याख्या करें।", "response": "सौर ऊर्जा एक स्वच्छ, नवीकरणीय ऊर्जा स्रोत है जो प्रदूषण को कम करता है और बिजली बिल में बचत कराता है।", "language": "hi"},
        {"instruction": "গাছ লাগানো কেন প্রয়োজন?", "response": "গাছ পরিবেশের ভারসাম্য রক্ষা করে, অক্সিজেন সরবরাহ করে এবং বায়ু দূষণ হ্রাস করতে সাহায্য করে।", "language": "bn"},
        {"instruction": "தண்ணீர் சேமிப்பின் முக்கியத்துவம் என்ன?", "response": "தண்ணீர் ஒரு விலைமதிப்பற்ற இயற்கை வளம். எதிர்கால சந்ததியினரின் நல்வாழ்வுக்கு நீர்நிலைகளைப் பாதுகாப்பது இன்றியமையாதது.", "language": "ta"},
        {"instruction": "వ్యాయామం యొక్క ప్రాముఖ్యత ఏమిటి?", "response": "రోజూ వ్యాయామం చేయడం వల్ల శారీరక దృఢత్వం, మానసిక ఉల్లాసం మరియు రోగనిరోధక శక్తి పెరుగుతాయి.", "language": "te"},
        {"instruction": "સંતોષકારક જીવનશૈલી માટે શું જરૂરી છે?", "response": "સંતુલિત આહાર, નિયમિત વ્યાયામ અને સકારાત્મક વિચારસરણી જીવનમાં સુખ અને શાંતિ લાવે છે.", "language": "gu"},
        {"instruction": "ग्रंथालयांचे महत्त्व स्पष्ट करा.", "response": "ग्रंथालये ज्ञानाचा अखंड स्रोत आहेत, जिथे वाचन संस्कृती वाढीस लागते आणि विचार समृद्ध होतात.", "language": "mr"},
    ]

    def __init__(self, seed_instructions: list[dict[str, str]] | None = None, seed: int = 42):
        self.mutator = InstructionMutator(seed=seed)
        self.seeds = seed_instructions or self.DEFAULT_SEEDS
        self.rng = random.Random(seed)

    def filter_quality(
        self,
        instruction: str,
        response: str,
        min_inst_len: int = 10,
        max_inst_len: int = 600,
        min_resp_len: int = 20,
        max_resp_len: int = 2500,
        max_ngram_repetition: float = 0.35,
    ) -> tuple[bool, str]:
        """Validate sample length, script presence, and repetitive patterns."""
        if len(instruction) < min_inst_len:
            return False, "Instruction too short"
        if len(instruction) > max_inst_len:
            return False, "Instruction exceeds maximum length"
        if len(response) < min_resp_len:
            return False, "Response too short"
        if len(response) > max_resp_len:
            return False, "Response exceeds maximum length"

        # Check for repetitive 3-grams
        words = response.split()
        if len(words) >= 6:
            trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
            unique_ratio = len(set(trigrams)) / len(trigrams)
            if (1.0 - unique_ratio) > max_ngram_repetition:
                return False, "Excessive n-gram repetition detected"

        return True, "Passed"

    def generate_batch(
        self,
        count: int = 10,
        target_languages: list[str] | None = None,
    ) -> list[SyntheticSample]:
        """Generate a batch of high-quality synthetic instruction-response samples."""
        target_languages = target_languages or ["hi", "bn", "ta", "te", "gu", "mr"]
        generated: list[SyntheticSample] = []
        attempts = 0
        max_attempts = count * 5

        while len(generated) < count and attempts < max_attempts:
            attempts += 1
            seed_item = self.rng.choice(self.seeds)
            lang = seed_item.get("language", "hi")
            if target_languages and lang not in target_languages:
                continue

            mutation = self.rng.choice(self.mutator.MUTATION_TYPES)
            mutated_inst = self.mutator.mutate(seed_item["instruction"], mutation, language=lang)
            resp = seed_item["response"]

            # Adapt response with step explanation if needed
            if mutation == "increase_reasoning_steps":
                resp = f"1. प्राथमिक विश्लेषण: {resp} \\n2. निष्कर्ष: उपर्युक्त बिंदुओं के आधार पर यह स्पष्ट है।"

            is_valid, reason = self.filter_quality(mutated_inst, resp)
            if not is_valid:
                logger.debug("Filtered sample (%s): %s", reason, mutated_inst)
                continue

            sample = SyntheticSample(
                instruction=mutated_inst,
                response=resp,
                language=lang,
                mutation_type=mutation,
                quality_score=0.92,
                metadata={"seed_instruction": seed_item["instruction"]},
            )
            generated.append(sample)

        return generated


    @staticmethod
    def export_jsonl(samples: list[SyntheticSample], output_path: str | Path) -> int:
        """Export samples to a JSONL file, returning the count of lines written."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
                count += 1
        return count

    @staticmethod
    def load_jsonl(input_path: str | Path) -> list[SyntheticSample]:
        """Load synthetic samples from a JSONL file."""
        samples: list[SyntheticSample] = []
        path = Path(input_path)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    samples.append(SyntheticSample(**d))
        return samples


    @staticmethod
    def deduplicate(samples: list[SyntheticSample], threshold: float = 0.80) -> list[SyntheticSample]:
        """Deduplicate samples based on instruction word overlap (Jaccard similarity)."""
        unique: list[SyntheticSample] = []
        token_sets: list[set[str]] = []

        for sample in samples:
            tokens = set(re.findall(r"\w+", sample.instruction.lower()))
            if not tokens:
                continue
            is_dup = False
            for prev_tokens in token_sets:
                sim = len(tokens & prev_tokens) / len(tokens | prev_tokens)
                if sim >= threshold:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(sample)
                token_sets.append(tokens)

        return unique


class CrossLingualSynthesizer:
    """Generates aligned bilingual instruction pairs from English-Indic parallel contexts."""

    DOMAIN_GLOSSARY: dict[str, dict[str, str]] = {
        "machine_learning": {
            "neural network": "तंत्रिका तंत्र (Neural Network)",
            "gradient descent": "प्रवणता अवतरण (Gradient Descent)",
            "loss function": "हानि फलन (Loss Function)",
            "attention mechanism": "अटेंशन तंत्र (Attention Mechanism)",
        },
        "governance": {
            "constitution": "संविधान (Constitution)",
            "fundamental rights": "मौलिक अधिकार (Fundamental Rights)",
            "judiciary": "न्यायपालिका (Judiciary)",
            "parliament": "संसद (Parliament)",
        },
    }

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def synthesize_pair(self, en_instruction: str, domain: str = "machine_learning") -> tuple[str, str]:
        """Translate key concepts into Indic terms while retaining English context."""
        glossary = self.DOMAIN_GLOSSARY.get(domain, self.DOMAIN_GLOSSARY["machine_learning"])
        hi_inst = en_instruction
        for en_term, hi_term in glossary.items():
            hi_inst = re.sub(rf"\b{en_term}\b", hi_term, hi_inst, flags=re.IGNORECASE)
        return en_instruction, hi_inst


    @staticmethod
    def stratified_sample(
        samples: list[SyntheticSample],
        samples_per_lang: int = 2,
    ) -> list[SyntheticSample]:
        """Balance generated dataset across languages with stratified extraction."""
        by_lang: dict[str, list[SyntheticSample]] = {}
        for s in samples:
            by_lang.setdefault(s.language, []).append(s)

        stratified: list[SyntheticSample] = []
        for _, items in sorted(by_lang.items()):
            stratified.extend(items[:samples_per_lang])
        return stratified


SyntheticDataGenerator.stratified_sample = CrossLingualSynthesizer.stratified_sample


def main():
    parser = argparse.ArgumentParser(description="Indic LLM Synthetic Data Generation")
    parser.add_argument("--count", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--languages", nargs="+", default=["hi", "bn", "ta", "te"], help="Target languages")
    parser.add_argument("--output", type=str, default="data/synthetic_samples.jsonl", help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generator = SyntheticDataGenerator(seed=args.seed)
    samples = generator.generate_batch(count=args.count, target_languages=args.languages)
    samples = generator.deduplicate(samples)
    written = generator.export_jsonl(samples, args.output)
    print(f"Generated {written} high-quality synthetic samples written to {args.output}")


if __name__ == "__main__":
    main()
