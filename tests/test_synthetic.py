"""Unit tests for Indic Synthetic Instruction Generator and Mutator."""

import pytest
from data.generate_synthetic import InstructionMutator, SyntheticSample


def test_synthetic_sample_dataclass():
    sample = SyntheticSample(
        instruction="भारत की राजधानी क्या है?",
        response="भारत की राजधानी नई दिल्ली है।",
        language="hi",
        mutation_type="seed",
        quality_score=0.95,
        metadata={"domain": "geography"},
    )
    d = sample.to_dict()
    assert d["instruction"] == "भारत की राजधानी क्या है?"
    assert d["language"] == "hi"
    assert d["quality_score"] == 0.95
    assert d["metadata"]["domain"] == "geography"


def test_instruction_mutator_operations():
    mutator = InstructionMutator(seed=123)
    seed_prompt = "जलवायु परिवर्तन के प्रभावों का वर्णन करें।"

    # Concretize
    c_mut = mutator.mutate(seed_prompt, "concretize", language="hi")
    assert len(c_mut) > len(seed_prompt)
    assert any(anchor in c_mut for anchor in InstructionMutator.CULTURAL_ANCHORS["hi"])

    # Deepen constraints
    d_mut = mutator.mutate(seed_prompt, "deepen_constraints", language="hi")
    assert "तीन मुख्य बिंदु" in d_mut

    # Increase reasoning
    r_mut = mutator.mutate(seed_prompt, "increase_reasoning_steps", language="hi")
    assert "चरण-दर-चरण" in r_mut

    # Cross-lingual
    x_mut = mutator.mutate(seed_prompt, "multilingual_cross_lingual", language="hi")
    assert "अंग्रेजी समानार्थी" in x_mut


def test_instruction_mutator_multilingual():
    mutator = InstructionMutator(seed=99)
    # Bengali
    bn_mut = mutator.mutate("কৃত্রিম বুদ্ধিমত্তা কী?", "concretize", language="bn")
    assert any(anchor in bn_mut for anchor in InstructionMutator.CULTURAL_ANCHORS["bn"])

    # Tamil
    ta_mut = mutator.mutate("வானிலை அறிக்கை", "deepen_constraints", language="ta")
    assert "முக்கிய குறிப்புகள்" in ta_mut


def test_synthetic_generator_batch():
    generator = SyntheticDataGenerator(seed=42)
    batch = generator.generate_batch(count=6, target_languages=["hi", "bn"])
    assert len(batch) == 6
    assert all(s.language in ["hi", "bn"] for s in batch)
    assert all(s.quality_score >= 0.70 for s in batch)


def test_quality_filter_bounds():
    generator = SyntheticDataGenerator(seed=42)
    # Too short
    ok, reason = generator.filter_quality("छोटा", "उत्तर")
    assert not ok
    assert "short" in reason.lower()

    # Repetition check
    repetitive_text = "बार बार बार बार बार बार बार बार बार बार बार बार बार बार बार बार बार बार"
    ok, reason = generator.filter_quality("यह एक परीक्षण निर्देश है?", repetitive_text)
    assert not ok
    assert "repetition" in reason.lower()
