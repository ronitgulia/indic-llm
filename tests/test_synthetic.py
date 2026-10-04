"""Unit tests for Indic Synthetic Instruction Generator and Mutator."""

from data.generate_synthetic import (
    CrossLingualSynthesizer,
    InstructionMutator,
    SyntheticDataGenerator,
    SyntheticSample,
)


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


def test_deduplication():
    s1 = SyntheticSample("भारत की राजधानी नई दिल्ली है।", "उत्तर", "hi", "seed")
    s2 = SyntheticSample("भारत की राजधानी नई दिल्ली है।", "भिन्न उत्तर", "hi", "seed")
    s3 = SyntheticSample("प्राकृतिक आपदा प्रबंधन पर निबंध लिखें।", "उत्तर", "hi", "seed")
    unique = SyntheticDataGenerator.deduplicate([s1, s2, s3], threshold=0.7)
    assert len(unique) == 2
    assert unique[0].instruction == s1.instruction
    assert unique[1].instruction == s3.instruction


def test_jsonl_export_import(tmp_path):
    p = tmp_path / "test_synth.jsonl"
    samples = [
        SyntheticSample("प्रश्न एक", "उत्तर एक", "hi", "seed"),
        SyntheticSample("প্রশ্ন দুই", "উত্তর দুই", "bn", "seed"),
    ]
    count = SyntheticDataGenerator.export_jsonl(samples, p)
    assert count == 2
    loaded = SyntheticDataGenerator.load_jsonl(p)
    assert len(loaded) == 2
    assert loaded[0].instruction == "प्रश्न एक"
    assert loaded[1].language == "bn"


def test_cross_lingual_synthesizer():
    syn = CrossLingualSynthesizer(seed=42)
    en_q = "Explain how neural network optimization uses gradient descent."
    en_orig, hi_synth = syn.synthesize_pair(en_q, domain="machine_learning")
    assert en_orig == en_q
    assert "तंत्रिका तंत्र" in hi_synth
    assert "प्रवणता अवतरण" in hi_synth


def test_stratified_sample():
    samples = [
        SyntheticSample("h1", "r1", "hi", "s"),
        SyntheticSample("h2", "r2", "hi", "s"),
        SyntheticSample("h3", "r3", "hi", "s"),
        SyntheticSample("b1", "r1", "bn", "s"),
        SyntheticSample("b2", "r2", "bn", "s"),
    ]
    balanced = SyntheticDataGenerator.stratified_sample(samples, samples_per_lang=2)
    assert len(balanced) == 4
    hi_count = sum(1 for s in balanced if s.language == "hi")
    bn_count = sum(1 for s in balanced if s.language == "bn")
    assert hi_count == 2
    assert bn_count == 2
