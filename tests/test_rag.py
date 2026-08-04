"""
Unit tests for Indic LLM Retrieval-Augmented Generation (RAG) module:
  - IndicTextSplitter sentence and chunk boundary handling (।, ॥, ?, !)
  - IndicBM25Retriever lexical indexing and relevance scoring
  - IndicDenseRetriever embedding computation and cosine ranking
  - HybridRetriever Reciprocal Rank Fusion (RRF)
  - Citation-attributed prompt formatting and end-to-end RAG answer synthesis
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest
import torch

# Ensure repo root and src/ are importable
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from model import IndicLLM, ModelConfig  # noqa: E402
from rag import (  # noqa: E402
    HybridRetriever,
    IndicBM25Retriever,
    IndicDenseRetriever,
    IndicDocument,
    IndicRAGEngine,
    IndicTextSplitter,
    RAGResponse,
    SearchResult,
    format_rag_prompt,
)


class DummyTokenizer:
    """Mock SentencePiece tokenizer for deterministic RAG tests."""

    def __init__(self, vocab_size: int = 100):
        self._vocab_size = vocab_size

    def encode(self, text: str, out_type: type = int) -> List[int]:
        tokens = [abs(hash(tok)) % (self._vocab_size - 3) + 3 for tok in text.split()]
        return tokens or [3]

    def decode(self, token_ids: List[int]) -> str:
        # Simple decode representation
        return " नई दिल्ली भारत की राजधानी है।"

    def bos_id(self) -> int:
        return 1

    def eos_id(self) -> int:
        return 2

    def pad_id(self) -> int:
        return 0


@pytest.fixture
def dummy_rag_model() -> IndicLLM:
    cfg = ModelConfig(
        vocab_size=100,
        dim=32,
        n_layers=1,
        n_heads=2,
        n_kv_heads=2,
        max_seq_len=64,
        dropout=0.0,
    )
    model = IndicLLM(cfg)
    model.eval()
    return model


@pytest.fixture
def sample_indic_docs() -> List[IndicDocument]:
    texts = [
        "भारत दक्षिण एशिया में स्थित एक विशाल देश है। इसकी राजधानी नई दिल्ली है।",
        "हिमालय पर्वतमाला भारत के उत्तर में स्थित है। गंगा नदी गंगोत्री से निकलती है।",
        "राजस्थान भारत का क्षेत्रफल की दृष्टि से सबसे बड़ा राज्य है। थार मरुस्थल यहीं स्थित है।",
    ]
    return [
        IndicDocument(doc_id=f"doc_{i}", text=text, metadata={"topic": "geography"})
        for i, text in enumerate(texts)
    ]


def test_indic_text_splitter_sentences():
    splitter = IndicTextSplitter(chunk_size=100, chunk_overlap=20)
    text = "भारत एक महान देश है। यहाँ कई भाषाएँ बोली जाती हैं॥ क्या आप जानते हैं? यह अत्यंत सुंदर है!"
    sentences = splitter.split_into_sentences(text)

    assert len(sentences) == 4
    assert "भारत एक महान देश है।" in sentences[0]
    assert "यहाँ कई भाषाएँ बोली जाती हैं॥" in sentences[1]
    assert "क्या आप जानते हैं?" in sentences[2]
    assert "यह अत्यंत सुंदर है!" in sentences[3]


def test_indic_text_splitter_chunking():
    splitter = IndicTextSplitter(chunk_size=60, chunk_overlap=15)
    text = (
        "पहला वाक्य यहाँ है। दूसरा वाक्य भी काफी महत्वपूर्ण है। "
        "तीसरा वाक्य थोड़ा लंबा है और इसमें जानकारी है। चौथा वाक्य अंत में आता है।"
    )
    chunks = splitter.split_text(text)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk) <= 120  # bounded chunk length


def test_indic_text_splitter_create_documents():
    splitter = IndicTextSplitter(chunk_size=100, chunk_overlap=20)
    texts = ["सूर्य पूर्व से निकलता है। यह प्रकाश देता है।", "चंद्रमा रात में चमकता है।"]
    docs = splitter.create_documents(texts, metadatas=[{"source": "astro1"}, {"source": "astro2"}])

    assert len(docs) >= 2
    assert docs[0].doc_id.startswith("doc_0_chunk_")
    assert docs[0].metadata["source"] == "astro1"


def test_indic_bm25_retriever(sample_indic_docs: List[IndicDocument]):
    retriever = IndicBM25Retriever(sample_indic_docs, k1=1.5, b=0.75)

    # Query for capital
    results = retriever.retrieve("राजधानी नई दिल्ली", top_k=2)
    assert len(results) == 2
    top_result = results[0]
    assert isinstance(top_result, SearchResult)
    assert top_result.document.doc_id == "doc_0"
    assert "नई दिल्ली" in top_result.document.text
    assert top_result.rank == 1
    assert top_result.score > 0.0

    # Query for mountains/rivers
    geo_results = retriever.retrieve("हिमालय और गंगा नदी", top_k=1)
    assert len(geo_results) == 1
    assert geo_results[0].document.doc_id == "doc_1"


def test_indic_dense_retriever(dummy_rag_model: IndicLLM, sample_indic_docs: List[IndicDocument]):
    tok = DummyTokenizer()
    retriever = IndicDenseRetriever(
        documents=sample_indic_docs,
        model=dummy_rag_model,
        tokenizer=tok,  # type: ignore[arg-type]
        device=torch.device("cpu"),
    )

    results = retriever.retrieve("भारत की राजधानी", top_k=2)
    assert len(results) == 2
    assert isinstance(results[0], SearchResult)
    assert results[0].score <= 1.0001  # Cosine similarity bound


def test_hybrid_retriever_rrf(dummy_rag_model: IndicLLM, sample_indic_docs: List[IndicDocument]):
    tok = DummyTokenizer()
    bm25 = IndicBM25Retriever(sample_indic_docs)
    dense = IndicDenseRetriever(
        documents=sample_indic_docs,
        model=dummy_rag_model,
        tokenizer=tok,  # type: ignore[arg-type]
        device=torch.device("cpu"),
    )
    hybrid = HybridRetriever(bm25, dense, bm25_weight=0.5, dense_weight=0.5, rrf_k=60)

    results = hybrid.retrieve("राजस्थान और थार मरुस्थल", top_k=2)
    assert len(results) == 2
    # doc_2 is specifically about Rajasthan and Thar
    assert results[0].document.doc_id == "doc_2"
    assert results[0].score > 0.0


def test_format_rag_prompt(sample_indic_docs: List[IndicDocument]):
    res1 = SearchResult(document=sample_indic_docs[0], score=1.8, rank=1)
    res2 = SearchResult(document=sample_indic_docs[1], score=1.2, rank=2)

    prompt = format_rag_prompt(
        query="भारत की राजधानी क्या है?",
        retrieved_docs=[res1, res2],
        system_prompt="सटीक उत्तर दें।",
    )

    assert "[संदर्भ 1] भारत दक्षिण एशिया" in prompt
    assert "[संदर्भ 2] हिमालय पर्वतमाला" in prompt
    assert "प्रश्न: भारत की राजधानी क्या है?" in prompt
    assert "निर्देश: सटीक उत्तर दें।" in prompt


def test_indic_rag_engine_end_to_end(dummy_rag_model: IndicLLM):
    tok = DummyTokenizer()
    splitter = IndicTextSplitter(chunk_size=150, chunk_overlap=30)
    bm25 = IndicBM25Retriever()

    engine = IndicRAGEngine(
        model=dummy_rag_model,
        tokenizer=tok,  # type: ignore[arg-type]
        retriever=bm25,
        splitter=splitter,
        device=torch.device("cpu"),
    )

    raw_texts = [
        "भारत की राजधानी नई दिल्ली है। यहाँ संसद भवन स्थित है।",
        "भारतीय रुपया भारत की आधिकारिक मुद्रा है।",
    ]
    n_chunks = engine.index_texts(raw_texts, source_id="knowledge")
    assert n_chunks >= 2

    response = engine.answer("भारत की राजधानी क्या है?", top_k=1, max_new_tokens=10)
    assert isinstance(response, RAGResponse)
    assert response.query == "भारत की राजधानी क्या है?"
    assert len(response.citations) == 1
    assert response.citations[0]["citation_tag"] == "[संदर्भ 1]"
    assert "[संदर्भ 1]" in response.prompt
    assert len(response.answer) > 0

