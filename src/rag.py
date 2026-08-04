"""
Indic LLM — Retrieval-Augmented Generation (RAG) Module
=========================================================
Implements sentence-aware document chunking, BM25 lexical search,
dense embedding retrieval, and citation-grounded RAG generation
tuned for Indic and Devanagari text:

  - IndicTextSplitter   : Splits text respecting Devanagari punctuation (।, ॥, ?, !)
  - IndicDocument       : Dataclass encapsulating text chunks and metadata
  - SearchResult        : Ranked retrieval match with score and rank
  - IndicBM25Retriever  : Pure-Python Okapi BM25 implementation for Indic corpora
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


@dataclass
class IndicDocument:
    """A text chunk indexed for retrieval."""

    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """A retrieved document with ranking metrics."""

    document: IndicDocument
    score: float
    rank: int = 1


class IndicTextSplitter:
    """
    Sentence-aware text chunker tailored for Devanagari and Indic scripts.
    Splits text along linguistic boundaries:
      - Purna Viram: '।' (U+0964)
      - Deergh Viram: '॥' (U+0965)
      - Latin/General punctuation: '.', '?', '!', '\n'
    """

    # Matches Indic and Latin sentence terminations followed by whitespace or end of string
    SENTENCE_SPLIT_REGEX = re.compile(r"([।॥?!]+(?:\s+|\Z)|\n{2,}|\.\s+)")

    def __init__(
        self,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        length_fn: Optional[Callable[[str], int]] = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_fn = length_fn or len

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into individual sentences preserving delimiters."""
        text = text.strip()
        if not text:
            return []

        parts = self.SENTENCE_SPLIT_REGEX.split(text)
        sentences: List[str] = []
        current = ""

        for part in parts:
            if not part:
                continue
            if self.SENTENCE_SPLIT_REGEX.match(part):
                current += part
                if current.strip():
                    sentences.append(current.strip())
                current = ""
            else:
                current += part

        if current.strip():
            sentences.append(current.strip())

        return sentences

    def split_text(self, text: str) -> List[str]:
        """Group sentences into overlapping chunks conforming to chunk_size."""
        sentences = self.split_into_sentences(text)
        if not sentences:
            return []

        chunks: List[str] = []
        current_chunk: List[str] = []
        current_len = 0

        for sentence in sentences:
            sent_len = self.length_fn(sentence)

            # If a single sentence exceeds chunk_size, split by whitespace words
            if sent_len > self.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                words = sentence.split()
                sub_chunk: List[str] = []
                sub_len = 0
                for w in words:
                    w_len = self.length_fn(w) + 1
                    if sub_len + w_len > self.chunk_size and sub_chunk:
                        chunks.append(" ".join(sub_chunk))
                        # Retain overlap words
                        overlap_words: List[str] = []
                        o_len = 0
                        for ow in reversed(sub_chunk):
                            if o_len + self.length_fn(ow) + 1 <= self.chunk_overlap:
                                overlap_words.insert(0, ow)
                                o_len += self.length_fn(ow) + 1
                            else:
                                break
                        sub_chunk = list(overlap_words)
                        sub_len = sum(self.length_fn(w) + 1 for w in sub_chunk)
                    sub_chunk.append(w)
                    sub_len += w_len
                if sub_chunk:
                    chunks.append(" ".join(sub_chunk))
                continue

            if current_len + sent_len + 1 > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))

                # Build overlap from the end of current_chunk
                overlap_sentences: List[str] = []
                overlap_len = 0
                for prev_sent in reversed(current_chunk):
                    p_len = self.length_fn(prev_sent) + 1
                    if overlap_len + p_len <= self.chunk_overlap:
                        overlap_sentences.insert(0, prev_sent)
                        overlap_len += p_len
                    else:
                        break

                current_chunk = list(overlap_sentences)
                current_len = sum(self.length_fn(s) + 1 for s in current_chunk)

            current_chunk.append(sentence)
            current_len += sent_len + 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def create_documents(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        source_id: str = "doc",
    ) -> List[IndicDocument]:
        """Convert raw texts into indexed IndicDocument instances."""
        docs: List[IndicDocument] = []
        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            chunks = self.split_text(text)
            for chunk_idx, chunk in enumerate(chunks):
                doc_id = f"{source_id}_{i}_chunk_{chunk_idx}"
                chunk_meta = dict(meta)
                chunk_meta["source_index"] = i
                chunk_meta["chunk_index"] = chunk_idx
                docs.append(IndicDocument(doc_id=doc_id, text=chunk, metadata=chunk_meta))
        return docs


class IndicBM25Retriever:
    """
    Okapi BM25 ranking algorithm optimized for Indic subwords and words.
    Scores relevance using Term Frequency (TF) and Inverse Document Frequency (IDF).
    """

    def __init__(
        self,
        documents: Optional[List[IndicDocument]] = None,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer_fn: Optional[Callable[[str], List[str]]] = None,
    ):
        self.k1 = k1
        self.b = b
        self.tokenizer_fn = tokenizer_fn or self._default_tokenizer
        self.documents: List[IndicDocument] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.term_freqs: List[Counter[str]] = []

        if documents:
            self.index_documents(documents)

    @staticmethod
    def _default_tokenizer(text: str) -> List[str]:
        """Simple Devanagari and Latin token extractor."""
        # Normalize punctuation and extract alphanumeric/Indic words
        cleaned = re.sub(r"[^\w\u0900-\u097F]+", " ", text.lower())
        tokens = cleaned.split()
        return [t for t in tokens if len(t) > 0]

    def index_documents(self, documents: List[IndicDocument]) -> None:
        """Build the inverted index and document length statistics."""
        self.documents = list(documents)
        n_docs = len(documents)
        if n_docs == 0:
            return

        self.term_freqs = []
        self.doc_lengths = []
        self.doc_freqs = Counter()

        total_tokens = 0
        for doc in self.documents:
            tokens = self.tokenizer_fn(doc.text)
            self.term_freqs.append(Counter(tokens))
            length = len(tokens)
            self.doc_lengths.append(length)
            total_tokens += length

            # Document frequency: unique tokens in this doc
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1

        self.avg_doc_len = total_tokens / max(1, n_docs)

        # Compute IDF: ln( (N - n + 0.5) / (n + 0.5) + 1 )
        self.idf = {}
        for token, freq in self.doc_freqs.items():
            self.idf[token] = math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1.0)

    def retrieve(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """Retrieve top_k documents scoring highest on query tokens."""
        if not self.documents:
            return []

        query_tokens = self.tokenizer_fn(query)
        scores = [0.0] * len(self.documents)

        for i, tf in enumerate(self.term_freqs):
            doc_len = self.doc_lengths[i]
            len_norm = 1.0 - self.b + self.b * (doc_len / max(1e-5, self.avg_doc_len))

            score = 0.0
            for q_tok in query_tokens:
                if q_tok in tf:
                    term_count = tf[q_tok]
                    numerator = term_count * (self.k1 + 1.0)
                    denominator = term_count + self.k1 * len_norm
                    idf_val = self.idf.get(q_tok, 0.0)
                    score += idf_val * (numerator / denominator)
            scores[i] = score

        # Rank documents descending by score
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)

        results: List[SearchResult] = []
        for rank, idx in enumerate(ranked_indices[:top_k], 1):
            results.append(
                SearchResult(
                    document=self.documents[idx],
                    score=float(scores[idx]),
                    rank=rank,
                )
            )

        return results
