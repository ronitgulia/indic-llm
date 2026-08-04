"""
Indic LLM — Retrieval-Augmented Generation (RAG) Module
=========================================================
Implements sentence-aware document chunking, BM25 lexical search,
dense embedding retrieval, hybrid rank fusion, and citation-grounded
RAG generation tuned for Indic and Devanagari text:

  - IndicTextSplitter   : Splits text respecting Devanagari punctuation (।, ॥, ?, !)
  - IndicDocument       : Dataclass encapsulating text chunks and metadata
  - SearchResult        : Ranked retrieval match with score and rank
  - IndicBM25Retriever  : Pure-Python Okapi BM25 implementation for Indic corpora
  - IndicDenseRetriever : Semantic vector retriever using IndicLLM representations
  - HybridRetriever     : Reciprocal Rank Fusion (RRF) combining BM25 and dense search
  - IndicRAGEngine      : End-to-end question answering pipeline with citation attribution
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import sentencepiece as spm
import torch
import torch.nn.functional as F

# Ensure src/ and repo root are importable
repo_root = Path(__file__).resolve().parent.parent
src_path = str(repo_root / "src")
inference_path = str(repo_root / "inference")
for p in (src_path, inference_path):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from model import IndicLLM, ModelConfig
except ImportError:
    from src.model import IndicLLM, ModelConfig  # type: ignore[no-redef]


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


@dataclass
class RAGResponse:
    """Synthesized answer with citation grounding."""

    query: str
    answer: str
    citations: List[Dict[str, Any]]
    retrieved_docs: List[SearchResult]
    prompt: str


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


class IndicDenseRetriever:
    """
    Dense semantic retriever utilizing IndicLLM representations.
    Computes mean-pooled token vectors and cosine similarity ranking.
    """

    def __init__(
        self,
        documents: Optional[List[IndicDocument]] = None,
        model: Optional[IndicLLM] = None,
        tokenizer: Optional[spm.SentencePieceProcessor] = None,
        device: Optional[torch.device] = None,
        max_seq_len: int = 256,
    ):
        self.documents = list(documents) if documents else []
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cpu")
        self.max_seq_len = max_seq_len
        self.doc_embeddings: Optional[torch.Tensor] = None

        if self.documents:
            self.index_documents(self.documents)

    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode texts into normalized embedding tensors (N, dim)."""
        if not texts:
            return torch.empty(0, 0)

        # Fallback pseudo-embedding if model/tokenizer not attached (for unit tests / mock setups)
        if self.model is None or self.tokenizer is None:
            dim = 64
            embs = torch.zeros((len(texts), dim), dtype=torch.float32)
            for i, text in enumerate(texts):
                for ch in text:
                    embs[i, hash(ch) % dim] += 1.0
            return F.normalize(embs, p=2, dim=-1)

        self.model.eval()
        embeddings: List[torch.Tensor] = []

        with torch.no_grad():
            for text in texts:
                toks = self.tokenizer.encode(text, out_type=int)
                if not toks:
                    toks = [0]
                toks = toks[: self.max_seq_len]
                input_tensor = torch.tensor([toks], dtype=torch.long, device=self.device)

                # Use token embeddings + mean-pooling
                tok_embs = self.model.tok_embeddings(input_tensor)  # (1, T, dim)
                mean_emb = tok_embs.mean(dim=1)  # (1, dim)
                norm_emb = F.normalize(mean_emb, p=2, dim=-1)
                embeddings.append(norm_emb.squeeze(0).cpu())

        return torch.stack(embeddings, dim=0)

    def index_documents(self, documents: List[IndicDocument]) -> None:
        """Index documents by computing and storing dense representation tensors."""
        self.documents = list(documents)
        if not self.documents:
            self.doc_embeddings = None
            return

        texts = [doc.text for doc in self.documents]
        self.doc_embeddings = self._encode_texts(texts)

    def retrieve(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """Retrieve top_k documents most semantically similar to query."""
        if not self.documents or self.doc_embeddings is None:
            return []

        query_emb = self._encode_texts([query])  # (1, dim)
        # Cosine similarity: (1, dim) @ (dim, N) -> (1, N)
        scores = (query_emb @ self.doc_embeddings.t()).squeeze(0)  # (N,)

        ranked_indices = torch.argsort(scores, descending=True).tolist()

        results: List[SearchResult] = []
        for rank, idx in enumerate(ranked_indices[:top_k], 1):
            results.append(
                SearchResult(
                    document=self.documents[idx],
                    score=float(scores[idx].item()),
                    rank=rank,
                )
            )
        return results


class HybridRetriever:
    """
    Hybrid retriever fusing lexical (BM25) and dense semantic search
    using Reciprocal Rank Fusion (RRF):
      RRF_score(d) = w_bm25 / (k + rank_bm25(d)) + w_dense / (k + rank_dense(d))
    """

    def __init__(
        self,
        bm25_retriever: IndicBM25Retriever,
        dense_retriever: IndicDenseRetriever,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        rrf_k: int = 60,
    ):
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k

    def index_documents(self, documents: List[IndicDocument]) -> None:
        """Index documents across both underlying retrievers."""
        self.bm25_retriever.index_documents(documents)
        self.dense_retriever.index_documents(documents)

    def retrieve(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """Retrieve top_k documents using reciprocal rank fusion."""
        if not self.bm25_retriever.documents:
            return []

        n_candidates = max(top_k * 2, 10)
        bm25_results = self.bm25_retriever.retrieve(query, top_k=n_candidates)
        dense_results = self.dense_retriever.retrieve(query, top_k=n_candidates)

        scores: Dict[str, float] = {}
        doc_map: Dict[str, IndicDocument] = {}

        # Accumulate BM25 ranks
        for res in bm25_results:
            d_id = res.document.doc_id
            doc_map[d_id] = res.document
            scores[d_id] = scores.get(d_id, 0.0) + self.bm25_weight / (self.rrf_k + res.rank)

        # Accumulate Dense ranks
        for res in dense_results:
            d_id = res.document.doc_id
            doc_map[d_id] = res.document
            scores[d_id] = scores.get(d_id, 0.0) + self.dense_weight / (self.rrf_k + res.rank)

        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        results: List[SearchResult] = []
        for rank, (d_id, fused_score) in enumerate(sorted_docs[:top_k], 1):
            results.append(
                SearchResult(
                    document=doc_map[d_id],
                    score=float(fused_score),
                    rank=rank,
                )
            )
        return results


def format_rag_prompt(
    query: str,
    retrieved_docs: List[SearchResult],
    system_prompt: Optional[str] = None,
) -> str:
    """
    Format citation-grounded RAG prompt in Hindi / Devanagari.
    Attaches [संदर्भ 1], [संदर्भ 2], etc. for verifiable answers.
    """
    default_sys = (
        "आप एक सहायक Indic AI हैं। केवल नीचे दिए गए संदर्भों के आधार पर प्रश्नों का सटीक और संक्षिप्त उत्तर दें।"
    )
    sys_text = system_prompt or default_sys

    context_lines = []
    for res in retrieved_docs:
        context_lines.append(f"[संदर्भ {res.rank}] {res.document.text}")

    context_block = "\n".join(context_lines) if context_lines else "कोई प्रासंगिक संदर्भ उपलब्ध नहीं है।"

    return (
        f"निर्देश: {sys_text}\n\n"
        f"संदर्भ सामग्री:\n"
        f"{context_block}\n\n"
        f"प्रश्न: {query}\n"
        f"उत्तर:"
    )


class IndicRAGEngine:
    """
    End-to-end Indic RAG Engine:
      1. Splits and indexes documents
      2. Retrieves relevant chunks via BM25, Dense, or Hybrid search
      3. Constructs citation-attributed prompt
      4. Generates factual answer via IndicLLM
    """

    def __init__(
        self,
        model: IndicLLM,
        tokenizer: spm.SentencePieceProcessor,
        retriever: Optional[Union[IndicBM25Retriever, IndicDenseRetriever, HybridRetriever]] = None,
        splitter: Optional[IndicTextSplitter] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.splitter = splitter or IndicTextSplitter(chunk_size=300, chunk_overlap=50)

        if retriever is not None:
            self.retriever = retriever
        else:
            bm25 = IndicBM25Retriever()
            dense = IndicDenseRetriever(model=self.model, tokenizer=self.tokenizer, device=self.device)
            self.retriever = HybridRetriever(bm25, dense)

    def index_texts(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        source_id: str = "doc",
    ) -> int:
        """Chunk and index an input corpus of texts."""
        docs = self.splitter.create_documents(texts, metadatas=metadatas, source_id=source_id)
        self.retriever.index_documents(docs)
        return len(docs)

    def retrieve(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """Retrieve relevant document chunks for a query."""
        return self.retriever.retrieve(query, top_k=top_k)

    @torch.no_grad()
    def answer(
        self,
        query: str,
        top_k: int = 3,
        max_new_tokens: int = 150,
        temperature: float = 0.3,
        top_p: float = 0.9,
        system_prompt: Optional[str] = None,
    ) -> RAGResponse:
        """Retrieve context, synthesize answer with citation tags, and return RAGResponse."""
        retrieved_docs = self.retrieve(query, top_k=top_k)
        prompt = format_rag_prompt(query, retrieved_docs, system_prompt=system_prompt)

        # Autoregressive generation
        self.model.eval()
        prompt_tokens = self.tokenizer.encode(prompt, out_type=int)
        bos_id = self.tokenizer.bos_id()
        if bos_id >= 0 and (not prompt_tokens or prompt_tokens[0] != bos_id):
            prompt_tokens = [bos_id] + prompt_tokens

        eos_id = self.tokenizer.eos_id()
        curr_tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)

        generated_ids: List[int] = []
        for _ in range(max_new_tokens):
            logits, _ = self.model(curr_tokens)
            next_logits = logits[0, -1, :]

            if temperature <= 1e-4:
                next_tok = int(torch.argmax(next_logits).item())
            else:
                scaled_logits = next_logits / temperature
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    sorted_logits[sorted_indices_to_remove] = float("-inf")
                    scaled_logits = torch.zeros_like(scaled_logits).scatter_(
                        0, sorted_indices, sorted_logits
                    )
                probs = F.softmax(scaled_logits, dim=-1)
                next_tok = int(torch.multinomial(probs, num_samples=1).item())

            if next_tok == eos_id:
                break

            generated_ids.append(next_tok)
            next_tensor = torch.tensor([[next_tok]], dtype=torch.long, device=self.device)
            curr_tokens = torch.cat([curr_tokens, next_tensor], dim=1)

            # Check stop conditions (double newline or prompt repeat)
            decoded_so_far = self.tokenizer.decode(generated_ids)
            if "\n\nप्रश्न:" in decoded_so_far or "\n\nनिर्देश:" in decoded_so_far:
                break

        answer_text = self.tokenizer.decode(generated_ids).strip()

        # Format citations
        citations = []
        for res in retrieved_docs:
            citations.append({
                "citation_tag": f"[संदर्भ {res.rank}]",
                "doc_id": res.document.doc_id,
                "score": res.score,
                "text_snippet": res.document.text[:120] + "...",
            })

        return RAGResponse(
            query=query,
            answer=answer_text,
            citations=citations,
            retrieved_docs=retrieved_docs,
            prompt=prompt,
        )


def _load_rag_model(ckpt_path: str, device: torch.device) -> IndicLLM:
    """Helper to load IndicLLM checkpoint for RAG CLI."""
    ckpt = torch.load(ckpt_path, map_location=device)
    raw_cfg = ckpt.get("config", {})
    model_cfg = ModelConfig(
        vocab_size=raw_cfg.get("vocab_size", 32000),
        dim=raw_cfg.get("model_dim", 512),
        n_layers=raw_cfg.get("n_layers", 8),
        n_heads=raw_cfg.get("n_heads", 8),
        n_kv_heads=raw_cfg.get("n_kv_heads", 4),
        max_seq_len=raw_cfg.get("max_seq_len", 512),
        dropout=0.0,
    )
    model = IndicLLM(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def main() -> None:
    """CLI entry point for Indic RAG indexing and question answering."""
    parser = argparse.ArgumentParser(description="Indic LLM Retrieval-Augmented Generation (RAG)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to IndicLLM checkpoint .pt")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to SentencePiece model file")
    parser.add_argument("--corpus", type=str, required=True, help="Path to text or JSONL document corpus")
    parser.add_argument("--query", type=str, default=None, help="Query string in Hindi / Indic language")
    parser.add_argument("--top_k", type=int, default=3, help="Number of retrieved chunks")
    parser.add_argument(
        "--retriever",
        type=str,
        choices=["bm25", "dense", "hybrid"],
        default="hybrid",
        help="Retrieval strategy",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    device = torch.device(args.device)

    # 1. Load tokenizer & model
    if not os.path.isfile(args.tokenizer):
        raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer}")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(args.tokenizer)

    model = _load_rag_model(args.checkpoint, device)

    # 2. Setup retrievers
    splitter = IndicTextSplitter(chunk_size=300, chunk_overlap=50)
    bm25 = IndicBM25Retriever()
    dense = IndicDenseRetriever(model=model, tokenizer=tokenizer, device=device)

    if args.retriever == "bm25":
        active_retriever = bm25
    elif args.retriever == "dense":
        active_retriever = dense
    else:
        active_retriever = HybridRetriever(bm25, dense)

    rag_engine = IndicRAGEngine(
        model=model,
        tokenizer=tokenizer,
        retriever=active_retriever,
        splitter=splitter,
        device=device,
    )

    # 3. Read and index corpus
    corpus_path = Path(args.corpus)
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    with open(corpus_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    num_chunks = rag_engine.index_texts(texts, source_id=corpus_path.stem)
    print(f"Indexed {len(texts)} documents into {num_chunks} chunks.")

    # 4. Answer query or run interactive loop
    if args.query:
        resp = rag_engine.answer(args.query, top_k=args.top_k)
        print(f"\nप्रश्न: {resp.query}")
        print(f"उत्तर: {resp.answer}\n")
        print("संदर्भ (Citations):")
        for cit in resp.citations:
            print(f"  {cit['citation_tag']} (Score: {cit['score']:.4f}) {cit['text_snippet']}")
    else:
        print("\n--- Indic RAG Interactive Mode (Type 'exit' to quit) ---")
        while True:
            try:
                user_q = input("\nप्रश्न > ").strip()
                if not user_q or user_q.lower() in ("exit", "quit"):
                    break
                resp = rag_engine.answer(user_q, top_k=args.top_k)
                print(f"उत्तर: {resp.answer}")
                print("संदर्भ:")
                for cit in resp.citations:
                    print(f"  {cit['citation_tag']} {cit['text_snippet']}")
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
