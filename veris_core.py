# veris_core.py
import re
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from io import BytesIO

import numpy as np
import faiss
from pypdf import PdfReader
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"

client = OpenAI()


@dataclass(frozen=True)
class Passage:
    text: str
    filename: str
    page: int


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _collapse_pdf_newlines(text: str) -> str:
    """
    Fix common PDF extraction issue where words appear separated by newlines.
    Keep paragraphs, remove line-wrapping newlines.
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)            # normalize big breaks
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)      # single newline -> space
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)     # de-hyphenate wraps
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _clean(text: str) -> str:
    text = (text or "").replace("\x00", " ")
    text = _collapse_pdf_newlines(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, size: int = 900, overlap: int = 180) -> List[str]:
    text = _clean(text)
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(n, start + size)
        chunk = text[start:end].strip()
        if len(chunk) > 80:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)

    return chunks


def pdf_bytes_to_passages(pdf_bytes: bytes, display_name: str) -> List[Passage]:
    reader = PdfReader(BytesIO(pdf_bytes))
    out: List[Passage] = []

    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        for chunk in chunk_text(page_text):
            out.append(Passage(text=chunk, filename=display_name, page=i + 1))

    return out


def embed(texts: List[str]) -> np.ndarray:
    res = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([r.embedding for r in res.data], dtype=np.float32)
    faiss.normalize_L2(vecs)
    return vecs


def build_store(passages: List[Passage]) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    texts = [p.text for p in passages]
    vecs = embed(texts)

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine sim because normalized
    index.add(vecs)

    meta = [{"text": p.text, "filename": p.filename, "page": p.page} for p in passages]
    return index, meta


def search(index: faiss.Index, meta: List[Dict[str, Any]], query: str, k: int = 6) -> List[Dict[str, Any]]:
    qv = embed([query])
    scores, ids = index.search(qv, k)

    results: List[Dict[str, Any]] = []
    for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if idx < 0:
            continue
        item = dict(meta[idx])
        item["score"] = float(score)
        results.append(item)
    return results


def make_context(results: List[Dict[str, Any]], limit_chars: int = 7000) -> Tuple[str, List[str]]:
    blocks: List[str] = []
    cites: List[str] = []
    used = 0

    for r in results:
        cite = f"{r['filename']} p.{r['page']}"
        block = f"[Source: {cite}]\n{r['text']}\n"
        if used + len(block) > limit_chars:
            break
        blocks.append(block)
        cites.append(cite)
        used += len(block)

    return "\n".join(blocks).strip(), cites


def answer_from_docs(question: str, context: str) -> str:
    res = client.responses.create(
        model=CHAT_MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "You are Veris, a document-grounded assistant.\n"
                    "Rules:\n"
                    "1) Use ONLY the provided context.\n"
                    "2) If the answer is not clearly supported, say exactly:\n"
                    "\"I don’t know based on the uploaded documents.\"\n"
                    "3) End with a Sources section listing filenames/pages used.\n"
                ),
            },
            {"role": "user", "content": f"Question:\n{question}\n\nContext:\n{context}"},
        ],
    )
    return res.output_text
