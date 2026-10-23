"""
RoadRisk AI — Grounded RAG Documentation & Knowledge Retrieval Pipeline
========================================================================
Indexes project documentation, dataset cards, model card, risk register,
and architecture specifications. Provides fast, 100% offline keyword & BM25
retrieval for grounded, hallucination-free AI assistant responses.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import re
import math
import logging
from typing import List, Dict, Any, Optional
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DOCS_DIR = PROJECT_ROOT / "docs"
_CACHED_INDEX: Optional[List[Dict[str, Any]]] = None


def chunk_markdown_document(filepath: Path) -> List[Dict[str, Any]]:
    """
    Split a Markdown document into structured semantic chunks based on header hierarchy.
    """
    if not filepath.exists():
        return []

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception as e:
        logger.warning(f"Could not read document {filepath}: {e}")
        return []

    # Relative path from project root
    try:
        rel_path = str(filepath.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        rel_path = filepath.name

    lines = text.splitlines()
    chunks = []
    current_title = filepath.stem.replace("_", " ").title()
    current_lines = []

    header_pattern = re.compile(r"^(#{1,4})\s+(.+)$")

    for line in lines:
        match = header_pattern.match(line)
        if match:
            # Save previous chunk if substantial
            if current_lines:
                chunk_body = "\n".join(current_lines).strip()
                if len(chunk_body) > 40:
                    chunks.append({
                        "source": rel_path,
                        "title": current_title,
                        "content": chunk_body,
                    })
                current_lines = []
            current_title = f"{filepath.stem.replace('_', ' ').title()} — {match.group(2).strip()}"
        else:
            current_lines.append(line)

    if current_lines:
        chunk_body = "\n".join(current_lines).strip()
        if len(chunk_body) > 40:
            chunks.append({
                "source": rel_path,
                "title": current_title,
                "content": chunk_body,
            })

    return chunks


def build_documentation_index(force_rebuild: bool = False) -> List[Dict[str, Any]]:
    """
    Scan project root and docs/ directory, building a comprehensive searchable index.
    Caches the index in memory.
    """
    global _CACHED_INDEX
    if _CACHED_INDEX is not None and not force_rebuild:
        return _CACHED_INDEX

    doc_files = []
    # Project README
    readme = PROJECT_ROOT / "README.md"
    if readme.exists():
        doc_files.append(readme)

    # All markdown files in docs/
    if DOCS_DIR.exists():
        doc_files.extend(list(DOCS_DIR.rglob("*.md")))

    # Specific raw readmes if present
    raw_readme = PROJECT_ROOT / "data" / "raw" / "URBAN_GB_README.md"
    if raw_readme.exists():
        doc_files.append(raw_readme)

    indexed_chunks = []
    for doc_path in doc_files:
        chunks = chunk_markdown_document(doc_path)
        indexed_chunks.extend(chunks)

    # Tokenize each chunk for BM25 search
    stop_words = {
        "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with",
        "by", "is", "are", "was", "were", "this", "that", "it", "from", "as", "be",
        "what", "when", "where", "which", "who", "whom", "how", "why", "can", "could",
        "should", "would", "does", "do", "did", "have", "has", "had", "been", "about",
        "any", "some", "more", "most", "other", "into", "over", "after"
    }

    for idx, c in enumerate(indexed_chunks):
        c["chunk_id"] = idx
        tokens = re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", (c["title"] + " " + c["content"]).lower())
        c["tokens"] = [t for t in tokens if t not in stop_words]
        c["token_counts"] = Counter(c["tokens"])
        c["doc_len"] = len(c["tokens"])

    logger.info(f"Built documentation index with {len(indexed_chunks)} chunks across {len(doc_files)} files.")
    _CACHED_INDEX = indexed_chunks
    return _CACHED_INDEX


def retrieve_project_context(
    query: str,
    top_k: int = 4,
    min_score: float = 2.0,
) -> List[Dict[str, Any]]:
    """
    Retrieve the most relevant documentation chunks for a user query using BM25 scoring.
    Works completely offline with zero API calls.
    """
    index = build_documentation_index()
    if not index or not query.strip():
        return []

    query_tokens = re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", query.lower())
    if not query_tokens:
        return []

    N = len(index)
    avg_dl = sum(c["doc_len"] for c in index) / max(1, N)
    k1 = 1.5
    b = 0.75

    # Document frequency per token
    doc_freqs = {}
    for token in set(query_tokens):
        doc_freqs[token] = sum(1 for c in index if token in c["token_counts"])

    scored_chunks = []
    for c in index:
        score = 0.0
        doc_len = c["doc_len"]
        tf_dict = c["token_counts"]

        for token in query_tokens:
            if token in tf_dict:
                tf = tf_dict[token]
                df = doc_freqs.get(token, 0)
                # BM25 IDF
                idf = math.log(1.0 + (N - df + 0.5) / (df + 0.5))
                # BM25 TF component
                num = tf * (k1 + 1.0)
                den = tf + k1 * (1.0 - b + b * (doc_len / avg_dl))
                score += idf * (num / den)

                # Bonus if token is in chunk title
                if token in c["title"].lower():
                    score += 1.5

        # Phrase match bonus
        if query.lower() in c["content"].lower():
            score += 3.0

        if score >= min_score:
            scored_chunks.append({
                "source": c["source"],
                "title": c["title"],
                "content": c["content"],
                "score": round(score, 3),
            })

    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    return scored_chunks[:top_k]


if __name__ == "__main__":
    print("Testing Project RAG Pipeline...")
    idx = build_documentation_index()
    print(f"Total chunks indexed: {len(idx)}")

    test_queries = [
        "What is UrbanGB and is it Indian data?",
        "How is CCTV data leakage mitigated?",
        "What are the target leakage exclusions for severity classification?",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        res = retrieve_project_context(q, top_k=2)
        for r in res:
            print(f"  [{r['score']}] {r['source']} ({r['title']})")
