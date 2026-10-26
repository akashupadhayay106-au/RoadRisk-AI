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

import re
import math
import logging
from typing import List, Dict, Any, Optional
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DOCS_DIR = PROJECT_ROOT / "docs"
_CACHED_INDEX: Optional[List[Dict[str, Any]]] = None

# Minimum share of distinct query terms a chunk must match to count as relevant.
# A pure score threshold is unsafe on this small corpus: with ~75 chunks a single
# term of document-frequency 1 scores IDF = log(1 + 74.5/1.5) ~= 3.93, so one
# incidental rare-word overlap would clear the score gate on its own. Requiring
# term coverage as well makes "matched one out of five words" fail, while every
# genuinely answerable query in the documentation set scores 50-100% coverage.
MIN_QUERY_COVERAGE = 0.5

# Index and query text share one stop-word list so that relevance coverage is
# measured over the same vocabulary that is actually searchable. Without this,
# counting "what/are/the/with" as query terms would deflate coverage and wrongly
# reject legitimately answerable questions.
STOP_WORDS = {
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "by", "is", "are", "was", "were", "this", "that", "it", "from", "as", "be",
    "what", "when", "where", "which", "who", "whom", "how", "why", "can", "could",
    "should", "would", "does", "do", "did", "have", "has", "had", "been", "about",
    "any", "some", "more", "most", "other", "into", "over", "after",
}


def _term_matched(term: str, tf_dict: Dict[str, int]) -> bool:
    """
    Check whether a query term occurs in a chunk, tolerating simple plural variants.

    The index intentionally has no stemmer, so "accident" and "accidents" are
    distinct tokens. Term coverage must not penalise a relevant chunk merely for
    using the other variant of the same word.
    """
    if term in tf_dict:
        return True
    if term.endswith("s") and term[:-1] in tf_dict:
        return True
    return f"{term}s" in tf_dict


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
    for idx, c in enumerate(indexed_chunks):
        c["chunk_id"] = idx
        tokens = re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", (c["title"] + " " + c["content"]).lower())
        c["tokens"] = [t for t in tokens if t not in STOP_WORDS]
        c["token_counts"] = Counter(c["tokens"])
        c["doc_len"] = len(c["tokens"])

    logger.info(f"Built documentation index with {len(indexed_chunks)} chunks across {len(doc_files)} files.")
    _CACHED_INDEX = indexed_chunks
    return _CACHED_INDEX


def retrieve_project_context(
    query: str,
    top_k: int = 4,
    min_score: float = 2.0,
    min_coverage: float = MIN_QUERY_COVERAGE,
) -> List[Dict[str, Any]]:
    """
    Retrieve the most relevant documentation chunks for a user query using BM25 scoring.
    Works completely offline with zero API calls.

    A chunk is returned only when it passes BOTH gates:
      1. ``score >= min_score``      — lexical relevance weight
      2. ``coverage >= min_coverage`` — share of distinct query terms actually present

    The coverage gate exists so the assistant can honestly report uncertainty:
    without it, a query about an unsupported topic can still retrieve a chunk that
    shares one incidental word, which would be presented to the user as a grounded
    answer. Returning ``[]`` instead lets the caller state that it could not find
    sufficient verified information.
    """
    index = build_documentation_index()
    if not index or not query.strip():
        return []

    query_tokens = re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", query.lower())
    if not query_tokens:
        return []

    # Coverage is measured over content terms only: stop words are removed from the
    # index, so including them here would make coverage unreachable for normal prose.
    distinct_terms = sorted({t for t in query_tokens if t not in STOP_WORDS})
    if not distinct_terms:
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

        matched_terms = [t for t in distinct_terms if _term_matched(t, tf_dict)]
        coverage = len(matched_terms) / len(distinct_terms)

        if score >= min_score and coverage >= min_coverage:
            scored_chunks.append({
                "source": c["source"],
                "title": c["title"],
                "content": c["content"],
                "score": round(score, 3),
                "coverage": round(coverage, 3),
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
