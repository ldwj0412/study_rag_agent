"""
retrieve.py — Hybrid search: dense (ChromaDB) + sparse (BM25) → RRF fusion

Usage as a library:
    from retrieve import retrieve
    results = retrieve("what is backpropagation", top_k=5)

Usage from CLI (for testing without generate.py):
    python retrieve.py "what is backpropagation" 5
"""

import logging
import pickle
from pathlib import Path

import chromadb
import numpy as np
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Paths — must mirror ingest.py exactly
# ---------------------------------------------------------------------------

INDEX_DIR       = Path("index")
CHROMA_PATH     = INDEX_DIR / "chroma"
BM25_PATH       = INDEX_DIR / "bm25_corpus.pkl"
CORPUS_IDS_PATH = INDEX_DIR / "corpus_ids.pkl"

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

N_CANDIDATES = 20
# Why 20 when default top_k=5?
# Fetching only top_k candidates from each retriever then fusing gives no
# re-ranking benefit. 4× top_k gives RRF enough overlap between the two
# ranked lists to actually re-order results. Standard empirical ratio.

RERANK_CANDIDATES = N_CANDIDATES
# Reduced from 2×N_CANDIDATES (40) to N_CANDIDATES (20).
# RRF already surfaces the true answer in the top 20 — reranking positions
# 21–40 adds latency with minimal quality gain.

RRF_K = 60
# Damping constant from Cormack, Clarke & Buettcher (2009).
# At k=60: rank 1 → 1/61 ≈ 0.0164, rank 20 → 1/80 = 0.0125.
# Meaningful spread without letting a single rank-1 result dominate.

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger("retrieve")
if not log.handlers:
    log.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    log.addHandler(h)

# ---------------------------------------------------------------------------
# Lazy singleton — loaded once on first retrieve() call, reused forever
# ---------------------------------------------------------------------------

_model:      BGEM3FlagModel | None  = None
_collection: chromadb.Collection | None = None
_bm25:       BM25Okapi | None     = None
_corpus:     list[list[str]] | None = None
_corpus_ids: list[str] | None     = None
_reranker:   FlagReranker | None  = None


def _load_resources() -> None:
    """
    Load all expensive resources on first call; do nothing on subsequent calls.

    Why lazy loading instead of loading at import time?
    BGE-M3 takes 10–30s to load into RAM. generate.py will import retrieve.py,
    so if we loaded at import time the model would load even for operations
    that don't need retrieval. Lazy loading means cost is paid only when
    retrieve() is first called.

    Why module-level globals rather than a class?
    retrieve.py exposes a single function. A singleton class would add
    indirection with no benefit. Module-level globals are idiomatic Python
    for process-scoped singletons.
    """
    global _model, _collection, _bm25, _corpus, _corpus_ids, _reranker

    if _model is not None:
        return  # fast path — already loaded

    log.info("Loading BGE-M3 model...")
    _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    # normalize_embeddings=True is the default in BGEM3FlagModel — query
    # vectors come out L2-normalized automatically.

    log.info("Opening ChromaDB collection...")
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    _collection = client.get_collection("lecture_notes")
    # get_collection, NOT get_or_create_collection — the collection must
    # already exist from ingest.py. Failing loudly here is better than
    # silently creating an empty collection and returning no results.

    log.info("Loading BM25 corpus...")
    with open(BM25_PATH, "rb") as f:
        _corpus = pickle.load(f)
    with open(CORPUS_IDS_PATH, "rb") as f:
        _corpus_ids = pickle.load(f)
    _bm25 = BM25Okapi(_corpus)
    # We reconstruct BM25Okapi from the raw corpus rather than pickling
    # the object itself. Reconstruction takes <1s and avoids pickle
    # version compatibility issues between ingest and retrieve runs.
    #
    # _corpus_ids is a parallel list where _corpus_ids[i] = the ChromaDB
    # string ID for BM25 position i. After incremental deletes/adds the
    # positional str(i) mapping breaks — _corpus_ids is the stable bridge.

    log.info("Loading BGE reranker...")
    _reranker = FlagReranker("BAAI/bge-reranker-base", use_fp16=True)
    # Using bge-reranker-base (~278M params) over bge-reranker-v2-m3 (~570M).
    # The corpus is English lecture notes — the multilingual advantage of v2-m3
    # doesn't apply. bge-reranker-base is ~2× faster with negligible quality
    # difference for English text.

    log.info(f"Ready. ChromaDB has {_collection.count()} chunks.")


# ---------------------------------------------------------------------------
# Step 1: Embed the query
# ---------------------------------------------------------------------------

def _embed_query(query: str) -> list[float]:
    """
    Embed the query string with BGE-M3.

    max_length=512 must match ingest.py's encode() call.
    Using a different truncation length would shift the query's position
    in embedding space relative to the stored chunk embeddings, degrading
    cosine similarity scores.
    """
    output = _model.encode(
        [query],
        batch_size=1,
        max_length=512,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    return output["dense_vecs"][0].tolist()
    # .tolist() converts numpy float32 → Python float, required by ChromaDB


# ---------------------------------------------------------------------------
# Step 2: Dense retrieval via ChromaDB
# ---------------------------------------------------------------------------

def _dense_search(query_embedding: list[float], n: int) -> list[dict]:
    """
    Return the top-n most semantically similar chunks from ChromaDB.

    ChromaDB with hnsw:space="cosine" returns distances in [0, 2]:
      0 = identical vectors, 2 = maximally dissimilar.
    Results are already sorted ascending (most similar first), so the
    first result has RRF rank 1, the second rank 2, etc.

    The distance value itself is NOT used in RRF — only the ordinal rank
    position matters. RRF treats both retrievers as black boxes that produce
    ordered lists; it doesn't care about score scales.
    """
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    # query() is a batch API — results have an outer batch dimension.
    # We sent one query, so we unwrap with [0].
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    return [
        {"text": doc, "metadata": meta}
        for doc, meta in zip(documents, metadatas)
    ]


# ---------------------------------------------------------------------------
# Step 3: Sparse retrieval via BM25
# ---------------------------------------------------------------------------

def _bm25_search(query: str, n: int) -> list[dict]:
    """
    Return the top-n keyword-matching chunks using BM25.

    CRITICAL: tokenize with query.lower().split() — the exact same method
    used in ingest.py when building the corpus:
        tokenized = [chunk.lower().split() for chunk in chunks]
    Any deviation (e.g., forgetting .lower()) causes BM25 to return
    near-zero scores for everything, silently breaking retrieval.

    BM25 complements dense search: dense finds semantically related chunks
    even without exact word overlap; BM25 finds chunks that contain the
    exact terms from the query even if the phrasing differs semantically.
    """
    query_tokens = query.lower().split()
    scores   = _bm25.get_scores(query_tokens)      # shape: (n_chunks,)
    top_idx  = np.argsort(scores)[::-1][:n].tolist()

    # Translate BM25 integer positions → stable ChromaDB string IDs.
    # Why not str(idx)? After incremental adds/deletes the BM25 corpus is
    # rebuilt from ChromaDB in whatever order ChromaDB returns documents.
    # That order is NOT guaranteed to match the original ingestion order,
    # so str(position) would point to the wrong chunk. _corpus_ids is a
    # parallel list written by rebuild_bm25() that maps each BM25 position
    # to its actual ChromaDB ID — safe across any number of partial runs.
    requested_ids = [_corpus_ids[i] for i in top_idx]
    chroma = _collection.get(
        ids=requested_ids,
        include=["documents", "metadatas"],
    )

    # ChromaDB .get() does NOT preserve the order of requested IDs —
    # it returns results in its own internal order. We re-sort by our
    # BM25 rank to restore the correct ordering for RRF.
    id_to_doc  = {cid: doc  for cid, doc  in zip(chroma["ids"], chroma["documents"])}
    id_to_meta = {cid: meta for cid, meta in zip(chroma["ids"], chroma["metadatas"])}

    hits = []
    for cid in requested_ids:
        if cid in id_to_doc:
            hits.append({"text": id_to_doc[cid], "metadata": id_to_meta[cid]})

    return hits


# ---------------------------------------------------------------------------
# Step 4: Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _rrf(
    dense_hits: list[dict],
    bm25_hits:  list[dict],
    k:          int = RRF_K,
) -> list[dict]:
    """
    Merge two ranked lists into a single ranked list using RRF.

    Formula: score(chunk) = Σ  1 / (k + rank_i(chunk))
                             i ∈ {dense, bm25}

    A chunk that appears at rank 3 in dense AND rank 5 in BM25 scores higher
    than a chunk that appears at rank 1 in dense only. This rewards consensus
    between the two retrievers.

    Dedup key = chunk text. Both retrievers point to the same chunks but
    identify them differently (BM25 by integer index, ChromaDB by string ID).
    Using chunk text as the universal key avoids an ID-translation layer.
    It is safe because each chunk has a unique [Source: ... | Page: ... |
    Section: ...] prefix injected by ingest.py.
    """
    scores: dict[str, float] = {}
    data:   dict[str, dict]  = {}

    for rank, hit in enumerate(dense_hits, start=1):
        key = hit["text"]
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        data[key]   = hit

    for rank, hit in enumerate(bm25_hits, start=1):
        key = hit["text"]
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        data[key]   = hit

    ranked_keys = sorted(scores, key=scores.__getitem__, reverse=True)

    # Return ALL ranked candidates — no top_k slice here.
    # retrieve() owns the final truncation, after _rerank() has re-ordered.
    # Pre-slicing to top_k would defeat reranking: a chunk at RRF rank 8
    # might be the most relevant once the cross-encoder sees both texts.
    results = []
    for key in ranked_keys:
        meta = data[key]["metadata"]
        results.append({
            "text":            key,
            "source_file":     meta.get("source_file", ""),
            "page_start":      meta.get("page_start", 0),
            "page_end":        meta.get("page_end", 0),
            "section_heading": meta.get("section_heading", ""),
            "rrf_score":       scores[key],
        })

    return results


# ---------------------------------------------------------------------------
# Step 5: Cross-encoder reranking
# ---------------------------------------------------------------------------

def _rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """
    Rerank candidates with a cross-encoder and return the top_k.

    WHY cross-encoder beats bi-encoder for precision:
    BGE-M3 (bi-encoder) embeds query and chunk independently, then compares
    with cosine similarity. Phrases like "data points" or "similarity" pull
    unrelated chunks (PageRank, Graph Data) near KNN queries in embedding
    space — the bi-encoder never sees the query while encoding a chunk.

    A cross-encoder receives [CLS] query [SEP] chunk [SEP] as one sequence.
    Every transformer layer attends from query tokens to chunk tokens and
    vice versa. When "KNN" and "PageRank" appear together, cross-attention
    detects the topic mismatch. This is impossible with a bi-encoder.
    The cost: no precomputation — every (query, chunk) pair needs a fresh
    forward pass. That is why we run it only over the ~40 RRF candidates,
    not the full corpus.

    WHY strip the source/page/section prefix before scoring:
    The prefix injected by ingest.py carries zero relevance signal for the
    reranker. It was fine-tuned on (query, passage) pairs from retrieval
    datasets — metadata tokens consume token budget and introduce noise.
    The source/page/section fields are preserved in the return dict.

    WHY keep rrf_score alongside rerank_score:
    Backward-compatibility with generate.py, and analytical value: a chunk
    with high rrf_score but low rerank_score is the false positive being
    eliminated. Having both scores makes the pipeline's behaviour debuggable.
    """
    if not candidates:
        return []

    def _strip_prefix(text: str) -> str:
        # Prefix format: "[Source: X | Page: Y | Section: Z]\n\n<content>"
        # Split on first double-newline; fall back to full text if absent.
        parts = text.split("\n\n", 1)
        return parts[1] if len(parts) == 2 else text

    pairs  = [[query, _strip_prefix(c["text"])] for c in candidates]
    scores = _reranker.compute_score(pairs, normalize=False)
    # normalize=False: raw logits are sufficient for ranking order.
    # Applying sigmoid (normalize=True) would add overhead with no benefit.

    scored = [
        {**c, "rerank_score": float(s)}
        for c, s in zip(candidates, scores)
    ]
    # float() handles both numpy scalar and Python float return types
    # depending on FlagEmbedding version.

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(query: str, top_k: int = 5, use_reranker: bool = True) -> list[dict]:
    """
    Retrieve the top_k most relevant chunks for the given query.

    Args:
        query:        natural language question
        top_k:        number of results to return (default 5)
        use_reranker: if True (default), apply cross-encoder reranking after RRF.
                      Set to False to compare RRF-only results for debugging.

    Returns a list of dicts (most relevant first):
        text            — full chunk text including source/page/section prefix
        source_file     — e.g. "lecture03_transformers.pdf"
        page_start      — 1-based page number where chunk starts
        page_end        — 1-based page number where chunk ends
        section_heading — nearest section heading, or "" if none
        rrf_score       — RRF fusion score (always present)
        rerank_score    — cross-encoder logit (present only when use_reranker=True)
    """
    _load_resources()
    query_vec  = _embed_query(query)
    dense_hits = _dense_search(query_vec, N_CANDIDATES)
    bm25_hits  = _bm25_search(query, N_CANDIDATES)
    candidates = _rrf(dense_hits, bm25_hits)   # full ranked list, no top_k yet

    if use_reranker:
        return _rerank(query, candidates[:RERANK_CANDIDATES], top_k)
    else:
        # RRF-only path — useful for A/B comparison without the reranker.
        return candidates[:top_k]


# ---------------------------------------------------------------------------
# CLI entry point — test retrieval without running the full pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print('Usage: python retrieve.py "your query" [top_k]')
        print('Example: python retrieve.py "what is backpropagation" 5')
        sys.exit(1)

    query        = sys.argv[1]
    top_k        = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    use_reranker = "--no-reranker" not in sys.argv

    results = retrieve(query, top_k=top_k, use_reranker=use_reranker)

    print(f'\nQuery: "{query}"')
    print(f"Reranker: {'ON' if use_reranker else 'OFF'}")
    print(f"Top {len(results)} results:\n")

    for i, r in enumerate(results, 1):
        if "rerank_score" in r:
            score_str = f"rerank={r['rerank_score']:.4f}  rrf={r['rrf_score']:.4f}"
        else:
            score_str = f"rrf={r['rrf_score']:.4f}"
        print(f"--- Result {i}  ({score_str}) ---")
        print(f"Source : {r['source_file']}  p.{r['page_start']}–{r['page_end']}")
        print(f"Section: {r['section_heading'] or '(none)'}")
        snippet = r["text"][:300]
        if len(r["text"]) > 300:
            snippet += "..."
        print(snippet)
        print()
