"""
ingest.py — RAG pipeline: PDF → chunks → embeddings → index

Run once to build the index, then re-run whenever notes change.
Incremental mode: only PDFs whose SHA-256 has changed are re-embedded.

Output artifacts (all in index/):
  chroma/          ← ChromaDB's files (dense vectors + metadata + text)
  bm25_corpus.pkl  ← tokenized chunks for BM25 keyword search in retrieve.py
  corpus_ids.pkl   ← parallel list: corpus_ids[i] = ChromaDB ID at BM25 position i
  manifest.json    ← {filename: {sha256, chunk_ids}} tracks what has been ingested

Design philosophy: every non-obvious decision is commented with WHY, not just WHAT.
"""

import hashlib
import json
import logging
import pickle
import time
from pathlib import Path


import chromadb
import fitz
import numpy as np
from FlagEmbedding import BGEM3FlagModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR  = Path("data")
INDEX_DIR = Path("index")
LOGS_DIR  = Path("logs")

CHROMA_PATH     = INDEX_DIR / "chroma"
BM25_PATH       = INDEX_DIR / "bm25_corpus.pkl"
CORPUS_IDS_PATH = INDEX_DIR / "corpus_ids.pkl"
MANIFEST_PATH   = INDEX_DIR / "manifest.json"

MAX_CHUNK_CHARS     = 2000  # safety cap for very text-heavy pages
CHUNK_OVERLAP_CHARS = 100   # overlap when a page exceeds MAX_CHUNK_CHARS
MARGIN_TOP          = 50    # points from top — running header zone
MARGIN_BOTTOM       = 50    # points from bottom — footer zone
MIN_PAGE_CHARS      = 30    # pages with fewer chars are image-only slides
EMBED_BATCH_SIZE    = 64    # 3080 Ti 12GB VRAM — safe at fp16


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    log = logging.getLogger("ingest")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    log.addHandler(logging.FileHandler(LOGS_DIR / "ingest.log", encoding="utf-8"))
    log.addHandler(logging.StreamHandler())
    for h in log.handlers:
        h.setFormatter(fmt)
    return log


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def hash_file(path: Path) -> str:
    """SHA-256 of file bytes — detects any content change."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest() -> dict:
    """Load manifest.json if it exists, otherwise return empty dict."""
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict) -> None:
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


# ---------------------------------------------------------------------------
# PDF utilities
# ---------------------------------------------------------------------------

def is_margin_noise(block: dict, page_height: float) -> bool:
    """
    Filter running headers/footers that repeat on every slide
    (course name, slide number, date). Detected by vertical position.
    """
    top_y    = block["bbox"][1]
    bottom_y = block["bbox"][3]
    return top_y < MARGIN_TOP or bottom_y > (page_height - MARGIN_BOTTOM)


def split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """
    Split text exceeding max_chars into overlapping sub-chunks.
    Used only for unusually dense pages. Overlap preserves sentences
    that land at split boundaries.
    """
    chunks, start = [], 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunks.append(text[start:])
            break
        split_at = text.rfind(" ", start, end)
        if split_at == -1:
            split_at = end
        chunks.append(text[start:split_at])
        start = max(split_at - overlap, 0)
    return [c.strip() for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Chunking — page-based
# ---------------------------------------------------------------------------

def extract_chunks_from_pdf(
    pdf_path: Path, log: logging.Logger
) -> tuple[list[str], list[dict], list[str]]:
    """
    Extract one chunk per page from a lecture slide PDF.

    Why page-based over heading-boundary chunking?
    Lecture slides are authored as one concept per slide. Heading detection
    with font-size heuristics was misclassifying body text as headings,
    producing tiny, incomplete chunks (e.g. a single sentence like "K-NN
    search is typically assisted by indices"). Page-based chunking is:
    - More reliable: page boundaries are exact, no font heuristics
    - Better aligned with slide structure: one slide = one idea = one chunk
    - Simpler: no heading detection, no font profiling

    Slide title extraction: the first non-empty text line on a page is
    almost always the slide title. We use it as section_heading for
    metadata and for the context prefix embedded with the chunk.

    Pages with fewer than MIN_PAGE_CHARS of text are image-only slides
    (diagrams, animations) and are skipped.
    """
    doc         = fitz.open(pdf_path)
    source_file = pdf_path.name
    chunks, metadata, ids = [], [], []
    skipped = 0

    for page in doc:
        page_num    = page.number + 1
        page_height = page.rect.height

        # Collect all text spans, filtering margin noise
        lines = []
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            if is_margin_noise(block, page_height):
                continue
            for line in block["lines"]:
                line_text = " ".join(
                    span["text"].strip()
                    for span in line["spans"]
                    if span["text"].strip()
                )
                if line_text:
                    lines.append(line_text)

        page_text = "\n".join(lines).strip()

        if len(page_text) < MIN_PAGE_CHARS:
            skipped += 1
            continue

        # First line = slide title (used as section heading in metadata + prefix)
        slide_title = lines[0].strip() if lines else ""

        prefix = (
            f"[Source: {source_file} | "
            f"Page: {page_num} | "
            f"Slide: {slide_title}]\n\n"
        )

        # Most slides fit in one chunk. Split only if unusually long.
        sub_bodies = (
            split_long_text(page_text, MAX_CHUNK_CHARS, CHUNK_OVERLAP_CHARS)
            if len(page_text) > MAX_CHUNK_CHARS else [page_text]
        )

        for sub_i, sub in enumerate(sub_bodies):
            full_text = prefix + sub
            # Content-addressed ID: stable across runs for the same page.
            # Single-chunk pages: "file.pdf_p6"
            # Sub-chunks of long pages: "file.pdf_p6_s0", "file.pdf_p6_s1"
            chunk_id = (
                f"{source_file}_p{page_num}"
                if len(sub_bodies) == 1
                else f"{source_file}_p{page_num}_s{sub_i}"
            )
            chunks.append(full_text)
            ids.append(chunk_id)
            metadata.append({
                "source_file":     source_file,
                "page_start":      page_num,
                "page_end":        page_num,
                "section_heading": slide_title,
                "char_length":     len(full_text),
            })

    doc.close()

    if skipped:
        log.warning(f"  {source_file}: skipped {skipped} image-only page(s)")

    return chunks, metadata, ids


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_chunks(chunks: list[str], log: logging.Logger) -> np.ndarray:
    """
    Embed all chunks with BGE-M3, returning dense vectors.

    BGE-M3 auto-detects GPU via PyTorch — no explicit device argument needed.
    use_fp16=True halves VRAM usage with negligible quality loss.
    """
    log.info("Loading BAAI/bge-m3...")
    model     = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    n_batches = (len(chunks) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
    all_dense = []

    log.info(f"Encoding {len(chunks)} chunks in {n_batches} batch(es)...")

    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch     = chunks[i : i + EMBED_BATCH_SIZE]
        batch_num = i // EMBED_BATCH_SIZE + 1
        output    = model.encode(
            batch,
            batch_size=EMBED_BATCH_SIZE,
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        all_dense.append(output["dense_vecs"])
        log.info(f"  Batch {batch_num}/{n_batches} done")

    return np.vstack(all_dense)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def add_to_chroma(
    collection: chromadb.Collection,
    chunks:     list[str],
    ids:        list[str],
    metadata:   list[dict],
    embeddings: np.ndarray,
    log:        logging.Logger,
) -> None:
    """Add (or upsert) chunks into the collection with their content-addressed IDs."""
    collection.upsert(
        ids        = ids,
        embeddings = embeddings.tolist(),
        documents  = chunks,
        metadatas  = metadata,
    )
    log.info(f"ChromaDB: {collection.count()} total chunks after upsert")


def rebuild_bm25(collection: chromadb.Collection, log: logging.Logger) -> None:
    """
    Rebuild BM25 corpus and corpus_ids from the current ChromaDB contents.

    Why rebuild from ChromaDB instead of from the local chunk lists?
    After incremental adds/deletes the ChromaDB collection is the single source
    of truth for what chunks exist. Rebuilding from it guarantees BM25 and
    ChromaDB are always in sync, even across multiple partial runs.

    corpus_ids[i] = the ChromaDB string ID for BM25 position i.
    retrieve.py uses this to translate BM25 rank → ChromaDB ID without
    relying on positional str(i) IDs, which break after deletions.
    """
    result    = collection.get(include=["documents"])
    ids       = result["ids"]
    docs      = result["documents"]
    tokenized = [doc.lower().split() for doc in docs]

    with open(BM25_PATH, "wb") as f:
        pickle.dump(tokenized, f)
    with open(CORPUS_IDS_PATH, "wb") as f:
        pickle.dump(ids, f)

    log.info(f"BM25 corpus rebuilt: {len(tokenized)} chunks")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log = setup_logging()
    t0  = time.time()

    INDEX_DIR.mkdir(exist_ok=True)

    # Open (or create) the ChromaDB collection.
    # get_or_create_collection instead of delete+create so incremental runs
    # preserve existing chunks.
    client     = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(
        name="lecture_notes",
        metadata={"hnsw:space": "cosine"},
    )

    manifest  = load_manifest()
    pdf_files = sorted(DATA_DIR.rglob("*.pdf"))

    if not pdf_files:
        log.error(f"No PDFs found in {DATA_DIR}/. Add your notes and re-run.")
        return
    log.info(f"Found {len(pdf_files)} PDF(s)")

    # Detect PDFs that were removed from data/ since last run
    current_names = {p.name for p in pdf_files}
    for removed_name in set(manifest) - current_names:
        old_ids = manifest[removed_name]["chunk_ids"]
        collection.delete(ids=old_ids)
        del manifest[removed_name]
        log.info(f"{removed_name}: removed from data/, deleted {len(old_ids)} chunk(s)")

    # Process each PDF
    new_chunks:   list[str]  = []
    new_ids:      list[str]  = []
    new_metadata: list[dict] = []

    for pdf_path in pdf_files:
        file_hash = hash_file(pdf_path)
        name      = pdf_path.name

        if manifest.get(name, {}).get("sha256") == file_hash:
            log.info(f"{name}: unchanged, skipping")
            continue

        log.info(f"{name}: changed, re-ingesting...")

        # Delete old chunks for this PDF (if any) before adding new ones
        if name in manifest:
            old_ids = manifest[name]["chunk_ids"]
            collection.delete(ids=old_ids)
            log.info(f"  Deleted {len(old_ids)} old chunk(s)")

        chunks, metadata, ids = extract_chunks_from_pdf(pdf_path, log)
        log.info(f"  → {len(chunks)} chunk(s)")

        new_chunks.extend(chunks)
        new_ids.extend(ids)
        new_metadata.extend(metadata)

        manifest[name] = {"sha256": file_hash, "chunk_ids": ids}

    # Embed and store new/changed chunks (skip if nothing changed)
    if new_chunks:
        log.info(f"Embedding {len(new_chunks)} new/changed chunk(s)...")
        embeddings = embed_chunks(new_chunks, log)
        add_to_chroma(collection, new_chunks, new_ids, new_metadata, embeddings, log)
    else:
        log.info("No PDFs changed — index is up to date.")

    # Always rebuild BM25 to stay in sync with ChromaDB
    log.info("Rebuilding BM25 corpus...")
    rebuild_bm25(collection, log)

    save_manifest(manifest)
    log.info(f"Ingest complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
