# Study RAG Agent

A retrieval-augmented generation (RAG) agent that answers questions from your study materials (PDFs). The main interface is a LangChain ReAct agent (`agent.py`) that decides when to search, can issue multiple searches for complex questions, and streams the answer token-by-token. A simpler fixed pipeline (`main.py`) is also included.

## Architecture

### Agent (`agent.py`) — primary interface

Follows the **ReAct** (Reason → Act → Observe) loop:

```
User query
    ↓
① REASON — LLM decides what to do next
    ↓ (needs notes)          ↓ (simple / off-topic)
② ACT — call search_notes    → Answer directly
    ↓
   Query expansion → Hybrid retrieval → Top 5 chunks
    ↓
③ OBSERVE — LLM reads chunks, reasons again (back to ①)
    ↓ (has enough context)
   Streaming answer with citations
```

The loop repeats until the LLM decides it has enough context. In practice the hybrid retriever surfaces relevant chunks in one call, so the loop exits after a single iteration. Multi-call behaviour kicks in only if the first search returns nothing useful.

**Conversational memory (two-tier)**

- **Short-term (per-thread)** — `SqliteSaver` persists every message to `index/checkpoints.db`. Each run starts a new thread (fresh UUID) by default, so follow-up questions like "what are its disadvantages?" or "explain that more simply" work without re-stating context. Threads survive restarts and can be resumed later.
- **Long-term (cross-session)** — `SqliteStore` persists user-saved facts to `index/memory.db`. When you ask the agent to remember something it calls `save_memory`; on every session start it calls `load_memories` and primes itself with those facts automatically.

**Slash commands** — type at the `You:` prompt before your question:

| Command | Effect |
|---|---|
| `/threads` | List all saved conversation threads with a preview of the first message |
| `/resume <n>` | Switch to thread n from the list and continue that conversation |
| `/new` | Start a fresh thread (default on startup) |

### Fixed Pipeline (`main.py`)
```
Question → Query Expansion → Hybrid Retrieval → Reranking → Generation → Answer
```

1. **Ingest** — Parse PDFs page by page, embed each slide with BGE-M3, store in ChromaDB. Build a BM25 index for keyword search. Incremental: only re-embeds changed files.
2. **Retrieve** — Expand query to lecture terminology, run dense + BM25 search, fuse with RRF, rerank top 20 with a cross-encoder.
3. **Generate** — Pass top 5 chunks as context to Gemini, answer using only the notes.

## Tech Stack

| Component | Choice |
|---|---|
| PDF parsing | pymupdf |
| Embedding | BAAI/bge-m3 (local) |
| Vector store | ChromaDB |
| Sparse retrieval | BM25 (rank-bm25) |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Reranker | BAAI/bge-reranker-base (local) |
| Generation | Gemini 3.1 Flash Lite → 2.5 Flash Lite (fallback) |
| Agent framework | LangChain `create_agent` |
| Short-term memory | SqliteSaver (langgraph-checkpoint-sqlite) |
| Long-term memory | SqliteStore (langgraph-checkpoint-sqlite) |

## Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows

# Install dependencies
pip install -r requirements.txt
pip install "transformers<4.46"  # FlagEmbedding compatibility

# Add your Gemini API key
# Create .env file (use UTF-8 encoding):
# GEMINI_API_KEY=your_key_here
```

Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com).

## Usage

```bash
# 1. Add your PDF lecture notes to data/
mkdir data
# copy your PDFs into data/

# 2. Build the index
python ingest.py

# 3a. Agent (LangChain ReAct) — recommended
python agent.py

# 3b. Fixed pipeline
python main.py
```

Incremental ingest — re-run `python ingest.py` after adding or changing PDFs. Only modified files are re-embedded.

## Project Structure

```
study-rag-agent/
├── data/               # PDF lecture notes (gitignored)
├── index/              # Generated index files (gitignored)
│   ├── chroma/         # ChromaDB vector store
│   ├── bm25_corpus.pkl # BM25 tokenized corpus
│   ├── corpus_ids.pkl  # BM25 position → ChromaDB ID mapping
│   ├── manifest.json   # SHA-256 manifest for incremental ingest
│   ├── checkpoints.db  # SqliteSaver conversation threads
│   └── memory.db       # SqliteStore long-term facts
├── ingest.py           # PDF → chunks → embeddings → index
├── retrieve.py         # Hybrid search + reranking
├── generate.py         # Query expansion + LLM generation
├── main.py             # Interactive Q&A loop (fixed pipeline)
├── agent.py            # LangChain ReAct agent (primary)
└── requirements.txt
```
