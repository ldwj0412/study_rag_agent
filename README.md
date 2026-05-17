# Personal RAG System

A retrieval-augmented generation (RAG) system that answers questions from university lecture notes (PDFs). Includes two interfaces: a fixed pipeline (`main.py`) and a LangChain agent (`agent.py`).

## Architecture

### Pipeline (`main.py`)
```
Question → Query Expansion → Hybrid Retrieval → Reranking → Generation → Answer
```

1. **Ingest** — Parse PDFs page by page, embed each slide with BGE-M3, store in ChromaDB. Build a BM25 index for keyword search. Incremental: only re-embeds changed files.
2. **Retrieve** — Expand query to lecture terminology, run dense + BM25 search, fuse with RRF, rerank top 20 with a cross-encoder.
3. **Generate** — Pass top 5 chunks as context to Gemini, answer using only the notes.

### Agent (`agent.py`)
A LangChain ReAct agent that decides when to search the notes. Skips retrieval for simple/off-topic questions, can issue multiple searches for complex ones. Uses the same hybrid retriever under the hood.

## Tech Stack

| Component | Choice |
|---|---|
| PDF parsing | pymupdf |
| Embedding | BAAI/bge-m3 (local) |
| Vector store | ChromaDB |
| Sparse retrieval | BM25 (rank-bm25) |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Reranker | BAAI/bge-reranker-base (local) |
| Generation | Gemini 2.5 Flash → 3.1 Flash Lite (fallback) |
| Agent framework | LangChain `create_agent` |

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

# 3a. Fixed pipeline
python main.py

# 3b. Agent (LangChain ReAct)
python agent.py
```

Incremental ingest — re-run `python ingest.py` after adding or changing PDFs. Only modified files are re-embedded.

## Project Structure

```
personal_rag/
├── data/               # PDF lecture notes (gitignored)
├── index/              # Generated index files (gitignored)
│   ├── chroma/         # ChromaDB vector store
│   ├── bm25_corpus.pkl # BM25 tokenized corpus
│   ├── corpus_ids.pkl  # BM25 position → ChromaDB ID mapping
│   └── manifest.json   # SHA-256 manifest for incremental ingest
├── ingest.py           # PDF → chunks → embeddings → index
├── retrieve.py         # Hybrid search + reranking
├── generate.py         # Query expansion + LLM generation
├── main.py             # Interactive Q&A loop (fixed pipeline)
├── agent.py            # LangChain ReAct agent
└── requirements.txt
```
