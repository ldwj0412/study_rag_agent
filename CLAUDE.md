# CLAUDE.md — study-rag-agent

## Environment
- Windows 10, PowerShell
- Virtual env: `.venv\Scripts\Activate.ps1`
- Always run Python via the activated venv

## Critical setup quirk
```
pip install "transformers<4.46"   # FlagEmbedding compatibility — do not upgrade
```

## Project layout
- `data/` — PDF lecture notes (gitignored)
- `index/` — all generated files (gitignored): ChromaDB, BM25, checkpoints.db, memory.db
- `agent.py` — primary entry point (ReAct agent)
- `retrieve.py` — hybrid retrieval (BGE-M3 + BM25 + RRF + reranker)
- `ingest.py` — PDF → embeddings → index
- `main.py` — fixed pipeline (secondary)

## Standard workflow after code changes
1. `git add <file> && git commit && git push`
2. Update Notion page `3637bc7f-b828-8166-ad0a-c8cf70b06e85`
3. Update README if architecture or usage changed

## Memory system
- Short-term: `SqliteSaver` → `index/checkpoints.db` (conversation threads)
- Long-term: `SqliteStore` → `index/memory.db` (user-saved facts)
- Slash commands: `/threads`, `/resume <n>`, `/new`

## Models
- Primary: `gemini-3.1-flash-lite` (thinking OFF, higher quota)
- Fallback: `gemini-2.5-flash-lite` (thinking OFF, faster first token)
- Do not use `gemini-2.5-flash` (non-lite) — thinking mode breaks streaming

## Key constraints
- Agent is strictly grounded to lecture notes — no fallback to LLM training knowledge
- `index/` files are gitignored — never commit them
- Do not upgrade `transformers` past 4.45.x
