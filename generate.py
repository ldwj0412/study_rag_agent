"""
generate.py — RAG answer generation using retrieved chunks + Gemini

Usage as a library:
    from generate import answer
    result = answer("what is KNN?")
    print(result["answer"])
    print(result["sources"])

Usage from CLI:
    python generate.py "what is KNN?"
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from retrieve import retrieve

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()
# Loads GEMINI_API_KEY from .env file in the project root.
# Why .env and not hardcoding? API keys in source code get committed to git
# accidentally. .env is in .gitignore so the key stays local.

MODEL_PRIMARY  = "gemini-2.5-flash"
MODEL_FALLBACK = "gemini-3.1-flash-lite"
# Primary: gemini-2.5-flash — faster, higher quality.
# Fallback: gemini-3.1-flash-lite — higher RPD quota (500/day vs 20/day).
# If the primary hits a 429 rate limit, the fallback is used automatically.

TOP_K      = 5    # number of chunks to retrieve per query
MAX_TOKENS = 1024 # max tokens in the generated answer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger("generate")
if not log.handlers:
    log.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    log.addHandler(h)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a study assistant helping a university student understand their course material.

Answer the question using ONLY the provided context excerpts from the student's lecture notes.
If the context does not contain enough information to answer the question, say so clearly — do not make up information.

Rules:
- Be concise and clear.
- Use bullet points or numbered steps where appropriate.
- If you reference specific content, mention which source it came from (e.g. "According to classification2.pdf...").
- Do not add information beyond what is in the context.
"""

def _generate(client: genai.Client, prompt: str, **kwargs) -> str:
    """Call Gemini with primary model, fall back to lite on 429."""
    from google.genai.errors import ClientError
    try:
        r = client.models.generate_content(model=MODEL_PRIMARY, contents=prompt, **kwargs)
        return r.text.strip()
    except ClientError as e:
        if e.code != 429:
            raise
        log.warning(f"Primary model rate-limited, falling back to {MODEL_FALLBACK}")
        r = client.models.generate_content(model=MODEL_FALLBACK, contents=prompt, **kwargs)
        return r.text.strip()


def _expand_query(query: str, client: genai.Client) -> str:
    """
    Rewrite the user's conversational question into lecture-style terminology.

    Why query expansion?
    BGE-M3 matches queries to chunks by semantic similarity. A conversational
    question like "how does KNN classify new data points?" may not embed close
    to a slide that says "nearest neighbor majority vote distance classification".
    Rewriting the query to use academic/lecture vocabulary closes this gap.

    Why use Gemini for this instead of a fixed template?
    The rewriting is domain-aware — Gemini knows that "classify" maps to
    "classification", "KNN" to "nearest neighbor", etc. A fixed template
    can't generalise across topics (KNN, clustering, decision trees, ...).

    The expanded query is used ONLY for retrieval. The original query is
    still used in the answer prompt so the LLM answers what the user asked,
    not the rewritten version.
    """
    prompt = (
        "Rewrite the following question as a short list of key technical terms "
        "and phrases that would appear in university lecture slides on this topic. "
        "Output only the rewritten query, no explanation, no bullet points.\n\n"
        f"Question: {query}\n"
        "Rewritten:"
    )
    return _generate(
        client, prompt,
        config=types.GenerateContentConfig(max_output_tokens=500, temperature=0.1),
    )


def _build_prompt(query: str, chunks: list[dict]) -> str:
    """
    Build the full prompt sent to Gemini.

    Why include source/page/section in the context block?
    The LLM can use this to cite where information came from, making the
    answer more trustworthy and easier to verify against the original notes.

    Why not just concatenate all chunk texts?
    Labelling each chunk with its source makes the model's citations accurate.
    Without labels, the model has no way to say "according to lecture X".
    """
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        header = (
            f"[Context {i}] "
            f"{chunk['source_file']} "
            f"p.{chunk['page_start']} — "
            f"{chunk['section_heading'] or 'General'}"
        )
        # Strip the machine-readable prefix injected by ingest.py —
        # the LLM prompt already has the header above; the prefix is redundant
        # and consumes tokens unnecessarily.
        body = chunk["text"].split("\n\n", 1)
        body_text = body[1] if len(body) == 2 else chunk["text"]
        context_blocks.append(f"{header}\n{body_text.strip()}")

    context_str = "\n\n---\n\n".join(context_blocks)

    return f"""{SYSTEM_PROMPT}

===== CONTEXT FROM LECTURE NOTES =====

{context_str}

===== QUESTION =====

{query}

===== ANSWER ====="""


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def answer(query: str, top_k: int = TOP_K) -> dict:
    """
    Retrieve relevant chunks and generate an answer with Gemini.

    Returns a dict:
        answer   — the LLM's response string
        sources  — list of dicts [{source_file, page_start, page_end, section_heading}]
        chunks   — the raw retrieved chunks (for inspection/debugging)
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found. "
            "Create a .env file with: GEMINI_API_KEY=your_key_here\n"
            "Get a free key at: https://aistudio.google.com"
        )
    client = genai.Client(api_key=api_key)
    # New google-genai SDK uses Client() instead of configure() + GenerativeModel().

    # Step 1: Expand query into lecture terminology for better retrieval
    expanded = _expand_query(query, client)
    log.info(f'Query expanded: "{expanded}"')

    # Step 2: Retrieve using expanded query, answer using original
    log.info(f'Retrieving top {top_k} chunks for: "{query}"')
    chunks = retrieve(expanded, top_k=top_k, use_reranker=True)

    if not chunks:
        return {
            "answer":  "No relevant content found in your lecture notes for this query.",
            "sources": [],
            "chunks":  [],
        }

    # Step 3: Build prompt (using original query, not expanded)
    prompt = _build_prompt(query, chunks)
    log.info(f"Prompt built ({len(prompt)} chars). Calling Gemini...")

    # Step 4: Generate
    answer_text = _generate(
        client, prompt,
        config=types.GenerateContentConfig(max_output_tokens=MAX_TOKENS, temperature=0.2),
    )
    log.info("Answer generated.")

    # Step 4: Build source list for the caller
    sources = [
        {
            "source_file":     c["source_file"],
            "page_start":      c["page_start"],
            "page_end":        c["page_end"],
            "section_heading": c["section_heading"],
        }
        for c in chunks
    ]

    return {
        "answer":         answer_text,
        "expanded_query": expanded,
        "sources":        sources,
        "chunks":         chunks,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print('Usage: python generate.py "your question"')
        print('Example: python generate.py "how does KNN classify new data points?"')
        sys.exit(1)

    query  = sys.argv[1]
    result = answer(query)

    print(f'\nQuestion: {query}')
    print(f'Expanded: {result["expanded_query"]}\n')
    print("=" * 60)
    print(result["answer"])
    print("=" * 60)
    print("\nSources used:")
    for i, s in enumerate(result["sources"], 1):
        print(
            f"  {i}. {s['source_file']}  "
            f"p.{s['page_start']}  "
            f"— {s['section_heading'] or 'General'}"
        )
