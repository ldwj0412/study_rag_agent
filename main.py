"""
main.py — Interactive Q&A loop over your lecture notes

Usage:
    python main.py

Models load once on startup, then each query is fast.
Type 'quit', 'exit', or 'q' to stop.
"""

import logging
import time

from generate import answer
from retrieve import retrieve

log = logging.getLogger("main")
if not log.handlers:
    log.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    log.addHandler(h)

DIVIDER = "=" * 62


def warmup() -> None:
    """
    Force all models to load before the user types their first question.

    Why a warmup call?
    retrieve.py uses lazy loading — models only load on the first retrieve()
    call. Without warmup, the first query would silently take 30+ seconds
    with no feedback. The warmup call triggers loading immediately on startup
    so the user sees the progress logs and knows the system is getting ready.
    """
    print("Loading models...", flush=True)
    t0 = time.time()
    retrieve("warmup", top_k=1, use_reranker=True)
    print(f"Ready ({time.time() - t0:.1f}s)\n")


def run() -> None:
    warmup()
    print("Ask a question about your lecture notes (or 'quit' to exit)\n")

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        try:
            result = answer(query)
        except Exception as e:
            print(f"Error: {e}\n")
            continue

        print(f"\nExpanded: {result['expanded_query']}\n")
        print(DIVIDER)
        print(result["answer"])
        print(DIVIDER)
        print("\nSources:")
        for i, s in enumerate(result["sources"], 1):
            print(
                f"  {i}. {s['source_file']}  "
                f"p.{s['page_start']}  "
                f"— {s['section_heading'] or 'General'}"
            )
        print()


if __name__ == "__main__":
    run()
