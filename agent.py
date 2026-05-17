"""
agent.py — RAG agent using LangChain create_agent over lecture notes.

Unlike the fixed pipeline in main.py, the agent decides when to call
retrieval, can search multiple times for complex questions, and answers
simple questions directly without hitting the index.

Usage:
    python agent.py
"""

from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from retrieve import retrieve

load_dotenv()

_genai_client = genai.Client()

# ---------------------------------------------------------------------------
# Query expansion — mirrors generate.py exactly
# ---------------------------------------------------------------------------

def _expand_query(query: str, client: genai.Client) -> str:
    prompt = (
        "Rewrite the following question as a short list of key technical terms "
        "and phrases that would appear in university lecture slides on this topic. "
        "Output only the rewritten query, no explanation, no bullet points.\n\n"
        f"Question: {query}\n"
        "Rewritten:"
    )
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=500, temperature=0.1),
    )
    return response.text.strip()


# ---------------------------------------------------------------------------
# Tool — wraps the existing hybrid retriever (BGE-M3 + BM25 + RRF + reranker)
# ---------------------------------------------------------------------------

@tool
def search_notes(query: str) -> str:
    """Search university lecture notes to find information relevant to the query."""
    expanded = _expand_query(query, _genai_client)
    chunks = retrieve(expanded, top_k=5, use_reranker=True)
    return "\n\n".join(
        f"[{c['source_file']} p.{c['page_start']} | {c['section_heading']}]\n{c['text']}"
        for c in chunks
    )


# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an assistant that answers questions about university lecture notes. "
    "Use the search_notes tool to retrieve relevant content before answering. "
    "Be concise and cite sources (filename + page number) in your answer."
)


def build_agent():
    primary  = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    fallback = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.2)
    # with_fallbacks: if primary hits a rate limit (429), LangChain automatically
    # retries the same call on the fallback model.
    llm = primary.with_fallbacks([fallback])
    return create_agent(llm, tools=[search_notes], system_prompt=SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------

def run() -> None:
    print("Loading models (first query may take ~30s)...")
    agent = build_agent()
    print("RAG Agent ready. Type your question (Ctrl+C or 'quit' to exit).\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not query or query.lower() in {"quit", "exit"}:
            break

        try:
            for event in agent.stream(
                {"messages": [{"role": "user", "content": query}]},
                stream_mode="values",
            ):
                event["messages"][-1].pretty_print()
            print()
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    run()
