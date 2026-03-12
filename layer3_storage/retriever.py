# layer3_storage/retriever.py
# PURPOSE: Natural language Q&A over your stored intel using RAG
# RAG = Retrieval Augmented Generation
# How it works:
#   1. Take a plain English question
#   2. Search your stored documents for relevant context
#   3. Send question + context to an LLM
#   4. LLM answers using YOUR data, not just its training
# This means answers are grounded in real intel you've collected!

from loguru import logger
from typing import Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from layer3_storage.hybrid_search import hybrid_search

# LLM clients
import ollama                           # Local: phi3:mini
from groq import Groq                   # Cloud fast: llama-3.1-8b-instant
import google.generativeai as genai     # Multimodal: gemini-1.5-flash
from dotenv import load_dotenv
import os

load_dotenv()

# ── LLM Configuration ─────────────────────────────────────────────────────────

# Local model (no API key needed, runs on your machine)
LOCAL_MODEL = "phi3:mini"

# Groq cloud models (fast, free tier)
GROQ_FAST_MODEL = "llama-3.1-8b-instant"
GROQ_ACCURATE_MODEL = "llama-3.3-70b-versatile"

# Google Gemini (multimodal, free tier)
GEMINI_MODEL = "gemini-1.5-flash"

# How many context chunks to feed to the LLM
MAX_CONTEXT_CHUNKS = 5      # More chunks = better answers but slower
MAX_CONTEXT_CHARS = 4000    # Max characters of context to send


# ── Context Builder ───────────────────────────────────────────────────────────

def _build_context(search_results: list[dict]) -> str:
    """
    Takes search results and formats them into a clean context string
    that gets sent to the LLM alongside the question.

    Each chunk is labeled with its source and date so the LLM can
    reference where information came from.
    """
    if not search_results:
        return "No relevant documents found in the intel database."

    context_parts = []
    total_chars = 0

    for i, result in enumerate(search_results[:MAX_CONTEXT_CHUNKS]):
        # Get the text content
        text = result.get("text", "")
        if not text:
            continue

        # Build source label
        source = result.get("source", "unknown")
        title = result.get("title", "")
        date = result.get("date", "")
        strategy = result.get("search_strategy", "")

        label = f"[Source {i+1}]"
        if title:
            label += f" {title}"
        if source:
            label += f" ({source})"
        if date:
            label += f" — {date}"

        chunk_text = f"{label}\n{text}"

        # Stop if we've hit the character limit
        if total_chars + len(chunk_text) > MAX_CONTEXT_CHARS:
            break

        context_parts.append(chunk_text)
        total_chars += len(chunk_text)

    return "\n\n---\n\n".join(context_parts)


def _build_prompt(question: str, context: str, task_type: str = "qa") -> str:
    """
    Builds the full prompt sent to the LLM.
    Different task types get different prompt styles.

    task_type options:
    - "qa"        : direct question answering
    - "summary"   : summarize what's known about a topic
    - "analysis"  : deeper analytical inference
    - "timeline"  : chronological summary of events
    """
    prompts = {
        "qa": f"""You are an intel analysis assistant. Answer the question using ONLY the provided context.
If the context doesn't contain enough information, say so clearly.
Be concise, factual, and cite which source you used.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:""",

        "summary": f"""You are an intel analysis assistant. Summarize what is known about the topic
based on the provided context. Be structured and factual.

CONTEXT:
{context}

TOPIC: {question}

SUMMARY:""",

        "analysis": f"""You are a senior intelligence analyst. Analyze the provided context and
draw key inferences, patterns, and insights related to the query.
Distinguish between facts and inferences clearly.

CONTEXT:
{context}

QUERY: {question}

ANALYSIS:""",

        "timeline": f"""You are an intel analysis assistant. Based on the provided context,
construct a chronological timeline of events related to the query.
Format each event as: [DATE] — Event description

CONTEXT:
{context}

TOPIC: {question}

TIMELINE:"""
    }

    return prompts.get(task_type, prompts["qa"])


# ── LLM Callers ───────────────────────────────────────────────────────────────

def _call_local(prompt: str) -> Optional[str]:
    """
    Calls phi3:mini running locally via Ollama.
    Slowest but free, private, works offline.
    Use for: testing, sensitive data, when APIs are down.
    """
    try:
        response = ollama.chat(
            model=LOCAL_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]
    except Exception as e:
        logger.error(f"Local LLM failed: {e}")
        return None


def _call_groq_fast(prompt: str) -> Optional[str]:
    """
    Calls llama-3.1-8b-instant via Groq.
    Very fast, good for simple Q&A and quick lookups.
    """
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model=GROQ_FAST_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Groq fast LLM failed: {e}")
        return None


def _call_groq_accurate(prompt: str) -> Optional[str]:
    """
    Calls llama-3.3-70b-versatile via Groq.
    Slower but much smarter — use for deep analysis tasks.
    """
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model=GROQ_ACCURATE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Groq accurate LLM failed: {e}")
        return None


def _call_gemini(prompt: str) -> Optional[str]:
    """
    Calls Gemini 1.5 Flash via Google API.
    Best for multimodal tasks and when you need Google's knowledge.
    """
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Gemini LLM failed: {e}")
        return None


# ── Main Retriever ────────────────────────────────────────────────────────────

def ask(
    question: str,
    query_embedding: Optional[list[float]] = None,  # From embedder.py
    llm: str = "groq_fast",                         # Which LLM to use
    task_type: str = "qa",                           # Type of task
    source: Optional[str] = None,                   # Filter by source
    language: Optional[str] = None,                 # Filter by language
    min_quality: float = 0.0,                        # Min quality filter
    top_k: int = 5                                   # Context chunks to use
) -> dict:
    """
    The MAIN function — ask a question, get an answer from your intel.

    Parameters:
    - question        : plain English question or topic
    - query_embedding : vector of the question (for semantic search)
    - llm             : which model to use:
                        "local"         → phi3:mini (offline, slow)
                        "groq_fast"     → llama-3.1-8b (fast, cloud)
                        "groq_accurate" → llama-3.3-70b (smart, cloud)
                        "gemini"        → gemini-1.5-flash (multimodal)
    - task_type       : "qa", "summary", "analysis", "timeline"
    - source          : only search this source type
    - language        : only search this language
    - min_quality     : minimum quality threshold
    - top_k           : how many context chunks to retrieve

    Returns a dict with:
    - "answer"    : the LLM's response
    - "sources"   : list of sources used as context
    - "llm_used"  : which LLM answered
    - "chunks_used": how many context chunks were used
    """

    logger.info(f"[Retriever] Question: '{question}' | LLM: {llm} | Task: {task_type}")

    # ── Step 1: Search for relevant context ──────────────────────────────────
    search_results = hybrid_search(
        query=question,
        query_embedding=query_embedding,
        source=source,
        language=language,
        min_quality=min_quality,
        top_k=top_k,
        use_rerank=True
    )

    logger.info(f"[Retriever] Found {len(search_results)} context chunks")

    # ── Step 2: Build context string ─────────────────────────────────────────
    context = _build_context(search_results)

    # ── Step 3: Build prompt ──────────────────────────────────────────────────
    prompt = _build_prompt(question, context, task_type)

    # ── Step 4: Call the chosen LLM ──────────────────────────────────────────
    llm_callers = {
        "local":         _call_local,
        "groq_fast":     _call_groq_fast,
        "groq_accurate": _call_groq_accurate,
        "gemini":        _call_gemini
    }

    caller = llm_callers.get(llm, _call_groq_fast)
    answer = caller(prompt)

    # Fallback chain: if chosen LLM fails, try next best
    if not answer:
        logger.warning(f"[Retriever] {llm} failed, trying groq_fast fallback...")
        answer = _call_groq_fast(prompt)

    if not answer:
        logger.warning("[Retriever] groq_fast failed, trying local fallback...")
        answer = _call_local(prompt)

    if not answer:
        answer = "Could not generate an answer — all LLMs unavailable."

    # ── Step 5: Build response ────────────────────────────────────────────────
    sources = []
    for r in search_results:
        source_info = {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "source": r.get("source", ""),
            "date": r.get("date", ""),
            "strategy": r.get("search_strategy", ""),
            "score": r.get("final_score", 0.0)
        }
        sources.append(source_info)

    result = {
        "question": question,
        "answer": answer,
        "sources": sources,
        "llm_used": llm,
        "task_type": task_type,
        "chunks_used": len(search_results)
    }

    logger.success(f"[Retriever] Answer generated using {len(search_results)} chunks.")
    return result


# ── Convenience Wrappers ──────────────────────────────────────────────────────

def quick_answer(question: str, query_embedding: Optional[list[float]] = None) -> str:
    """
    Fast Q&A using Groq fast model. Returns just the answer string.
    Use for simple factual questions.
    """
    result = ask(question, query_embedding, llm="groq_fast", task_type="qa")
    return result["answer"]


def deep_analysis(question: str, query_embedding: Optional[list[float]] = None) -> str:
    """
    Deep analysis using the accurate 70B model.
    Use for complex analytical tasks where quality matters most.
    """
    result = ask(question, query_embedding, llm="groq_accurate", task_type="analysis")
    return result["answer"]


def summarize_topic(topic: str, query_embedding: Optional[list[float]] = None) -> str:
    """
    Summarizes everything known about a topic from your intel.
    Use for research briefings and topic overviews.
    """
    result = ask(topic, query_embedding, llm="groq_accurate", task_type="summary")
    return result["answer"]


def build_timeline(topic: str, query_embedding: Optional[list[float]] = None) -> str:
    """
    Builds a chronological timeline of events for a topic.
    Use for tracking how a story developed over time.
    """
    result = ask(topic, query_embedding, llm="groq_accurate", task_type="timeline")
    return result["answer"]


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run this file directly to test the RAG pipeline.
    Command: python layer3_storage/retriever.py
    Make sure:
    - Docker + Qdrant running
    - GROQ_API_KEY in your .env file
    - storage_manager.py test has been run (so there's data)
    """
    logger.info("Testing Retriever (RAG pipeline)...")

    # Fake embedding — in real use this comes from embedder.py
    fake_embedding = [0.1] * 768

    # Test 1: Basic Q&A
    logger.info("--- Test 1: Basic Q&A ---")
    result = ask(
        question="What has SpaceX been doing?",
        query_embedding=fake_embedding,
        llm="groq_fast",
        task_type="qa"
    )
    print(f"\nQuestion: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Sources used: {result['chunks_used']}")
    print(f"LLM: {result['llm_used']}")

    # Test 2: Summary
    logger.info("--- Test 2: Topic Summary ---")
    summary = summarize_topic("Elon Musk", fake_embedding)
    print(f"\nSummary:\n{summary}")

    # Test 3: Analysis
    logger.info("--- Test 3: Deep Analysis ---")
    analysis = deep_analysis("What are the latest space developments?", fake_embedding)
    print(f"\nAnalysis:\n{analysis}")

    logger.success("Retriever test complete!")