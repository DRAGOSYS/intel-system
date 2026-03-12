"""
Layer 4 Intelligence - Prompt Templates
All prompts used by LLM components. Centralised for consistency and easy tuning.
Temperature is always <= 0.1 to minimise hallucination.
"""

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Dataclass wrappers
# ---------------------------------------------------------------------------

@dataclass
class PromptTemplate:
    """A named prompt with an optional system message and user template."""
    name: str
    system: str
    user_template: str  # Use {placeholders} for substitution
    temperature: float = 0.1
    max_tokens: int = 1024


# ---------------------------------------------------------------------------
# RAG / Retrieval prompts
# ---------------------------------------------------------------------------

RAG_ANSWER = PromptTemplate(
    name="rag_answer",
    system=(
        "You are a precise research assistant. "
        "Answer ONLY using the context provided below. "
        "Every factual claim must be followed by its source in square brackets, e.g. [Source: doc_id]. "
        "If the answer cannot be found in the context, respond exactly with: "
        "\"NO_DATA_FOUND: The retrieved context does not contain enough information to answer this question.\" "
        "Do NOT speculate or use prior knowledge. Temperature is fixed at 0.1."
    ),
    user_template=(
        "CONTEXT:\n"
        "{context}\n\n"
        "QUESTION:\n"
        "{question}\n\n"
        "Provide a concise, well-cited answer based strictly on the context above."
    ),
    temperature=0.1,
    max_tokens=1024,
)

RAG_SUMMARISE = PromptTemplate(
    name="rag_summarise",
    system=(
        "You are a document summarisation assistant. "
        "Summarise only what is explicitly stated in the provided text. "
        "Do not add information, interpretations, or opinions. "
        "Cite the source document for each key point."
    ),
    user_template=(
        "DOCUMENT (source: {source_id}):\n"
        "{document_text}\n\n"
        "Provide a bullet-point summary of the key facts."
    ),
    temperature=0.1,
    max_tokens=512,
)


# ---------------------------------------------------------------------------
# Research agent prompts
# ---------------------------------------------------------------------------

RESEARCH_PLAN = PromptTemplate(
    name="research_plan",
    system=(
        "You are a research planner. Given a user question, decompose it into "
        "3-5 specific sub-queries that together will fully answer the question. "
        "Output ONLY a JSON array of strings. No commentary."
    ),
    user_template=(
        "MAIN QUESTION:\n{question}\n\n"
        "Return a JSON array of sub-queries."
    ),
    temperature=0.1,
    max_tokens=256,
)

RESEARCH_SYNTHESISE = PromptTemplate(
    name="research_synthesise",
    system=(
        "You are a senior research analyst. "
        "Synthesise the provided sub-answers into one coherent final answer. "
        "Preserve all source citations from the sub-answers. "
        "If any sub-answer contains NO_DATA_FOUND, note the gap explicitly. "
        "Do not introduce any information not present in the sub-answers."
    ),
    user_template=(
        "ORIGINAL QUESTION:\n{question}\n\n"
        "SUB-ANSWERS:\n{sub_answers}\n\n"
        "Write the final synthesised answer with citations."
    ),
    temperature=0.1,
    max_tokens=1024,
)


# ---------------------------------------------------------------------------
# Analysis agent prompts
# ---------------------------------------------------------------------------

DOCUMENT_ANALYSIS = PromptTemplate(
    name="document_analysis",
    system=(
        "You are a document analysis expert. Analyse the document deeply and extract: "
        "1. Key entities (people, organisations, dates, figures). "
        "2. Main arguments or findings. "
        "3. Contradictions or ambiguities. "
        "4. Data points and statistics. "
        "Ground every extraction in a direct quote or paraphrase with [page/section ref]. "
        "Output in structured JSON."
    ),
    user_template=(
        "DOCUMENT (id={doc_id}, type={doc_type}):\n"
        "{document_text}\n\n"
        "Return structured JSON analysis."
    ),
    temperature=0.05,
    max_tokens=1024,
)

COMPARATIVE_ANALYSIS = PromptTemplate(
    name="comparative_analysis",
    system=(
        "You are a comparative analyst. Compare the provided documents on the dimensions listed. "
        "Use only information explicitly stated in the documents. "
        "Cite [doc_id] for every comparative claim. "
        "Output a JSON object with a 'comparison_table' and a 'verdict' field."
    ),
    user_template=(
        "DOCUMENTS:\n{documents}\n\n"
        "DIMENSIONS TO COMPARE:\n{dimensions}\n\n"
        "Return JSON comparison."
    ),
    temperature=0.1,
    max_tokens=1024,
)

SENTIMENT_ANALYSIS = PromptTemplate(
    name="sentiment_analysis",
    system=(
        "You are a sentiment analysis model. Classify the sentiment of the provided text "
        "as POSITIVE, NEGATIVE, or NEUTRAL with a confidence score 0.0-1.0. "
        "Extract the top 3 sentiment-driving phrases. "
        "Output JSON: {sentiment, confidence, driving_phrases}."
    ),
    user_template="TEXT:\n{text}\n\nReturn JSON sentiment analysis.",
    temperature=0.05,
    max_tokens=256,
)


# ---------------------------------------------------------------------------
# Media agent prompts
# ---------------------------------------------------------------------------

IMAGE_ANALYSIS = PromptTemplate(
    name="image_analysis",
    system=(
        "You are a vision analysis assistant. Describe only what is visually present in the image. "
        "Do not speculate about context or make assumptions. "
        "Extract: objects, text (OCR), people count (no identification), colours, layout. "
        "Output structured JSON."
    ),
    user_template=(
        "Analyse the provided image.\n"
        "USER QUESTION (if any): {question}\n"
        "Return JSON with keys: objects, ocr_text, people_count, dominant_colours, layout_description."
    ),
    temperature=0.1,
    max_tokens=512,
)

VIDEO_FRAME_ANALYSIS = PromptTemplate(
    name="video_frame_analysis",
    system=(
        "You are a video content analyser. You will receive sampled frames from a video. "
        "Describe the visual content of each frame and identify transitions, text overlays, "
        "and key events. Do not speculate beyond what is visible. Output JSON array per frame."
    ),
    user_template=(
        "VIDEO METADATA: {metadata}\n"
        "FRAME TIMESTAMPS: {timestamps}\n"
        "USER QUESTION: {question}\n"
        "Analyse the frames and return structured JSON."
    ),
    temperature=0.1,
    max_tokens=768,
)

AUDIO_TRANSCRIPTION_CLEANUP = PromptTemplate(
    name="audio_transcription_cleanup",
    system=(
        "You are a transcription editor. Clean the raw ASR transcript: "
        "fix punctuation, capitalisation, and obvious mis-recognitions. "
        "Do NOT change the meaning or add content. "
        "Return the cleaned transcript only, no commentary."
    ),
    user_template="RAW TRANSCRIPT:\n{raw_transcript}\n\nReturn cleaned transcript.",
    temperature=0.05,
    max_tokens=1024,
)

PDF_EXTRACTION_QA = PromptTemplate(
    name="pdf_extraction_qa",
    system=(
        "You are a PDF reading assistant. Answer questions using only the extracted PDF text provided. "
        "Cite page numbers where available [p.N]. "
        "If the answer is not in the text, respond: NO_DATA_FOUND."
    ),
    user_template=(
        "PDF CONTENT (pages {page_range}):\n"
        "{pdf_text}\n\n"
        "QUESTION: {question}\n\n"
        "Answer with page citations."
    ),
    temperature=0.1,
    max_tokens=1024,
)


# ---------------------------------------------------------------------------
# Hallucination guard prompts
# ---------------------------------------------------------------------------

HALLUCINATION_CHECK = PromptTemplate(
    name="hallucination_check",
    system=(
        "You are a fact-verification specialist. Given an answer and its source context, "
        "determine whether every claim in the answer is supported by the context. "
        "Output JSON: {"
        "  \"verdict\": \"SUPPORTED\" | \"PARTIAL\" | \"UNSUPPORTED\", "
        "  \"unsupported_claims\": [list of strings], "
        "  \"confidence\": float 0.0-1.0"
        "}. "
        "Be strict: any claim without clear context support is UNSUPPORTED."
    ),
    user_template=(
        "SOURCE CONTEXT:\n{context}\n\n"
        "ANSWER TO VERIFY:\n{answer}\n\n"
        "Return JSON verification result."
    ),
    temperature=0.05,
    max_tokens=512,
)

CONFIDENCE_SCORING = PromptTemplate(
    name="confidence_scoring",
    system=(
        "You are a confidence calibration model. Given a question, a retrieved context, "
        "and a generated answer, score the answer's confidence 0.0-1.0. "
        "Consider: context relevance, answer completeness, citation coverage. "
        "Output JSON: {confidence_score, reasoning, missing_information}."
    ),
    user_template=(
        "QUESTION: {question}\n\n"
        "CONTEXT SIMILARITY SCORE: {similarity_score}\n\n"
        "ANSWER: {answer}\n\n"
        "Return JSON confidence assessment."
    ),
    temperature=0.05,
    max_tokens=256,
)


# ---------------------------------------------------------------------------
# Registry for easy lookup
# ---------------------------------------------------------------------------

TEMPLATE_REGISTRY: dict[str, PromptTemplate] = {
    t.name: t for t in [
        RAG_ANSWER,
        RAG_SUMMARISE,
        RESEARCH_PLAN,
        RESEARCH_SYNTHESISE,
        DOCUMENT_ANALYSIS,
        COMPARATIVE_ANALYSIS,
        SENTIMENT_ANALYSIS,
        IMAGE_ANALYSIS,
        VIDEO_FRAME_ANALYSIS,
        AUDIO_TRANSCRIPTION_CLEANUP,
        PDF_EXTRACTION_QA,
        HALLUCINATION_CHECK,
        CONFIDENCE_SCORING,
    ]
}


def get_template(name: str) -> PromptTemplate:
    """Retrieve a prompt template by name. Raises KeyError if not found."""
    if name not in TEMPLATE_REGISTRY:
        raise KeyError(
            f"Prompt template '{name}' not found. "
            f"Available: {list(TEMPLATE_REGISTRY.keys())}"
        )
    return TEMPLATE_REGISTRY[name]


def render_prompt(name: str, **kwargs) -> tuple[str, str]:
    """
    Render a prompt template with the given keyword arguments.

    Returns:
        (system_prompt, user_prompt) tuple ready for LLM calls.
    """
    template = get_template(name)
    try:
        user_prompt = template.user_template.format(**kwargs)
    except KeyError as e:
        raise ValueError(
            f"Missing placeholder {e} for template '{name}'. "
            f"Template requires: {_extract_placeholders(template.user_template)}"
        )
    return template.system, user_prompt


def _extract_placeholders(template_str: str) -> list[str]:
    """Extract {placeholder} names from a template string."""
    import re
    return re.findall(r"\{(\w+)\}", template_str)