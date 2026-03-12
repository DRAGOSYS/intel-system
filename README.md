# Intel System — AI Powered Data Gathering and Analysis System

A locally hosted, multi-layer AI pipeline that ingests data from heterogeneous sources, processes and stores it in a vector database, and answers queries using a mix of local and cloud LLMs with strict anti-hallucination guardrails.

Built entirely on a mid-range laptop (AMD Ryzen 5, 8GB RAM, no GPU) ensuring that the model works with good speed & great accuracy.

---

## Architecture
```
Layer 1 — Data Ingestion      → Web, RSS, News, PDF, Audio, Video, Images
Layer 2 — Processing Pipeline → Chunking, Embedding, NER, Quality Filtering  
Layer 3 — Storage & Retrieval → Qdrant (vectors) + SQLite (metadata)
Layer 4 — Intelligence        → RAG Pipeline, LLM Routing, Anti-Hallucination
Layer 5 — Output              → API + Dashboard (in development)
```

---

## Key Features

- **Multi-source ingestion** — RSS feeds, news APIs, web scraping, Wayback Machine, PDF, audio, video, images
- **Smart LLM routing** — local Phi3-mini for quick queries, Groq Llama 3.3-70B for heavy analysis, Gemini 1.5 Flash for multimodal
- **Anti-hallucination RAG** — every response is source-cited, confidence-scored, and grounded strictly in retrieved context
- **Fully local** — runs without internet using Ollama + Qdrant via Docker

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Vector Database | Qdrant |
| Local LLM | Phi3-mini via Ollama |
| Cloud LLMs | Groq (Llama 3.1/3.3), Gemini 1.5 Flash |
| Embeddings | nomic-embed-text |
| Audio Processing | Faster-Whisper |
| Web Scraping | Playwright, BeautifulSoup |
| API | FastAPI |
| Dashboard | Streamlit |
| Containerization | Docker |

---

## Setup

1. Clone the repository
```bash
git clone https://github.com/DRAGOSYS/intel-system.git
cd intel-system
```

2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory
```
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
```

5. Start Qdrant and Ollama
```bash
docker start qdrant
ollama serve
```

6. Run the system
```bash
python main.py
```

---

## Project Status

| Layer | Status |
|---|---|
| Layer 1 — Data Ingestion | ✅ Complete |
| Layer 2 — Processing Pipeline | ✅ Complete |
| Layer 3 — Storage & Retrieval | ✅ Complete |
| Layer 4 — Intelligence | ✅ Complete |
| Layer 5 — Output Interface | 🔄 In Development |