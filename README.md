# NUGuide — AI-Powered Tour Guide Assistant

A production-grade RAG (Retrieval-Augmented Generation) chatbot built for Niagara University's admissions tour guides. NUGuide helps guides quickly look up accurate university information during campus tours — from academic programs and scholarships to parking costs and the tour route — without searching through handbooks or training materials.

Architechture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend                             │
│              Next.js · React · Tailwind CSS                 │
│         github.com/[username]/nuguide-frontend              │
└──────────────────────────┬──────────────────────────────────┘
                           │ REST API
┌──────────────────────────▼──────────────────────────────────┐
│                     FastAPI Backend                          │
│                                                             │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────┐ │
│  │   Question   │──▶│  Hybrid      │──▶│  LLM (GPT-5.4   │ │
│  │   + History  │   │  Retrieval   │   │  mini)           │ │
│  └─────────────┘   └──────┬───────┘   └──────────────────┘ │
│                     ┌─────┴──────┐                          │
│               ┌─────▼───┐  ┌─────▼────┐                    │
│               │ Semantic │  │  BM25    │                    │
│               │ Search   │  │  Keyword │                    │
│               │(ChromaDB)│  │  Search  │                    │
│               └──────────┘  └──────────┘                    │
└─────────────────────────────────────────────────────────────┘
         │                          │
    Chroma Cloud               Local chunks
   (hosted vectors)            (BM25 index)
```

## Key Features

- **Hybrid Retrieval** — Combines semantic vector search (ChromaDB + OpenAI embeddings) with BM25 keyword search, then merges, deduplicates, and ranks results using a custom source-priority function
- **Source-Priority Ranking** — Cheat sheets and curated documents are ranked above handbooks and web sources, ensuring the most reliable information surfaces first
- **Multi-Source Knowledge Base** — Ingests 4 internal documents, 5 curated reference files, and 14 live university web pages with differentiated chunking strategies per source type
- **Hallucination Prevention** — Strict prompt engineering ensures the chatbot only answers from retrieved context, with explicit fallback responses and source citations on every answer
- **Cloud Vector Database** — Embeddings stored on Chroma Cloud for instant startup and decoupled architecture
- **Session-Based Chat History** — Maintains per-session conversation history with history-aware question reformulation for natural follow-up questions
- **Feedback Logging** — Thumbs up/down feedback stored per response for continuous quality evaluation
- **Knowledge Gap Tracking** — Automatically logs unanswered questions with gap-type classification (`missing_content` vs `insufficient_detail`) to guide knowledge base improvements

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API Framework | FastAPI |
| LLM | OpenAI GPT-5.4-mini (temperature=0) |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Database | Chroma Cloud |
| Orchestration | LangChain (LCEL) |
| Keyword Search | BM25Retriever |
| Document Parsing | PDFPlumber, Unstructured, WebBaseLoader |

## Project Structure

```
src/
  api.py              — FastAPI server with /chat, /feedback, /session endpoints
  chatbot.py           — Hybrid retrieval pipeline, RAG chain, ask() function
  ingest.py            — Document loading, chunking, embedding, Chroma Cloud upload
  test_retrieval.py    — Retrieval sanity check script
knowledge_base/
  documents/           — Source documents (PDFs, DOCX, TXT)
.env.example           — Required environment variables (no secrets)
requirements.txt       — Python dependencies
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/chat` | Send a question, receive an answer with sources |
| `POST` | `/feedback` | Submit thumbs up/down rating for a response |
| `DELETE` | `/session/{id}` | Clear chat history for a session |

### Example Request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How much does parking cost?", "session_id": "demo"}'
```

### Example Response

```json
{
  "answer": "Resident parking pass: $105. Commuter parking pass: $95.",
  "sources": ["general_facts.txt", "campus_tour_guide_handbook.docx"],
  "session_id": "demo"
}
```

## Setup

### Prerequisites

- Python 3.11+
- OpenAI API key
- Chroma Cloud account (free tier)

### Installation

```bash
git clone https://github.com/[username]/nuguide-backend.git
cd nuguide-backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Copy the example environment file and add your keys:

```bash
cp .env.example .env
```

Then edit `.env` with your actual API keys.

### Ingest Knowledge Base

```bash
cd src
python ingest.py
```

This loads all documents, chunks them with source-specific strategies, generates embeddings, and uploads to Chroma Cloud.

### Run the Server

```bash
cd src
uvicorn api:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

## Retrieval Pipeline

1. **Question Reformulation** — Follow-up questions are reformulated into standalone queries using chat history
2. **Semantic Search** — ChromaDB returns top 4 similar chunks by embedding distance
3. **BM25 Keyword Search** — Returns top 8 chunks by keyword relevance
4. **Merge & Deduplicate** — Results from both retrievers are combined, removing duplicate content
5. **Source-Priority Ranking** — Chunks are sorted by source reliability (cheat sheet → curated files → handbooks → websites)
6. **Top-12 Selection** — Best 12 chunks are passed as context to the LLM
7. **Answer Generation** — GPT-5.4-mini generates a response strictly from the provided context

## Frontend

The companion frontend repository is available at [github.com/[username]/nuguide-frontend](https://github.com/[username]/nuguide-frontend).

## License

This project was built as part of coursework and professional development at Niagara University. All university-specific content and data belongs to Niagara University.