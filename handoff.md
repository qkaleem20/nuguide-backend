Here's the full updated handoff document. Copy this entire thing into a Google Doc and save it.

---

# NUGuide — Complete Project Handoff (Updated May 2026)

## Project Overview

NUGuide is a production-grade AI-powered RAG chatbot built for Niagara University's admissions tour guides. It helps guides quickly look up accurate university information during campus tours — academic programs, scholarships, campus facts, tour route, policies — without searching through handbooks. Secondary purpose: help guides prepare answers for prospective student questions.

Live URL: www.nuguide.info
Backend API: https://nuchatbot.azurewebsites.net
Admin endpoint: https://nuchatbot.azurewebsites.net/admin/unanswered
GitHub Backend: https://github.com/qkaleem20/nuguide-backend
GitHub Frontend: https://github.com/qkaleem20/nuguide-frontend

## Tech Stack

Backend: Python 3.12, FastAPI, LangChain (LCEL), Chroma Cloud, BM25, OpenAI (GPT-5.4-mini, text-embedding-3-small)
Frontend: Next.js 16 (App Router, static export), React, Tailwind CSS v4
Fonts: Playfair Display (headings), Inter (body)
Vector Database: Chroma Cloud (hosted, collection: nuguide_kb, database: NUGuide)
Deployment: Microsoft Azure (App Service for backend, Static Web Apps for frontend)
CI/CD: GitHub Actions (auto-deploy on push to main for both repos)
Custom Domain: www.nuguide.info via Squarespace DNS with CNAME to Azure
Local Python: 3.12 via Homebrew (not system Python 3.14 which has compatibility issues)

## Backend Structure (nu_chatbot/)

```
src/
  api.py              — FastAPI server, /chat, /feedback, /session, /admin/unanswered endpoints
  chatbot.py           — hybrid retrieval pipeline, RAG chain, ask() with unanswered detection
  ingest.py            — document loading, chunking, embedding, Chroma Cloud upload in batches of 200
  manage_kb.py         — knowledge base management utilities
  test.py              — test script
knowledge_base/
  documents/           — all source documents (PDFs, DOCX, TXT)
  chunks.pkl           — local BM25 cache (gitignored)
  unanswered_log.json  — local unanswered log (gitignored)
.env                   — API keys (gitignored)
.env.example           — template showing required variables
.gitignore
Procfile               — Azure startup command
README.md
requirements.txt
.github/workflows/main_nuchatbot.yml — Azure deployment workflow
```

## Frontend Structure (nuguide-frontend/)

```
app/
  page.js            — main page, state management, welcome screen + chat view
  layout.js          — fonts, metadata
  globals.css        — design tokens, animations, mobile fixes
components/
  Sidebar.js         — collapsible panel, quick prompts, contacts, full-width on mobile
  ChatWindow.js      — message list, auto-scroll, background glow
  Message.js         — user/assistant bubbles, sources toggle, feedback buttons
  ChatInput.js       — auto-resize textarea, no auto-focus on mobile
  TypingIndicator.js — animated 3-dot bounce
  SourceTag.js       — clickable blue pills for web sources, static purple pills for docs
  StatusIndicator.js — green/red dot, hides label on mobile
lib/
  api.js             — checkHealth(), sendMessage(), submitFeedback(), clearSession()
  constants.js       — SESSION_ID, QUICK_PROMPTS, CATEGORY_CARDS, STARTER_QUESTIONS
  useWindowSize.js   — useWindowSize() and useIsMobile() hooks
next.config.mjs      — output: 'export', images: { unoptimized: true }
.env.local           — NEXT_PUBLIC_API_URL (gitignored)
.env.example
.github/workflows/azure-static-web-apps-agreeable-river-0a705300f.yml
```

## Knowledge Base Documents (current)

student_room_handbook.docx — office operations, phone scripts (source_type: handbook)
spring_2026_training.pdf — training slides, residence life FAQs (source_type: handbook)
academic_cheat_sheet.pdf — program-specific facts, placement rates, 45 pages (source_type: handbook, but PRIORITY 0 in ranking)
nu_clubs.pdf — full clubs list (source_type: nu_clubs)
campus_tour_guide_handbook.docx — primary source, full tour route, buildings, policies, 71K chars (source_type: handbook)
tour_route.txt — curated 30-stop route (source_type: tour_route)
general_facts.txt — key stats, parking prices, contacts, campus buildings info (source_type: key_facts)
useful_links.txt — official NU URLs (source_type: useful_links)
college_bridges.txt — quick reference summaries per college for tour guides (source_type: college_bridges)
tour_guide_roles.txt — guide operations info (source_type: operations)
arts_and_sciences.docx — COAS majors, minors, and program details (source_type: academics)
college_of_business.docx — HCBA majors, minors, and program details (source_type: academics)
college_of_education.docx — education majors, minors, and program details (source_type: academics)
college_of_hospitality.docx — COHST majors, minors, and program details (source_type: academics)
college_of_nursing.docx — nursing majors and program details (source_type: academics)
majors_and_minors.docx — complete list of all majors and minors by college (source_type: academics)
pre_professional_programs.docx — pre-law, pre-health, etc. (source_type: academics)

College-specific docx files use this format:
```
COLLEGE NAME — UNDERGRADUATE MAJORS AND PROGRAMS
[List of majors]
[List of minors]
PROGRAM NAME: [name]
TYPE: Major/Minor (Degree type)
COLLEGE: [college]
OVERVIEW: [description]
CAREERS: [career paths]
NOTES: [additional info]
```

## Website Sources (21 pages)

https://www.niagara.edu/tuition-aid/scholarships-grants/
https://www.niagara.edu/tuition-aid/financial-aid/
https://www.niagara.edu/academics/
https://www.niagara.edu/programs/
https://www.niagara.edu/colleges/college-of-arts-and-sciences/
https://www.niagara.edu/colleges/college-of-education/
https://www.niagara.edu/colleges/college-of-hospitality-sport-and-tourism-management/
https://www.niagara.edu/colleges/college-of-nursing/
https://www.niagara.edu/colleges/holzschuh-college-of-business-administration/
https://www.niagara.edu/academics/academic-and-career-exploration-program/
https://www.niagara.edu/academics/honors-program/
https://www.niagara.edu/admissions/international-admissions/
https://www.niagara.edu/admissions/canadian-students/
https://www.niagara.edu/current-students/student-life/clubs-organizations/
https://www.niagara.edu/about/nu-history/
https://www.niagara.edu/future-students/admitted-students/
https://www.niagara.edu/athletics/
https://www.niagara.edu/current-students/student-life/kiernan-recreation-center/
https://catalog.niagara.edu/undergraduate/student-affairs/residence-life/
https://catalog.niagara.edu/undergraduate/curriculum/foundation-courses/
https://www.niagara.edu/visit/tour-guides/
Plus 7 education-specific program page URLs

## RAG Pipeline

1. User sends question through frontend
2. If chat history exists, question is reformulated into standalone query using the LLM
3. Semantic search on Chroma Cloud — top 4 chunks by embedding similarity (OpenAI text-embedding-3-small)
4. BM25 keyword search from local chunks.pkl — top 8 chunks
5. Merge, deduplicate
6. Source-priority ranking using source_type metadata:
   - Priority 0: academic_cheat_sheet.pdf
   - Priority 1: source_type in [academics, key_facts, tour_route, useful_links, college_bridges, nu_clubs, operations]
   - Priority 2: source_type == handbook
   - Priority 3: everything else (websites)
7. Top 12 chunks passed as context to GPT-5.4-mini at temperature=0
8. Answer generated strictly from context, returned with source list

## Chunking Strategy

PDFs/DOCX: chunk_size=1500, overlap=150
Web pages: chunk_size=800, overlap=80
TXT files: chunk_size=512, overlap=50
Chunks under 50 characters filtered out
Current total: 831 chunks

## System Prompt Rules

- Answer ONLY from provided context
- Never use markdown formatting (no bold **, no italics *, no headers #)
- Fallback: "I don't have that in my materials. Please reach out to Sara, the Co-Directors, or call the admissions office at 716-286-8700."
- Graduate programs: redirect to 716-286-7360
- NEVER answer tuition cost questions — redirect to niagara.edu/tuition-aid
- Cheat sheet is highest priority for academic questions
- Proactively offer more info after academic/facility answers (one sentence)
- Ask ONE clarifying question for vague program questions
- No follow-up questions for factual lookups

## API Endpoints

GET / — health check
POST /chat — {question, session_id} → {answer, sources, session_id}
POST /feedback — {question, answer, rating, session_id} → {status}
DELETE /session/{id} — clears chat history
GET /admin/unanswered — returns all unanswered questions with gap classifications

## Unanswered Question Tracking

Detects fallback phrases in LLM response
Logs: timestamp, session_id, question, sources_found, gap_type
gap_type "missing_content" = sources_found is 0
gap_type "insufficient_detail" = sources_found > 0
Writes to /home/unanswered_log.json on Azure, knowledge_base/unanswered_log.json locally
Wrapped in try/except so logging failure never crashes the response

## Feedback System

Thumbs up/down on every response, prevents double-click
Writes to /home/feedback_log.json on Azure, feedback_log.json locally
Also wrapped in try/except

## Frontend Design

Dark purple-black theme (#080511), primary #7C3AED, gold accent #D4A017
Playfair Display headings, Inter body
Welcome screen: hero, category cards (2-col mobile, 4-col desktop), starter questions
Chat view: message bubbles, typing indicator, auto-scroll
Source tags: blue clickable pills for web (open new tab with ↗), purple static for docs
NUGuide logo in navbar is clickable — starts new conversation
Status indicator hides label on mobile
100dvh viewport height for mobile browser bar
Safe area padding for iPhone home bar
No auto-focus on mobile (prevents keyboard popup)
Sidebar full-width on mobile with larger touch targets

## Chroma Cloud Setup

Collection: nuguide_kb
Database: NUGuide
Connected via chromadb.CloudClient with API key, tenant, database from env vars
Ingestion deletes existing collection before re-uploading (prevents duplicates)
Uploads in batches of 200 (Chroma Cloud limits 300 per request)
load_chunks() in chatbot.py uses local chunks.pkl if available, fetches from Chroma Cloud if not (for Azure)

## Azure Deployment

Backend: Azure App Service, Linux, Python 3.12, B1 tier
Startup command: cd src && gunicorn api:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
pysqlite3-binary patch at top of api.py (conditional — only runs on Azure)
Frontend: Azure Static Web Apps, Free tier, static export
Next.js builds with output: 'export', output to out/ folder
NEXT_PUBLIC_API_URL baked in at build time via GitHub Actions env variable
CORS via ALLOWED_ORIGINS env var on backend

## Environment Variables — Backend (Azure)

OPENAI_API_KEY, CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE, ALLOWED_ORIGINS, SCM_DO_BUILD_DURING_DEPLOYMENT

## Environment Variables — Frontend

NEXT_PUBLIC_API_URL (set in GitHub Actions workflow, baked at build time)

## Local Development

```bash
cd /Users/qutaibakaleem/Desktop/nu_chatbot
source venv/bin/activate  (Python 3.12 via Homebrew)
cd src
uvicorn api:app --reload --port 8000
```

```bash
cd /Users/qutaibakaleem/Desktop/nuguide-frontend
npm run dev
```

## Confirmed Data Facts

Resident parking: $105, Commuter: $95
First Day Complete: $265/semester
ACEP in ACAD building (not Timon Hall)
19 D1 teams, 16 club sports, 100+ clubs, ~4000 students, 10:1 ratio

## Key Contacts

Sara Villnave: ext. 8714
Alexis Kadlecik: ext. 8713
Admissions: 716-286-8700
Graduate Studies: 716-286-7360
Campus Activities: 716-286-8510

## Known Issues Resolved

Python 3.14 incompatibility — rebuilt venv with Python 3.12 via Homebrew
httptools crash — used --http h11 flag (only on 3.14)
Missing packages in requirements.txt — added langchain-chroma, langchain-text-splitters, pdfplumber, unstructured, fastapi, uvicorn, gunicorn, pysqlite3-binary, python-docx
SQLite too old on Azure — conditional pysqlite3 patch
Chroma Cloud 300 records per request — batch uploads at 200
Log paths not writable on Azure — write to /home/ on Azure
Logging crashes killing responses — wrapped in try/except
SSR hydration mismatch — useWindowSize uses static default (1024)
Markdown in responses — added "no markdown" rule to system prompt
Collection soft deleted error — app restart after re-ingestion fixes it
majors_and_minors.docx retrieval issue — each college file now has its own majors list at the top

## What's Next (planned)

1. Input document updates — continuing to improve and expand knowledge base content
2. Google Drive integration — auto-sync documents when admissions staff update them
3. Admin dashboard — web interface for supervisor to manage documents, view gaps, trigger re-ingestion
4. Additional improvements TBD

## Resume Entry

```
AI-Powered Tour Guide Chatbot (NUGuide) — www.nuguide.info
Python · FastAPI · LangChain · Chroma Cloud · OpenAI · Next.js · React · Microsoft Azure

• Built production-grade RAG chatbot for NU admissions tour guides integrating
  15+ documents and 21 web pages into a unified hybrid knowledge base with 831 indexed chunks
• Engineered hybrid retrieval combining semantic vector search (Chroma Cloud,
  OpenAI embeddings) with BM25 keyword search and source-priority ranking
• Designed multi-layer document ingestion with differentiated chunking strategies
  by source type and curated bridging documents to resolve retrieval gaps
• Implemented hallucination prevention through temperature-0 prompt engineering,
  source citations, and automatic knowledge gap tracking classifying unanswered
  questions as missing content vs insufficient detail
• Built responsive Next.js frontend with dark-themed NU-branded design,
  clickable source citations, feedback collection, and mobile-optimized layout
• Deployed on Microsoft Azure (App Service + Static Web Apps) with Chroma Cloud,
  GitHub Actions CI/CD, and custom domain
```

---

Save that. Start a new chat, paste it at the top, and tell me what you want to work on.