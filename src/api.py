import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import json
import os
from collections import Counter
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, AIMessage
from chatbot import load_vectorstore, build_chain, ask
from drive_sync import sync_drive
from ingest import run_ingest

chain = None
chat_sessions = {}

ALLOWED_ORIGINS = json.loads(os.getenv("ALLOWED_ORIGINS", '["http://localhost:3000"]'))

# Secret that protects every /admin route. Set it on Azure (Application settings)
# and in your local .env. If it's unset, admin endpoints fail closed (401).
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")

# In-process state for the "Sync now" button. Single-worker only (see note in
# the sync endpoint).
sync_state = {
    "running": False,
    "last_run": None,
    "last_result": None,
    "error": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global chain
    print("Loading Knowledge base...")
    vectorstore = load_vectorstore()
    chain = build_chain(vectorstore)
    print("NUGuide API is ready.")
    yield
    print("Shutting Down")

app = FastAPI(
    title="NUGuide API",
    description="RAG chatbot API for Niagara University tour guides",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    session_id: str

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: str
    session_id: str = "default"

class FeedbackResponse(BaseModel):
    status: str

class StatusUpdate(BaseModel):
    timestamp: str   # the entry's timestamp acts as its id
    status: str      # unanswered: needs_content | off_topic | hidden ; feedback: hidden | active

class DeleteRequest(BaseModel):
    timestamp: str


# ── Shared helpers ──────────────────────────────────────────────────────────

def _log_path(filename):
    """Azure persists /home; locally fall back to knowledge_base/."""
    if os.path.exists("/home"):
        base = "/home"
    else:
        base = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "knowledge_base",
        )
    return os.path.join(base, filename)


def _feedback_path():
    return "/home/feedback_log.json" if os.path.exists("/home") else "feedback_log.json"


def _load_json(path):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def require_admin(x_admin_token: str = Header(None)):
    """Dependency that guards admin routes. Fails closed if ADMIN_TOKEN unset."""
    if not ADMIN_TOKEN or x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


def log_question(question, session_id):
    """Lean, append-only log of EVERY question (timestamp + text only).

    One JSON object per line (JSONL) so we never read+rewrite a growing file.
    The full answer is not stored here — it's already saved in the feedback log
    when a guide rates a reply.
    """
    path = _log_path("questions_log.jsonl")
    try:
        with open(path, "a") as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "question": question,
            }) + "\n")
    except Exception as e:
        print(f"Warning: could not log question: {e}")


# ── Public routes ───────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Health check - confirms the APIT is running."""
    return {"status": "NUGuide API is runnning"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if request.session_id not in chat_sessions:
        chat_sessions[request.session_id] = []

    # Log every question (lean, append-only) for usage count + most-asked.
    log_question(request.question, request.session_id)

    chat_history = chat_sessions[request.session_id]

    answer, sources, is_unanswered, sources_found = ask(chain, request.question, chat_history)

    if is_unanswered:
        try:
            log_unanswered(request.question, request.session_id, sources_found)
        except Exception as e:
            print(f"Warning: Could not log unanswered question: {e}")

    chat_sessions[request.session_id].append(
        HumanMessage(content=request.question)
    )
    chat_sessions[request.session_id].append(
        AIMessage(content=answer)
    )

    return ChatResponse(
        answer=answer,
        sources=sources,
        session_id=request.session_id
    )

@app.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest):
    if os.path.exists("/home"):
        log_path = "/home/feedback_log.json"
    else:
        log_path = "feedback_log.json"
    try:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": request.session_id,
            "question": request.question,
            "answer": request.answer,
            "rating": request.rating
        }

        logs = []
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                try:
                    logs = json.load(f)
                except json.JSONDecodeError:
                    logs = []

        logs.append(entry)

        with open(log_path, "w") as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not log feedback: {e}")

    return FeedbackResponse(status="recorded")

@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """
    Clears chat history for a session.
    Called when the user clicks New COnversation in the frontend
    """

    if session_id in chat_sessions:
        del chat_sessions[session_id]
    return {"status": "cleared"}

def log_unanswered(question, session_id, sources_found):
    """Logs questions the chatbot couldn't answer for knowledge base improvement."""
    log_path = _log_path("unanswered_log.json")

    entry = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "question": question,
        "sources_found": sources_found,
        "gap_type": "missing_content" if sources_found == 0 else "insufficient_detail",
    }

    logs = []
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []

    logs.append(entry)

    with open(log_path, "w") as f:
        json.dump(logs, f, indent=2)


# ── Admin routes (all guarded by require_admin) ─────────────────────────────

def _run_sync_job():
    """Background job: pull Drive -> re-ingest -> reload the live chain."""
    global chain
    try:
        sync_state["error"] = None
        drive_result = sync_drive()
        ingest_result = run_ingest()
        # Reload the in-process chain so the live bot serves the new KB without
        # a manual restart (this is what fixes the "rebuild breaks prod" problem).
        vectorstore = load_vectorstore()
        chain = build_chain(vectorstore)
        sync_state["last_result"] = {"drive": drive_result, "ingest": ingest_result}
        sync_state["last_run"] = datetime.now().isoformat()
    except Exception as e:
        sync_state["error"] = str(e)
    finally:
        sync_state["running"] = False


@app.post("/admin/sync")
def admin_sync(background_tasks: BackgroundTasks, _: None = Depends(require_admin)):
    # NOTE: sync_state and the chain reload live in THIS process. Run the backend
    # single-worker (one uvicorn/gunicorn worker) or this state won't be shared.
    if sync_state["running"]:
        return {"status": "already_running"}
    sync_state["running"] = True
    background_tasks.add_task(_run_sync_job)
    return {"status": "started"}


@app.get("/admin/sync/status")
def admin_sync_status(_: None = Depends(require_admin)):
    return sync_state


@app.get("/admin/unanswered")
def get_unanswered(_: None = Depends(require_admin)):
    logs = _load_json(_log_path("unanswered_log.json"))

    out = []
    for e in logs:
        # Manual status wins; otherwise pre-sort by the only real signal we have:
        # 0 sources found = nothing in the KB matched = almost certainly off-topic.
        status = e.get("status")
        if not status:
            status = "off_topic" if e.get("sources_found", 1) == 0 else "needs_content"
        out.append({**e, "status": status})

    return {"logs": out}


@app.post("/admin/unanswered/update")
def update_unanswered(req: StatusUpdate, _: None = Depends(require_admin)):
    """Set an entry's status: needs_content | off_topic | hidden."""
    path = _log_path("unanswered_log.json")
    logs = _load_json(path)
    for e in logs:
        if e.get("timestamp") == req.timestamp:
            e["status"] = req.status
    _save_json(path, logs)
    return {"status": "ok"}


@app.post("/admin/unanswered/delete")
def delete_unanswered(req: DeleteRequest, _: None = Depends(require_admin)):
    """Permanently remove an entry from the log."""
    path = _log_path("unanswered_log.json")
    logs = [e for e in _load_json(path) if e.get("timestamp") != req.timestamp]
    _save_json(path, logs)
    return {"status": "deleted"}


@app.get("/admin/feedback")
def get_feedback(_: None = Depends(require_admin)):
    logs = _load_json(_feedback_path())

    negative = [
        {**e, "status": e.get("status", "active")}
        for e in logs if e.get("rating") == "negative"
    ]
    positive_count = sum(1 for e in logs if e.get("rating") == "positive")

    return {
        "negative": negative,  # the thumbs-down replies Sara reviews
        "counts": {
            "positive": positive_count,
            "negative": len(negative),
            "total": len(logs),
        },
    }


@app.post("/admin/feedback/update")
def update_feedback(req: StatusUpdate, _: None = Depends(require_admin)):
    """Hide/unhide a thumbs-down entry: hidden | active."""
    path = _feedback_path()
    logs = _load_json(path)
    for e in logs:
        if e.get("timestamp") == req.timestamp:
            e["status"] = req.status
    _save_json(path, logs)
    return {"status": "ok"}


@app.post("/admin/feedback/delete")
def delete_feedback(req: DeleteRequest, _: None = Depends(require_admin)):
    """Permanently remove a feedback entry from the log."""
    path = _feedback_path()
    logs = [e for e in _load_json(path) if e.get("timestamp") != req.timestamp]
    _save_json(path, logs)
    return {"status": "deleted"}


@app.get("/admin/stats")
def get_stats(_: None = Depends(require_admin)):
    """Usage count (total + last 7 days) and most-asked questions."""
    path = _log_path("questions_log.jsonl")

    questions = []
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    questions.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    cutoff = datetime.now() - timedelta(days=7)
    last_7_days = 0
    counter = Counter()

    for q in questions:
        text = (q.get("question") or "").strip()
        if text:
            counter[text.lower()] += 1
        ts = q.get("timestamp")
        if ts:
            try:
                if datetime.fromisoformat(ts) >= cutoff:
                    last_7_days += 1
            except ValueError:
                pass

    top_questions = [
        {"question": text, "count": count}
        for text, count in counter.most_common(10)
    ]

    return {
        "total": len(questions),
        "last_7_days": last_7_days,
        "top_questions": top_questions,
    }