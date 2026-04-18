import json 
import os
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, AIMessage

from chatbot import load_vectorstore, build_chain, ask 

chain = None
chat_sessions = {}

ALLOWED_ORIGINS = json.loads(os.getenv("ALLOWED_ORIGINS", '["http://localhost:3000"]'))

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

@app.get("/")
def root():
    """Health check - confirms the APIT is running."""
    return {"status": "NUGuide API is runnning"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if request.session_id not in chat_sessions:
        chat_sessions[request.session_id] = []

    chat_history = chat_sessions[request.session_id]

    answer, sources, is_unanswered, sources_found = ask(chain, request.question, chat_history)

    if is_unanswered:
        log_unanswered(request.question, request.session_id, sources_found)

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
    """
    Feedback endpoint.
    Logs thumbs up/down to feedback_log.json for evaluate purposes.
    """
    log_path = "feedback_log.json"
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
    """Logs questions the chatbot couldn't answer for Knowlegde Base improvement."""
    log_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "knowledge_base", "unanswered_log.json"

    )

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