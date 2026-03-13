"""
rag/bot.py — RAG pipeline for Cloud AI Assistant.

Adapted from Bot.py. Stateless — safe to call from FastAPI threads.
Uses ChatOllama (llama3.2) + ChromaDB + HuggingFace embeddings.

Usage:
    from rag.bot import ask, set_db_path

    set_db_path("/path/to/chroma_db", "/path/to/interaction_guidelines.txt")
    answer = ask("Why are cloud boundaries coarse?", app_context={...})
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Defaults — override via environment variables or set_db_path() ────────────
_DB_PATH = "/app/rag/Chroma dir"
_GUIDELINES_PATH = "/app/rag/interaction_guidelines.txt"

EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME  = "cloud_prediction_kb"
OLLAMA_MODEL     = os.environ.get("OLLAMA_MODEL", "llama3.2")

# ── Lazy-loaded singletons ─────────────────────────────────────────────────────
_db             = None
_llm            = None
_system_prompt  = None


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

def set_db_path(db_path: str, guidelines_path: str):
    """Call once at startup to override default paths."""
    global _DB_PATH, _GUIDELINES_PATH, _db, _llm, _system_prompt
    _DB_PATH         = db_path
    _GUIDELINES_PATH = guidelines_path
    # Force reload on next ask()
    _db = _llm = _system_prompt = None


# ══════════════════════════════════════════════════════════════════════════════
# LAZY INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════

def _get_db():
    global _db
    if _db is None:
        from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
        from langchain_chroma import Chroma

        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        _db = Chroma(
            persist_directory=_DB_PATH,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
        logger.info(f"[RAG] ChromaDB loaded — {_db._collection.count()} chunks")
    return _db


def _get_llm():
    global _llm
    if _llm is None:
        from langchain_ollama import ChatOllama
        _llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0.3,
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434"),
        )
        logger.info(f"[RAG] LLM ready: {OLLAMA_MODEL}")
    return _llm


def _get_system_prompt() -> str:
    global _system_prompt
    if _system_prompt is None:
        p = Path(_GUIDELINES_PATH)
        if p.exists():
            _system_prompt = p.read_text(encoding="utf-8")
        else:
            logger.warning(f"[RAG] Guidelines not found at {_GUIDELINES_PATH} — using default")
            _system_prompt = (
                "You are a cloud motion prediction AI assistant. "
                "Answer questions about the CREvNet model, INSAT-3DR satellite data, "
                "and cloud prediction metrics. Be concise and technical."
            )
    return _system_prompt


# ══════════════════════════════════════════════════════════════════════════════
# INTENT DETECTION  (from Bot.py verbatim)
# ══════════════════════════════════════════════════════════════════════════════

INTENT_MAP = {
    "Metrices":     ["csi", "metric", "score", "accuracy", "precision", "f1"],
    "Failure":      ["blurry", "coarse", "problem", "bad", "fail", "wrong", "miss"],
    "Improvement":  ["improve", "better", "fix", "enhance", "upgrade", "optimise", "optimize"],
    "Architecture": ["model", "architecture", "attention", "encoder", "decoder", "crevnet", "layer"],
    "Overview":     [],
}

def detect_intent(question: str) -> str:
    q = question.lower()
    for category, keywords in INTENT_MAP.items():
        if any(k in q for k in keywords):
            return category
    return "Overview"


# ══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_docs(question: str, k: int = 4) -> tuple[list, str]:
    intent    = detect_intent(question)
    db        = _get_db()
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k, "filter": {"category": intent}},
    )
    docs = retriever.invoke(question)
    logger.info(f"[RAG] Intent={intent}  chunks={len(docs)}")
    return docs, intent


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(question: str, docs: list, app_context: dict) -> str:
    retrieved = "\n\n---\n\n".join(d.page_content for d in docs) if docs else "(no relevant chunks found)"

    ctx_lines = "\n".join(f"  {k:15}: {v}" for k, v in app_context.items())

    return f"""{_get_system_prompt()}

CURRENT APP STATE:
{ctx_lines}

RETRIEVED KNOWLEDGE:
{retrieved}

USER QUESTION:
{question}
"""


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def ask(question: str, app_context: Optional[dict] = None) -> dict:
    """
    Full RAG pipeline: question → intent → retrieve → prompt → LLM → answer.

    Returns:
        {
            "answer":  str,
            "intent":  str,
            "chunks":  int,
        }
    """
    if app_context is None:
        app_context = {}

    try:
        docs, intent = retrieve_docs(question)
        prompt       = build_prompt(question, docs, app_context)
        llm          = _get_llm()
        response     = llm.invoke(prompt)
        return {
            "answer": response.content,
            "intent": intent,
            "chunks": len(docs),
        }
    except Exception as e:
        logger.error(f"[RAG] ask() failed: {e}")
        raise
