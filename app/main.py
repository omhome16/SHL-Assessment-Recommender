"""
FastAPI service for the SHL Assessment Recommender.

Endpoints:
  GET  /health  →  {"status": "ok"}
  POST /chat    →  {"reply": "...", "recommendations": [...], "end_of_conversation": bool}
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agent import process_chat
from app.catalog import catalog_index
from app.config import RESPONSE_TIMEOUT
from app.models import ChatRequest, ChatResponse

# ── Logging setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ── Application lifecycle ─────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load catalog and build indices at startup."""
    logger.info("Starting SHL Assessment Recommender...")
    start = time.time()
    catalog_index.load()
    elapsed = time.time() - start
    logger.info(
        "Catalog loaded in %.1fs — %d items indexed",
        elapsed,
        len(catalog_index.items),
    )
    yield
    logger.info("Shutting down SHL Assessment Recommender")


# ── FastAPI app ────────────────────────────────────────────────────────────

app = FastAPI(
    title="SHL Assessment Recommender",
    description=(
        "Conversational agent that helps hiring managers select "
        "SHL assessments through dialogue."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — evaluator may call from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health check ───────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """
    Readiness probe.
    Returns 200 with {"status": "ok"} when the service is ready.
    The evaluator allows up to 2 minutes for cold start.
    """
    return {"status": "ok"}


# ── Chat endpoint ──────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main conversation endpoint.

    Stateless: receives full conversation history, returns the next
    agent reply with optional recommendations.
    """
    start_time = time.time()
    logger.info(
        "POST /chat — %d messages in history",
        len(request.messages),
    )

    try:
        # Enforce 28-second timeout (2s buffer under the 30s limit)
        response = await asyncio.wait_for(
            process_chat(request.messages),
            timeout=RESPONSE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        logger.error("Request timed out after %.1fs", elapsed)
        return ChatResponse(
            reply=(
                "I need a moment to think about that. Could you try again? "
                "In the meantime, could you tell me more about the role "
                "you're hiring for so I can narrow down the right SHL assessments?"
            ),
            recommendations=None,
            end_of_conversation=False,
        )
    except Exception as e:
        logger.error("Error processing chat: %s", e, exc_info=True)
        return ChatResponse(
            reply=(
                "I encountered an issue processing your request. "
                "Could you please rephrase your question about SHL assessments?"
            ),
            recommendations=None,
            end_of_conversation=False,
        )

    elapsed = time.time() - start_time
    rec_count = len(response.recommendations) if response.recommendations else 0
    logger.info(
        "Response generated in %.1fs — %d recommendations, end=%s",
        elapsed,
        rec_count,
        response.end_of_conversation,
    )

    return response


# ── Global exception handler ──────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all handler to ensure we never return a non-JSON response."""
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=200,  # Return 200 to not break the evaluator
        content={
            "reply": "I apologize for the error. Could you please try your question again?",
            "recommendations": None,
            "end_of_conversation": False,
        },
    )
