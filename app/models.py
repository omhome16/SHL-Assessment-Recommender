"""
Pydantic models for the SHL Assessment Recommender API.
Enforces the exact schema required by the automated evaluator.
"""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ── Request Models ─────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    """A single message in the conversation history."""
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    """
    Incoming chat request. The API is stateless: every call carries
    the full conversation history.
    """
    messages: list[ChatMessage] = Field(..., min_length=1)


# ── Response Models ────────────────────────────────────────────────────────

class Recommendation(BaseModel):
    """A single assessment recommendation from the SHL catalog."""
    name: str = Field(..., description="Exact assessment name from the SHL catalog")
    url: str = Field(..., description="Exact catalog URL for this assessment")
    test_type: str = Field(..., description="Assessment type code (e.g. K, P, A, B, S, C, D)")


class ChatResponse(BaseModel):
    """
    The agent's reply. Schema is non-negotiable per the evaluator spec.

    - recommendations is None (null) when the agent is still gathering context
      or refusing an off-topic request.
    - recommendations is a list of 1-10 items when the agent has committed.
    - end_of_conversation is True only when the agent considers the task done.
    """
    reply: str = Field(..., min_length=1)
    recommendations: Optional[list[Recommendation]] = None
    end_of_conversation: bool = False

    @field_validator("recommendations")
    @classmethod
    def validate_recommendation_count(
        cls, v: Optional[list[Recommendation]]
    ) -> Optional[list[Recommendation]]:
        if v is not None and len(v) == 0:
            # Empty list should be None per spec
            return None
        if v is not None and len(v) > 10:
            # Hard cap at 10
            v = v[:10]
        return v
