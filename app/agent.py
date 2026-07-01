"""
Agent core: orchestrates conversation flow, retrieval, and LLM calls.

Architecture:
  1. Extract search context from conversation history
  2. Retrieve relevant catalog items via hybrid search
  3. Build grounded prompt with catalog context
  4. Single LLM call (Groq) → structured JSON response
  5. Validate all recommendations against catalog
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from groq import Groq

from app.catalog import CatalogIndex, CatalogItem, catalog_index
from app.config import GROQ_API_KEY, GROQ_MODEL, TOP_K_RETRIEVAL
from app.models import ChatMessage, ChatResponse, Recommendation
from app.prompts import build_system_prompt

logger = logging.getLogger(__name__)


# ── Groq client (lazy init) ───────────────────────────────────────────────
_groq_client: Optional[Groq] = None


def _get_groq_client() -> Groq:
    """Get or create the Groq client."""
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set — cannot make LLM calls")
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def _extract_search_context(messages: list[ChatMessage]) -> str:
    """
    Build a combined search query from conversation context.

    Rather than a separate LLM call for query extraction (which adds latency),
    we concatenate all user messages with keyword boosting. This is simpler,
    faster, and works well with the hybrid retrieval approach.
    """
    parts = []
    for msg in messages:
        if msg.role == "user":
            parts.append(msg.content)

    combined = " ".join(parts)
    return combined


def _extract_mentioned_assessments(messages: list[ChatMessage]) -> list[str]:
    """
    Extract assessment names mentioned in the conversation for
    refinement and comparison scenarios.
    """
    mentioned = []
    for msg in messages:
        content = msg.content
        # Look for patterns like "OPQ32r", "Verify G+", "DSI", etc.
        for pattern in [
            r"OPQ\w*",
            r"DSI",
            r"GSA",
            r"SVAR",
            r"Verify\s+\w+",
            r"G\+",
        ]:
            matches = re.findall(pattern, content, re.IGNORECASE)
            mentioned.extend(matches)
    return mentioned


def _build_catalog_context(
    catalog: CatalogIndex,
    messages: list[ChatMessage],
) -> str:
    """
    Retrieve and format catalog items relevant to the conversation.

    Uses hybrid search to find the most relevant assessments,
    then formats them for injection into the system prompt.
    """
    query = _extract_search_context(messages)

    if not query.strip():
        items = catalog.get_all_items()[:20]
        return "\n\n".join(item.to_context_string() for item in items)

    # Hybrid search
    results = catalog.search(query, top_k=TOP_K_RETRIEVAL)

    if not results:
        items = catalog.get_all_items()[:20]
        return "\n\n".join(item.to_context_string() for item in items)

    # Also do targeted searches for any specific keywords
    additional_items: list[CatalogItem] = []
    mentioned = _extract_mentioned_assessments(messages)
    for name_fragment in mentioned:
        item = catalog.get_item_by_name(name_fragment)
        if item and item not in [r[0] for r in results]:
            additional_items.append(item)

    # Format results
    context_parts = []
    seen_ids = set()

    for item, score in results:
        if item.entity_id not in seen_ids:
            context_parts.append(item.to_context_string())
            seen_ids.add(item.entity_id)

    for item in additional_items:
        if item.entity_id not in seen_ids:
            context_parts.append(item.to_context_string())
            seen_ids.add(item.entity_id)

    return "\n\n".join(context_parts)


def _parse_llm_response(raw_text: str) -> dict:
    """
    Parse the LLM's JSON response, handling common formatting issues.

    The LLM sometimes wraps JSON in markdown code fences or adds
    extra text. This function robustly extracts the JSON object.
    """
    text = raw_text.strip()

    # Remove markdown code fences if present
    if text.startswith("```"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]
    elif not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse LLM response as JSON: %s", e)
        logger.error("Raw text: %s", raw_text[:500])
        return {
            "reply": "I apologize for the technical difficulty. Could you please rephrase your question about SHL assessments?",
            "recommendations": None,
            "end_of_conversation": False,
        }


def _validate_and_fix_recommendations(
    recommendations: Optional[list[dict]],
    catalog: CatalogIndex,
) -> Optional[list[Recommendation]]:
    """
    Validate every recommendation against the catalog.

    - If a recommendation matches a catalog item, use the catalog's exact name and URL.
    - If it doesn't match, drop it (never return hallucinated items).
    - If all recommendations are invalid, return None.
    """
    if recommendations is None:
        return None

    if not isinstance(recommendations, list):
        return None

    validated = []
    for rec in recommendations:
        if not isinstance(rec, dict):
            continue

        name = rec.get("name", "")
        url = rec.get("url", "")

        # Validate against catalog
        item = catalog.validate_recommendation(name, url)

        if item:
            validated.append(Recommendation(
                name=item.name,
                url=item.link,
                test_type=item.test_type_code,
            ))
        else:
            logger.warning(
                "Dropping hallucinated recommendation: name='%s' url='%s'",
                name, url,
            )

    # Deduplicate by URL
    seen_urls = set()
    deduped = []
    for rec in validated:
        if rec.url not in seen_urls:
            seen_urls.add(rec.url)
            deduped.append(rec)

    if not deduped:
        return None

    return deduped[:10]


def append_recommendations_table(
    reply: str,
    recommendations: Optional[list[Recommendation]],
    catalog: CatalogIndex,
) -> str:
    """
    Append a markdown formatted table of recommendations to the reply text.
    Matches the format expected in the sample conversation traces.
    """
    if not recommendations:
        return reply

    table_lines = [
        "",
        "| # | Name | Test Type | Keys | Duration | Languages | URL |",
        "|---|------|-----------|------|----------|-----------|-----|",
    ]

    for idx, rec in enumerate(recommendations, 1):
        item = catalog.get_item_by_url(rec.url) or catalog.get_item_by_name(rec.name)
        if item:
            name = item.name
            test_type = item.test_type_code
            keys = ", ".join(item.keys) if item.keys else "—"
            duration = item.duration if item.duration else "—"

            if not item.languages:
                languages = "—"
            elif len(item.languages) <= 4:
                languages = ", ".join(item.languages)
            else:
                languages = (
                    ", ".join(item.languages[:4])
                    + f" _(+{len(item.languages) - 4} more)_"
                )

            url = f"<{item.link}>"
        else:
            name = rec.name
            test_type = rec.test_type
            keys = "—"
            duration = "—"
            languages = "—"
            url = f"<{rec.url}>"

        table_lines.append(
            f"| {idx} | {name} | {test_type} | {keys} | {duration} | {languages} | {url} |"
        )

    return reply.strip() + "\n\n" + "\n".join(table_lines)


async def process_chat(
    messages: list[ChatMessage],
    catalog: Optional[CatalogIndex] = None,
) -> ChatResponse:
    """
    Process a chat request and return the agent's response.

    This is the main entry point for the agent. It:
    1. Extracts search context from conversation history
    2. Retrieves relevant catalog items
    3. Builds a grounded prompt
    4. Makes a single LLM call via Groq
    5. Validates all recommendations
    6. Returns a schema-compliant response
    """
    if catalog is None:
        catalog = catalog_index

    # ── Step 1: Build catalog context via retrieval ─────────────────────
    catalog_context = _build_catalog_context(catalog, messages)

    # ── Step 2: Build system prompt with turn budget ───────────────────
    turn_count = len(messages)
    system_prompt = build_system_prompt(
        catalog_context=catalog_context,
        turn_count=turn_count,
    )

    # ── Step 3: Build conversation for LLM (OpenAI-compatible format) ──
    llm_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        llm_messages.append({
            "role": msg.role,
            "content": msg.content,
        })

    # ── Step 4: Call Groq LLM ──────────────────────────────────────────
    try:
        client = _get_groq_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=llm_messages,
            temperature=0.3,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )
        raw_text = response.choices[0].message.content
        logger.info("LLM response received (%d chars)", len(raw_text))

    except Exception as e:
        logger.error("LLM call failed: %s", e, exc_info=True)
        return ChatResponse(
            reply="I'm experiencing a temporary issue. Could you please try again?",
            recommendations=None,
            end_of_conversation=False,
        )

    # ── Step 5: Parse and validate response ────────────────────────────
    parsed = _parse_llm_response(raw_text)

    reply = parsed.get("reply", "")
    if not reply:
        reply = "I can help you find the right SHL assessments. Could you tell me more about the role you're hiring for?"

    raw_recommendations = parsed.get("recommendations")
    end_of_conversation = bool(parsed.get("end_of_conversation", False))

    # Validate all recommendations against catalog
    validated_recommendations = _validate_and_fix_recommendations(
        raw_recommendations, catalog
    )

    # Programmatically append the markdown table to the reply text if recommendations exist
    reply_with_table = append_recommendations_table(reply, validated_recommendations, catalog)

    # ── Step 6: Build response ─────────────────────────────────────────
    return ChatResponse(
        reply=reply_with_table,
        recommendations=validated_recommendations,
        end_of_conversation=end_of_conversation,
    )
