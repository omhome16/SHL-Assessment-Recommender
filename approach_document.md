# SHL Assessment Recommender — Technical Approach Document
**Candidate:** Om Home | **Role:** AI Intern | **Submitting Organization:** SHL Labs

---

## 1. Executive Summary & Architecture Overview

The **SHL Assessment Recommender** is an intelligent conversational agent designed to guide recruiters and hiring managers from vague hiring goals (e.g., *"I need an assessment"*) to a shortlist of 1–10 validated, canonical assessments selected from the SHL product catalog. 

The system is architected as a **stateless FastAPI service** exposing two public endpoints: `GET /health` and `POST /chat`. Each request to `/chat` carries the full conversation history. No session state or conversation logs are stored on the server. The execution pipeline per request runs synchronously in a single lifecycle:

```
[ POST /chat Request ]
        │
        ▼
 1. Query Consolidation  ──► (Aggregates all user messages)
        │
        ▼
 2. Hybrid Retrieval    ──► (BM25 + all-MiniLM-L6-v2 Embeddings)
        │
        ▼
 3. Context Grounding   ──► (Extracts top-25 items metadata)
        │
        ▼
 4. Prompt Assembly     ──► (Bakes warning caps and catalog data)
        │
        ▼
 5. LLM JSON Inference  ──► (Calls Llama-3.3-70B via Groq API)
        │
        ▼
 6. Post-Validation     ──► (Drops hallucinations, verifies URLs)
        │
        ▼
 7. Table Rendering     ──► (Appends formatted markdown table to reply)
        │
        ▼
[ JSON ChatResponse ]
```

---

## 2. Design Choices & Rationale

### LLM Selection (Llama-3.3-70b-versatile via Groq)
We selected `llama-3.3-70b-versatile` running on Groq LPUs. Groq’s hardware delivers ultra-low latencies (~1.5s–2s response times), which easily satisfies the 30-second API limit. By utilizing Groq’s native `response_format={"type": "json_object"}`, we enforce schema compliance at the hardware level, preventing formatting errors that often crash traditional regex-based JSON parsers.

### Hybrid Retrieval (BM25 + Dense Embeddings)
Pure semantic search struggles with exact technology keywords, while pure keyword search misses conceptual constraints.
*   **BM25 (rank_bm25)**: Essential for exact-match technical keywords (e.g., "Java 8", "AWS", "SQL").
*   **Dense Embeddings (all-MiniLM-L6-v2)**: Vital for mapping soft skills or behavioral traits (e.g., "leadership benchmark", "personality audit") to cognitive or personality tests.
*   **Weighted Scoring**: We combine scores using the formula:
    $$\text{Score} = 0.4 \times \text{Score}_{\text{BM25}} + 0.6 \times \text{Similarity}_{\text{Cosine}}$$
    This combined scoring outperformed either retrieval method alone on the 10 conversation traces.

### State Persistence via Markdown Tables
In a stateless API, previous turns' recommendations are not sent back in a structured array. To keep the agent aware of what it previously recommended, the system programmatically generates a formatted markdown table of the active recommendations and appends it to the assistant's `reply` string. When the evaluator sends the conversation history in the next turn, the agent parses the table and remains fully aware of past recommendations, enabling correct mid-conversation edits (e.g., *"drop the first one"*).

### Turn Budget Enforcement
To respect the strict 8-turn budget limit, the system prompt calculates remaining turns:
*   At $\le$ 2 turns remaining, a critical system prompt override is injected: `⚠️ CRITICAL: Only {turns_left} turns remain. You MUST output recommendations now. Do not ask clarifying questions.` This prevents the LLM from asking unnecessary questions and failing the trace budget.

---

## 3. Retrieval & Indexing Setup

The indexing pipeline runs once during application startup:
1.  **Catalog Ingestion**: Loads the 377 SHL product catalog items from `SHL_product_catlog.json`.
2.  **BM25 Indexing**: Extracts tokenized representations of each item's name, description, keywords (`keys`), and target job levels.
3.  **Semantic Indexing**: Generates 384-dimensional dense vectors using a localized `sentence-transformers/all-MiniLM-L6-v2` instance.
4.  **Startup Time**: Loading and indexing the entire catalog finishes in under 5.0 seconds.

---

## 4. Prompt Engineering & Intent Routing

The agent system prompt in [prompts.py](file:///d:/AI/AIML/SUNRISE%20COUNTDOWN/ai-craftsman-portfolio/projects/SHL/app/prompts.py) defines a strict intent routing structure:
*   **CLARIFY**: Runs if the role family, seniority level, or business purpose (selection vs. development) is missing. It asks 1–2 clarifying questions and returns `recommendations: null`.
*   **RECOMMEND**: Triggers only when role details and purpose are both known. Recommends 1–10 items from the top-25 grounded context.
*   **REFINE**: Processes modifications (e.g. *"add personality tests"*) by updating the active shortlist instead of restarting the query.
*   **COMPARE**: Evaluates differences between assessments using *only* facts present in the injected catalog metadata to prevent model priors from hallucinating differences.
*   **CONFIRM**: Sets `end_of_conversation: true` once the user accepts the shortlist.
*   **REFUSE**: Rejects off-topic prompts (e.g. salary ranges, legal hiring guidelines) and prompt injections.

---

## 5. Evaluation & Validation Methodology

We built a turn-by-turn simulation harness ([replay_live_conversations.py](file:///d:/AI/AIML/SUNRISE%20COUNTDOWN/ai-craftsman-portfolio/projects/SHL/tests/replay_live_conversations.py)) that replays the 10 sample dialogue traces against our live API endpoint and scores:
1.  **Schema Compliance**: Verifies that every response contains the correct JSON fields.
2.  **Catalog URL/Name Integrity**: Verifies that every recommended assessment name and URL is present in the catalog.
3.  **Recall@10**: Calculates the fraction of expected assessments returned by the agent.
4.  **Request Latency**: Measures the round-trip API processing time (average is **2.02 seconds**).

---

## 6. What Did Not Work & Solutions

*   **Pytorch Memory Limits (OOM) on Free Hosting**: Loading PyTorch and sentence-transformers in Python allocated ~600MB of RAM, causing Render's 512MB free tier to trigger Out-of-Memory (OOM) crashes.
    *   *Solution*: Deployed to Hugging Face Spaces (Docker SDK), which provides a free 16GB RAM CPU instance, eliminating OOM issues.
*   **Eager Turn-1 Recommendations**: The LLM frequently suggested tests on the very first turn of vague inputs (e.g., *"We need a solution for senior leadership"*), failing the clarification behavior probe.
    *   *Solution*: Strengthened prompt guardrails to require at least **2–3 key facts** (explicitly requiring both role context AND selection vs. development purpose) before allowing a recommendation.
*   **Duplicate URLs**: The LLM occasionally recommended the same test under different names (e.g. "Java 8" and "Core Java 8").
    *   *Solution*: Added a post-LLM deduplication layer in the agent script that keeps only unique URLs.

---

## 7. Final Live API Performance Summary

A final automated trace replay against the live Space URL (`https://omhome-shl-assessment-recommender.hf.space`) yielded the following metrics:
*   **Mean Recall@10 Score**: **`100%`** (10/10 traces successfully matched)
*   **Schema Compliance Rate**: **`100%`** (0 parsing errors)
*   **Catalog Integrity Rate**: **`100%`** (0 hallucinated links)
*   **Average API Latency**: **`2.02 seconds`** (well below the 28.0-second safety timeout)
*   **Timeout & Turn-Cap Adherence**: **`Passed`**
