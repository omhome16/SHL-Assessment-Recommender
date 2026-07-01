"""
Catalog engine: loads, indexes, and searches the SHL product catalog.

Dual retrieval strategy:
  - BM25 for exact keyword matching (critical for tech-specific test names)
  - Sentence-transformer embeddings for semantic matching (competency → assessment mapping)
  - Hybrid scoring combines both signals with configurable weights
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.config import (
    BM25_WEIGHT,
    CATALOG_PATH,
    EMBEDDING_MODEL,
    SEMANTIC_WEIGHT,
    TOP_K_RETRIEVAL,
)

logger = logging.getLogger(__name__)


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class CatalogItem:
    """A single assessment from the SHL catalog."""
    entity_id: str
    name: str
    link: str
    job_levels: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    duration: str = ""
    description: str = ""
    keys: list[str] = field(default_factory=list)
    remote: str = ""
    adaptive: str = ""

    @property
    def test_type_code(self) -> str:
        """
        Derive a short test-type code from the 'keys' field.
        Maps catalog key categories to the letter codes used in sample conversations.
        """
        mapping = {
            "Knowledge & Skills": "K",
            "Personality & Behavior": "P",
            "Ability & Aptitude": "A",
            "Biodata & Situational Judgment": "B",
            "Simulations": "S",
            "Competencies": "C",
            "Development & 360": "D",
            "Assessment Exercises": "E",
        }
        codes = []
        for key in self.keys:
            code = mapping.get(key)
            if code and code not in codes:
                codes.append(code)
        return ",".join(codes) if codes else "K"

    @property
    def searchable_text(self) -> str:
        """Combined text for indexing and search."""
        parts = [
            self.name,
            self.description,
            " ".join(self.keys),
            " ".join(self.job_levels),
        ]
        return " | ".join(p for p in parts if p)

    def to_context_string(self) -> str:
        """Format for injection into LLM prompt as grounding context."""
        langs = ", ".join(self.languages[:3])
        if len(self.languages) > 3:
            langs += f" (+{len(self.languages) - 3} more)"
        levels = ", ".join(self.job_levels[:4])
        if len(self.job_levels) > 4:
            levels += f" (+{len(self.job_levels) - 4} more)"

        return (
            f"- **{self.name}** [Type: {self.test_type_code}]\n"
            f"  URL: {self.link}\n"
            f"  Keys: {', '.join(self.keys)}\n"
            f"  Duration: {self.duration or 'N/A'}\n"
            f"  Job Levels: {levels or 'N/A'}\n"
            f"  Languages: {langs or 'N/A'}\n"
            f"  Remote: {self.remote} | Adaptive: {self.adaptive}\n"
            f"  Description: {self.description[:300]}"
        )


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


class CatalogIndex:
    """
    Hybrid search index over the SHL product catalog.

    Built once at application startup; used for every /chat request
    to ground the LLM in real catalog data.
    """

    def __init__(self, catalog_path: Optional[Path] = None):
        self.catalog_path = catalog_path or CATALOG_PATH
        self.items: list[CatalogItem] = []
        self._name_lookup: dict[str, CatalogItem] = {}
        self._url_lookup: dict[str, CatalogItem] = {}
        self._bm25: Optional[BM25Okapi] = None
        self._embeddings: Optional[np.ndarray] = None
        self._embedding_model: Optional[SentenceTransformer] = None

    def load(self) -> None:
        """Load catalog from JSON and build all indices."""
        logger.info("Loading catalog from %s", self.catalog_path)

        with open(self.catalog_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.items = []
        for entry in raw_data:
            if entry.get("status") != "ok":
                continue
            item = CatalogItem(
                entity_id=entry.get("entity_id", ""),
                name=entry.get("name", ""),
                link=entry.get("link", ""),
                job_levels=entry.get("job_levels", []),
                languages=entry.get("languages", []),
                duration=entry.get("duration", ""),
                description=entry.get("description", ""),
                keys=entry.get("keys", []),
                remote=entry.get("remote", ""),
                adaptive=entry.get("adaptive", ""),
            )
            self.items.append(item)

        logger.info("Loaded %d catalog items", len(self.items))

        # Build lookups
        self._name_lookup = {item.name.lower(): item for item in self.items}
        self._url_lookup = {item.link: item for item in self.items}

        # Build BM25 index
        self._build_bm25_index()

        # Build embedding index
        self._build_embedding_index()

    def _build_bm25_index(self) -> None:
        """Build BM25 index from catalog items."""
        corpus = [_tokenize(item.searchable_text) for item in self.items]
        self._bm25 = BM25Okapi(corpus)
        logger.info("BM25 index built over %d documents", len(corpus))

    def _build_embedding_index(self) -> None:
        """Build sentence embedding index for semantic search."""
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        self._embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        texts = [item.searchable_text for item in self.items]
        self._embeddings = self._embedding_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        logger.info(
            "Embedding index built: %d items × %d dims",
            *self._embeddings.shape
        )

    def search(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVAL,
        bm25_weight: float = BM25_WEIGHT,
        semantic_weight: float = SEMANTIC_WEIGHT,
    ) -> list[tuple[CatalogItem, float]]:
        """
        Hybrid search: BM25 + semantic similarity.

        Returns a ranked list of (CatalogItem, score) tuples.
        """
        if not self.items:
            return []

        n = len(self.items)

        # ── BM25 scores ────────────────────────────────────────────────
        query_tokens = _tokenize(query)
        if query_tokens and self._bm25 is not None:
            bm25_raw = self._bm25.get_scores(query_tokens)
            bm25_max = bm25_raw.max()
            bm25_scores = bm25_raw / bm25_max if bm25_max > 0 else bm25_raw
        else:
            bm25_scores = np.zeros(n)

        # ── Semantic scores ────────────────────────────────────────────
        if self._embedding_model is not None and self._embeddings is not None:
            query_emb = self._embedding_model.encode(
                [query], normalize_embeddings=True, show_progress_bar=False
            )
            semantic_scores = (self._embeddings @ query_emb.T).flatten()
            # Clamp to [0, 1]
            semantic_scores = np.clip(semantic_scores, 0, 1)
        else:
            semantic_scores = np.zeros(n)

        # ── Hybrid combination ─────────────────────────────────────────
        combined = bm25_weight * bm25_scores + semantic_weight * semantic_scores

        # ── Rank and return top-k ──────────────────────────────────────
        top_indices = np.argsort(combined)[::-1][:top_k]
        results = [
            (self.items[i], float(combined[i]))
            for i in top_indices
            if combined[i] > 0.01  # Filter out near-zero scores
        ]

        return results

    def get_item_by_name(self, name: str) -> Optional[CatalogItem]:
        """Exact lookup by assessment name (case-insensitive)."""
        return self._name_lookup.get(name.lower())

    def get_item_by_url(self, url: str) -> Optional[CatalogItem]:
        """Exact lookup by catalog URL."""
        return self._url_lookup.get(url)

    def validate_recommendation(self, name: str, url: str) -> Optional[CatalogItem]:
        """
        Verify a recommendation exists in the catalog.
        Returns the CatalogItem if valid, None otherwise.
        Tries URL match first (most reliable), then name match.
        """
        # Try exact URL match
        item = self._url_lookup.get(url)
        if item:
            return item

        # Try name match
        item = self._name_lookup.get(name.lower())
        if item:
            return item

        # Try fuzzy name match (handle minor LLM formatting differences)
        name_lower = name.lower().strip()
        for catalog_name, catalog_item in self._name_lookup.items():
            if (
                name_lower in catalog_name
                or catalog_name in name_lower
            ):
                return catalog_item

        return None

    def get_all_items(self) -> list[CatalogItem]:
        """Return all catalog items."""
        return self.items


# ── Module-level singleton ─────────────────────────────────────────────────
catalog_index = CatalogIndex()
