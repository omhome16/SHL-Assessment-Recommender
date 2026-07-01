"""
Conversation replay tests.
Replays the 10 sample conversations against the agent and checks:
  - Schema compliance on every response
  - Catalog integrity (all URLs from catalog)
  - Recall@10 on final recommendations
"""
import asyncio
import json
import re
import pytest
from pathlib import Path

from app.agent import process_chat
from app.catalog import CatalogIndex, catalog_index
from app.config import CATALOG_PATH
from app.models import ChatMessage, ChatResponse


# ── Parse expected assessments from sample conversation files ──────────────

CONVERSATIONS_DIR = Path(__file__).resolve().parent.parent / "sample_conversations" / "GenAI_SampleConversations"


def parse_expected_assessments(md_text: str) -> list[str]:
    """
    Extract the final set of expected assessment names from a conversation MD file.
    Takes the last table in the file as the final shortlist.
    """
    # Find all tables in the markdown
    table_pattern = re.compile(
        r"\|.*?\|.*?\|.*?\|.*?\|.*?\|.*?\|.*?\|\n"
        r"\|[-\s|]+\|\n"
        r"((?:\|.*?\|.*?\|.*?\|.*?\|.*?\|.*?\|.*?\|\n)*)",
        re.MULTILINE,
    )

    matches = list(table_pattern.finditer(md_text))
    if not matches:
        return []

    # Use the last table (final shortlist)
    last_table = matches[-1]
    rows_text = last_table.group(0)

    names = []
    for line in rows_text.split("\n"):
        line = line.strip()
        if not line or line.startswith("|---") or line.startswith("| #"):
            continue
        cells = [c.strip() for c in line.split("|")]
        # Filter empty cells
        cells = [c for c in cells if c]
        if len(cells) >= 2:
            # Second cell is the name
            name = cells[1].strip()
            if name and not name.startswith("---") and name != "Name":
                names.append(name)

    return names


def load_conversation_trace(filepath: Path) -> tuple[list[dict], list[str]]:
    """
    Load a conversation file and return:
      - The first user message as the starting point
      - Expected assessment names from the final table
    """
    text = filepath.read_text(encoding="utf-8")
    expected = parse_expected_assessments(text)

    # Extract the first user message
    user_msg_match = re.search(r"\*\*User\*\*\s*\n\s*>\s*(.+?)(?:\n|$)", text)
    first_user_msg = user_msg_match.group(1).strip() if user_msg_match else ""

    return [{"role": "user", "content": first_user_msg}], expected


def compute_recall_at_k(recommended: list[str], expected: list[str], k: int = 10) -> float:
    """
    Compute Recall@K: fraction of expected assessments that appear
    in the top K recommendations.
    """
    if not expected:
        return 1.0  # No expected items → trivially recalled

    recommended_lower = [r.lower() for r in recommended[:k]]
    expected_lower = [e.lower() for e in expected]

    found = sum(1 for e in expected_lower if e in recommended_lower)
    return found / len(expected_lower)


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def catalog():
    """Ensure catalog is loaded."""
    if not catalog_index.items:
        catalog_index.load()
    return catalog_index


# ── Tests ──────────────────────────────────────────────────────────────────

class TestConversationReplay:
    """Test that the agent produces valid responses for sample conversations."""

    @pytest.mark.parametrize("conv_file", sorted(CONVERSATIONS_DIR.glob("C*.md")))
    def test_first_turn_schema_compliance(self, conv_file: Path, catalog):
        """Every response must match the ChatResponse schema."""
        messages, _ = load_conversation_trace(conv_file)
        if not messages[0]["content"]:
            pytest.skip(f"No user message found in {conv_file.name}")

        chat_messages = [ChatMessage(**m) for m in messages]
        response = asyncio.get_event_loop().run_until_complete(
            process_chat(chat_messages, catalog)
        )

        # Schema compliance
        assert isinstance(response, ChatResponse)
        assert response.reply  # Non-empty reply
        assert isinstance(response.end_of_conversation, bool)

        # If recommendations exist, validate them
        if response.recommendations:
            assert len(response.recommendations) <= 10
            for rec in response.recommendations:
                assert rec.name
                assert rec.url.startswith("https://www.shl.com/")
                assert rec.test_type
                # Validate against catalog
                item = catalog.validate_recommendation(rec.name, rec.url)
                assert item is not None, (
                    f"Recommendation not in catalog: {rec.name} ({rec.url})"
                )


class TestExpectedAssessmentParsing:
    """Test that we can correctly parse expected assessments from markdown."""

    def test_parse_c1(self):
        filepath = CONVERSATIONS_DIR / "C1.md"
        if filepath.exists():
            text = filepath.read_text(encoding="utf-8")
            expected = parse_expected_assessments(text)
            assert len(expected) > 0
            assert any("OPQ" in name for name in expected)

    def test_parse_c9(self):
        filepath = CONVERSATIONS_DIR / "C9.md"
        if filepath.exists():
            text = filepath.read_text(encoding="utf-8")
            expected = parse_expected_assessments(text)
            assert len(expected) > 0
            assert any("Java" in name for name in expected)


class TestRecallComputation:
    """Test the Recall@K metric computation."""

    def test_perfect_recall(self):
        recommended = ["A", "B", "C"]
        expected = ["A", "B", "C"]
        assert compute_recall_at_k(recommended, expected) == 1.0

    def test_partial_recall(self):
        recommended = ["A", "B", "D"]
        expected = ["A", "B", "C"]
        assert abs(compute_recall_at_k(recommended, expected) - 2 / 3) < 0.01

    def test_no_recall(self):
        recommended = ["X", "Y", "Z"]
        expected = ["A", "B", "C"]
        assert compute_recall_at_k(recommended, expected) == 0.0

    def test_empty_expected(self):
        assert compute_recall_at_k(["A"], []) == 1.0

    def test_case_insensitive(self):
        recommended = ["java 8 (new)"]
        expected = ["Java 8 (New)"]
        assert compute_recall_at_k(recommended, expected) == 1.0
