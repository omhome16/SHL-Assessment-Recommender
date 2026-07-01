"""
Tests for schema compliance.
Ensures every response matches the exact ChatResponse schema required by the evaluator.
"""
import pytest
from pydantic import ValidationError

from app.models import ChatMessage, ChatRequest, ChatResponse, Recommendation


class TestChatMessage:
    def test_valid_user_message(self):
        msg = ChatMessage(role="user", content="I need an assessment")
        assert msg.role == "user"
        assert msg.content == "I need an assessment"

    def test_valid_assistant_message(self):
        msg = ChatMessage(role="assistant", content="Happy to help!")
        assert msg.role == "assistant"

    def test_invalid_role(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="system", content="test")

    def test_empty_content(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="user", content="")


class TestChatRequest:
    def test_valid_request(self):
        req = ChatRequest(messages=[
            ChatMessage(role="user", content="Hello"),
        ])
        assert len(req.messages) == 1

    def test_empty_messages(self):
        with pytest.raises(ValidationError):
            ChatRequest(messages=[])

    def test_max_turns_exceeded(self):
        """8 messages = 8 turns; the next reply would be turn 9 → reject."""
        messages = [
            ChatMessage(role="user" if i % 2 == 0 else "assistant", content=f"msg {i}")
            for i in range(8)
        ]
        with pytest.raises(ValidationError):
            ChatRequest(messages=messages)

    def test_seven_messages_accepted(self):
        """7 messages + the reply = 8 turns → OK."""
        messages = [
            ChatMessage(role="user" if i % 2 == 0 else "assistant", content=f"msg {i}")
            for i in range(7)
        ]
        req = ChatRequest(messages=messages)
        assert len(req.messages) == 7


class TestRecommendation:
    def test_valid_recommendation(self):
        rec = Recommendation(
            name="Java 8 (New)",
            url="https://www.shl.com/products/product-catalog/view/java-8-new/",
            test_type="K",
        )
        assert rec.name == "Java 8 (New)"
        assert rec.test_type == "K"

    def test_multi_type(self):
        rec = Recommendation(name="Test", url="https://example.com", test_type="K,S")
        assert rec.test_type == "K,S"


class TestChatResponse:
    def test_clarification_response(self):
        """No recommendations when clarifying."""
        resp = ChatResponse(
            reply="What role are you hiring for?",
            recommendations=None,
            end_of_conversation=False,
        )
        assert resp.recommendations is None
        assert resp.end_of_conversation is False

    def test_recommendation_response(self):
        resp = ChatResponse(
            reply="Here are my recommendations.",
            recommendations=[
                Recommendation(name="Test1", url="https://example.com/1", test_type="K"),
                Recommendation(name="Test2", url="https://example.com/2", test_type="P"),
            ],
            end_of_conversation=False,
        )
        assert len(resp.recommendations) == 2

    def test_empty_list_becomes_none(self):
        """Empty recommendations list should become None per spec."""
        resp = ChatResponse(
            reply="Still gathering context.",
            recommendations=[],
            end_of_conversation=False,
        )
        assert resp.recommendations is None

    def test_end_of_conversation(self):
        resp = ChatResponse(
            reply="Confirmed. Your shortlist is ready.",
            recommendations=[
                Recommendation(name="OPQ32r", url="https://example.com", test_type="P"),
            ],
            end_of_conversation=True,
        )
        assert resp.end_of_conversation is True

    def test_over_10_capped(self):
        """More than 10 recommendations gets capped."""
        recs = [
            Recommendation(name=f"Test{i}", url=f"https://example.com/{i}", test_type="K")
            for i in range(15)
        ]
        resp = ChatResponse(
            reply="Large list",
            recommendations=recs,
            end_of_conversation=False,
        )
        assert len(resp.recommendations) == 10

    def test_serialization_schema(self):
        """Verify JSON output matches expected schema."""
        resp = ChatResponse(
            reply="Here are the results.",
            recommendations=[
                Recommendation(name="Test", url="https://example.com", test_type="K"),
            ],
            end_of_conversation=False,
        )
        data = resp.model_dump()
        assert "reply" in data
        assert "recommendations" in data
        assert "end_of_conversation" in data
        assert isinstance(data["recommendations"], list)
        assert data["recommendations"][0]["name"] == "Test"
        assert data["recommendations"][0]["url"] == "https://example.com"
        assert data["recommendations"][0]["test_type"] == "K"
