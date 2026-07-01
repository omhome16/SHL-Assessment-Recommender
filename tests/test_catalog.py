"""
Tests for catalog engine.
Validates catalog loading, indexing, search quality, and integrity checks.
"""
import pytest
from pathlib import Path

from app.catalog import CatalogIndex, CatalogItem, _tokenize
from app.config import CATALOG_PATH


@pytest.fixture(scope="module")
def catalog():
    """Load catalog once for all tests in this module."""
    idx = CatalogIndex(catalog_path=CATALOG_PATH)
    idx.load()
    return idx


class TestCatalogLoading:
    def test_catalog_loads(self, catalog: CatalogIndex):
        assert len(catalog.items) > 300
        assert len(catalog.items) < 500

    def test_all_items_have_names(self, catalog: CatalogIndex):
        for item in catalog.items:
            assert item.name, f"Item {item.entity_id} has no name"

    def test_all_items_have_links(self, catalog: CatalogIndex):
        for item in catalog.items:
            assert item.link.startswith("https://www.shl.com/"), (
                f"Item '{item.name}' has invalid link: {item.link}"
            )

    def test_all_items_have_keys(self, catalog: CatalogIndex):
        for item in catalog.items:
            assert len(item.keys) > 0, f"Item '{item.name}' has no keys"


class TestTestTypeMapping:
    def test_knowledge_type(self, catalog: CatalogIndex):
        item = catalog.get_item_by_name(".NET Framework 4.5")
        assert item is not None
        assert "K" in item.test_type_code

    def test_personality_type(self, catalog: CatalogIndex):
        item = catalog.get_item_by_name("Occupational Personality Questionnaire OPQ32r")
        if item:
            assert "P" in item.test_type_code


class TestSearch:
    def test_java_search(self, catalog: CatalogIndex):
        """Searching 'Java' should return Java-related assessments."""
        results = catalog.search("Java developer assessment", top_k=10)
        assert len(results) > 0
        names = [r[0].name.lower() for r in results]
        assert any("java" in n for n in names), f"No Java results in: {names}"

    def test_personality_search(self, catalog: CatalogIndex):
        """Searching for personality should return personality assessments."""
        results = catalog.search("personality assessment for leadership", top_k=10)
        assert len(results) > 0
        # At least one result should have Personality key
        found = any(
            "Personality & Behavior" in r[0].keys
            for r in results
        )
        assert found, "No personality assessment found"

    def test_safety_search(self, catalog: CatalogIndex):
        """Safety-related search should find DSI and safety assessments."""
        results = catalog.search("safety dependability plant operator", top_k=10)
        assert len(results) > 0
        names = [r[0].name.lower() for r in results]
        assert any("safety" in n or "dsi" in n or "dependab" in n for n in names)

    def test_empty_query(self, catalog: CatalogIndex):
        """Empty query should not crash."""
        results = catalog.search("", top_k=5)
        # May return empty or fallback results
        assert isinstance(results, list)


class TestCatalogIntegrity:
    def test_name_lookup(self, catalog: CatalogIndex):
        """Should find items by exact name."""
        item = catalog.get_item_by_name("Global Skills Development Report")
        assert item is not None
        assert item.entity_id == "4302"

    def test_url_lookup(self, catalog: CatalogIndex):
        """Should find items by exact URL."""
        url = "https://www.shl.com/products/product-catalog/view/global-skills-development-report/"
        item = catalog.get_item_by_url(url)
        assert item is not None
        assert item.name == "Global Skills Development Report"

    def test_validate_valid_recommendation(self, catalog: CatalogIndex):
        """Valid name + URL should pass validation."""
        item = catalog.validate_recommendation(
            name="Global Skills Development Report",
            url="https://www.shl.com/products/product-catalog/view/global-skills-development-report/",
        )
        assert item is not None

    def test_validate_invalid_recommendation(self, catalog: CatalogIndex):
        """Invented assessment should fail validation."""
        item = catalog.validate_recommendation(
            name="Totally Made Up Test 9000",
            url="https://www.shl.com/products/product-catalog/view/does-not-exist/",
        )
        assert item is None

    def test_no_duplicate_urls(self, catalog: CatalogIndex):
        """All catalog items should have unique URLs."""
        urls = [item.link for item in catalog.items]
        assert len(urls) == len(set(urls)), "Duplicate URLs found in catalog"


class TestTokenizer:
    def test_basic_tokenization(self):
        tokens = _tokenize("Java 8 developer")
        assert "java" in tokens
        assert "developer" in tokens

    def test_punctuation_removal(self):
        tokens = _tokenize("Hello, world! (test)")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_single_char_filtered(self):
        tokens = _tokenize("I a test")
        assert "test" in tokens
        # Single chars should be filtered
        assert "a" not in tokens
