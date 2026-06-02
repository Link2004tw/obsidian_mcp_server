"""Tests for indexer.py — pure functions (no external dependencies)."""
from unittest.mock import patch

from obsidian_ai.indexer import (
    chunk_text,
    split_by_headings,
    chunk_text_heading_aware,
    _sanitize,
    _word_count,
    _extract_tags,
    _extract_frontmatter_fields,
    _tags_to_meta,
    _links_to_meta,
)


# ── _word_count ────────────────────────────────────────────────────


def test_word_count_basic():
    assert _word_count("hello world") == 2


def test_word_count_empty():
    assert _word_count("") == 0


def test_word_count_whitespace():
    assert _word_count("  lots   of   spaces  ") == 3


def test_word_count_single():
    assert _word_count("one") == 1


# ── _sanitize ──────────────────────────────────────────────────────


def test_sanitize_removes_urls():
    result = _sanitize("visit https://example.com for more")
    assert "https://" not in result
    assert "visit" in result
    assert "for more" in result


def test_sanitize_removes_empty_parens():
    result = _sanitize("hello () world")
    assert "()" not in result
    assert "hello" in result
    assert "world" in result


def test_sanitize_collapses_whitespace():
    result = _sanitize("too   many    spaces")
    assert "  " not in result
    assert result == "too many spaces"


def test_sanitize_strips_edges():
    result = _sanitize("  trimmed  ")
    assert result == "trimmed"


def test_sanitize_preserves_content():
    result = _sanitize("This is a normal sentence.")
    assert result == "This is a normal sentence."


# ── chunk_text ─────────────────────────────────────────────────────


def test_chunk_text_short_text():
    """Text shorter than chunk_size returns a single chunk."""
    chunks = chunk_text("one two three", size=500, overlap=100)
    assert len(chunks) == 1
    assert chunks[0] == "one two three"


def test_chunk_text_exact_size():
    """Text exactly chunk_size returns one chunk (start advances to len(words), loop ends)."""
    text = " ".join([f"w{i}" for i in range(500)])
    chunks = chunk_text(text, size=500, overlap=100)
    # start=0 → chunk[0:500], start=400 → chunk[400:500] (100 words)
    assert len(chunks) == 2


def test_chunk_text_multiple_chunks():
    """Text exceeding chunk_size produces overlapping chunks."""
    text = " ".join([f"w{i}" for i in range(1000)])
    chunks = chunk_text(text, size=500, overlap=100)
    assert len(chunks) == 3  # 1000 words, 500 size, 100 overlap → 0-499, 400-899, 800-999


def test_chunk_text_overlap():
    """Consecutive chunks share overlap words."""
    text = " ".join([f"w{i}" for i in range(600)])
    chunks = chunk_text(text, size=500, overlap=100)
    # Chunk 0: words 0-499, Chunk 1: words 400-599
    assert len(chunks) == 2
    # Last 100 words of chunk 0 should equal first 100 words of chunk 1
    words_0 = chunks[0].split()
    words_1 = chunks[1].split()
    assert words_0[-100:] == words_1[:100]


def test_chunk_text_custom_params():
    text = " ".join([f"w{i}" for i in range(20)])
    chunks = chunk_text(text, size=10, overlap=3)
    # 20 words, size 10, overlap 3 → step=7: 0-9, 7-16, 14-19
    assert len(chunks) == 3


def test_chunk_text_empty():
    chunks = chunk_text("")
    assert chunks == []


# ── _extract_tags ──────────────────────────────────────────────────


def test_extract_tags_with_frontmatter():
    content = "---\ntags:\n  - python\n  - ai\n---\nBody here"
    assert _extract_tags(content) == ["python", "ai"]


def test_extract_tags_string_tag():
    content = "---\ntags: python\n---\nBody"
    assert _extract_tags(content) == ["python"]


def test_extract_tags_no_frontmatter():
    assert _extract_tags("Just plain text") == []


def test_extract_tags_no_tags_field():
    content = "---\ntitle: test\n---\nBody"
    assert _extract_tags(content) == []


def test_extract_tags_empty_tags():
    content = "---\ntags: []\n---\nBody"
    assert _extract_tags(content) == []


# ── _tags_to_meta ─────────────────────────────────────────────────


def test_tags_to_meta_basic():
    assert _tags_to_meta(["python", "ai"]) == ",python,ai,"


def test_tags_to_meta_single():
    assert _tags_to_meta(["python"]) == ",python,"


def test_tags_to_meta_empty():
    assert _tags_to_meta([]) == ",,"


# ── split_by_headings ──────────────────────────────────────────────


def test_split_by_headings_no_headings():
    """Text without headings returns single section with empty heading path."""
    result = split_by_headings("Just plain text here.")
    assert result == [("", "Just plain text here.")]


def test_split_by_headings_single_heading():
    """Single heading returns one section with that heading as path."""
    result = split_by_headings("# Introduction\n\nSome intro text.")
    assert len(result) == 1
    assert result[0][0] == "# Introduction"
    assert "Some intro text" in result[0][1]


def test_split_by_headings_multiple_levels():
    """Nested headings accumulate parent paths."""
    text = """# Setup

Setup content.

## Configuration

Config content.

### Advanced

Advanced content."""
    result = split_by_headings(text)
    assert len(result) == 3
    assert result[0][0] == "# Setup"
    assert "Setup content" in result[0][1]
    assert result[1][0] == "# Setup > ## Configuration"
    assert "Config content" in result[1][1]
    assert result[2][0] == "# Setup > ## Configuration > ### Advanced"
    assert "Advanced content" in result[2][1]


def test_split_by_headings_text_before_first_heading():
    """Text before the first heading is a separate section."""
    text = "Intro paragraph.\n\n## Section\n\nSection body."
    result = split_by_headings(text)
    assert len(result) == 2
    assert result[0][0] == ""
    assert "Intro paragraph" in result[0][1]
    assert result[1][0] == "## Section"
    assert "Section body" in result[1][1]


def test_split_by_headings_heading_reset():
    """A lower-level heading after a deeper one resets the hierarchy."""
    text = """# Top

Top content.

## Middle

Middle content.

# New Top

New top content."""
    result = split_by_headings(text)
    assert len(result) == 3
    assert result[0][0] == "# Top"
    assert result[1][0] == "# Top > ## Middle"
    assert result[2][0] == "# New Top"


# ── chunk_text_heading_aware ──────────────────────────────────────


def test_chunk_text_heading_aware_no_headings():
    """No headings returns single chunk with empty heading path."""
    result = chunk_text_heading_aware("Short text.", size=500)
    assert len(result) == 1
    assert result[0][0] == ""
    assert result[0][1] == "Short text."


def test_chunk_text_heading_aware_small_sections():
    """Sections smaller than chunk size are kept intact."""
    text = "# Intro\n\nHello world.\n\n## Details\n\nSome details here."
    result = chunk_text_heading_aware(text, size=500)
    assert len(result) == 2
    assert result[0][0] == "# Intro"
    assert "# Intro" in result[0][1]
    assert result[1][0] == "# Intro > ## Details"
    assert "## Details" in result[1][1]


def test_chunk_text_heading_aware_large_section():
    """Sections larger than chunk size are split with heading prefix."""
    words = " ".join([f"word{i}" for i in range(600)])
    text = f"# Big Section\n\n{words}"
    result = chunk_text_heading_aware(text, size=500, overlap=100)
    # 600 words > 500 size, so should split into 2 chunks
    assert len(result) == 2
    # Both chunks should have the heading prefix
    assert "# Big Section" in result[0][1]
    assert "# Big Section" in result[1][1]


def test_chunk_text_heading_aware_empty_section():
    """Empty heading sections are skipped."""
    text = "# Empty\n\n\n## Has Content\n\nSome text."
    result = chunk_text_heading_aware(text, size=500)
    assert len(result) == 1
    assert result[0][0] == "# Empty > ## Has Content"


# ── _extract_frontmatter_fields ───────────────────────────────────


def test_extract_frontmatter_fields_all_present():
    content = """---
created: 2024-01-15
modified: 2024-06-20
aliases:
  - my note
  - alias two
cssclasses:
  - wide
  - center
title: My Custom Title
tags:
  - test
---
Body text."""
    fields = _extract_frontmatter_fields(content)
    assert fields["created"] == "2024-01-15"
    assert fields["modified"] == "2024-06-20"
    assert "my note" in fields["aliases_str"]
    assert "wide" in fields["cssclasses_str"]
    assert fields["fm_title"] == "My Custom Title"


def test_extract_frontmatter_fields_partial():
    content = "---\ntitle: Partial Title\naliases: single-alias\n---\nBody"
    fields = _extract_frontmatter_fields(content)
    assert fields["fm_title"] == "Partial Title"
    assert fields["aliases_str"] == ",single-alias,"
    assert "created" not in fields
    assert "modified" not in fields


def test_extract_frontmatter_fields_no_frontmatter():
    fields = _extract_frontmatter_fields("Just plain text, no frontmatter.")
    assert fields == {}


def test_extract_frontmatter_fields_empty_frontmatter():
    content = "---\ntags: []\n---\nBody"
    fields = _extract_frontmatter_fields(content)
    assert fields == {}


def test_extract_frontmatter_fields_string_alias():
    content = "---\naliases: single\n---\nBody"
    fields = _extract_frontmatter_fields(content)
    assert fields["aliases_str"] == ",single,"

