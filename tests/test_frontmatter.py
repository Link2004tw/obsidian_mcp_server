from obsidian_ai.frontmatter import add_tags, build, parse, validate


def test_parse_no_frontmatter():
    meta, body = parse("Just content")
    assert meta == {}
    assert body == "Just content"


def test_parse_with_frontmatter():
    content = "---\ntags: [python, test]\n---\n\nBody text"
    meta, body = parse(content)
    assert meta == {"tags": ["python", "test"]}
    assert body == "Body text"


def test_build_empty_meta():
    assert build({}, "Body") == "Body"


def test_build_with_meta():
    result = build({"tags": ["a"]}, "Body")
    assert "---" in result
    assert "tags:" in result
    assert "Body" in result


def test_add_tags_creates_frontmatter():
    result = add_tags("Just content", ["newtag"])
    assert "newtag" in result
    assert "---" in result


def test_add_tags_appends():
    content = "---\ntags: [existing]\n---\n\nBody"
    result = add_tags(content, ["newtag"])
    assert "existing" in result
    assert "newtag" in result


def test_add_tags_no_duplicates():
    content = "---\ntags: [tag1]\n---\n\nBody"
    result = add_tags(content, ["tag1"])
    assert result.count("tag1") == 1


def test_add_tags_converts_string_to_list():
    content = "---\ntags: single\n---\n\nBody"
    result = add_tags(content, ["extra"])
    assert "extra" in result
    assert "- single" in result
    assert "- extra" in result


# ── Validation ───────────────────────────────────────────────────────


def test_validate_no_frontmatter():
    assert validate("Just content") == []


def test_validate_valid_frontmatter():
    content = "---\ntags: [python, test]\naliases: [py]\n---\n\nBody"
    assert validate(content) == []


def test_validate_tags_as_string():
    content = "---\ntags: python\n---\n\nBody"
    warnings = validate(content)
    assert len(warnings) == 1
    assert "string" in warnings[0]


def test_validate_tags_as_null():
    content = "---\ntags:\n---\n\nBody"
    warnings = validate(content)
    assert any("null" in w for w in warnings)


def test_validate_tags_as_number():
    content = "---\ntags: 42\n---\n\nBody"
    warnings = validate(content)
    assert any("number" in w for w in warnings)


def test_validate_tags_non_string_in_list():
    content = "---\ntags: [good, 42]\n---\n\nBody"
    warnings = validate(content)
    assert any("non-string" in w for w in warnings)


def test_validate_aliases_not_list():
    content = "---\naliases: wrong\n---\n\nBody"
    warnings = validate(content)
    assert any("list" in w for w in warnings)


def test_validate_duplicate_keys():
    content = "---\ntags: [a]\ntags: [b]\n---\n\nBody"
    warnings = validate(content)
    assert any("Duplicate" in w for w in warnings)


def test_validate_unclosed():
    content = "---\ntags: [a]\n"
    warnings = validate(content)
    assert any("closed" in w for w in warnings)
