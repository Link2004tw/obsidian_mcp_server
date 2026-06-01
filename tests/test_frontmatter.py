from obsidian_ai.frontmatter import parse, build, add_tags


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
