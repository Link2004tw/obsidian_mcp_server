from obsidian_ai.wiki_links import extract_wiki_links, normalize_wiki_link_target


def test_normalize_wiki_link_target_strips_display_text():
    assert normalize_wiki_link_target("Project Alpha|display text") == "project alpha"


def test_normalize_wiki_link_target_strips_section_link():
    assert normalize_wiki_link_target("Project Alpha#Setup") == "project alpha"


def test_normalize_wiki_link_target_normalizes_folder_path():
    assert normalize_wiki_link_target("/Folder\\Project Alpha.md") == "folder/project alpha"


def test_extract_wiki_links_handles_common_forms():
    content = "See [[Project Alpha]], [[Project Beta|Beta]], and [[Folder/Project Gamma#Setup]]."

    assert extract_wiki_links(content) == [
        "project alpha",
        "project beta",
        "folder/project gamma",
    ]


def test_extract_wiki_links_deduplicates_in_first_seen_order():
    content = "[[Project Alpha]] [[project alpha#Later]] [[Project Beta]]"

    assert extract_wiki_links(content) == ["project alpha", "project beta"]


def test_extract_wiki_links_ignores_image_embeds_and_empty_targets():
    content = "![[image.png]] [[]] [[#Only Section]] [[Visible]]"

    assert extract_wiki_links(content) == ["visible"]
