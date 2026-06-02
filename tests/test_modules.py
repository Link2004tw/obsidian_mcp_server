import sys

sys.path.insert(0, "src")


def test_import_config():
    from obsidian_ai import config
    assert config.ollama_embed_model == "nomic-embed-text"


def test_import_logger():
    from obsidian_ai.logger import get_logger, log_error
    assert callable(get_logger)
    assert callable(log_error)


def test_import_frontmatter():
    from obsidian_ai.frontmatter import add_tags, build, parse
    assert callable(parse)
    assert callable(build)
    assert callable(add_tags)


def test_import_chroma_store():
    from obsidian_ai.chroma_store import (
        count,
        dedup_paths,
        get_by_title,
        upsert,
    )
    assert callable(upsert)
    assert callable(count)
    assert callable(dedup_paths)
    assert callable(get_by_title)


def test_import_mcp_server():
    from obsidian_ai.mcp_server import mcp
    assert mcp.name == "obsidian-ai"


def test_import_entity_store():
    from obsidian_ai.entity_store import add, search, clear, save, stats, entity_types
    assert callable(add)
    assert callable(search)
    assert callable(clear)
    assert callable(save)
    assert callable(stats)
    assert callable(entity_types)


def test_import_pipelines_extract():
    from obsidian_ai.pipelines import extract_entities
    assert callable(extract_entities)
