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
    from obsidian_ai.frontmatter import parse, build, add_tags
    assert callable(parse)
    assert callable(build)
    assert callable(add_tags)


def test_import_chroma_store():
    from obsidian_ai.chroma_store import upsert, delete_by_path, query, get_by_path, count, dedup_paths
    assert callable(upsert)
    assert callable(count)
    assert callable(dedup_paths)


def test_import_mcp_server():
    from obsidian_ai.mcp_server import mcp
    assert mcp.name == "obsidian-ai"
