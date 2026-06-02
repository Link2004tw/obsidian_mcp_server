"""Obsidian AI Knowledge System — semantic search, tagging, and backlinks for Obsidian vaults."""

# Lightweight imports only — heavy deps (chroma_store, indexer, llm_client,
# mcp_server, pipelines) are imported lazily to avoid loading ChromaDB and
# other heavy dependencies when only config or obsidian_client is needed.
from . import config, obsidian_client

__all__ = [
    "config",
    "obsidian_client",
]
