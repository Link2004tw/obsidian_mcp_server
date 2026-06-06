"""Obsidian AI MCP server — tool registration, re-exports, and entry point."""

from fastmcp import FastMCP

from . import config
from .logger import get_logger
from .tools import register_all
from .tools._shared import (  # noqa: F401
    _EXPAND_QUERY_CACHE,
    _build_search_where,
    _expand_query,
    _get_vault_terminology,
    _group_by_note,
    _hybrid_search,
    _matches_where,
    _normalize_path,
    _rewrite_query,
    _truncate_snippet,
)
from .tools.graph import (  # noqa: F401
    add_entity,
    entity_timeline,
    export_graph,
    get_backlinks,
    get_broken_links,
    get_communities,
    get_entity_aliases,
    get_entity_types,
    get_graph_stats,
    get_linked_notes,
    get_note_community,
    get_note_entities,
    get_orphan_notes,
    get_ranking_weights,
    get_shortest_path,
    import_entities,
    list_entities,
    merge_entities,
    multi_hop_traversal,
    related_entities,
    related_notes,
    search_entities,
    set_ranking_weights,
)
from .tools.misc import (  # noqa: F401
    get_clusters,
    health_check,
)
from .tools.notes import (  # noqa: F401
    add_note_to_subject,
    add_tags,
    batch_tag_notes,
    create_backlink,
    list_all_notes,
    list_folder,
    list_folder_deep,
    read_note,
    read_note_by_title,
    remove_tags,
    search_by_tags,
    set_tags,
    switch_embedding_model,
    sync_index,
    write_note,
)
from .tools.search import (  # noqa: F401
    ask_agent,
    ask_vault,
    batch_search,
    composite_search,
    find_duplicate_notes,
    get_index_stats,
    get_subject,
    retrieve_notes,
    search_notes,
    summarize_topic,
    tag_notes,
)
from .tools.todos import (  # noqa: F401
    add_todo,
    add_todo_from_natural_language,
    ask_vault_about_todo,
    ask_vault_about_todos,
    complete_todo,
    delete_todo,
    ensure_todo_file,
    estimate_completion_date,
    get_notes_for_todo,
    get_overdue_summary,
    get_todo_stats,
    get_todos,
    get_todos_by_priority,
    get_todos_for_note,
    link_todo_to_notes,
    suggest_due_date,
    suggest_task_priority,
    suggest_task_splitting,
    sync_todos,
    update_todo,
)

log = get_logger("obsidian_ai.mcp_server", log_file="mcp_calls.log")

mcp = FastMCP("obsidian-ai")
register_all(mcp)

if __name__ == "__main__":
    cfg_warnings = config.validate(verbose=True)
    if cfg_warnings:
        log.warning(f"Startup config validation found {len(cfg_warnings)} issue(s)")
    log.info("Starting MCP server")
    mcp.run()
