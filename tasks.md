# Tool Simplification Plan

Consolidate ~50 MCP tools into 9 unified tools so small models don't get confused.

## Target Tools

| # | Tool | Actions | Replaces |
|---|------|---------|----------|
| 1 | `ask` | (single `query` param) | search_notes, batch_search, ask_vault, retrieve_notes, ask_agent, tag_notes, summarize_topic, get_subject, composite_search, get_index_stats, get_clusters, search_entities, related_notes, get_backlinks, get_linked_notes, get_broken_links, get_communities, get_note_community, get_orphan_notes, get_shortest_path, get_graph_stats, multi_hop_traversal, export_graph, entity_timeline, related_entities |
| 2 | `notes` | read, write, list, list_folder, search_by_tags, read_by_title, add_note_to_subject | read_note, write_note, list_all_notes, list_folder, list_folder_deep, search_by_tags, read_note_by_title, add_note_to_subject |
| 3 | `tags` | add, remove, set, batch_add, auto_suggest | add_tags, remove_tags, set_tags, batch_tag_notes, tag_notes |
| 4 | `links` | create, backlinks, outgoing, broken | create_backlink, get_backlinks, get_linked_notes, get_broken_links |
| 5 | `graph` | communities, community_of, orphans, path, stats, related, traverse, export | get_communities, get_note_community, get_orphan_notes, get_shortest_path, get_graph_stats, multi_hop_traversal, related_notes, export_graph |
| 6 | `entities` | search, note_entities, list, aliases, timeline, related, add, merge, change_type, types, weights, import | search_entities, get_note_entities, get_entity_types, get_entity_aliases, merge_entities, import_entities, entity_timeline, related_entities, list_entities, add_entity, add_aliases, change_entity_type, get_ranking_weights, set_ranking_weights |
| 7 | `todo` | list, add, complete, update, delete, stats, suggest_priority, suggest_date, suggest_split, overdue_summary, link, ask | get_todos, add_todo, complete_todo, update_todo, delete_todo, get_todo_stats, get_todos_by_priority, add_todo_from_natural_language, suggest_task_priority, suggest_due_date, suggest_task_splitting, get_overdue_summary, get_todos_for_note, get_notes_for_todo, link_todo_to_notes, ask_vault_about_todo |
| 8 | `admin` | health, reindex, stats, switch_model, sync_todos | health_check, sync_index, switch_embedding_model, ensure_todo_file, sync_todos |
| 9 | `tools` | (single call) | list_all_tools |

---

## Implementation Tasks

### Phase 1 — Foundation ✅

- [x] **Task 1: Create `_tool_base.py`** — Build a `build_tool()` decorator that provides logging, action dispatch foundation.

- [x] **Task 2: Create `ask.py`** — New consolidated `ask` tool with single `query: str` param. Enhanced agent prompt covers all discovery scenarios. Auto-routes to 20 internal capabilities.

- [x] **Task 3: Create `notes.py` (new)** — Consolidated notes tool. 7 actions: read / write / list / list_folder / search_by_tags / read_by_title / add_note_to_subject.

- [x] **Task 4: Create `tags.py`** — Consolidated tag management. 5 actions: add / remove / set / batch_add / auto_suggest.

- [x] **Task 5: Create `links.py`** — Consolidated wiki-link tool. 4 actions: create / backlinks / outgoing / broken.

- [x] **Task 6: Create `graph.py` (new)** — Consolidated graph exploration. 8 actions: communities / community_of / orphans / path / stats / related / traverse / export.

- [x] **Task 7: Create `entities.py`** — Consolidated entity management. 13 actions: search / note_entities / list / aliases / timeline / related / add / merge / change_type / types / weights_get / weights_set / import.

- [x] **Task 8: Create `todo.py` (new)** — Consolidated todo tool. 12 actions: list / add / complete / update / delete / stats / suggest_priority / suggest_date / suggest_split / overdue_summary / link / ask.

- [x] **Task 9: Create `admin.py`** — Consolidated admin tool. 5 actions: health / reindex / stats / switch_model / sync_todos.

- [x] **Task 10: Create `tools.py`** — Tool discovery. Same as old `list_all_tools`.

### Phase 2 — Connect ✅

- [x] **Task 11: Update `tools/__init__.py`** — Replace old module list with new 9 tool modules. Use `TOOL_MODULES` in `_tool_base.py` to avoid circular imports.

### Phase 3 — Cleanup

- [ ] **Task 12: Remove old tool files** — Delete unused `search.py`, old `notes.py`, old `graph.py`, old `todos.py`, `misc.py`. **Important:** `_shared.py` must be kept.

- [ ] **Task 13: Remove orphaned imports** — Scan entire codebase for imports referencing deleted tool functions. Update `pipelines.py` to use new consolidated tools.

- [ ] **Task 14: Update docs** — Update `docs/` and any README references listing the old 50 tools.

### Phase 4 — Test

- [ ] **Task 15: Verify tool registration** — Run the MCP server and confirm only 9 tools are registered. ✅ *(9 tools confirmed)*

- [ ] **Task 16: Test `ask` with diverse queries** — Verify it correctly routes to: semantic search, entity lookup, Q&A, timeline, graph communities, backlinks, related notes, summary, etc.

- [ ] **Task 17: Test each action on every tool** — Ensure every action in notes/tags/links/graph/entities/todo/admin works identically to the old separate tools.

- [ ] **Task 18: Test backward compatibility** — Check that existing `.env`, config, Obsidian vault access, and data files are unaffected.
