"""Comprehensive tests for all consolidated MCP tool actions.
Covers ask tool routing, all untested actions, and backward compatibility patterns."""
import json
from unittest.mock import patch, MagicMock

from obsidian_ai import mcp_server

# =========================================================================
# ask tool — universal discovery routing
# =========================================================================


@patch("obsidian_ai.llm_client")
@patch("obsidian_ai.tools.ask.pipelines")
def test_ask_search_capability(mock_pipelines, mock_llm):
    """ask routes to semantic search via LLM intent detection."""
    mock_llm.chat.return_value = '{"capability": "search", "params": {"query": "machine learning", "n": 5}}'
    with patch("obsidian_ai.tools._shared._hybrid_search") as mock_search:
        mock_search.return_value = [{"path": "ml.md", "similarity_score": 0.95}]
        result = mcp_server.ask(query="find notes about machine learning")
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["path"] == "ml.md"


@patch("obsidian_ai.llm_client")
@patch("obsidian_ai.tools.ask.pipelines")
def test_ask_qa_capability(mock_pipelines, mock_llm):
    """ask routes to Q&A (pipelines.query) when appropriate."""
    mock_llm.chat.return_value = '{"capability": "ask", "params": {"question": "What do I know about ESP32?"}}'
    mock_pipelines.query.return_value = "You have 3 notes about ESP32."
    result = mcp_server.ask(query="What do I know about ESP32?")
    assert "3 notes" in result
    mock_pipelines.query.assert_called_once_with(ask="What do I know about ESP32?")


@patch("obsidian_ai.llm_client")
@patch("obsidian_ai.tools.ask.pipelines")
def test_ask_summary_capability(mock_pipelines, mock_llm):
    """ask routes to topic summarization."""
    mock_llm.chat.return_value = '{"capability": "summary", "params": {"topic": "neural networks"}}'
    mock_pipelines.summarize_topic.return_value = "Summary of neural network notes."
    result = mcp_server.ask(query="summarize neural networks")
    assert "Summary" in result


@patch("obsidian_ai.llm_client")
@patch("obsidian_ai.entity_store")
def test_ask_entity_capability(mock_entity_store, mock_llm):
    """ask routes to entity lookup."""
    mock_llm.chat.return_value = '{"capability": "entity", "params": {"entity_name": "ESP32"}}'
    mock_entity_store.search.return_value = [
        {"path": "esp32.md", "entity_name": "ESP32", "entity_type": "Hardware",
         "snippet": "using ESP32", "confidence": 0.95}
    ]
    result = mcp_server.ask(query="find notes about ESP32")
    data = json.loads(result)
    assert data[0]["entity_name"] == "ESP32"


@patch("obsidian_ai.llm_client")
@patch("obsidian_ai.entity_store")
def test_ask_entity_timeline(mock_entity_store, mock_llm):
    """ask routes to entity timeline."""
    mock_llm.chat.return_value = '{"capability": "entity_timeline", "params": {"name": "ESP32"}}'
    mock_entity_store.get_timeline.return_value = [
        {"date": "2024-01-01", "event": "Started ESP32 project", "note": "Notes/esp32.md"}
    ]
    result = mcp_server.ask(query="timeline of ESP32")
    assert "Timeline" in result
    assert "ESP32" in result


@patch("obsidian_ai.llm_client")
@patch("obsidian_ai.entity_relations")
def test_ask_related_entities(mock_relations, mock_llm):
    """ask routes to related entity discovery."""
    mock_llm.chat.return_value = '{"capability": "related_entities", "params": {"name": "Python"}}'
    mock_relations.get_related.return_value = [
        {"entity_name": "Django", "relation_type": "uses", "confidence": 0.9}
    ]
    result = mcp_server.ask(query="what is Python related to")
    assert "Django" in result


@patch("obsidian_ai.llm_client")
@patch("obsidian_ai.graph_store")
@patch("obsidian_ai.obsidian_client")
def test_ask_backlinks(mock_obsidian, mock_graph, mock_llm):
    """ask routes to backlink lookup."""
    mock_llm.chat.return_value = '{"capability": "backlinks", "params": {"path": "Notes/main.md"}}'
    mock_graph.get_backlinks.return_value = ["Notes/a.md", "Notes/b.md"]
    result = mcp_server.ask(query="what links to main")
    data = json.loads(result)
    assert len(data) == 2


@patch("obsidian_ai.llm_client")
@patch("obsidian_ai.graph_store")
def test_ask_communities(mock_graph, mock_llm):
    """ask routes to community detection."""
    mock_llm.chat.return_value = '{"capability": "communities", "params": {}}'
    mock_graph.label_propagation.return_value = {
        "Notes/a.md": "0", "Notes/b.md": "0", "Notes/c.md": "1"
    }
    result = mcp_server.ask(query="show me note communities")
    assert "2 communities" in result


@patch("obsidian_ai.llm_client")
@patch("obsidian_ai.graph_store")
@patch("obsidian_ai.obsidian_client")
def test_ask_broken_links(mock_obsidian, mock_graph, mock_llm):
    """ask routes to broken link detection."""
    mock_llm.chat.return_value = '{"capability": "broken_links", "params": {}}'
    mock_obsidian.list_all_notes.return_value = []
    mock_graph.get_broken_links.return_value = [
        {"source_path": "a.md", "link_target": "missing.md"}
    ]
    result = mcp_server.ask(query="find broken links")
    data = json.loads(result)
    assert data[0]["link_target"] == "missing.md"


@patch("obsidian_ai.llm_client")
def test_ask_fallback_on_bad_json(mock_llm):
    """ask falls back to Q&A when LLM returns unparseable JSON."""
    mock_llm.chat.return_value = "I don't know what you mean"
    with patch("obsidian_ai.tools.ask.pipelines.query") as mock_q:
        mock_q.return_value = "Fallback answer"
        result = mcp_server.ask(query="something vague")
        assert result == "Fallback answer"


@patch("obsidian_ai.llm_client")
def test_ask_stats_capability(mock_llm):
    """ask routes to index stats."""
    mock_llm.chat.return_value = '{"capability": "stats", "params": {}}'
    with (
        patch("obsidian_ai.chroma_store.get_index_stats") as mock_stats,
        patch("obsidian_ai.llm_client.embed_cache_info") as mock_cache,
        patch("obsidian_ai.entity_store.stats") as mock_ent,
    ):
        mock_stats.return_value = {"total_chunks": 100, "unique_notes": 50}
        mock_cache.return_value = {"currsize": 10, "maxsize": 1000, "hits": 5, "misses": 3}
        mock_ent.return_value = {"total_entities": 20, "total_mentions": 45}
        result = mcp_server.ask(query="vault stats")
        assert "100" in result
        assert "50" in result


# =========================================================================
# notes tool — all actions
# =========================================================================


@patch("obsidian_ai.chroma_store")
@patch("obsidian_ai.tools.notes.obsidian_client")
def test_notes_search_by_tags(mock_obsidian, mock_chroma):
    mock_chroma.search_by_tags.return_value = [
        {"path": "a.md", "title": "A", "tags_str": ",python,", "snippet": "content"},
        {"path": "b.md", "title": "B", "tags_str": ",python,", "snippet": "content"},
    ]
    result = mcp_server.notes(action="search_by_tags", tags=["python"])
    data = json.loads(result)
    assert len(data) == 2
    assert data[0]["path"] == "a.md"


@patch("obsidian_ai.chroma_store")
@patch("obsidian_ai.tools.notes.obsidian_client")
def test_notes_read_by_title(mock_obsidian, mock_chroma):
    mock_chroma.get_by_title.return_value = [{"path": "Notes/Found.md"}]
    mock_obsidian.get_note.return_value = "# Found\nDiscovered!"
    result = mcp_server.notes(action="read_by_title", title="Found")
    assert "Discovered" in result


@patch("obsidian_ai.chroma_store")
@patch("obsidian_ai.tools.notes.obsidian_client")
def test_notes_read_by_title_not_found(mock_obsidian, mock_chroma):
    mock_chroma.get_by_title.return_value = []
    result = mcp_server.notes(action="read_by_title", title="Missing")
    assert "No note found" in result


@patch("obsidian_ai.tools.notes.obsidian_client")
@patch("obsidian_ai.entity_store")
@patch("obsidian_ai.graph_store")
@patch("obsidian_ai.tools.notes.indexer")
def test_notes_add_note_to_subject(mock_indexer, mock_graph, mock_entity, mock_obsidian):
    mock_obsidian.get_note.side_effect = Exception("not found")  # hub doesn't exist yet
    result = mcp_server.notes(
        action="add_note_to_subject",
        subject="ESP32",
        title="Getting Started",
        content="Notes about ESP32",
    )
    assert "Note created" in result
    assert "Subjects/ESP32/Getting Started.md" in result
    calls = mock_obsidian.put_note.call_args_list
    paths = [c[0][0] for c in calls]
    assert "Subjects/ESP32/ESP32.md" in paths  # hub created
    assert "Subjects/ESP32/Getting Started.md" in paths  # note created


# =========================================================================
# tags tool — remaining actions
# =========================================================================


@patch("obsidian_ai.tools.tags.obsidian_client")
@patch("obsidian_ai.tools.tags.indexer")
def test_tags_batch_add(mock_indexer, mock_obsidian):
    mock_obsidian.get_note.side_effect = [
        "---\ntags: []\n---\nBody A",
        "---\ntags: []\n---\nBody B",
    ]
    result = mcp_server.tags(action="batch_add", note_paths=["a.md", "b.md"], tags=["python"])
    data = json.loads(result)
    assert "a.md" in data
    assert "b.md" in data


@patch("obsidian_ai.pipelines")
@patch("obsidian_ai.tools.tags.indexer")
def test_tags_auto_suggest(mock_indexer, mock_pipelines):
    mock_pipelines.tag_notes.return_value = 'Tagged: {"a.md": ["ml"]}'
    result = mcp_server.tags(action="auto_suggest", query="machine learning")
    assert "Tagged" in result
    mock_pipelines.tag_notes.assert_called_once_with(ask="machine learning", top_k=5)


# =========================================================================
# entities tool — remaining actions
# =========================================================================


@patch("obsidian_ai.tools.entities.entity_store")
def test_entities_list(mock_entity_store):
    mock_entity_store.list_entities.return_value = [
        {"name": "ESP32", "type": "Hardware", "confidence": 0.95, "note_count": 5},
        {"name": "Python", "type": "Technology", "confidence": 0.9, "note_count": 12},
    ]
    result = mcp_server.entities(action="list")
    data = json.loads(result)
    assert len(data) == 2
    assert data[0]["name"] == "ESP32"


@patch("obsidian_ai.tools.entities.entity_store")
def test_entities_list_filtered(mock_entity_store):
    mock_entity_store.list_entities.return_value = [{"name": "Alice", "type": "Person", "note_count": 3}]
    result = mcp_server.entities(action="list", entity_type="Person")
    data = json.loads(result)
    assert data[0]["type"] == "Person"


@patch("obsidian_ai.tools.entities.entity_store")
def test_entities_aliases_found(mock_entity_store):
    mock_entity_store.get_aliases.return_value = {
        "canonical": "ESP32", "type": "Hardware",
        "aliases": ["esp32", "ESP-32"], "mention_count": 10,
    }
    result = mcp_server.entities(action="aliases", name="ESP32")
    assert "esp32" in result
    assert "ESP-32" in result


@patch("obsidian_ai.tools.entities.entity_store")
def test_entities_aliases_not_found(mock_entity_store):
    mock_entity_store.get_aliases.return_value = None
    result = mcp_server.entities(action="aliases", name="Ghost")
    assert "Entity not found" in result


@patch("obsidian_ai.tools.entities.entity_store")
def test_entities_timeline(mock_entity_store):
    mock_entity_store.get_timeline.return_value = [
        {"date": "2024-01", "event": "Started project", "note": "Notes/project.md"},
        {"date": "2024-06", "event": "Completed milestone", "note": "Notes/project.md"},
    ]
    result = mcp_server.entities(action="timeline", name="ProjectX")
    assert "Timeline" in result
    assert "Started project" in result
    assert "Completed milestone" in result


@patch("obsidian_ai.tools.entities.entity_store")
def test_entities_timeline_not_found(mock_entity_store):
    mock_entity_store.get_timeline.return_value = None
    result = mcp_server.entities(action="timeline", name="Ghost")
    assert "not found" in result


@patch("obsidian_ai.tools.entities.entity_store")
def test_entities_timeline_empty(mock_entity_store):
    mock_entity_store.get_timeline.return_value = []
    result = mcp_server.entities(action="timeline", name="NewEntity")
    assert "No timeline entries" in result


@patch("obsidian_ai.tools.entities.entity_relations")
def test_entities_related(mock_relations):
    mock_relations.get_related.return_value = [
        {"entity_name": "Django", "relation_type": "uses", "confidence": 0.9, "depth": 1}
    ]
    result = mcp_server.entities(action="related", name="Python")
    assert "Django" in result
    assert "uses" in result


@patch("obsidian_ai.tools.entities.entity_relations")
def test_entities_related_not_found(mock_relations):
    mock_relations.get_related.return_value = []
    result = mcp_server.entities(action="related", name="Unknown")
    assert "No related entities" in result


@patch("obsidian_ai.tools.entities.entity_store")
@patch("obsidian_ai.tools.entities.entity_relations")
@patch("obsidian_ai.tools.entities.indexer")
def test_entities_add(mock_indexer, mock_relations, mock_entity_store):
    mock_entity_store.add_manual_entity.return_value = {
        "entity_name": "NewProject",
        "entity_type": "Project",
        "aliases": ["NP"],
        "mention_count": 0,
    }
    result = mcp_server.entities(action="add", name="NewProject", entity_type="Project", aliases=["NP"])
    assert "Entity added" in result
    assert "NewProject" in result
    assert "Project" in result


@patch("obsidian_ai.tools.entities.entity_store")
def test_entities_merge(mock_entity_store):
    mock_entity_store.merge.return_value = {
        "canonical": "ESP32", "type": "Hardware",
        "aliases": ["esp32", "esp-32"], "mention_count": 15,
    }
    result = mcp_server.entities(action="merge", primary="ESP32", secondary="esp_32")
    assert "Merged" in result
    assert "ESP32" in result


@patch("obsidian_ai.tools.entities.entity_store")
def test_entities_merge_not_found(mock_entity_store):
    mock_entity_store.merge.return_value = None
    result = mcp_server.entities(action="merge", primary="Real", secondary="Ghost")
    assert "Could not merge" in result


@patch("obsidian_ai.tools.entities.entity_store")
def test_entities_change_type(mock_entity_store):
    mock_entity_store.change_entity_type.return_value = {
        "entity_name": "ESP32", "entity_type": "Technology", "mention_count": 10,
    }
    result = mcp_server.entities(action="change_type", name="ESP32", new_type="Technology")
    assert "Entity type changed" in result
    assert "Technology" in result


@patch("obsidian_ai.tools.entities.entity_store")
def test_entities_change_type_not_found(mock_entity_store):
    mock_entity_store.change_entity_type.return_value = None
    result = mcp_server.entities(action="change_type", name="Ghost", new_type="Person")
    assert "not found" in result


@patch("obsidian_ai.ranker")
def test_entities_weights_get(mock_ranker):
    mock_ranker.weights.return_value = {"semantic": 0.4, "entity": 0.3, "graph": 0.2, "keyword": 0.1}
    result = mcp_server.entities(action="weights_get")
    assert "0.40" in result
    assert "0.30" in result


@patch("obsidian_ai.ranker")
def test_entities_weights_set(mock_ranker):
    mock_ranker.set_weights.return_value = {"semantic": 0.5, "entity": 0.3, "graph": 0.1, "keyword": 0.1}
    result = mcp_server.entities(action="weights_set", semantic=0.5)
    assert "Updated" in result
    assert "0.50" in result
    mock_ranker.set_weights.assert_called_once_with(semantic=0.5, entity=None, graph=None, keyword=None)


@patch("obsidian_ai.tools.entities.entity_resolver.EntityResolver")
def test_entities_import(mock_resolver_cls):
    mock_resolver = MagicMock()
    mock_resolver_cls.return_value = mock_resolver
    mock_resolver.resolve.return_value = {
        "total_incoming": 5, "merged": 2, "added": 3, "skipped": 0, "relations_added": 1,
    }
    data = json.dumps([{"name": "Entity1", "type": "Person"}])
    result = mcp_server.entities(action="import", data=data)
    assert "Entity import complete" in result
    assert "Merged:" in result
    assert "Added:" in result
    assert "2" in result
    assert "3" in result


# =========================================================================
# links tool — remaining actions
# =========================================================================


@patch("obsidian_ai.tools.links.graph_store")
def test_links_backlinks(mock_graph):
    mock_graph.get_backlinks.return_value = ["Notes/a.md", "Notes/b.md"]
    result = mcp_server.links(action="backlinks", path="Notes/main.md")
    data = json.loads(result)
    assert len(data) == 2
    assert data[0]["path"] == "Notes/a.md"


@patch("obsidian_ai.tools.links.graph_store")
def test_links_outgoing(mock_graph):
    mock_graph.get_outgoing.return_value = ["Notes/target1.md", "Notes/target2.md"]
    result = mcp_server.links(action="outgoing", path="Notes/main.md")
    data = json.loads(result)
    assert len(data) == 2
    assert data[0]["path"] == "Notes/target1.md"


@patch("obsidian_ai.tools.links.graph_store")
@patch("obsidian_ai.tools.links.obsidian_client")
def test_links_broken(mock_obsidian, mock_graph):
    mock_obsidian.list_all_notes.return_value = ["a.md", "b.md"]
    mock_obsidian.get_note.return_value = "[[missing]]"
    mock_graph.get_broken_links.return_value = [
        {"source_path": "a.md", "link_target": "missing.md"}
    ]
    result = mcp_server.links(action="broken")
    data = json.loads(result)
    assert data[0]["link_target"] == "missing.md"


# =========================================================================
# graph tool — all actions
# =========================================================================


@patch("obsidian_ai.tools.graph.graph_store")
def test_graph_communities(mock_graph):
    mock_graph.label_propagation.return_value = {"Notes/a.md": "0", "Notes/b.md": "0", "Notes/c.md": "1"}
    result = mcp_server.graph(action="communities")
    data = json.loads(result)
    assert "0" in data
    assert "1" in data


@patch("obsidian_ai.tools.graph.graph_store")
def test_graph_community_of(mock_graph):
    mock_graph.get_community_info.return_value = {"community_id": "0", "members": ["Notes/neighbor.md"]}
    result = mcp_server.graph(action="community_of", path="Notes/main.md")
    assert "Community 0" in result
    assert "neighbor" in result


@patch("obsidian_ai.tools.graph.graph_store")
def test_graph_community_of_not_found(mock_graph):
    mock_graph.get_community_info.return_value = None
    result = mcp_server.graph(action="community_of", path="Notes/alone.md")
    assert "does not belong" in result


@patch("obsidian_ai.tools.graph.graph_store")
def test_graph_orphans(mock_graph):
    mock_graph.get_orphans.return_value = ["Notes/orphan1.md", "Notes/orphan2.md"]
    result = mcp_server.graph(action="orphans")
    data = json.loads(result)
    assert len(data) == 2
    assert "orphan1" in result


@patch("obsidian_ai.tools.graph.graph_store")
def test_graph_path_found(mock_graph):
    mock_graph.shortest_path.return_value = ["Notes/a.md", "Notes/b.md", "Notes/c.md"]
    result = mcp_server.graph(action="path", start="Notes/a.md", end="Notes/c.md")
    assert "Shortest path" in result
    assert "2 hops" in result


@patch("obsidian_ai.tools.graph.graph_store")
def test_graph_path_not_found(mock_graph):
    mock_graph.shortest_path.return_value = None
    result = mcp_server.graph(action="path", start="Notes/a.md", end="Notes/z.md")
    assert "No path found" in result


@patch("obsidian_ai.tools.graph.graph_store")
def test_graph_stats(mock_graph):
    mock_graph.stats.return_value = {"nodes": 10, "edges": 15, "avg_degree": 3.0}
    result = mcp_server.graph(action="stats")
    data = json.loads(result)
    assert data["nodes"] == 10
    assert data["edges"] == 15


@patch("obsidian_ai.tools.graph.graph_store")
@patch("obsidian_ai.tools.graph.obsidian_client")
@patch("obsidian_ai.tools.graph._hybrid_search")
def test_graph_related(mock_search, mock_obsidian, mock_graph):
    mock_obsidian.get_note.return_value = "# Source\nContent here"
    mock_graph.get_outgoing.return_value = ["Notes/neighbor.md"]
    mock_graph.get_backlinks.return_value = []
    mock_search.return_value = [{"path": "Notes/neighbor.md", "title": "neighbor", "similarity_score": 0.8}]
    result = mcp_server.graph(action="related", path="Notes/main.md")
    data = json.loads(result)
    assert len(data) >= 1


@patch("obsidian_ai.tools.graph.graph_store")
def test_graph_traverse(mock_graph):
    mock_graph.bfs.return_value = {
        "Notes/b.md": ["Notes/a.md", "Notes/b.md"],
        "Notes/c.md": ["Notes/a.md", "Notes/c.md"],
    }
    result = mcp_server.graph(action="traverse", path="Notes/a.md", max_depth=2)
    data = json.loads(result)
    assert len(data) == 2


@patch("obsidian_ai.tools.graph.graph_store")
def test_graph_export_json(mock_graph):
    mock_graph.to_dict.return_value = {"nodes": ["a.md", "b.md"], "edges": [["a.md", "b.md"]]}
    result = mcp_server.graph(action="export", format="json")
    data = json.loads(result)
    assert "nodes" in data


@patch("obsidian_ai.tools.graph.graph_store")
def test_graph_export_dot(mock_graph):
    mock_graph.to_dot.return_value = 'digraph G { "a.md" -> "b.md"; }'
    result = mcp_server.graph(action="export", format="dot")
    assert 'digraph' in result


# =========================================================================
# todo tool — all actions
# =========================================================================


@patch("obsidian_ai.tools.todo._impl")
def test_todo_list(mock_impl):
    mock_impl.get_todos.return_value = [
        {"id": "1", "task": "Review PR", "project": "Work", "status": "pending"},
        {"id": "2", "task": "Buy groceries", "project": "Personal", "status": "pending"},
    ]
    result = mcp_server.todo(action="list")
    data = json.loads(result)
    assert len(data) == 2
    assert data[0]["task"] == "Review PR"


@patch("obsidian_ai.tools.todo._impl")
@patch("obsidian_ai.tools.todo.indexer")
def test_todo_add(mock_indexer, mock_impl):
    mock_impl.add_todo.return_value = {"success": True, "id": "42", "project": "Work"}
    result = mcp_server.todo(action="add", project="Work", task="Fix bug")
    data = json.loads(result)
    assert data["success"] is True
    assert data["id"] == "42"


@patch("obsidian_ai.tools.todo._impl")
@patch("obsidian_ai.tools.todo.indexer")
def test_todo_complete(mock_indexer, mock_impl):
    mock_impl.complete_todo.return_value = {"success": True, "id": "42"}
    result = mcp_server.todo(action="complete", todo_id="42")
    data = json.loads(result)
    assert data["success"] is True


@patch("obsidian_ai.tools.todo._impl")
def test_todo_complete_not_found(mock_impl):
    mock_impl.complete_todo.return_value = None
    result = mcp_server.todo(action="complete", todo_id="999")
    data = json.loads(result)
    assert "error" in data


@patch("obsidian_ai.tools.todo._impl")
@patch("obsidian_ai.tools.todo.indexer")
def test_todo_update(mock_indexer, mock_impl):
    mock_impl.update_todo.return_value = {"success": True, "id": "42", "status": "completed"}
    result = mcp_server.todo(action="update", todo_id="42", status="completed")
    data = json.loads(result)
    assert data["success"] is True


@patch("obsidian_ai.tools.todo._impl")
def test_todo_update_not_found(mock_impl):
    mock_impl.update_todo.return_value = None
    result = mcp_server.todo(action="update", todo_id="999", task="new desc")
    data = json.loads(result)
    assert "error" in data


@patch("obsidian_ai.tools.todo._impl")
@patch("obsidian_ai.tools.todo.indexer")
def test_todo_delete(mock_indexer, mock_impl):
    mock_impl.delete_todo.return_value = True
    result = mcp_server.todo(action="delete", todo_id="42")
    data = json.loads(result)
    assert data["success"] is True


@patch("obsidian_ai.tools.todo._impl")
def test_todo_delete_not_found(mock_impl):
    mock_impl.delete_todo.return_value = False
    result = mcp_server.todo(action="delete", todo_id="999")
    data = json.loads(result)
    assert data["success"] is False


@patch("obsidian_ai.tools.todo._impl")
def test_todo_stats(mock_impl):
    mock_impl.get_todo_stats.return_value = {"total": 10, "completed": 4, "pending": 6}
    result = mcp_server.todo(action="stats")
    data = json.loads(result)
    assert data["total"] == 10
    assert data["pending"] == 6


@patch("obsidian_ai.tools.todo._impl")
def test_todo_suggest_priority(mock_impl):
    mock_impl.suggest_task_priority.return_value = "high"
    result = mcp_server.todo(action="suggest_priority", task="Fix production outage")
    assert result == "high"


@patch("obsidian_ai.tools.todo._impl")
def test_todo_suggest_date(mock_impl):
    mock_impl.suggest_due_date.return_value = "2026-06-10"
    result = mcp_server.todo(action="suggest_date", task="Submit report")
    assert result == "2026-06-10"


@patch("obsidian_ai.tools.todo._impl")
def test_todo_suggest_split(mock_impl):
    mock_impl.suggest_task_splitting.return_value = ["Step 1", "Step 2", "Step 3"]
    result = mcp_server.todo(action="suggest_split", task="Build feature X")
    data = json.loads(result)
    assert len(data) == 3


@patch("obsidian_ai.tools.todo._impl")
def test_todo_overdue_summary(mock_impl):
    mock_impl.get_overdue_summary.return_value = {"overdue": 3, "tasks": ["Fix bugs", "Write docs"]}
    result = mcp_server.todo(action="overdue_summary")
    assert "overdue" in result


@patch("obsidian_ai.tools.todo._impl")
@patch("obsidian_ai.tools.todo.indexer")
def test_todo_link(mock_indexer, mock_impl):
    mock_impl.link_todo_to_notes.return_value = {"success": True, "links": 2}
    result = mcp_server.todo(action="link", todo_id="42", note_paths=["a.md", "b.md"])
    data = json.loads(result)
    assert data["success"] is True


@patch("obsidian_ai.tools.todo._impl")
def test_todo_ask_by_id(mock_impl):
    mock_impl.ask_vault_about_todo.return_value = "This todo is about fixing the login bug."
    result = mcp_server.todo(action="ask", todo_id="42")
    assert "login bug" in result


@patch("obsidian_ai.tools.todo._impl")
def test_todo_ask_query(mock_impl):
    mock_impl.ask_vault_about_todos.return_value = "You have 5 overdue tasks."
    result = mcp_server.todo(action="ask", query="What's overdue?")
    assert "5 overdue" in result


# =========================================================================
# admin tool — remaining actions
# =========================================================================


@patch("obsidian_ai.tools.admin.indexer")
@patch("obsidian_ai.tools.admin.entity_store")
@patch("obsidian_ai.tools.admin.llm_client")
@patch("obsidian_ai.tools.admin.keyword_search")
def test_admin_reindex(mock_keyword, mock_llm, mock_entity_store, mock_indexer):
    result = mcp_server.admin(action="reindex")
    assert "Index sync complete" in result


@patch("obsidian_ai.tools.admin.indexer")
@patch("obsidian_ai.tools.admin.entity_store")
@patch("obsidian_ai.tools.admin.llm_client")
@patch("obsidian_ai.tools.admin.keyword_search")
def test_admin_reindex_subject(mock_keyword, mock_llm, mock_entity_store, mock_indexer):
    mock_entity_store.search.return_value = [
        {"path": "a.md", "confidence": 0.9},
        {"path": "b.md", "confidence": 0.8},
    ]
    mock_indexer.index_note.return_value = True
    result = mcp_server.admin(action="reindex", subject="ESP32")
    assert "Re-indexed 2 notes" in result


@patch("obsidian_ai.tools.admin.indexer")
@patch("obsidian_ai.tools.admin.entity_store")
@patch("obsidian_ai.tools.admin.llm_client")
@patch("obsidian_ai.tools.admin.keyword_search")
def test_admin_reindex_subject_not_found(mock_keyword, mock_llm, mock_entity_store, mock_indexer):
    mock_entity_store.search.return_value = []
    result = mcp_server.admin(action="reindex", subject="Ghost")
    assert "No notes found" in result


@patch("obsidian_ai.tools.admin.requests")
@patch("obsidian_ai.tools.admin.llm_client")
@patch("obsidian_ai.tools.admin.chroma_store")
@patch("obsidian_ai.tools.admin.indexer")
@patch("obsidian_ai.tools.admin.keyword_search")
@patch("obsidian_ai.tools.admin.entity_store")
def test_admin_switch_model(mock_entity_store, mock_keyword, mock_indexer, mock_chroma, mock_llm, mock_requests):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_requests.post.return_value = mock_resp
    result = mcp_server.admin(action="switch_model", model_name="new-embed-model")
    assert "Switched" in result
    assert "new-embed-model" in result
    mock_chroma.reset_collection.assert_called_once()


@patch("obsidian_ai.tools.admin.requests")
def test_admin_switch_model_not_found(mock_requests):
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_requests.post.return_value = mock_resp
    result = mcp_server.admin(action="switch_model", model_name="nonexistent")
    assert "not found" in result


@patch("obsidian_ai.tools.admin._impl")
@patch("obsidian_ai.tools.admin.indexer")
def test_admin_sync_todos(mock_indexer, mock_impl):
    mock_impl.sync_todos.return_value = {"success": True, "projects": {"Work": 3}}
    result = mcp_server.admin(action="sync_todos")
    data = json.loads(result)
    assert data["success"] is True


# =========================================================================
# Invalid action handling (all tools)
# =========================================================================


def test_all_tools_invalid_action():
    assert "Error" in mcp_server.notes(action="fly")
    assert "Error" in mcp_server.tags(action="fly")
    assert "Error" in mcp_server.links(action="fly")
    assert "Error" in mcp_server.graph(action="fly")
    assert "Error" in mcp_server.entities(action="fly")
    assert "Error" in mcp_server.todo(action="fly")
    assert "Error" in mcp_server.admin(action="fly")


# =========================================================================
# tools() — self-discovery introspection
# =========================================================================


def test_tools_self_discovery():
    result = mcp_server.tools()
    data = json.loads(result)
    names = [t["name"] for t in data]
    assert "ask" in names
    assert "notes" in names
    assert "tags" in names
    assert "links" in names
    assert "graph" in names
    assert "entities" in names
    assert "todo" in names
    assert "admin" in names
    assert "tools" in names
    assert len(data) == 9, f"Expected 9 tools, got {len(data)}"


# =========================================================================
# Backward compatibility — module re-exports, path normalization
# =========================================================================


def test_backward_compat_all_tools_reexported():
    """All 9 tools are re-exported from mcp_server module."""
    assert hasattr(mcp_server, "ask")
    assert hasattr(mcp_server, "notes")
    assert hasattr(mcp_server, "tags")
    assert hasattr(mcp_server, "links")
    assert hasattr(mcp_server, "graph")
    assert hasattr(mcp_server, "entities")
    assert hasattr(mcp_server, "todo")
    assert hasattr(mcp_server, "admin")
    assert hasattr(mcp_server, "tools")


def test_backward_compat_shared_reexported():
    """Shared helpers are still re-exported from mcp_server for backward compat."""
    assert hasattr(mcp_server, "_normalize_path")
    assert hasattr(mcp_server, "_expand_query")
    assert hasattr(mcp_server, "_hybrid_search")
    assert hasattr(mcp_server, "_truncate_snippet")
    assert hasattr(mcp_server, "_build_search_where")
    assert hasattr(mcp_server, "_matches_where")
    assert hasattr(mcp_server, "_get_vault_terminology")
    assert hasattr(mcp_server, "_group_by_note")
    assert hasattr(mcp_server, "_rewrite_query")
    assert hasattr(mcp_server, "_EXPAND_QUERY_CACHE")
