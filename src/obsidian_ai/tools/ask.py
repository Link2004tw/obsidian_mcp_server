"""Universal discovery tool — understands any vault query and routes it internally."""

import json
import re

from .. import pipelines
from ..logger import get_logger, log_error
from ._tool_base import build_tool

log = get_logger("obsidian_ai.tools.ask")

_AGENT_SYSTEM = """You are an intelligent routing agent for an Obsidian vault AI assistant.
Given a user's natural language query, decide which capability to use and with what parameters.

Available capabilities:

1. **search** — Semantic + keyword search across all notes.
   Use for: "find notes about X", "what does my vault say about Y", "show me notes related to Z"
   Returns matching passages with similarity scores.

2. **ask** — Direct Q&A using RAG (retrieve relevant notes and answer with LLM).
   Use for: "what do I know about X", "what did I write about Y", direct questions about vault content

3. **summary** — LLM-generated consolidated summary of a topic across multiple notes.
   Use for: "summarize what I have about X", "give me an overview of Y"

4. **entity** — Find notes mentioning a specific named entity (person, project, hardware, etc.).
   Entity types: Person, Project, Hardware, Technology, Location, Concept, Event.
   Use for: "what notes mention Alice", "find everything about ESP32", "what do I have on Project X"

5. **entity_timeline** — Chronological timeline of events for a person or project.
   Use for: "what happened with X", "timeline of events for Y"

6. **related_entities** — Discover entities connected to a given entity (works_on, uses, part_of, etc.).
   Use for: "what is X related to", "who works on Y", "what projects use Z"

7. **related_notes** — Find semantically similar or wiki-link-connected notes to a given note.
   Use for: "what notes are similar to X", "find related notes to Y"

8. **backlinks** — List all notes that link TO a given note.
   Use for: "what notes reference X", "which pages link to Y"

9. **outgoing** — List all notes that a given note links TO.
   Use for: "what does this note link to", "outgoing links from X"

10. **communities** — Detect densely connected groups of notes in the wiki-link graph.
    Use for: "show me note communities", "how are my notes clustered topically"

11. **community_of** — Find which community a note belongs to and its neighbors.
    Use for: "what community is X in", "what is X connected to"

12. **orphans** — Find notes with no wiki-links (disconnected).
    Use for: "find orphan notes", "which notes are isolated"

13. **path** — Find shortest wiki-link path between two notes.
    Use for: "how is X connected to Y", "find path from A to B"

14. **graph_stats** — Return summary statistics about the wiki-link graph.
    Use for: "vault graph stats", "how many links in my vault", "vault connectivity"

15. **subject** — Explore notes about a broad subject/person/concept using LLM expansion.
    Use for: "tell me about X", "explore subject Y"

16. **stats** — Get diagnostic stats about the index (note count, chunk count, entities).
    Use for: "index stats", "how many notes are indexed", "vault status"

17. **clusters** — Group notes into semantic topic clusters.
    Use for: "cluster my notes", "show topic groups", "thematic groupings"

18. **tags** — Auto-suggest YAML frontmatter tags for notes matching a query.
    Use for: "suggest tags for notes about X"

19. **broken_links** — Find wiki-links pointing to non-existent notes.
    Use for: "find broken links", "dead links in my vault"

20. **graph_export** — Export the wiki-link graph as JSON or DOT.
    Use for: "export graph", "visualize vault structure"

Return your decision as JSON with this exact structure:
{"capability": "capability_name", "params": {"param1": "value1", "param2": "value2"}}

Only include relevant params. Use reasonable defaults for omitted params."""


def _route(query: str) -> str:
    """Route a query to the appropriate internal capability via LLM agent."""
    from .. import llm_client

    messages = [
        {"role": "system", "content": _AGENT_SYSTEM},
        {"role": "user", "content": query},
    ]
    try:
        response = llm_client.chat(messages, think=False)
        decision = json.loads(response)
        if not isinstance(decision, dict) or "capability" not in decision:
            raise ValueError("Unexpected format")
    except (json.JSONDecodeError, ValueError):
        match = re.search(r"\{[^{}]+\}", response, re.DOTALL)
        if match:
            try:
                decision = json.loads(match.group())
            except json.JSONDecodeError:
                decision = {"capability": "ask", "params": {"question": query}}
        else:
            decision = {"capability": "ask", "params": {"question": query}}

    capability = decision.get("capability", "ask")
    params = decision.get("params", {})

    normalizers = {
        "search": {"query": ["query", "q", "search", "question"]},
        "ask": {"question": ["question", "query", "ask"]},
        "summary": {"topic": ["topic", "subject", "query", "question"]},
        "entity": {"entity_name": ["entity_name", "name", "entity", "query"]},
        "entity_timeline": {"name": ["name", "entity", "entity_name"]},
        "related_entities": {"name": ["name", "entity"]},
        "related_notes": {"path": ["path", "note", "note_path"]},
        "backlinks": {"path": ["path", "note", "note_path"]},
        "outgoing": {"path": ["path", "note", "note_path"]},
        "community_of": {"path": ["path", "note", "note_path"]},
        "subject": {"subject": ["subject", "topic", "query"]},
        "path": {"start": ["start", "from"], "end": ["end", "to"]},
    }

    norm = normalizers.get(capability, {})
    for canonical, aliases in norm.items():
        if canonical not in params:
            for alias in aliases:
                if alias in params:
                    params[canonical] = params.pop(alias)
                    break

    log.info("ask — routed to %s with %s", capability, params)
    return _execute(capability, params, query)


def _execute(capability: str, params: dict, fallback_query: str) -> str:
    """Execute a capability with given params, falling back to Q&A on error."""
    from .. import (
        chroma_store,
        config,
        entity_store,
        graph_store,
        llm_client,
        obsidian_client,
        ranker,
    )
    from ._shared import _expand_query, _hybrid_search

    try:
        if capability == "search":
            query = params.get("query", fallback_query)
            n = params.get("n", 5)
            results = _hybrid_search(queries=[query], n=n)
            return json.dumps(results, ensure_ascii=False, indent=2)

        elif capability == "ask":
            return pipelines.query(ask=params.get("question", fallback_query))

        elif capability == "summary":
            return pipelines.summarize_topic(topic=params.get("topic", fallback_query))

        elif capability == "entity":
            name = params.get("entity_name", fallback_query)
            n = params.get("n", 10)
            entity_type = params.get("entity_type")
            results = entity_store.search(name, type=entity_type, n=n)
            return json.dumps(results, ensure_ascii=False, indent=2) if results else f"No notes found mentioning \"{name}\"."

        elif capability == "entity_timeline":
            name = params.get("name", fallback_query)
            entries = entity_store.get_timeline(name)
            if not entries:
                return f"No timeline entries for \"{name}\"."
            lines = [f"Timeline for {name} ({len(entries)} entries):"]
            for entry in entries:
                note = entry.get("note", "")
                title = note.rsplit("/", 1)[-1].replace(".md", "") if note else "?"
                lines.append(f"  {entry['date']}  {entry['event']}  ({title})")
            return "\n".join(lines)

        elif capability == "related_entities":
            from .. import entity_relations
            name = params.get("name", fallback_query)
            results = entity_relations.get_related(name)
            if not results:
                return f"No related entities found for \"{name}\"."
            lines = [f"Entities related to \"{name}\":"]
            for r in results:
                lines.append(f"  \u2022 {r['entity_name']}  [{r.get('relation_type', '?')}]  conf={r.get('confidence', 0):.2f}")
            return "\n".join(lines)

        elif capability == "related_notes":
            from ._shared import _hybrid_search
            path = params.get("path", "")
            if not path:
                return "No note path provided."
            content = obsidian_client.get_note(path)
            body = content.split("---", 2)[-1] if content.startswith("---") else content
            results = _hybrid_search(queries=[body[:500]], n=10, keyword_weight=0.3)
            results = [r for r in results if r["path"] != path]
            return json.dumps(results[:10], ensure_ascii=False, indent=2)

        elif capability == "backlinks":
            path = params.get("path", "")
            if not path:
                return "No note path provided."
            sources = graph_store.get_backlinks(path)
            if not sources:
                return f"No backlinks found for \"{path}\"."
            result = [{"path": s, "title": s.rsplit("/", 1)[-1].replace(".md", "")} for s in sources]
            return json.dumps(result, ensure_ascii=False, indent=2)

        elif capability == "outgoing":
            path = params.get("path", "")
            if not path:
                return "No note path provided."
            targets = graph_store.get_outgoing(path)
            if not targets:
                return f"No outgoing links found for \"{path}\"."
            result = [{"path": t, "title": t.rsplit("/", 1)[-1].replace(".md", "")} for t in targets]
            return json.dumps(result, ensure_ascii=False, indent=2)

        elif capability == "communities":
            communities = graph_store.label_propagation()
            if not communities:
                return "No communities detected."
            grouped: dict[str, list[str]] = {}
            for p, cid in communities.items():
                grouped.setdefault(str(cid), []).append(p)
            lines = [f"Found {len(grouped)} communities:"]
            for cid, members in sorted(grouped.items()):
                lines.append(f"  Community {cid}: {len(members)} members")
                for m in members[:5]:
                    lines.append(f"    - {m}")
                if len(members) > 5:
                    lines.append(f"    ... and {len(members) - 5} more")
            return "\n".join(lines)

        elif capability == "community_of":
            path = params.get("path", "")
            if not path:
                return "No note path provided."
            info = graph_store.get_community_info(path)
            if info is None:
                return f"\"{path}\" does not belong to any community."
            members = info["members"]
            lines = [f"Community {info['community_id']} — {len(members) + 1} members"]
            lines.append(f"  Self: {path}")
            for m in members[:10]:
                lines.append(f"    - {m}")
            return "\n".join(lines)

        elif capability == "orphans":
            orphans = graph_store.get_orphans()
            if not orphans:
                return "No orphan notes found."
            return f"Orphan notes ({len(orphans)}):\n" + "\n".join(f"  - {p}" for p in orphans)

        elif capability == "path":
            start = params.get("start", "")
            end = params.get("end", "")
            if not start or not end:
                return "Both start and end note paths are required."
            path_result = graph_store.shortest_path(start, end)
            if path_result is None:
                return f"No path found between \"{start}\" and \"{end}\"."
            lines = [f"Shortest path ({len(path_result) - 1} hops):"]
            for i, p in enumerate(path_result):
                marker = "\u2192" if i > 0 else " "
                lines.append(f"  {marker} {p}")
            return "\n".join(lines)

        elif capability == "graph_stats":
            stats = graph_store.stats()
            return json.dumps(stats, ensure_ascii=False, indent=2)

        elif capability == "subject":
            from ._shared import _expand_query
            subject = params.get("subject", fallback_query)
            expanded = _expand_query(subject)
            queries = [subject] + expanded
            results = _hybrid_search(queries=queries, n=10, keyword_weight=0.3)
            return json.dumps(results, ensure_ascii=False, indent=2) if results else f"No notes found about \"{subject}\"."

        elif capability == "stats":
            stats = chroma_store.get_index_stats()
            cache = llm_client.embed_cache_info()
            ent_stats = entity_store.stats()
            lines = [
                f"Unique notes indexed: {stats['unique_notes']}",
                f"Total chunks stored:  {stats['total_chunks']}",
                f"Embedding model:      {config.ollama_embed_model}",
                f"ChromaDB path:        {config.chroma_path}",
                f"Embedding cache:      {cache['currsize']}/{cache['maxsize']} (hits={cache['hits']}, misses={cache['misses']})",
                f"Entities extracted:   {ent_stats['total_entities']} ({ent_stats['total_mentions']} mentions)",
            ]
            return "\n".join(lines)

        elif capability == "clusters":
            from ..clustering import get_clusters
            clusters = get_clusters()
            return json.dumps(clusters, ensure_ascii=False, indent=2) if clusters else "No clusters found."

        elif capability == "tags":
            return pipelines.tag_notes(ask=params.get("query", fallback_query))

        elif capability == "broken_links":
            notes = obsidian_client.list_all_notes()
            all_contents: dict[str, str] = {}
            import contextlib
            for note_path in notes:
                with contextlib.suppress(Exception):
                    all_contents[note_path] = obsidian_client.get_note(note_path)
            broken = graph_store.get_broken_links(all_contents)
            if not broken:
                return "No broken links found."
            return json.dumps(broken, ensure_ascii=False, indent=2)

        elif capability == "graph_export":
            fmt = params.get("format", "json")
            if fmt == "dot":
                return graph_store.to_dot()
            data = graph_store.to_dict()
            return json.dumps(data, ensure_ascii=False, indent=2)

        else:
            log.warning("ask — unknown capability %s, falling back to Q&A", capability)
            return pipelines.query(ask=fallback_query)

    except Exception as e:
        log_error(log, f"ask — {capability} failed", exc=e)
        return pipelines.query(ask=fallback_query)


@build_tool("ask")
def ask(query: str) -> str:
    """Ask anything about your vault — search, Q&A, entity lookup, graph exploration, summaries, and more.

    Automatically understands your intent and routes to the right internal capability.
    Handles: semantic search, question answering, entity/relationship lookup,
    wiki-link graph traversal, topic summaries, tag suggestions, index stats,
    orphan detection, community discovery, shortest path, and more.

    Args:
        query: Your natural language request — anything about your vault.

    Returns:
        A string result — either formatted text, JSON data, or an LLM-generated answer.
    """
    return _route(query)


__all_tools__ = [ask]
