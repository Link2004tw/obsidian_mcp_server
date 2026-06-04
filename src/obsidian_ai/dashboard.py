"""Standalone HTML dashboard for Obsidian AI knowledge graph.

Usage:
    from obsidian_ai.dashboard import generate, serve

    # Static HTML file
    html = generate()
    open("dashboard.html", "w").write(html)

    # Live server
    serve(host="localhost", port=8765)
"""

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from . import chroma_store, config, graph_store, llm_client, todos

TITLE = "Obsidian AI Dashboard"
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


# ── Data gathering ──────────────────────────────────────────────────


def gather_data() -> dict[str, Any]:
    """Collect all dashboard data from stores."""
    gs = graph_store
    g_stats = gs.stats()
    communities = gs.label_propagation()

    # Community breakdown
    comm_members: dict[int, list[str]] = {}
    for path, cid in communities.items():
        comm_members.setdefault(cid, []).append(path)
    community_list = [
        {"id": cid, "size": len(members), "members": sorted(members)[:50]}
        for cid, members in sorted(comm_members.items())
    ]

    # Graph export (note-to-note edges only)
    raw = gs.to_dict()
    nodes_list = sorted(raw.get("edges", {}).keys())
    edges_list: list[dict] = []
    for src, tgts in raw.get("edges", {}).items():
        for tgt in tgts:
            edges_list.append({"from": src, "to": tgt})

    # Index stats
    try:
        idx_stats = chroma_store.get_index_stats()
    except Exception:
        idx_stats = {"total_chunks": 0, "unique_notes": 0}

    # Entity stats
    try:
        entity_list = gs.get_entity_nodes()
        entity_count = len(entity_list)
        entity_type_counts: dict[str, int] = {}
        for e in entity_list:
            entity_type_counts[e["entity_type"]] = entity_type_counts.get(e["entity_type"], 0) + 1
    except Exception:
        entity_list = []
        entity_count = 0
        entity_type_counts = {}

    # Health
    try:
        health = llm_client.check_health()
    except Exception:
        health = {"ollama": {"available": False}, "overall": "unknown"}

    # Orphans
    try:
        orphans = gs.get_orphans()
    except Exception:
        orphans = []

    # Hubs
    hubs = g_stats.get("hubs", [])

    # Todo stats
    try:
        td_stats = todos.td_stats()
    except Exception:
        td_stats = {"total": 0, "overdue": 0, "completed": 0}

    # Semantic clusters
    try:
        from .clustering import get_clusters as _get_clusters
        cluster_data = _get_clusters()
    except Exception:
        cluster_data = []

    return {
        "vault_name": os.path.basename(config.vault_dir) if hasattr(config, "vault_dir") and config.vault_dir else "Obsidian Vault",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "graph": {
            "nodes": g_stats.get("nodes", 0),
            "edges": g_stats.get("edges", 0),
            "avg_degree": g_stats.get("avg_degree", 0),
            "isolated_count": g_stats.get("isolated_count", 0),
            "community_count": g_stats.get("community_count", 0),
            "modularity": g_stats.get("modularity", 0),
        },
        "graph_edges": edges_list,
        "graph_nodes": nodes_list,
        "communities": community_list,
        "index": idx_stats,
        "entities": {
            "total": entity_count,
            "by_type": entity_type_counts,
            "list": sorted(entity_list, key=lambda e: e["entity_name"]),
        },
        "hubs": hubs,
        "orphans": orphans,
        "health": health,
        "todos": td_stats,
        "clusters": cluster_data,
    }


# ── Template ────────────────────────────────────────────────────────


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{TITLE}}</title>
<script src="https://unpkg.com/vis-network@9.1.6/standalone/umd/vis-network.min.js"></script>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg: #0f0f1a;
  --surface: #1a1a2e;
  --surface2: #16213e;
  --border: #2a2a4a;
  --text: #e0e0e0;
  --text-dim: #8888aa;
  --accent: #7c5cfc;
  --accent2: #f28e2b;
  --green: #4ade80;
  --red: #f87171;
  --yellow: #facc15;
}
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: var(--bg); color: var(--text); font-size: 14px; line-height: 1.5;
  min-height: 100vh;
}
header {
  background: var(--surface); border-bottom: 1px solid var(--border);
  padding: 16px 24px; display: flex; align-items: center; gap: 16px;
  flex-wrap: wrap;
}
header h1 { font-size: 20px; font-weight: 600; }
.header-meta { color: var(--text-dim); font-size: 12px; margin-left: auto; }
.health-badge {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 500;
}
.health-ok { background: rgba(74,222,128,.15); color: var(--green); }
.health-degraded { background: rgba(250,204,21,.15); color: var(--yellow); }
.health-err { background: rgba(248,113,113,.15); color: var(--red); }
.container { max-width: 1440px; margin: 0 auto; padding: 16px 20px; }

/* Stats cards */
.stats-row {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 10px; margin-bottom: 16px;
}
.stat-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 12px 14px; text-align: center;
}
.stat-card .value { font-size: 28px; font-weight: 700; color: var(--accent); }
.stat-card .label { font-size: 11px; color: var(--text-dim); text-transform: uppercase; letter-spacing: .5px; margin-top: 2px; }
.stat-card .sublabel { font-size: 11px; color: var(--text-dim); margin-top: 2px; }

/* Main layout */
.main-grid {
  display: grid; grid-template-columns: 1fr 320px; gap: 16px;
  margin-bottom: 16px;
}
#graph-container {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; overflow: hidden; min-height: 500px; position: relative;
}
#graph { width: 100%; height: 500px; }

/* Sidebar */
#sidebar {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 14px; overflow-y: auto; max-height: 500px;
}
#sidebar h3 { font-size: 13px; color: var(--text-dim); text-transform: uppercase; letter-spacing: .5px; margin-bottom: 10px; }
.comm-group { margin-bottom: 8px; }
.comm-header {
  display: flex; align-items: center; gap: 8px; padding: 6px 8px;
  background: var(--surface2); border-radius: 6px; cursor: pointer;
  font-size: 13px; user-select: none; transition: background .15s;
}
.comm-header:hover { background: var(--border); }
.comm-header .badge {
  background: var(--accent); color: #fff; font-size: 10px; font-weight: 600;
  padding: 1px 7px; border-radius: 8px; margin-left: auto;
}
.comm-members { display: none; padding: 4px 0 4px 12px; }
.comm-members.open { display: block; }
.comm-member {
  padding: 3px 6px; font-size: 12px; color: var(--text-dim);
  border-radius: 4px; cursor: pointer; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.comm-member:hover { background: var(--surface2); color: var(--text); }

/* Bottom panels */
.bottom-grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
}
.panel {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 14px;
}
.panel h3 { font-size: 13px; color: var(--text-dim); text-transform: uppercase; letter-spacing: .5px; margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid var(--border); }
.panel table { width: 100%; border-collapse: collapse; font-size: 12px; }
.panel th, .panel td { padding: 6px 8px; text-align: left; border-bottom: 1px solid var(--border); }
.panel th { color: var(--text-dim); font-weight: 500; }
.panel td { color: var(--text); }
.entity-tag {
  display: inline-block; padding: 1px 8px; border-radius: 8px; font-size: 11px;
  background: var(--surface2); color: var(--accent2); margin-right: 4px; margin-bottom: 2px;
}

/* Community color palette */
.comm-color { display: inline-block; width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.comm-colors { display: flex; gap: 3px; flex-wrap: wrap; }
.comm-colors span { width: 12px; height: 12px; border-radius: 3px; }

/* Search */
.search-bar {
  display: flex; gap: 8px; margin-bottom: 14px;
}
.search-bar input {
  flex: 1; background: var(--surface2); border: 1px solid var(--border);
  border-radius: 6px; padding: 8px 12px; color: var(--text); font-size: 13px;
  outline: none;
}
.search-bar input:focus { border-color: var(--accent); }
.search-bar button {
  background: var(--accent); color: #fff; border: none; border-radius: 6px;
  padding: 8px 16px; cursor: pointer; font-size: 13px; font-weight: 500;
}
.search-bar button:hover { opacity: .9; }
.search-results { margin-top: 8px; }
.search-result {
  padding: 6px 8px; border-radius: 6px; margin-bottom: 4px;
  background: var(--surface2); font-size: 12px; cursor: pointer;
}
.search-result:hover { background: var(--border); }
.search-result .score { color: var(--accent2); font-weight: 500; float: right; }
.search-result .signals { color: var(--text-dim); font-size: 11px; }

/* Mode badge */
.mode-badge {
  display: inline-block; padding: 2px 8px; border-radius: 4px;
  font-size: 10px; font-weight: 600; text-transform: uppercase;
  background: var(--surface2); color: var(--text-dim); margin-left: 8px;
}

/* Responsive */
@media (max-width: 900px) {
  .main-grid { grid-template-columns: 1fr; }
  .bottom-grid { grid-template-columns: 1fr; }
  #sidebar { max-height: none; }
}
</style>
</head>
<body>

<header>
  <h1>Obsidian AI Dashboard</h1>
  <span class="header-meta" id="header-meta">loading...</span>
  <span class="health-badge" id="health-badge">...</span>
  <span class="mode-badge" id="mode-badge"></span>
</header>

<div class="container">

<div class="stats-row" id="stats-row">
  <div class="stat-card"><div class="value" id="stat-nodes">-</div><div class="label">Nodes</div></div>
  <div class="stat-card"><div class="value" id="stat-edges">-</div><div class="label">Edges</div></div>
  <div class="stat-card"><div class="value" id="stat-communities">-</div><div class="label">Communities</div></div>
  <div class="stat-card"><div class="value" id="stat-modularity">-</div><div class="label">Modularity</div></div>
  <div class="stat-card"><div class="value" id="stat-notes">-</div><div class="label">Notes</div></div>
  <div class="stat-card"><div class="value" id="stat-entities">-</div><div class="label">Entities</div></div>
  <div class="stat-card"><div class="value" id="stat-isolated">-</div><div class="label">Isolated</div></div>
  <div class="stat-card"><div class="value" id="stat-todos">-</div><div class="label">Todos</div></div>
</div>

<div class="main-grid">
  <div id="graph-container">
    <div id="graph"></div>
  </div>
  <div id="sidebar">
    <h3>Communities</h3>
    <div id="community-list"></div>
  </div>
</div>

<div class="bottom-grid">
  <div class="panel" id="hubs-panel">
    <h3>Top Hubs (by degree)</h3>
    <div id="hubs-body"><p style="color:var(--text-dim);font-size:12px;">loading...</p></div>
  </div>
  <div class="panel" id="orphans-panel">
    <h3>Orphan Notes</h3>
    <div id="orphans-body"><p style="color:var(--text-dim);font-size:12px;">loading...</p></div>
  </div>
  <div class="panel" id="entities-panel">
    <h3>Entity Nodes</h3>
    <div id="entities-body"><p style="color:var(--text-dim);font-size:12px;">loading...</p></div>
  </div>
  <div class="panel" id="search-panel">
    <h3>Search Notes</h3>
    <div class="search-bar">
      <input type="text" id="search-input" placeholder="Search query...">
      <button id="search-btn">Search</button>
    </div>
    <div id="search-results"></div>
  </div>
</div>

</div><!-- .container -->

<script>
// ── Embedded data (populated in static mode) ──────────────────
var EMBEDDED_DATA = null; // __EMBEDDED_DATA__

// ── Community color palette ──────────────────────────────────
var COMMUNITY_COLORS = [
  '#7c5cfc', '#f28e2b', '#4ade80', '#f87171', '#60a5fa',
  '#facc15', '#a78bfa', '#34d399', '#fb923c', '#2dd4bf',
  '#e879f9', '#22d3ee', '#fbbf24', '#818cf8', '#6ee7b7',
];

// ── Data loading ────────────────────────────────────────────
var STORE = {}; // populated by loadData

function render(data) {
  STORE = data;
  document.getElementById('header-meta').textContent = data.vault_name + ' · ' + data.generated_at;
  renderStats(data);
  renderGraph(data);
  renderCommunities(data);
  renderHubs(data);
  renderOrphans(data);
  renderEntities(data);
}

function renderStats(d) {
  var g = d.graph;
  setText('stat-nodes', g.nodes);
  setText('stat-edges', g.edges);
  setText('stat-communities', g.community_count);
  setText('stat-modularity', g.modularity.toFixed(3));
  setText('stat-notes', d.index.unique_notes);
  setText('stat-entities', d.entities.total);
  setText('stat-isolated', g.isolated_count);
  setText('stat-todos', d.todos.total);
}

function setText(id, val) {
  var el = document.getElementById(id);
  if (el) el.textContent = val;
}

function renderGraph(d) {
  var container = document.getElementById('graph');
  if (!container) return;
  if (!d.graph_edges || d.graph_edges.length === 0) {
    container.innerHTML = '<p style="padding:40px;text-align:center;color:var(--text-dim)">No graph data</p>';
    return;
  }

  // Build community map: path -> cid
  var pathToComm = {};
  d.communities.forEach(function(c) {
    c.members.forEach(function(p) { pathToComm[p] = c.id; });
  });

  var nodes = {};
  d.graph_edges.forEach(function(e) {
    if (!nodes[e.from]) nodes[e.from] = {id: e.from, label: labelOf(e.from), community: pathToComm[e.from], degree: 0};
    if (!nodes[e.to]) nodes[e.to] = {id: e.to, label: labelOf(e.to), community: pathToComm[e.to], degree: 0};
    nodes[e.from].degree++;
    nodes[e.to].degree++;
  });
  // Also include isolated nodes
  d.graph_nodes.forEach(function(p) {
    if (!nodes[p]) nodes[p] = {id: p, label: labelOf(p), community: pathToComm[p], degree: 0};
  });

  var maxDeg = 1;
  Object.keys(nodes).forEach(function(k) { if (nodes[k].degree > maxDeg) maxDeg = nodes[k].degree; });

  var visNodes = Object.keys(nodes).map(function(k) {
    var n = nodes[k];
    var size = 8 + (n.degree / maxDeg) * 22;
    var color = COMMUNITY_COLORS[n.community % COMMUNITY_COLORS.length] || '#888';
    return {
      id: n.id, label: n.label, size: size,
      color: {background: color, border: color},
      font: {size: 10 + (n.degree / maxDeg) * 6, color: '#e0e0e0'},
      title: n.id + ' (deg ' + n.degree + ')',
    };
  });

  var visEdges = d.graph_edges.map(function(e) {
    return {from: e.from, to: e.to, color: {opacity: 0.35}, width: 1};
  });

  var network = new vis.Network(container, {
    nodes: new vis.DataSet(visNodes),
    edges: new vis.DataSet(visEdges),
  }, {
    physics: {solver: 'forceAtlas2Based', forceAtlas2Based: {gravitationalConstant: -60, springLength: 160}},
    edges: {smooth: false},
    interaction: {hover: true, tooltipDelay: 200},
  });
}

function labelOf(path) {
  var parts = path.replace(/\\/g, '/').split('/');
  return parts[parts.length - 1].replace(/\.md$/i, '');
}

function renderCommunities(d) {
  var el = document.getElementById('community-list');
  if (!el) return;
  if (!d.communities || d.communities.length === 0) {
    el.innerHTML = '<p style="color:var(--text-dim);font-size:12px;">No communities</p>';
    return;
  }
  var html = '';
  d.communities.forEach(function(c, i) {
    var color = COMMUNITY_COLORS[i % COMMUNITY_COLORS.length];
    html += '<div class="comm-group">';
    html += '<div class="comm-header" onclick="toggleComm(this)">';
    html += '<span class="comm-color" style="background:' + color + '"></span>';
    html += 'Community ' + c.id + ' <span class="badge">' + c.size + '</span>';
    html += '</div>';
    html += '<div class="comm-members">';
    c.members.forEach(function(m) {
      html += '<div class="comm-member" title="' + m + '">' + labelOf(m) + '</div>';
    });
    html += '</div></div>';
  });
  el.innerHTML = html;
}

function toggleComm(el) {
  var members = el.nextElementSibling;
  if (members) members.classList.toggle('open');
}

function renderHubs(d) {
  var el = document.getElementById('hubs-body');
  if (!el) return;
  if (!d.hubs || d.hubs.length === 0) {
    el.innerHTML = '<p style="color:var(--text-dim);font-size:12px;">No hubs</p>';
    return;
  }
  var html = '<table><thead><tr><th>Note</th><th>Degree</th></tr></thead><tbody>';
  d.hubs.forEach(function(h) {
    html += '<tr><td>' + labelOf(h.path) + '</td><td>' + h.degree + '</td></tr>';
  });
  html += '</tbody></table>';
  el.innerHTML = html;
}

function renderOrphans(d) {
  var el = document.getElementById('orphans-body');
  if (!el) return;
  if (!d.orphans || d.orphans.length === 0) {
    el.innerHTML = '<p style="color:var(--text-dim);font-size:12px;">No orphans</p>';
    return;
  }
  var html = '<table><thead><tr><th>Note</th></tr></thead><tbody>';
  d.orphans.forEach(function(o) {
    html += '<tr><td>' + o + '</td></tr>';
  });
  html += '</tbody></table>';
  el.innerHTML = html;
}

function renderEntities(d) {
  var el = document.getElementById('entities-body');
  if (!el) return;
  if (!d.entities || !d.entities.list || d.entities.list.length === 0) {
    el.innerHTML = '<p style="color:var(--text-dim);font-size:12px;">No entities</p>';
    return;
  }
  var html = '<table><thead><tr><th>Name</th><th>Type</th><th>Notes</th></tr></thead><tbody>';
  d.entities.list.slice(0, 50).forEach(function(e) {
    html += '<tr><td>' + e.entity_name + '</td><td><span class="entity-tag">' + e.entity_type + '</span></td><td>' + e.linked_notes + '</td></tr>';
  });
  html += '</tbody></table>';
  if (d.entities.list.length > 50) {
    html += '<p style="color:var(--text-dim);font-size:11px;margin-top:6px;">... and ' + (d.entities.list.length - 50) + ' more</p>';
  }
  el.innerHTML = html;
}

// ── Initialize ──────────────────────────────────────────────
(function init() {
  if (EMBEDDED_DATA) {
    document.getElementById('mode-badge').textContent = 'static';
    render(EMBEDDED_DATA);
  } else {
    document.getElementById('mode-badge').textContent = 'live';
    fetch('/api/data')
      .then(function(r) { return r.json(); })
      .then(function(d) { render(d); })
      .catch(function(e) { console.error('Dashboard load error:', e); });
  }

  // Search
  var searchInput = document.getElementById('search-input');
  var searchBtn = document.getElementById('search-btn');
  var searchResults = document.getElementById('search-results');
  function doSearch() {
    var q = searchInput.value.trim();
    if (!q) { searchResults.innerHTML = ''; return; }
    if (EMBEDDED_DATA) {
      // Static mode: simple client-side filter
      var results = [];
      var pat = q.toLowerCase();
      Object.keys(STORE).forEach(function(k) {
        if (k === 'graph_edges') return;
      });
      // Filter graph nodes
      if (STORE.graph_nodes) {
        STORE.graph_nodes.forEach(function(p) {
          if (p.toLowerCase().indexOf(pat) !== -1) {
            results.push({path: p, score: 1, signals: ['name']});
          }
        });
      }
      if (STORE.entities && STORE.entities.list) {
        STORE.entities.list.forEach(function(e) {
          if (e.entity_name.toLowerCase().indexOf(pat) !== -1) {
            results.push({path: '__entity:' + e.entity_type + ':' + e.entity_name, score: 1, signals: ['entity']});
          }
        });
      }
      renderSearchResults(results);
    } else {
      fetch('/api/search?q=' + encodeURIComponent(q))
        .then(function(r) { return r.json(); })
        .then(function(r) { renderSearchResults(r); })
        .catch(function(e) { console.error('Search error:', e); });
    }
  }
  searchBtn.addEventListener('click', doSearch);
  searchInput.addEventListener('keydown', function(e) { if (e.key === 'Enter') doSearch(); });
})();

function renderSearchResults(results) {
  var el = document.getElementById('search-results');
  if (!results || results.length === 0) {
    el.innerHTML = '<p style="color:var(--text-dim);font-size:12px;">No results</p>';
    return;
  }
  var html = '';
  results.slice(0, 20).forEach(function(r) {
    html += '<div class="search-result">';
    html += '<span>' + labelOf(r.path || r.title || r.entity_name || '?') + '</span>';
    if (r.score) html += '<span class="score">' + r.score.toFixed(3) + '</span>';
    if (r.signals) html += '<div class="signals">' + r.signals.join(', ') + '</div>';
    html += '</div>';
  });
  el.innerHTML = html;
}
</script>
</body>
</html>"""


# ── HTML Generation (static) ────────────────────────────────────────


def generate(data: dict[str, Any] | None = None) -> str:
    """Return dashboard HTML with embedded data."""
    if data is None:
        data = gather_data()

    # Embed data as JSON
    data_json = json.dumps(data, indent=1, ensure_ascii=False, default=str)

    html = TEMPLATE
    html = html.replace("{{TITLE}}", TITLE)
    html = html.replace("var EMBEDDED_DATA = null; // __EMBEDDED_DATA__",
                        f"var EMBEDDED_DATA = {data_json};")
    return html


# ── Live Server ─────────────────────────────────────────────────────


class _DashboardHandler(BaseHTTPRequestHandler):
    """HTTP handler that serves the dashboard HTML and API endpoints."""

    _html: str = ""
    _data: dict[str, Any] = {}

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_html()
        elif self.path == "/api/data":
            self._serve_json(self._gather_fresh_data())
        elif self.path.startswith("/api/search"):
            self._handle_search()
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"404 not found")

    def _serve_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(self._html.encode("utf-8"))

    def _serve_json(self, data: dict):
        body = json.dumps(data, indent=1, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _gather_fresh_data(self) -> dict[str, Any]:
        return gather_data()

    def _handle_search(self):
        from urllib.parse import parse_qs, urlparse
        params = parse_qs(urlparse(self.path).query)
        q = params.get("q", [""])[0]
        if not q:
            self._serve_json([])
            return
        try:
            from .ranker import composite_search
            results = composite_search(query=q, n=10, retrieval_depth=2)
            self._serve_json(results)
        except Exception as e:
            self._serve_json({"error": str(e)})

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[dashboard] {fmt % args}\n")


def serve(host: str = "localhost", port: int = 8765) -> None:
    """Start the live dashboard server.

    Args:
        host: bind address (default localhost).
        port: bind port (default 8765).
    """
    _DashboardHandler._html = TEMPLATE.replace("{{TITLE}}", TITLE)
    _DashboardHandler._html = _DashboardHandler._html.replace(
        "var EMBEDDED_DATA = null; // __EMBEDDED_DATA__",
        "var EMBEDDED_DATA = null;"
    )

    server = HTTPServer((host, port), _DashboardHandler)
    print(f"Obsidian AI Dashboard → http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()
