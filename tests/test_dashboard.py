"""Tests for the dashboard module."""
import json

from obsidian_ai.dashboard import gather_data, generate, serve


def test_gather_data_returns_expected_keys():
    data = gather_data()
    assert "vault_name" in data
    assert "generated_at" in data
    assert "graph" in data
    assert "graph_edges" in data
    assert "graph_nodes" in data
    assert "communities" in data
    assert "index" in data
    assert "entities" in data
    assert "hubs" in data
    assert "orphans" in data
    assert "health" in data
    assert "todos" in data


def test_gather_data_graph_stats():
    data = gather_data()
    g = data["graph"]
    assert isinstance(g["nodes"], int)
    assert isinstance(g["edges"], int)
    assert isinstance(g["avg_degree"], (int, float))
    assert isinstance(g["community_count"], int)
    assert isinstance(g["modularity"], (int, float))


def test_gather_data_edges_format():
    data = gather_data()
    for edge in data["graph_edges"]:
        assert "from" in edge
        assert "to" in edge


def test_gather_data_communities_format():
    data = gather_data()
    for comm in data["communities"]:
        assert "id" in comm
        assert "size" in comm
        assert "members" in comm
        assert isinstance(comm["id"], int)
        assert isinstance(comm["size"], int)
        assert isinstance(comm["members"], list)


def test_generate_returns_html():
    data = gather_data()
    html = generate(data)
    assert html.startswith("<!DOCTYPE html>")
    assert "</html>" in html
    assert "EMBEDDED_DATA" in html
    assert "vis-network" in html


def test_generate_embeds_data():
    data = gather_data()
    html = generate(data)
    assert "var EMBEDDED_DATA =" in html
    # Extract the embedded JSON
    start = html.index("var EMBEDDED_DATA = ") + len("var EMBEDDED_DATA = ")
    end = html.index(";", start)
    embedded = json.loads(html[start:end])
    assert embedded["graph"]["nodes"] == data["graph"]["nodes"]


def test_generate_without_data():
    html = generate()
    assert html.startswith("<!DOCTYPE html>")
    assert "EMBEDDED_DATA" in html


def test_serve_starts_and_stops():
    """Verify serve() can start and be shut down via KeyboardInterrupt."""
    import threading
    import time

    errors = []

    def runner():
        try:
            serve(host="127.0.0.1", port=18765)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            errors.append(e)

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    time.sleep(0.5)

    # Verify server is listening
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = s.connect_ex(("127.0.0.1", 18765))
    s.close()
    assert result == 0, f"Server did not start: {errors}"

    if errors:
        raise errors[0]
