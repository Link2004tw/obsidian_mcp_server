"""CLI wrapper for obsidian-ai tools."""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def cmd_index(args):
    from obsidian_ai.indexer import run_index
    import time
    start = time.time()
    run_index()
    print(f"Elapsed: {time.time() - start:.1f}s")


def cmd_watch(args):
    from obsidian_ai.indexer import watch
    watch()


def cmd_search(args):
    from obsidian_ai import llm_client, chroma_store
    embedding = llm_client.embed(args.query)
    results = chroma_store.query(embedding, n=args.n)
    seen = {}
    for r in results:
        path = r["metadata"]["path"]
        if path not in seen:
            seen[path] = r["metadata"].get("title", "")
    for path, title in seen.items():
        print(f"  {path} — {title}")


def cmd_tag(args):
    from obsidian_ai.pipelines import tag_notes
    result = tag_notes(args.query, top_k=args.n)
    print(result)


def cmd_stats(args):
    from obsidian_ai import chroma_store
    print(f"Notes in index: {chroma_store.count()}")


def main():
    parser = argparse.ArgumentParser(description="obsidian-ai CLI")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("index", help="Run full index")
    sub.add_parser("watch", help="Start file watcher")

    p_search = sub.add_parser("search", help="Semantic search")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("-n", type=int, default=5, help="Number of results")

    p_tag = sub.add_parser("tag", help="Auto-tag notes")
    p_tag.add_argument("query", help="Query to find notes to tag")
    p_tag.add_argument("-n", type=int, default=5, help="Number of notes")

    sub.add_parser("stats", help="Show index stats")

    args = parser.parse_args()
    if args.command == "index":
        cmd_index(args)
    elif args.command == "watch":
        cmd_watch(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "tag":
        cmd_tag(args)
    elif args.command == "stats":
        cmd_stats(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
