import argparse
from collections import Counter
from codearchguard.analysis.extractor import extract_graph


def main(argv=None):
    p = argparse.ArgumentParser(prog="cag")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="Parse a repo and print graph stats / optionally load Neo4j")
    a.add_argument("repo")
    a.add_argument("--source-root", action="append", help="Source directory relative to the repository; repeat for multiple roots")
    a.add_argument("--neo4j-uri")
    a.add_argument("--user", default="neo4j")
    a.add_argument("--password", default="password")

    u = sub.add_parser("ui", help="Launch interactive CodeArchGuard Demo Web UI")
    u.add_argument("--port", type=int, default=7860, help="Web server port (default: 7860)")
    u.add_argument("--share", action="store_true", help="Generate public Gradio share link")

    args = p.parse_args(argv)

    if args.cmd == "ui":
        from codearchguard.ui import main as ui_main
        ui_main(port=args.port, share=args.share)
        return

    g = extract_graph(args.repo, source_roots=args.source_root)
    print("Nodes:", dict(Counter(n.kind.value for n in g.nodes.values())))
    print("Edges:", dict(Counter(e.kind.value for e in g.edges)))
    if args.neo4j_uri:
        from codearchguard.graph.neo4j_store import Neo4jStore
        store = Neo4jStore(args.neo4j_uri, args.user, args.password)
        store.init_schema(); store.save(g); store.close()
        print("Loaded into Neo4j.")


if __name__ == "__main__":
    main()
