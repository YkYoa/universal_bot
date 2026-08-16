#!/usr/bin/env python3
"""Command-line front end for the qvic_2026 sequence store.

    sequence_store_cli.py import
    sequence_store_cli.py import --file config/sequence.yaml
    sequence_store_cli.py list
    sequence_store_cli.py show qvic_2026_both
    sequence_store_cli.py export --file /tmp/roundtrip.yaml
    sequence_store_cli.py delete some_sequence

Point it at a different database with --db or the QVIC_DB_PATH env var.
`import` with no --file seeds/refreshes store.DEFAULT_DB_PATH's sqlite file
from this project's own config/sequence.yaml - store.connect() creates the
file and schema on demand, so this is the one command that takes a bare
checkout to a working sequence store.
"""

import argparse
import json
import os
import sys

# Installed to lib/qvic_2026 next to no package dir, so make the source tree's
# module importable when running straight out of the checkout.
try:
    from qvic_2026 import store, yaml_sync
except ImportError:  # pragma: no cover - only hit before colcon build
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from qvic_2026 import store, yaml_sync

# Sibling of store.DEFAULT_DB_PATH - this in-development repo only ever runs
# from this one checkout (same reasoning as store.py/qvic_2026.launch.py).
DEFAULT_SEQUENCE_YAML = os.path.join(
    os.path.dirname(os.path.dirname(store.DEFAULT_DB_PATH)), "config", "sequence.yaml")


def cmd_import(args):
    summary = yaml_sync.import_yaml(args.file, path=args.db, replace=not args.no_replace)
    print(f"waypoints: {summary['waypoints']} across {summary['sections']} sections")
    print(f"sequences: {', '.join(summary['sequences']) or '(none)'}")
    if summary["skipped"]:
        print(f"skipped (already exist): {', '.join(summary['skipped'])}")


def cmd_export(args):
    summary = yaml_sync.export_yaml(args.file, path=args.db)
    print(f"wrote {summary['waypoints']} waypoints, "
          f"{len(summary['sequences'])} sequences to {args.file}")


def cmd_list(args):
    rows = store.list_sequences(path=args.db)
    if not rows:
        print("(no sequences - run `import` first)")
        return
    width = max(len(r["name"]) for r in rows)
    for r in rows:
        mode = r["required_control_mode"]
        loop = "forever" if r["repeat"] == -1 else f"x{r['repeat']}"
        print(f"{r['name']:<{width}}  {r['num_steps']:>3} steps  {r['arm']:<10} "
              f"{loop:<8} mode={mode}")


def cmd_sections(args):
    for section, counts in store.list_sections(path=args.db).items():
        parts = ", ".join(f"{n} {kind}" for kind, n in counts.items() if n)
        print(f"{section:<24} {parts}")


def cmd_show(args):
    seq = store.get_sequence(args.name, path=args.db)
    print(json.dumps(seq, indent=2))


def cmd_delete(args):
    store.delete_sequence(args.name, path=args.db)
    print(f"deleted '{args.name}'")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=None,
                        help="database file (default: QVIC_DB_PATH or the "
                             "package's data/sequences.db)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("import", help="load a sequence.yaml into the store")
    p.add_argument("--file", default=DEFAULT_SEQUENCE_YAML,
                   help=f"default: {DEFAULT_SEQUENCE_YAML}")
    p.add_argument("--no-replace", action="store_true",
                   help="skip sequences that already exist instead of overwriting")
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("export", help="write the store back out as sequence.yaml")
    p.add_argument("--file", required=True)
    p.set_defaults(func=cmd_export)

    sub.add_parser("list", help="list sequences").set_defaults(func=cmd_list)
    sub.add_parser("sections", help="list waypoint sections").set_defaults(func=cmd_sections)

    p = sub.add_parser("show", help="dump one sequence as JSON")
    p.add_argument("name")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("delete", help="delete one sequence")
    p.add_argument("name")
    p.set_defaults(func=cmd_delete)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except store.StoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
