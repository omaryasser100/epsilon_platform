"""CLI for managing channels.

Channels represent a company or tenant; every PDF (`reports` row) and
every chunk (`report_chunks` row) belongs to exactly one channel. The
ingest and query CLIs both take a `--channel-id <uuid>`, which is what
this script returns from `create`.

Subcommands
===========
    python channels.py create "Acme Corp"
        Create a new channel and print its UUID. Optional flags:
            --desc "..."           free-form description
            --meta key=value       repeatable; stored in the metadata jsonb

    python channels.py list
        Print every channel as a table: UUID, name, created_at.

    python channels.py show <channel-id>
        Print one channel plus the reports (PDFs) it contains.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid

from core.config import settings
from core.db import close_pool, init_pool
from pipeline import registry

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("channels")


# ── Shared helpers ──────────────────────────────────────────────────────────

def _parse_meta(items: list[str]) -> dict[str, str]:
    """Parse repeated `--meta key=value` into a flat dict. Mirrors
    main.py so the flag behaves identically in both CLIs."""
    out: dict[str, str] = {}
    for raw in items:
        if "=" not in raw:
            raise SystemExit(f"--meta value must be key=value, got: {raw}")
        key, _, value = raw.partition("=")
        key = key.strip()
        if not key:
            raise SystemExit(f"--meta key must be non-empty: {raw}")
        out[key] = value
    return out


def _validate_uuid(value: str, label: str) -> str:
    try:
        return str(uuid.UUID(value))
    except ValueError as exc:
        raise SystemExit(f"{label} is not a valid UUID: {exc}")


# ── Subcommand handlers ─────────────────────────────────────────────────────

def _cmd_create(args: argparse.Namespace) -> int:
    metadata = _parse_meta(args.meta)
    try:
        channel_id = registry.create_channel(
            name=args.name,
            description=args.desc,
            metadata=metadata,
        )
    except Exception as exc:
        # Most common cause: channels.Name UNIQUE violation.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(channel_id)
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    rows = registry.list_channels()
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    if not rows:
        print("(no channels yet — create one with `channels.py create \"<name>\"`)")
        return 0
    # Compact human-readable table. Name is the most useful column day-
    # to-day; UUID first so it's easy to copy for the next CLI call.
    print(f"  {'UUID':<38}{'Name':<32}{'Created':<25}")
    print("  " + "─" * (38 + 32 + 25))
    for ch in rows:
        created = (ch["created_at"] or "")[:19]
        print(f"  {ch['id']:<38}{ch['name'][:30]:<32}{created:<25}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    channel_id = _validate_uuid(args.channel_id, "channel-id")
    channel = registry.get_channel(channel_id)
    if channel is None:
        print(f"error: channel {channel_id} not found", file=sys.stderr)
        return 1

    reports = registry.list_reports(channel_id)

    if args.json:
        print(json.dumps({"channel": channel, "reports": reports},
                         ensure_ascii=False, indent=2))
        return 0

    print()
    print("─" * 72)
    print(f"  Channel    {channel['name']}")
    print(f"  UUID       {channel['id']}")
    print(f"  Created    {channel['created_at']}")
    if channel["description"]:
        print(f"  Desc       {channel['description']}")
    if channel["metadata"]:
        print(f"  Metadata   {json.dumps(channel['metadata'], ensure_ascii=False)}")
    print("─" * 72)
    if not reports:
        print("  (no reports ingested yet)")
    else:
        print(f"  Reports ({len(reports)}):")
        for r in reports:
            created = (r["created_at"] or "")[:19]
            print(f"    • {r['id']}  {r['filename']}  ({created})")
            if r["title"] and r["title"] != r["filename"]:
                print(f"        title: {r['title']}")
            if r["metadata"]:
                print(f"        meta:  {json.dumps(r['metadata'], ensure_ascii=False)}")
    print()
    return 0


# ── CLI plumbing ────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage channels (the per-company scope for ingest + query).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_create = sub.add_parser("create", help="Create a new channel.")
    p_create.add_argument("name", type=str, help="Human-readable channel name (unique).")
    p_create.add_argument(
        "--desc", type=str, default="",
        help="Optional description.",
    )
    p_create.add_argument(
        "--meta", action="append", default=[],
        metavar="KEY=VALUE",
        help="Free-form metadata key=value pair. Repeat for multiple.",
    )

    p_list = sub.add_parser("list", help="List all channels.")
    p_list.add_argument("--json", action="store_true", help="Emit JSON.")

    p_show = sub.add_parser("show", help="Show one channel and its reports.")
    p_show.add_argument("channel_id", type=str, help="Channel UUID.")
    p_show.add_argument("--json", action="store_true", help="Emit JSON.")

    return parser.parse_args(argv)


# ── Entry point ─────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if not settings.database_url:
        print(
            "error: DATABASE_URL is not configured — set it in .env or the "
            "environment before running the CLI.",
            file=sys.stderr,
        )
        return 2

    init_pool()
    try:
        handlers = {
            "create": _cmd_create,
            "list":   _cmd_list,
            "show":   _cmd_show,
        }
        return handlers[args.cmd](args)
    finally:
        close_pool()


if __name__ == "__main__":
    sys.exit(main())
