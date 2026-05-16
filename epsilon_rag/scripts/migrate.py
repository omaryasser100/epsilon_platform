"""Apply pending migrations from migrations/ to the configured DB.

Idempotent: every applied file is recorded in `schema_migrations` so
re-running on a fully-migrated DB is a no-op. Migration files run in
lexicographic order, which is why they're prefixed with zero-padded
numbers.

Usage
=====
    python scripts/migrate.py           # apply pending, exit 0
    python scripts/migrate.py --status  # show what's applied vs pending
    python scripts/migrate.py --dry-run # list pending without applying
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Import compatibility: this script lives in scripts/ but reuses the
# core/db.py pool and core/config.py settings. Add the parent so the
# `core` and `pipeline` imports resolve when run as `python scripts/migrate.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import settings
from core.db import close_pool, get_conn, init_pool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("migrate")


_MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


def _list_files() -> list[Path]:
    """Return migration files sorted by filename. The numeric prefix
    is what enforces order; we never sort by mtime."""
    files = sorted(_MIGRATIONS_DIR.glob("*.sql"))
    if not files:
        raise SystemExit(f"no .sql files found in {_MIGRATIONS_DIR}")
    return files


def _applied_versions() -> set[str]:
    """Read every row from schema_migrations. Returns an empty set if
    the table doesn't exist yet (first run)."""
    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name = 'schema_migrations'
            )
            """,
        )
        exists = cur.fetchone()[0]
        if not exists:
            return set()
        cur = conn.execute("SELECT version FROM schema_migrations")
        return {row[0] for row in cur.fetchall()}


def _apply(path: Path) -> None:
    """Run one migration file in its own transaction and record the
    version. Any exception rolls back and the version stays unrecorded
    so the next run will retry this file."""
    sql = path.read_text(encoding="utf-8")
    version = path.stem
    logger.info("applying %s …", path.name)
    with get_conn() as conn:
        conn.execute(sql)
        # 000_schema_migrations creates the ledger table; record the
        # version too. For every subsequent migration the table exists
        # already and this INSERT is the audit trail.
        conn.execute(
            "INSERT INTO schema_migrations (version) VALUES (%s) "
            "ON CONFLICT (version) DO NOTHING",
            [version],
        )
        conn.commit()
    logger.info("applied %s", path.name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true",
                        help="Show applied vs pending and exit.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List pending without applying.")
    args = parser.parse_args()

    if not settings.database_url:
        print("error: DATABASE_URL is not configured.", file=sys.stderr)
        return 2

    init_pool()
    try:
        files = _list_files()
        applied = _applied_versions()

        if args.status:
            for path in files:
                marker = "✓" if path.stem in applied else "·"
                print(f"  {marker} {path.name}")
            return 0

        pending = [p for p in files if p.stem not in applied]
        if not pending:
            logger.info("schema is up to date (%d applied)", len(applied))
            return 0

        if args.dry_run:
            print(f"  {len(pending)} pending migration(s):")
            for path in pending:
                print(f"    · {path.name}")
            return 0

        for path in pending:
            _apply(path)
        logger.info("applied %d migration(s)", len(pending))
        return 0
    finally:
        close_pool()


if __name__ == "__main__":
    sys.exit(main())
