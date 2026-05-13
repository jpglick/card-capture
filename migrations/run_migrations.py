import sqlite3
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).parent


def apply_migrations(db_path: Path) -> None:
    """Apply every *.sql migration file in sorted order.

    Idempotent: already-applied files are skipped. ``ALTER TABLE … ADD COLUMN``
    failures caused by duplicate column names are silently swallowed; any other
    ``sqlite3.OperationalError`` is re-raised.
    """
    sql_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            "CREATE TABLE IF NOT EXISTS _migrations "
            "(filename TEXT PRIMARY KEY, "
            "applied_at TEXT NOT NULL DEFAULT (datetime('now')))"
        )
        applied = {r[0] for r in conn.execute("SELECT filename FROM _migrations").fetchall()}
        for sql_file in sql_files:
            if sql_file.name in applied:
                continue
            sql = sql_file.read_text()
            all_ok = True
            for statement in _split_statements(sql):
                try:
                    conn.execute(statement)
                except sqlite3.OperationalError as exc:
                    msg = str(exc)
                    if "duplicate column name" in msg:
                        continue  # idempotency for ADD COLUMN
                    
                    # If a table is missing, we can't apply the migration.
                    # We skip the statement but mark all_ok = False so we don't
                    # record this file as applied yet.
                    if "no such table" in msg:
                        all_ok = False
                        continue
                    raise
            
            if all_ok:
                print(f"Applied migration: {sql_file.name}")
                conn.execute("INSERT INTO _migrations(filename) VALUES (?)", (sql_file.name,))
            else:
                print(f"Skipped migration (table missing, will retry): {sql_file.name}")
        conn.commit()


def _split_statements(sql: str) -> list:
    """Split a SQL script into individual statements by semicolons.

    Note: splitting is done naively on ``;`` characters. This is sufficient for
    migration files whose SQL does not contain semicolons inside string literals
    or block comments. Each returned segment has leading/trailing whitespace
    stripped; empty segments are omitted.
    """
    return [s.strip() for s in sql.split(";") if s.strip()]
