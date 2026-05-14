import logging
import sqlite3
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).parent
logger = logging.getLogger(__name__)


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


def assert_migrations_complete(db_path: Path) -> None:
    """Raise RuntimeError if any expected migration file has not been applied.

    Called at app startup after apply_migrations() so that a partially-applied
    migration (e.g. due to a 'no such table' skip) is surfaced immediately
    rather than producing a silent 500 on the first write.
    """
    sql_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not sql_files:
        return
    with sqlite3.connect(db_path) as conn:
        try:
            applied = {r[0] for r in conn.execute("SELECT filename FROM _migrations").fetchall()}
        except sqlite3.OperationalError:
            applied = set()

    missing = [f.name for f in sql_files if f.name not in applied]
    if missing:
        msg = (
            f"The following database migrations have not been applied: {missing}. "
            "Run `apply_migrations(db_path)` or restart the app after resolving "
            "any migration prerequisites."
        )
        logger.error(msg)
        raise RuntimeError(msg)


def _split_statements(sql: str) -> list:
    """Split a SQL script into individual statements.

    Handles single-quoted strings, double-quoted identifiers, line comments
    (``--``), block comments (``/* ... */``), and ``BEGIN...END`` blocks so
    that semicolons inside these constructs are never treated as statement
    separators. Empty segments are omitted.
    """
    statements: list[str] = []
    buf: list[str] = []
    depth = 0  # nesting depth for BEGIN...END blocks (used in triggers)
    i = 0
    n = len(sql)

    while i < n:
        ch = sql[i]

        # Line comment: consume until end of line
        if ch == '-' and i + 1 < n and sql[i + 1] == '-':
            end = sql.find('\n', i)
            end = n if end == -1 else end
            buf.append(sql[i:end])
            i = end
            continue

        # Block comment: consume until */
        if ch == '/' and i + 1 < n and sql[i + 1] == '*':
            end = sql.find('*/', i + 2)
            end = n if end == -1 else end + 2
            buf.append(sql[i:end])
            i = end
            continue

        # Single-quoted string literal (handles '' escapes)
        if ch == "'":
            j = i + 1
            while j < n:
                if sql[j] == "'" and j + 1 < n and sql[j + 1] == "'":
                    j += 2
                elif sql[j] == "'":
                    j += 1
                    break
                else:
                    j += 1
            buf.append(sql[i:j])
            i = j
            continue

        # Double-quoted identifier (handles "" escapes)
        if ch == '"':
            j = i + 1
            while j < n:
                if sql[j] == '"' and j + 1 < n and sql[j + 1] == '"':
                    j += 2
                elif sql[j] == '"':
                    j += 1
                    break
                else:
                    j += 1
            buf.append(sql[i:j])
            i = j
            continue

        # BEGIN keyword — increases nesting depth for trigger bodies
        word5 = sql[i:i + 5].upper()
        prev_alnum = i > 0 and sql[i - 1].isalnum()
        next5_alnum = i + 5 < n and sql[i + 5].isalnum()
        if word5 == 'BEGIN' and not prev_alnum and not next5_alnum:
            depth += 1
            buf.append(sql[i:i + 5])
            i += 5
            continue

        # END keyword
        word3 = sql[i:i + 3].upper()
        next3_alnum = i + 3 < n and sql[i + 3].isalnum()
        if word3 == 'END' and not prev_alnum and not next3_alnum:
            depth = max(0, depth - 1)
            buf.append(sql[i:i + 3])
            i += 3
            continue

        # Statement separator (only at depth 0)
        if ch == ';' and depth == 0:
            stmt = ''.join(buf).strip()
            if stmt:
                statements.append(stmt)
            buf = []
            i += 1
            continue

        buf.append(ch)
        i += 1

    # Trailing statement without a closing semicolon
    stmt = ''.join(buf).strip()
    if stmt:
        statements.append(stmt)

    return statements
