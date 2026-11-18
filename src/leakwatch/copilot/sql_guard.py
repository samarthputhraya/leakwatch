"""SQL safety for the copilot's ``run_sql`` tool (PRD §9.2, §9.4).

A query is accepted only if it parses (sqlglot, DuckDB dialect) into exactly one read-only query:

- one statement, whose root is a ``SELECT`` or a set operation of selects (``UNION``,
  ``INTERSECT``, ``EXCEPT``); CTEs (``WITH``) and subqueries are fine;
- no DDL/DML or session commands anywhere in the tree (``CREATE``, ``INSERT``, ``COPY``,
  ``ATTACH``, ``PRAGMA``, ``SET``, ``INSTALL``/``LOAD``, ``SELECT … INTO`` …);
- tables are the lake views (or CTEs of the same query), unqualified; table functions other than
  ``range``/``generate_series``/``unnest`` are rejected, and so are file paths in ``FROM``;
- no file, network, extension or settings functions (``read_parquet``, ``read_csv*``,
  ``read_json*``, ``glob``, ``query``, ``getenv``, ``http*`` …);
- the result is capped: no ``LIMIT`` → ``LIMIT max_rows``; a larger literal limit is lowered; any
  other limit expression is wrapped in ``SELECT * FROM (…) LIMIT max_rows``.

What runs is the SQL regenerated from the checked tree, so the statement that was checked is the
statement that executes. DuckDB itself is a second layer: the copilot's connection has external
access disabled outside the lake and its configuration locked (``copilot.tools``).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

#: Function-name prefixes (lower case) that touch files, the network, extensions or settings.
DENIED_FUNCTION_PREFIXES: tuple[str, ...] = (
    "read_",
    "parquet_",
    "sniff_",
    "glob",
    "http",
    "s3",
    "gcs",
    "azure",
    "iceberg",
    "delta_",
    "sqlite",
    "postgres",
    "mysql",
    "duckdb_",
    "pragma",
    "query",
    "install",
    "load",
    "copy",
    "attach",
    "detach",
    "export",
    "import",
    "write_",
    "getenv",
    "current_setting",
    "set_config",
    "checkpoint",
    "force_checkpoint",
    "system",
    "json_execute",
    "from_substrait",
    "get_substrait",
    "which_secret",
    "secret",
    "enable_",
    "disable_",
)

#: Table functions that only generate values.
ALLOWED_TABLE_FUNCTIONS: frozenset[str] = frozenset({"range", "generate_series", "unnest"})

_FORBIDDEN_NODE_NAMES: tuple[str, ...] = (
    "Insert",
    "Update",
    "Delete",
    "Create",
    "Drop",
    "Alter",
    "Merge",
    "Copy",
    "Command",
    "Pragma",
    "Set",
    "Use",
    "Attach",
    "Detach",
    "Into",
    "Transaction",
    "Commit",
    "Rollback",
    "TruncateTable",
    "LoadData",
    "Export",
    "Describe",
    "Show",
    "Kill",
    "Cache",
    "Uncache",
    "Refresh",
    "Summarize",
)
FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = tuple(
    getattr(exp, name) for name in _FORBIDDEN_NODE_NAMES if hasattr(exp, name)
)


class SQLGuardError(ValueError):
    """The query is not a single safe SELECT; the message says why (shown to the model)."""


@dataclass(frozen=True)
class GuardedSQL:
    """A query that passed the guard."""

    sql: str  # what will run (regenerated from the checked tree, LIMIT enforced)
    original: str
    max_rows: int
    limit_changed: bool  # True if a LIMIT was added, lowered or wrapped


def _function_name(node: exp.Func) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.name).lower()
    return node.sql_name().lower()


def _cte_names(tree: exp.Expression) -> set[str]:
    return {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}


def _check_tables(tree: exp.Expression, allowed: set[str]) -> None:
    ctes = _cte_names(tree)
    for table in tree.find_all(exp.Table):
        target = table.this
        if not isinstance(target, exp.Identifier):
            name = _function_name(target) if isinstance(target, exp.Func) else type(target).__name__
            if name.lower() not in ALLOWED_TABLE_FUNCTIONS:
                raise SQLGuardError(
                    f"table function {name!r} is not allowed; query the lake views instead"
                )
            continue
        if table.args.get("db") or table.args.get("catalog"):
            raise SQLGuardError(f"use unqualified view names (got {table.sql(dialect='duckdb')!r})")
        name = str(target.this).lower()
        if name not in allowed and name not in ctes:
            views = ", ".join(sorted(allowed)) or "none"
            raise SQLGuardError(f"unknown table {target.this!r}; available views: {views}")


def _check_functions(tree: exp.Expression) -> None:
    for node in tree.find_all(exp.Func):
        name = _function_name(node)
        if name.startswith(DENIED_FUNCTION_PREFIXES):
            raise SQLGuardError(f"function {name!r} is not allowed (file/network/settings access)")


def _enforce_limit(tree: exp.Query, max_rows: int) -> tuple[exp.Query, bool]:
    limit = tree.args.get("limit")
    if tree.args.get("fetch") is not None:
        limit = tree.args["fetch"]
    if limit is None:
        return tree.limit(max_rows, copy=True), True
    value = limit.args.get("expression")
    if isinstance(limit, exp.Limit) and isinstance(value, exp.Literal) and value.is_int:
        if int(value.this) <= max_rows:
            return tree, False
        new = tree.copy()
        new.args["limit"].set("expression", exp.Literal.number(max_rows))
        return new, True
    wrapped = exp.select("*").from_(tree.subquery("q", copy=True)).limit(max_rows)
    return wrapped, True


def guard_sql(
    query: str,
    allowed_tables: Iterable[str],
    max_rows: int = 1000,
    max_chars: int = 8000,
) -> GuardedSQL:
    """Check ``query`` and return the SQL to execute.

    Args:
        query: SQL text from the model.
        allowed_tables: names of the lake views that exist (case-insensitive).
        max_rows: LIMIT enforced on the result.
        max_chars: longest query text accepted.

    Raises:
        SQLGuardError: the query is not a single safe SELECT (the message says why).

    """
    text = (query or "").strip()
    if not text:
        raise SQLGuardError("empty query")
    if len(text) > max_chars:
        raise SQLGuardError(f"query is longer than {max_chars} characters")
    try:
        statements = [s for s in sqlglot.parse(text, read="duckdb") if s is not None]
    except (ParseError, TokenError) as exc:
        first = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        raise SQLGuardError(f"could not parse the query: {first}") from exc
    if len(statements) != 1:
        raise SQLGuardError(f"exactly one statement is allowed (got {len(statements)})")
    tree = statements[0]
    if isinstance(tree, exp.Subquery):
        tree = tree.unnest()
    if not isinstance(tree, exp.Select | exp.SetOperation):
        raise SQLGuardError(f"only SELECT queries are allowed (got {type(tree).__name__.upper()})")
    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            raise SQLGuardError(
                f"{type(node).__name__.upper()} is not allowed; read-only SELECT only"
            )
    _check_tables(tree, {t.lower() for t in allowed_tables})
    _check_functions(tree)
    limited, changed = _enforce_limit(tree, max_rows)
    return GuardedSQL(
        sql=limited.sql(dialect="duckdb"), original=text, max_rows=max_rows, limit_changed=changed
    )
