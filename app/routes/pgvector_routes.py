# app/routes/pgvector_routes.py
import json
from fastapi import APIRouter, HTTPException, Request
from app.services.database import PSQLDatabase
from app.routes.document_routes import _redacted_metadata

router = APIRouter()


# --- Entitlement gating for debug record dumps (D-KSPT-1, reviewer F1) -------
# These routes are mounted only in debug_mode, but must still never disclose
# vectors across entities/tenants. Authority is the signed token entitlement.


def require_read_entitlement(request: Request) -> dict:
    ent = getattr(request.state, "entitlement", None)
    if ent is None or "read" not in ent["actions"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    return ent


def filter_rows_by_entitlement(rows: list, ent: dict) -> list:
    """Keep only rows whose cmetadata user_id is in ent and (if stored)
    tenant_id == tid. Rows without an entitled user_id are dropped (never
    disclosed)."""
    out = []
    for row in rows:
        meta = row.get("cmetadata")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (ValueError, TypeError):
                meta = {}
        meta = meta or {}
        if meta.get("user_id") not in ent["entity_ids"]:
            continue
        tid = meta.get("tenant_id")
        if tid is not None and str(tid) != ent["tenant_id"]:
            continue
        out.append(row)
    return out


def _redact_row_cmetadata(rows: list) -> list:
    """Apply the #116 fail-closed redactor (`document_routes._redacted_metadata`) to each row's
    `cmetadata` (RV-197D item 5). These debug dumps do a raw ``SELECT *`` including `cmetadata`; a
    legacy row whose `quarantined_link` is still a raw string (or any non-canonical shape) would
    otherwise leak the raw URL/credentials/path/token here exactly as GET /documents did before F1.

    The redactor is REUSED, never re-implemented. `cmetadata` arrives as a dict or a JSON string
    (asyncpg JSONB), matching `filter_rows_by_entitlement`. A clean or already-canonical row is
    returned UNCHANGED (the redactor returns the same object, so no churn and the container type is
    preserved -- ``retrieved == stored``); only a row that actually carried a raw marker is rewritten,
    with the redacted `cmetadata` re-serialized to a string iff it arrived as one. A `cmetadata` that
    is neither a dict nor a parseable JSON object has no structured marker to strip and is left as-is.
    """
    out = []
    for row in rows:
        meta = row.get("cmetadata")
        was_str = isinstance(meta, str)
        parsed = meta
        if was_str:
            try:
                parsed = json.loads(meta)
            except (ValueError, TypeError):
                out.append(row)
                continue
        if not isinstance(parsed, dict):
            out.append(row)
            continue
        redacted = _redacted_metadata(parsed)
        if redacted is parsed:
            out.append(row)  # nothing under quarantine / already canonical -> untouched, no churn
            continue
        new_row = dict(row)
        new_row["cmetadata"] = json.dumps(redacted) if was_str else redacted
        out.append(new_row)
    return out


async def check_index_exists(table_name: str, column_name: str) -> bool:
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        result = await conn.fetch(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_indexes
                WHERE tablename = $1 
                AND indexdef LIKE '%' || $2 || '%'
            );
            """,
            table_name,
            column_name,
        )
    return result[0]['exists']


@router.get("/test/check_index")
async def check_file_id_index(table_name: str, column_name: str):
    if await check_index_exists(table_name, column_name):
        return {"message": f"Index on {column_name} exists in the table {table_name}."}
    else:
        return HTTPException(status_code=404, detail=f"No index on {column_name} found in the table {table_name}.")


@router.get("/db/tables")
async def get_table_names(schema: str = "public"):
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        table_names = await conn.fetch(
            """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = $1
            """,
            schema,
        )
    # Extract table names from records
    tables = [record['table_name'] for record in table_names]
    return {"schema": schema, "tables": tables}


@router.get("/db/tables/columns")
async def get_table_columns(table_name: str, schema: str = "public"):
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        columns = await conn.fetch(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = $1 AND table_name = $2
            ORDER BY ordinal_position;
            """,
            schema, table_name,
        )
    column_names = [col['column_name'] for col in columns]
    return {"table_name": table_name, "columns": column_names}


@router.get("/records/all")
async def get_all_records(request: Request, table_name: str):
    ent = require_read_entitlement(request)
    # Validate that the table name is one of the expected ones to prevent SQL injection
    if table_name not in ["langchain_pg_collection", "langchain_pg_embedding"]:
        raise HTTPException(status_code=400, detail="Invalid table name")

    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        # Use SQLAlchemy core or raw SQL queries to fetch all records
        records = await conn.fetch(f"SELECT * FROM {table_name};")

    # Convert records to JSON serializable format, assuming records can be directly serialized
    records_json = [dict(record) for record in records]

    # Never disclose vectors across entities/tenants (D-KSPT-1); never leak a raw quarantined
    # link (RV-197D item 5) -- redact AFTER the entitlement filter so a dropped row is never
    # merely redacted.
    return _redact_row_cmetadata(filter_rows_by_entitlement(records_json, ent))


@router.get("/records")
async def get_records_filtered_by_custom_id(request: Request, custom_id: str, table_name: str = "langchain_pg_embedding"):
    ent = require_read_entitlement(request)
    # Validate that the table name is one of the expected ones to prevent SQL injection
    if table_name not in ["langchain_pg_collection", "langchain_pg_embedding"]:
        raise HTTPException(status_code=400, detail="Invalid table name")

    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        # Use parameterized queries to prevent SQL Injection
        query = f"SELECT * FROM {table_name} WHERE custom_id=$1;"
        records = await conn.fetch(query, custom_id)

    # Convert records to JSON serializable format, assuming the Record class has a dict method.
    records_json = [dict(record) for record in records]

    # Never disclose vectors across entities/tenants (D-KSPT-1); never leak a raw quarantined
    # link (RV-197D item 5) -- redact AFTER the entitlement filter so a dropped row is never
    # merely redacted.
    return _redact_row_cmetadata(filter_rows_by_entitlement(records_json, ent))
