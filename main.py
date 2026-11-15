import duckdb
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# 1. Old Global Database
GLOBAL_PARQUET_URL = "https://huggingface.co/datasets/PhisherJR/GlobalTG/resolve/main/idx_tgid_in.parquet"

# 2. Sharded Database Base Pattern
SHARD_BASE_URL = "https://huggingface.co/datasets/00000N0/Tgdata/resolve/main/TG_DATA_PARTS/part_id%3D{part_id}/data_0.parquet"

db_conn = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_conn
    db_conn = duckdb.connect(database=":memory:")
    # Metadata cache enable taaki remote Parquet fast load ho
    db_conn.execute("SET enable_http_metadata_cache=true;")
    yield
    if db_conn:
        db_conn.close()

app = FastAPI(
    title="Telegram Multi-Source Parquet API",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_shard_url_from_query(query_str: str) -> str | None:
    """Extracts last 2 digits from query and formats shard URL."""
    digits = "".join(filter(str.isdigit, str(query_str)))
    if not digits:
        return None
    # Last 2 digits ko integer me convert karte hain (e.g. '00' -> 0, '05' -> 5, '28' -> 28)
    part_id = int(digits[-2:]) if len(digits) >= 2 else int(digits)
    return SHARD_BASE_URL.format(part_id=part_id)

def execute_parquet_search(file_url: str, search_val: str, explicit_col: str | None = None, limit: int = 10):
    """Dynamically detects matching ID column and queries remote parquet."""
    try:
        # Schema inspect
        schema_rows = db_conn.execute(f"DESCRIBE SELECT * FROM '{file_url}' LIMIT 1;").fetchall()
        columns = [row[0] for row in schema_rows]

        # Target column select
        if explicit_col and explicit_col in columns:
            target_cols = [explicit_col]
        else:
            # Common Telegram ID field names prioritize
            priority_names = ["tg_id", "id", "user_id", "telegram_id", "phone", "username"]
            target_cols = [c for c in priority_names if c in columns]
            if not target_cols:
                target_cols = columns  # Fallback to all columns if no standard ID field

        where_conditions = " OR ".join([f'CAST("{col}" AS VARCHAR) = ?' for col in target_cols])
        sql = f"SELECT * FROM '{file_url}' WHERE {where_conditions} LIMIT {limit};"

        cursor = db_conn.execute(sql, [str(search_val)] * len(target_cols))
        result_cols = [c[0] for c in cursor.description]
        rows = cursor.fetchall()

        # Clean empty / null fields from records
        results = []
        for row in rows:
            record = {k: v for k, v in zip(result_cols, row) if v is not None and v != ""}
            results.append(record)

        return {"status": "success", "count": len(results), "data": results, "error": None}

    except Exception as e:
        return {"status": "error", "count": 0, "data": [], "error": str(e)}

@app.get("/")
def root():
    return {
        "status": "online",
        "description": "Multi-source Parquet Search API (Global + Sharded 0-99)",
        "endpoints": {
            "/search?q=12345678": "Search across both Global and Sharded databases"
        }
    }

@app.get("/search")
def search(
    q: str = Query(..., description="Telegram ID / Phone / Query value"),
    column: str = Query(default=None, description="Optional: specific column to search"),
    limit: int = Query(default=10, le=50, description="Max records to return per database")
):
    q_clean = q.strip()
    if not q_clean:
        raise HTTPException(status_code=400, detail="Query parameter 'q' cannot be empty.")

    # 1. Query Global Parquet (Old Data)
    global_res = execute_parquet_search(
        file_url=GLOBAL_PARQUET_URL,
        search_val=q_clean,
        explicit_col=column,
        limit=limit
    )

    # 2. Query Sharded Parquet (New Data based on last 2 digits)
    shard_url = get_shard_url_from_query(q_clean)
    if shard_url:
        shard_res = execute_parquet_search(
            file_url=shard_url,
            search_val=q_clean,
            explicit_col=column,
            limit=limit
        )
    else:
        shard_res = {
            "status": "skipped",
            "count": 0,
            "data": [],
            "error": "Query contains no numeric digits to determine shard ID."
        }

    # Combined total count
    total_found = global_res["count"] + shard_res["count"]

    return {
        "query": q_clean,
        "total_results": total_found,
        "sources": {
            "global_db": {
                "count": global_res["count"],
                "results": global_res["data"],
                "error": global_res["error"]
            },
            "sharded_db": {
                "shard_part": int("".join(filter(str.isdigit, q_clean))[-2:]) if shard_url else None,
                "count": shard_res["count"],
                "results": shard_res["data"],
                "error": shard_res["error"]
            }
        }
    }
