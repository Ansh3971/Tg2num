import duckdb
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# 1st Database (Column: telegram_id)
DB1_GLOBAL_URL = "https://huggingface.co/datasets/PhisherJR/GlobalTG/resolve/main/idx_tgid_in.parquet"

# 2nd Database - Sharded by last 2 digits (Column: user_id)
DB2_SHARD_BASE_URL = "https://huggingface.co/datasets/00000N0/Tgdata/resolve/main/TG_DATA_PARTS/part_id%3D{part_id}/data_0.parquet"

# 3rd Database - Merged All (Column: user_id)
DB3_MERGED_URL = "https://huggingface.co/datasets/00000N0/Tg/resolve/main/simple_all/merged_all.parquet"

db_conn = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_conn
    db_conn = duckdb.connect(database=":memory:")
    db_conn.execute("SET enable_http_metadata_cache=true;")
    yield
    if db_conn:
        db_conn.close()

app = FastAPI(
    title="Telegram Multi-Endpoint Parquet API",
    version="4.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def clean_user_id(raw_id: str | None) -> str:
    """Extracts numeric digits and validates the ID."""
    if not raw_id:
        raise HTTPException(status_code=400, detail="Please provide a Telegram ID using ?id=123456 or ?q=123456")
    clean_id = "".join(filter(str.isdigit, str(raw_id).strip()))
    if not clean_id:
        raise HTTPException(status_code=400, detail="Invalid ID. Only numeric Telegram IDs are allowed.")
    return clean_id

def get_shard_info(clean_id: str) -> tuple[int, str]:
    """Extracts last 2 digits from ID and returns (part_id, shard_url)."""
    part_id = int(clean_id[-2:]) if len(clean_id) >= 2 else int(clean_id)
    return part_id, DB2_SHARD_BASE_URL.format(part_id=part_id)

def query_parquet(file_url: str, column_name: str, user_id_val: str, limit: int = 10):
    """Executes DuckDB query on the specified remote Parquet file and column."""
    try:
        local_cursor = db_conn.cursor()
        sql = f'SELECT * FROM \'{file_url}\' WHERE "{column_name}" = ? LIMIT ?'
        
        try:
            # Try integer match first for fast Parquet row-group pushdown
            param = int(user_id_val)
            res = local_cursor.execute(sql, [param, limit])
        except Exception:
            # Fallback to string matching if column type is VARCHAR
            sql_fallback = f'SELECT * FROM \'{file_url}\' WHERE CAST("{column_name}" AS VARCHAR) = ? LIMIT ?'
            res = local_cursor.execute(sql_fallback, [str(user_id_val), limit])

        col_names = [col[0] for col in res.description]
        rows = res.fetchall()

        results = []
        for row in rows:
            record = {k: v for k, v in zip(col_names, row) if v is not None and v != ""}
            results.append(record)

        return {
            "status": "success",
            "found": len(results) > 0,
            "searched_column": column_name,
            "count": len(results),
            "results": results if results else "No record found",
            "error": None
        }
    except Exception as e:
        return {
            "status": "error",
            "found": False,
            "searched_column": column_name,
            "count": 0,
            "results": [],
            "error": str(e)
        }

@app.get("/")
def root():
    return {
        "status": "online",
        "endpoints": {
            "db1_only": "/db1?id=123456789 (or /db1/123456789) -> Searches GlobalTG (telegram_id)",
            "db2_only": "/db2?id=123456789 (or /db2/123456789) -> Searches Sharded DB (user_id, last 2 digits)",
            "db3_only": "/db3?id=123456789 (or /db3/123456789) -> Searches Merged DB (user_id)",
            "all_combined": "/search?id=123456789 (or /search/123456789) -> Searches all 3 DBs"
        }
    }

# ==========================================
# ENDPOINT 1: DB 1 ONLY (GlobalTG - telegram_id)
# ==========================================
@app.get("/db1")
def search_db1_query(
    id: str = Query(default=None, description="Telegram ID"),
    q: str = Query(default=None, description="Alternative param for Telegram ID"),
    limit: int = Query(default=10, le=50)
):
    clean_id = clean_user_id(id or q)
    res = query_parquet(DB1_GLOBAL_URL, "telegram_id", clean_id, limit)
    return {"database": "db1_global", "id": int(clean_id), **res}

@app.get("/db1/{user_id}")
def search_db1_path(user_id: str, limit: int = Query(default=10, le=50)):
    clean_id = clean_user_id(user_id)
    res = query_parquet(DB1_GLOBAL_URL, "telegram_id", clean_id, limit)
    return {"database": "db1_global", "id": int(clean_id), **res}


# ==========================================
# ENDPOINT 2: DB 2 ONLY (Sharded 0-99 - user_id)
# ==========================================
@app.get("/db2")
def search_db2_query(
    id: str = Query(default=None, description="Telegram ID"),
    q: str = Query(default=None, description="Alternative param for Telegram ID"),
    limit: int = Query(default=10, le=50)
):
    clean_id = clean_user_id(id or q)
    part_id, shard_url = get_shard_info(clean_id)
    res = query_parquet(shard_url, "user_id", clean_id, limit)
    return {"database": "db2_sharded", "shard_part": part_id, "id": int(clean_id), **res}

@app.get("/db2/{user_id}")
def search_db2_path(user_id: str, limit: int = Query(default=10, le=50)):
    clean_id = clean_user_id(user_id)
    part_id, shard_url = get_shard_info(clean_id)
    res = query_parquet(shard_url, "user_id", clean_id, limit)
    return {"database": "db2_sharded", "shard_part": part_id, "id": int(clean_id), **res}


# ==========================================
# ENDPOINT 3: DB 3 ONLY (Merged All - user_id)
# ==========================================
@app.get("/db3")
def search_db3_query(
    id: str = Query(default=None, description="Telegram ID"),
    q: str = Query(default=None, description="Alternative param for Telegram ID"),
    limit: int = Query(default=10, le=50)
):
    clean_id = clean_user_id(id or q)
    res = query_parquet(DB3_MERGED_URL, "user_id", clean_id, limit)
    return {"database": "db3_merged", "id": int(clean_id), **res}

@app.get("/db3/{user_id}")
def search_db3_path(user_id: str, limit: int = Query(default=10, le=50)):
    clean_id = clean_user_id(user_id)
    res = query_parquet(DB3_MERGED_URL, "user_id", clean_id, limit)
    return {"database": "db3_merged", "id": int(clean_id), **res}


# ==========================================
# BONUS ENDPOINT: ALL 3 COMBINED (/search)
# ==========================================
@app.get("/search")
def search_all_query(
    id: str = Query(default=None, description="Telegram ID"),
    q: str = Query(default=None, description="Alternative param for Telegram ID"),
    limit: int = Query(default=10, le=50)
):
    clean_id = clean_user_id(id or q)
    return run_all_databases(clean_id, limit)

@app.get("/search/{user_id}")
def search_all_path(user_id: str, limit: int = Query(default=10, le=50)):
    clean_id = clean_user_id(user_id)
    return run_all_databases(clean_id, limit)

def run_all_databases(clean_id: str, limit: int):
    part_id, shard_url = get_shard_info(clean_id)

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_db1 = executor.submit(query_parquet, DB1_GLOBAL_URL, "telegram_id", clean_id, limit)
        future_db2 = executor.submit(query_parquet, shard_url, "user_id", clean_id, limit)
        future_db3 = executor.submit(query_parquet, DB3_MERGED_URL, "user_id", clean_id, limit)

        res_db1 = future_db1.result()
        res_db2 = future_db2.result()
        res_db3 = future_db3.result()

    list_db1 = res_db1["results"] if isinstance(res_db1["results"], list) else []
    list_db2 = res_db2["results"] if isinstance(res_db2["results"], list) else []
    list_db3 = res_db3["results"] if isinstance(res_db3["results"], list) else []

    combined_records = list_db1 + list_db2 + list_db3
    total_found = len(combined_records)

    return {
        "id": int(clean_id),
        "found": total_found > 0,
        "total_results": total_found,
        "combined_data": combined_records if combined_records else "No record found",
        "sources": {
            "db1_global": res_db1,
            "db2_sharded": {"shard_part": part_id, **res_db2},
            "db3_merged": res_db3
        }
    }
