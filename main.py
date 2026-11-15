import time
import traceback
import duckdb
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# 1st Database (Column: telegram_id)
DB1_GLOBAL_URL = "https://huggingface.co/datasets/PhisherJR/GlobalTG/resolve/main/idx_tgid_in.parquet"

# 2nd Database - Sharded by last 2 digits (Column: user_id)
DB2_SHARD_BASE_URL = "https://huggingface.co/datasets/00000N0/Tgdata/resolve/main/TG_DATA_PARTS/part_id%3D{part_id}/data_0.parquet"

# 3rd Database - Merged All (Column: user_id)
DB3_MERGED_URL = "https://huggingface.co/datasets/00000N0/Tg/resolve/main/simple_all/merged_all.parquet"

app = FastAPI(
    title="Telegram Multi-Endpoint Parquet API",
    version="5.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def log(msg: str):
    """Prints immediately to Render logs without buffering."""
    print(f"[API-LOG] {msg}", flush=True)

def create_connection():
    """Creates a lightweight, Render-safe DuckDB connection."""
    conn = duckdb.connect(database=":memory:")
    # Render Free Tier (512MB RAM) safe settings
    safe_settings = [
        "SET memory_limit='256MB';",
        "SET threads=2;",
        "SET enable_http_metadata_cache=true;",
        "SET http_keep_alive=true;",
        "SET http_retries=2;",
        "SET http_timeout=20;"
    ]
    for s in safe_settings:
        try:
            conn.execute(s)
        except Exception:
            pass
    return conn

# Global connection initialized at startup
db_conn = create_connection()

def clean_user_id(raw_id: str | None) -> str:
    if not raw_id:
        raise HTTPException(status_code=400, detail="Please provide a Telegram ID.")
    clean_id = "".join(filter(str.isdigit, str(raw_id).strip()))
    if not clean_id:
        raise HTTPException(status_code=400, detail="Invalid ID. Only numeric Telegram IDs are allowed.")
    return clean_id

def get_shard_info(clean_id: str) -> tuple[int, str]:
    part_id = int(clean_id[-2:]) if len(clean_id) >= 2 else int(clean_id)
    return part_id, DB2_SHARD_BASE_URL.format(part_id=part_id)

def query_parquet(db_label: str, file_url: str, col_name: str, clean_id: str, limit: int = 10):
    """
    Executes fast Parquet lookup with live logging.
    Uses string literal comparison so DuckDB auto-casts to BIGINT if the column is numeric,
    or compares directly as VARCHAR if the column is string (avoiding INT32 overflow).
    """
    start_time = time.time()
    log(f"[{db_label}] START -> ID: {clean_id} | Column: {col_name} | URL: {file_url}")

    try:
        cursor = db_conn.cursor()

        # clean_id contains strictly digits (0-9), so literal formatting is 100% SQLi-safe
        # and avoids Python INT32 parameter binding issues in DuckDB.
        sql = f"SELECT * FROM '{file_url}' WHERE \"{col_name}\" = '{clean_id}' LIMIT {int(limit)}"
        
        try:
            res = cursor.execute(sql)
            col_names = [col[0] for col in res.description]
            rows = res.fetchall()
        except Exception as inner_err:
            log(f"[{db_label}] Primary query warning: {inner_err}. Trying TRY_CAST fallback...")
            sql_fallback = f"SELECT * FROM '{file_url}' WHERE TRY_CAST(\"{col_name}\" AS BIGINT) = {int(clean_id)} LIMIT {int(limit)}"
            res = cursor.execute(sql_fallback)
            col_names = [col[0] for col in res.description]
            rows = res.fetchall()

        elapsed = round(time.time() - start_time, 2)
        log(f"[{db_label}] SUCCESS in {elapsed}s -> Rows found: {len(rows)}")

        results = []
        for row in rows:
            record = {k: v for k, v in zip(col_names, row) if v is not None and v != ""}
            results.append(record)

        return {
            "status": "success",
            "found": len(results) > 0,
            "time_taken_sec": elapsed,
            "searched_column": col_name,
            "count": len(results),
            "results": results if results else "No record found",
            "error": None
        }

    except Exception as e:
        elapsed = round(time.time() - start_time, 2)
        err_msg = str(e)
        log(f"[{db_label}] ERROR after {elapsed}s -> {err_msg}")
        log(traceback.format_exc())
        return {
            "status": "error",
            "found": False,
            "time_taken_sec": elapsed,
            "searched_column": col_name,
            "count": 0,
            "results": "No record found",
            "error": err_msg
        }

@app.get("/")
def root():
    return {
        "status": "online",
        "version": "5.0.0",
        "endpoints": {
            "debug_check": "/debug (Open this to test all 3 DBs and see exact errors/columns)",
            "db1": ["/db1/723625545", "/search/db1/723625545", "/db1?id=723625545"],
            "db2": ["/db2/723625545", "/search/db2/723625545", "/db2?id=723625545"],
            "db3": ["/db3/723625545", "/search/db3/723625545", "/db3?id=723625545"],
            "all_combined": ["/search/723625545", "/search?id=723625545"]
        }
    }

# =========================================================
# DIAGNOSTIC ENDPOINT: /debug (Pata lagane ke liye kya error hai)
# =========================================================
@app.get("/debug")
def debug_databases():
    """Tests connection and schema of all 3 Parquet URLs and returns exact details."""
    log("Running /debug diagnostic check on all 3 databases...")
    test_urls = {
        "db1_global": DB1_GLOBAL_URL,
        "db2_shard_sample": DB2_SHARD_BASE_URL.format(part_id=14),
        "db3_merged": DB3_MERGED_URL
    }
    report = {}
    for name, url in test_urls.items():
        t0 = time.time()
        try:
            cursor = db_conn.cursor()
            schema = cursor.execute(f"DESCRIBE SELECT * FROM '{url}' LIMIT 1;").fetchall()
            sample = cursor.execute(f"SELECT * FROM '{url}' LIMIT 1;").fetchall()
            elapsed = round(time.time() - t0, 2)
            report[name] = {
                "reachable": True,
                "ping_sec": elapsed,
                "columns": [{r[0]: r[1]} for r in schema],
                "sample_row_loaded": len(sample) > 0,
                "error": None
            }
            log(f"[DEBUG] {name} OK ({elapsed}s)")
        except Exception as e:
            elapsed = round(time.time() - t0, 2)
            report[name] = {
                "reachable": False,
                "ping_sec": elapsed,
                "columns": [],
                "error": str(e)
            }
            log(f"[DEBUG] {name} FAILED ({elapsed}s): {e}")
    return report

# ==========================================
# DB 1 ENDPOINTS
# ==========================================
@app.get("/db1")
@app.get("/search/db1")
def search_db1_query(id: str = Query(default=None), q: str = Query(default=None), limit: int = Query(default=10, le=50)):
    clean_id = clean_user_id(id or q)
    res = query_parquet("DB1", DB1_GLOBAL_URL, "telegram_id", clean_id, limit)
    return {"database": "db1_global", "id": int(clean_id), **res}

@app.get("/db1/{user_id}")
@app.get("/search/db1/{user_id}")
def search_db1_path(user_id: str, limit: int = Query(default=10, le=50)):
    clean_id = clean_user_id(user_id)
    res = query_parquet("DB1", DB1_GLOBAL_URL, "telegram_id", clean_id, limit)
    return {"database": "db1_global", "id": int(clean_id), **res}

# ==========================================
# DB 2 ENDPOINTS
# ==========================================
@app.get("/db2")
@app.get("/search/db2")
def search_db2_query(id: str = Query(default=None), q: str = Query(default=None), limit: int = Query(default=10, le=50)):
    clean_id = clean_user_id(id or q)
    part_id, shard_url = get_shard_info(clean_id)
    res = query_parquet(f"DB2-Part-{part_id}", shard_url, "user_id", clean_id, limit)
    return {"database": "db2_sharded", "shard_part": part_id, "id": int(clean_id), **res}

@app.get("/db2/{user_id}")
@app.get("/search/db2/{user_id}")
def search_db2_path(user_id: str, limit: int = Query(default=10, le=50)):
    clean_id = clean_user_id(user_id)
    part_id, shard_url = get_shard_info(clean_id)
    res = query_parquet(f"DB2-Part-{part_id}", shard_url, "user_id", clean_id, limit)
    return {"database": "db2_sharded", "shard_part": part_id, "id": int(clean_id), **res}

# ==========================================
# DB 3 ENDPOINTS
# ==========================================
@app.get("/db3")
@app.get("/search/db3")
def search_db3_query(id: str = Query(default=None), q: str = Query(default=None), limit: int = Query(default=10, le=50)):
    clean_id = clean_user_id(id or q)
    res = query_parquet("DB3", DB3_MERGED_URL, "user_id", clean_id, limit)
    return {"database": "db3_merged", "id": int(clean_id), **res}

@app.get("/db3/{user_id}")
@app.get("/search/db3/{user_id}")
def search_db3_path(user_id: str, limit: int = Query(default=10, le=50)):
    clean_id = clean_user_id(user_id)
    res = query_parquet("DB3", DB3_MERGED_URL, "user_id", clean_id, limit)
    return {"database": "db3_merged", "id": int(clean_id), **res}

# ==========================================
# ALL 3 COMBINED (/search) WITH TIMEOUT PROTECTION
# ==========================================
@app.get("/search")
def search_all_query(id: str = Query(default=None), q: str = Query(default=None), limit: int = Query(default=10, le=50)):
    clean_id = clean_user_id(id or q)
    return run_all_databases(clean_id, limit)

@app.get("/search/{user_id}")
def search_all_path(user_id: str, limit: int = Query(default=10, le=50)):
    clean_id = clean_user_id(user_id)
    return run_all_databases(clean_id, limit)

def safe_future_result(future, db_name: str, timeout_sec: int = 20):
    try:
        return future.result(timeout=timeout_sec)
    except FuturesTimeoutError:
        log(f"[{db_name}] TIMEOUT after {timeout_sec}s!")
        return {
            "status": "timeout",
            "found": False,
            "time_taken_sec": timeout_sec,
            "count": 0,
            "results": "No record found",
            "error": f"{db_name} query took longer than {timeout_sec}s (file too large for free tier)."
        }
    except Exception as e:
        return {
            "status": "error",
            "found": False,
            "count": 0,
            "results": "No record found",
            "error": str(e)
        }

def run_all_databases(clean_id: str, limit: int):
    log(f"[/search ALL] Starting combined search for ID: {clean_id}")
    part_id, shard_url = get_shard_info(clean_id)

    executor = ThreadPoolExecutor(max_workers=3)
    future_db1 = executor.submit(query_parquet, "DB1", DB1_GLOBAL_URL, "telegram_id", clean_id, limit)
    future_db2 = executor.submit(query_parquet, f"DB2-Part-{part_id}", shard_url, "user_id", clean_id, limit)
    future_db3 = executor.submit(query_parquet, "DB3", DB3_MERGED_URL, "user_id", clean_id, limit)

    res_db1 = safe_future_result(future_db1, "DB1", timeout_sec=20)
    res_db2 = safe_future_result(future_db2, "DB2", timeout_sec=20)
    res_db3 = safe_future_result(future_db3, "DB3", timeout_sec=20)
    executor.shutdown(wait=False)

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
