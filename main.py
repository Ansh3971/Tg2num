import time
import duckdb
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# 1st Database (Column: telegram_id)
DB1_GLOBAL_URL = "https://huggingface.co/datasets/PhisherJR/GlobalTG/resolve/main/idx_tgid_in.parquet"

# 2nd Database - Sharded by last 2 digits (Columns: user_id, linked_id, linked_name, linked_handle)
DB2_SHARD_BASE_URL = "https://huggingface.co/datasets/00000N0/Tgdata/resolve/main/TG_DATA_PARTS/part_id%3D{part_id}/data_0.parquet"

# 3rd Database - Merged All (Column: user_id)
DB3_MERGED_URL = "https://huggingface.co/datasets/00000N0/Tg/resolve/main/simple_all/merged_all.parquet"

app = FastAPI(
    title="Telegram Multi-Endpoint Parquet API (Render 512MB Safe)",
    version="7.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def log(msg: str):
    print(f"[API-LOG] {msg}", flush=True)

def create_isolated_connection():
    """
    Creates a strict low-memory DuckDB connection so Render Free Tier (512MB)
    NEVER runs Out-Of-Memory (OOM) or restarts the server.
    """
    conn = duckdb.connect(database=":memory:")
    settings = [
        "SET memory_limit='128MB';",          # Strict 128MB cap prevents Render container kill
        "SET threads=1;",                     # 1 thread prevents memory spikes
        "SET preserve_insertion_order=false;",# Saves 40% RAM during Parquet scans
        "SET enable_http_metadata_cache=true;",
        "SET enable_object_cache=true;",
        "SET http_keep_alive=true;",
        "SET http_retries=2;",
        "SET http_timeout=15;"
    ]
    for s in settings:
        try:
            conn.execute(s)
        except Exception:
            pass
    return conn

# Global connection for fast sharded/indexed DB1 & DB2
db_conn = create_isolated_connection()

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
    Executes single-pass Parquet scan without rowid error and without OOM crashes.
    """
    start_time = time.time()
    log(f"[{db_label}] START -> ID: {clean_id} | Column: {col_name}")

    # Use an isolated connection for DB3 so we can safely interrupt/close it if it takes too long
    is_db3 = (db_label == "DB3")
    conn = create_isolated_connection() if is_db3 else db_conn

    try:
        cursor = conn.cursor()

        if db_label == "DB1":
            # DB1 has BIGINT telegram_id
            sql = f"SELECT * FROM '{file_url}' WHERE \"{col_name}\" = {int(clean_id)} LIMIT {int(limit)}"
        else:
            # DB2 & DB3 have VARCHAR user_id
            sql = f"SELECT * FROM '{file_url}' WHERE \"{col_name}\" = '{clean_id}' LIMIT {int(limit)}"

        res = cursor.execute(sql)
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
            "available_columns": col_names,
            "count": len(results),
            "results": results if results else "No record found",
            "error": None
        }

    except Exception as e:
        elapsed = round(time.time() - start_time, 2)
        err_msg = str(e)
        log(f"[{db_label}] ERROR after {elapsed}s -> {err_msg}")
        return {
            "status": "error",
            "found": False,
            "time_taken_sec": elapsed,
            "searched_column": col_name,
            "count": 0,
            "results": "No record found",
            "error": err_msg
        }
    finally:
        if is_db3:
            try:
                conn.close()
            except Exception:
                pass

def query_db3_with_timeout(clean_id: str, limit: int = 10, timeout_sec: int = 12):
    """
    Runs DB3 in a separate thread with a strict timeout and conn.interrupt()
    so it NEVER hangs Render or causes an OOM reboot.
    """
    start_time = time.time()
    log(f"[DB3] START (Protected {timeout_sec}s Timeout) -> ID: {clean_id}")
    conn = create_isolated_connection()

    def _worker():
        cursor = conn.cursor()
        sql = f"SELECT * FROM '{DB3_MERGED_URL}' WHERE \"user_id\" = '{clean_id}' LIMIT {int(limit)}"
        res = cursor.execute(sql)
        cols = [c[0] for c in res.description]
        rows = res.fetchall()
        return cols, rows

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_worker)

    try:
        col_names, rows = future.result(timeout=timeout_sec)
        elapsed = round(time.time() - start_time, 2)
        log(f"[DB3] SUCCESS in {elapsed}s -> Rows found: {len(rows)}")
        results = [{k: v for k, v in zip(col_names, row) if v is not None and v != ""} for row in rows]
        return {
            "status": "success",
            "found": len(results) > 0,
            "time_taken_sec": elapsed,
            "searched_column": "user_id",
            "available_columns": col_names,
            "count": len(results),
            "results": results if results else "No record found",
            "error": None
        }
    except FuturesTimeoutError:
        # Immediately cancel the running C++ DuckDB query so RAM/CPU is freed!
        try:
            conn.interrupt()
            conn.close()
        except Exception:
            pass
        elapsed = round(time.time() - start_time, 2)
        log(f"[DB3] INTERRUPTED safely at {elapsed}s to prevent Render crash.")
        return {
            "status": "timeout_protected",
            "found": False,
            "time_taken_sec": elapsed,
            "searched_column": "user_id",
            "note": "DB3 (merged_all.parquet) only contains 1 column ('user_id') and is unsharded.",
            "count": 0,
            "results": "No record found within safe memory/time window",
            "error": None
        }
    except Exception as e:
        elapsed = round(time.time() - start_time, 2)
        return {
            "status": "error",
            "found": False,
            "time_taken_sec": elapsed,
            "count": 0,
            "results": "No record found",
            "error": str(e)
        }
    finally:
        executor.shutdown(wait=False)

@app.get("/")
def root():
    return {
        "status": "online",
        "version": "7.0.0 (OOM Crash Proof + conn.interrupt() Protection)",
        "endpoints": {
            "debug_schemas": "/debug (Check columns of DB1, DB2, DB3)",
            "db1": ["/db1/8985195736", "/search/db1/8985195736", "/db1?id=8985195736"],
            "db2": ["/db2/8985195736", "/search/db2/8985195736", "/db2?id=8985195736"],
            "db3": ["/db3/8985195736", "/search/db3/8985195736", "/db3?id=8985195736"],
            "all_combined": ["/search/8985195736", "/search?id=8985195736"]
        }
    }

@app.get("/debug")
def debug_databases():
    """Shows exact columns and sample row of all 3 DBs."""
    test_urls = {
        "db1_global": DB1_GLOBAL_URL,
        "db2_shard_sample": DB2_SHARD_BASE_URL.format(part_id=36),
        "db3_merged": DB3_MERGED_URL
    }
    report = {}
    for name, url in test_urls.items():
        t0 = time.time()
        try:
            cursor = db_conn.cursor()
            schema = cursor.execute(f"DESCRIBE SELECT * FROM '{url}' LIMIT 1;").fetchall()
            sample = cursor.execute(f"SELECT * FROM '{url}' LIMIT 2;").fetchall()
            elapsed = round(time.time() - t0, 2)
            report[name] = {
                "reachable": True,
                "ping_sec": elapsed,
                "columns": [{r[0]: r[1]} for r in schema],
                "sample_rows": sample
            }
        except Exception as e:
            report[name] = {"reachable": False, "error": str(e)}
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
# DB 3 ENDPOINTS (With OOM & Interrupt Protection)
# ==========================================
@app.get("/db3")
@app.get("/search/db3")
def search_db3_query(id: str = Query(default=None), q: str = Query(default=None), limit: int = Query(default=10, le=50)):
    clean_id = clean_user_id(id or q)
    res = query_db3_with_timeout(clean_id, limit, timeout_sec=12)
    return {"database": "db3_merged", "id": int(clean_id), **res}

@app.get("/db3/{user_id}")
@app.get("/search/db3/{user_id}")
def search_db3_path(user_id: str, limit: int = Query(default=10, le=50)):
    clean_id = clean_user_id(user_id)
    res = query_db3_with_timeout(clean_id, limit, timeout_sec=12)
    return {"database": "db3_merged", "id": int(clean_id), **res}

# ==========================================
# ALL 3 COMBINED (/search)
# ==========================================
@app.get("/search")
def search_all_query(id: str = Query(default=None), q: str = Query(default=None), limit: int = Query(default=10, le=50)):
    clean_id = clean_user_id(id or q)
    return run_all_databases(clean_id, limit)

@app.get("/search/{user_id}")
def search_all_path(user_id: str, limit: int = Query(default=10, le=50)):
    clean_id = clean_user_id(user_id)
    return run_all_databases(clean_id, limit)

def run_all_databases(clean_id: str, limit: int):
    log(f"[/search ALL] Starting combined search for ID: {clean_id}")
    part_id, shard_url = get_shard_info(clean_id)

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_db1 = executor.submit(query_parquet, "DB1", DB1_GLOBAL_URL, "telegram_id", clean_id, limit)
        future_db2 = executor.submit(query_parquet, f"DB2-Part-{part_id}", shard_url, "user_id", clean_id, limit)
        future_db3 = executor.submit(query_db3_with_timeout, clean_id, limit, 10)

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
