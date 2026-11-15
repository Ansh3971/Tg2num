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
    # Remote Parquet metadata caching for faster repeat lookups
    db_conn.execute("SET enable_http_metadata_cache=true;")
    yield
    if db_conn:
        db_conn.close()

app = FastAPI(
    title="Telegram 3-in-1 Parquet Search API",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_shard_info(clean_id: str) -> tuple[int, str]:
    """Extracts last 2 digits from ID and returns (part_id, shard_url)."""
    part_id = int(clean_id[-2:]) if len(clean_id) >= 2 else int(clean_id)
    return part_id, DB2_SHARD_BASE_URL.format(part_id=part_id)

def query_parquet(file_url: str, column_name: str, user_id_val: str, limit: int = 10):
    """Thread-safe DuckDB query targeting the exact column."""
    try:
        # Har thread ke liye alag cursor taaki parallel queries safe rahein
        local_cursor = db_conn.cursor()

        # Bare column filter rakha hai taaki Parquet row-group pushdown fast kaam kare
        sql = f'SELECT * FROM \'{file_url}\' WHERE "{column_name}" = ? LIMIT ?'
        
        try:
            # Agar column BIGINT/INT hai toh integer parameter fast match karega
            param = int(user_id_val) if user_id_val.isdigit() else user_id_val
            res = local_cursor.execute(sql, [param, limit])
        except Exception:
            # Fallback agar column VARCHAR string format me ho
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
            "searched_column": column_name,
            "count": len(results),
            "results": results,
            "error": None
        }
    except Exception as e:
        return {
            "status": "error",
            "searched_column": column_name,
            "count": 0,
            "results": [],
            "error": str(e)
        }

@app.get("/")
def root():
    return {
        "status": "online",
        "usage": {
            "by_id_param": "/search?id=123456789",
            "by_q_param": "/search?q=123456789",
            "direct_path": "/search/123456789"
        }
    }

@app.get("/search")
def search_by_query(
    id: str = Query(default=None, description="Telegram User ID"),
    q: str = Query(default=None, description="Alternative query param for ID"),
    limit: int = Query(default=10, le=50)
):
    target_id = id or q
    if not target_id:
        raise HTTPException(status_code=400, detail="Please provide an ID using ?id=123456 or ?q=123456")
    return run_all_databases(target_id, limit)

@app.get("/search/{user_id}")
def search_by_path(user_id: str, limit: int = Query(default=10, le=50)):
    return run_all_databases(user_id, limit)

def run_all_databases(raw_id: str, limit: int):
    # Sirf digits extract karein (agar koi space ya symbol aa jaye toh clean ho jaye)
    clean_id = "".join(filter(str.isdigit, str(raw_id).strip()))
    if not clean_id:
        raise HTTPException(status_code=400, detail="Invalid ID. Only numeric Telegram IDs are allowed.")

    part_id, shard_url = get_shard_info(clean_id)

    # Teeno databases ko ek saath (parallel) query karein
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_db1 = executor.submit(query_parquet, DB1_GLOBAL_URL, "telegram_id", clean_id, limit)
        future_db2 = executor.submit(query_parquet, shard_url, "user_id", clean_id, limit)
        future_db3 = executor.submit(query_parquet, DB3_MERGED_URL, "user_id", clean_id, limit)

        res_db1 = future_db1.result()
        res_db2 = future_db2.result()
        res_db3 = future_db3.result()

    # Sabhi results ki ek combined list bhi bana dete hain aasan access ke liye
    combined_records = res_db1["results"] + res_db2["results"] + res_db3["results"]
    total_found = len(combined_records)

    return {
        "id": int(clean_id),
        "found": total_found > 0,
        "total_results": total_found,
        "combined_data": combined_records,
        "sources": {
            "db1_global": res_db1,
            "db2_sharded": {
                "shard_part": part_id,
                **res_db2
            },
            "db3_merged": res_db3
        }
    }
