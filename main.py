import duckdb
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

PARQUET_URL = "https://huggingface.co/datasets/PhisherJR/GlobalTG/resolve/main/idx_tgid_in.parquet"

# DuckDB read-only connection
db_conn = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_conn
    # DuckDB in-memory initialization
    db_conn = duckdb.connect(database=":memory:")
    # httpfs automatically enabled in DuckDB >= 1.0, explicit enable for safety
    db_conn.execute("SET enable_http_metadata_cache=true;")
    yield
    if db_conn:
        db_conn.close()

app = FastAPI(
    title="Parquet Query API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS allow for frontend / web integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "status": "online",
        "endpoints": {
            "/schema": "Check columns and data types in the parquet file",
            "/search?q=VALUE": "Search records by specific ID/column"
        }
    }

@app.get("/schema")
def get_schema():
    """Inspects table columns and types without reading the whole file."""
    try:
        schema_query = f"DESCRIBE SELECT * FROM '{PARQUET_URL}' LIMIT 1;"
        result = db_conn.execute(schema_query).fetchall()
        columns = [{"name": row[0], "type": row[1]} for row in result]
        return {"columns": columns}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/search")
def search(
    column: str = Query(default="tg_id", description="Column to search in (e.g., tg_id, id, etc.)"),
    q: str = Query(..., description="Query value to search"),
    limit: int = Query(default=10, le=100, description="Max records to return")
):
    try:
        # Validate column name to prevent SQL injection
        allowed_columns = [
            row[0] for row in db_conn.execute(f"DESCRIBE SELECT * FROM '{PARQUET_URL}' LIMIT 1;").fetchall()
        ]
        
        if column not in allowed_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid column '{column}'. Available columns: {allowed_columns}"
            )

        # Parameterized query execution
        query_sql = f"""
            SELECT * 
            FROM '{PARQUET_URL}' 
            WHERE CAST("{column}" AS VARCHAR) = ? 
            LIMIT ?
        """
        cursor = db_conn.execute(query_sql, [str(q), limit])
        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()

        if not rows:
            return {"count": 0, "results": [], "message": "No records found"}

        data = [dict(zip(columns, row)) for row in rows]
        return {
            "count": len(data),
            "results": data
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")
