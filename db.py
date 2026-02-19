import sqlite3
from config import DB_PATH


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            email TEXT UNIQUE,
            pwd_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS detections (
            id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            stream_url TEXT,
            snapshot_url TEXT,
            image_path TEXT NOT NULL,
            thumb_path TEXT NOT NULL,
            count INTEGER NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            bundle_info TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )

    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_det_user_time ON detections(user_id, timestamp DESC)"
    )

    conn.commit()
    conn.close()
    
    # Also try to add column to existing table (if table already exists)
    try:
        conn = get_conn()
        cur = conn.cursor()
        
        # Check if column exists
        cur.execute("PRAGMA table_info(detections)")
        columns = [col[1] for col in cur.fetchall()]
        
        if 'bundle_info' not in columns:
            cur.execute("ALTER TABLE detections ADD COLUMN bundle_info TEXT")
            conn.commit()
            print("✅ Added bundle_info column to existing detections table")
    except Exception as e:
        print(f"Note: {e}")
    finally:
        conn.close()
