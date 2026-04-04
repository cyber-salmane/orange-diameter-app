import sqlite3
from pathlib import Path
from typing import Optional, Dict, List
import logging

from config import DB_PATH

logger = logging.getLogger(__name__)

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id           TEXT PRIMARY KEY,
            username     TEXT UNIQUE NOT NULL,
            email        TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            ip           TEXT DEFAULT '',
            first_seen   TEXT NOT NULL,
            last_seen    TEXT NOT NULL,
            is_banned    INTEGER DEFAULT 0,
            is_verified  INTEGER DEFAULT 0,
            verification_token TEXT,
            created_at   TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS password_resets (
            id          TEXT PRIMARY KEY,
            user_id     TEXT NOT NULL,
            token       TEXT UNIQUE NOT NULL,
            created_at  TEXT NOT NULL,
            used        INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS login_attempts (
            id          TEXT PRIMARY KEY,
            username    TEXT NOT NULL,
            ip          TEXT NOT NULL,
            success     INTEGER NOT NULL,
            timestamp   TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS uploads (
            id          TEXT PRIMARY KEY,
            user_id     TEXT NOT NULL,
            filename    TEXT NOT NULL,
            path        TEXT NOT NULL,
            upload_type TEXT NOT NULL,
            file_size   INTEGER NOT NULL,
            timestamp   TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS analyses (
            id             TEXT PRIMARY KEY,
            user_id        TEXT NOT NULL,
            num_oranges    INTEGER NOT NULL,
            avg_diameter   REAL,
            avg_confidence REAL,
            timestamp      TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id          TEXT PRIMARY KEY,
            user_id     TEXT NOT NULL,
            token       TEXT UNIQUE NOT NULL,
            created_at  TEXT NOT NULL,
            expires_at  TEXT NOT NULL,
            ip          TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE INDEX IF NOT EXISTS idx_login_attempts_username ON login_attempts(username);
        CREATE INDEX IF NOT EXISTS idx_login_attempts_timestamp ON login_attempts(timestamp);
        CREATE INDEX IF NOT EXISTS idx_analyses_user_id ON analyses(user_id);
        CREATE INDEX IF NOT EXISTS idx_uploads_user_id ON uploads(user_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token);
    """)
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")

def get_user_by_id(user_id: str) -> Optional[Dict]:
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def get_user_by_username(username: str) -> Optional[Dict]:
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    conn.close()
    return dict(row) if row else None

def get_user_by_email(email: str) -> Optional[Dict]:
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    conn.close()
    return dict(row) if row else None

def get_user_analyses(user_id: str, limit: int = 50) -> List[Dict]:
    conn = get_db()
    rows = conn.execute(
        """SELECT id, num_oranges, avg_diameter, avg_confidence, timestamp
           FROM analyses WHERE user_id=?
           ORDER BY timestamp DESC LIMIT ?""",
        (user_id, limit)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_user_stats(user_id: str) -> Dict:
    conn = get_db()
    total_analyses = conn.execute(
        "SELECT COUNT(*) FROM analyses WHERE user_id=?", (user_id,)
    ).fetchone()[0]

    total_oranges = conn.execute(
        "SELECT COALESCE(SUM(num_oranges), 0) FROM analyses WHERE user_id=?", (user_id,)
    ).fetchone()[0]

    avg_diameter_row = conn.execute(
        "SELECT AVG(avg_diameter) FROM analyses WHERE user_id=? AND avg_diameter IS NOT NULL",
        (user_id,)
    ).fetchone()[0]

    conn.close()

    return {
        "total_analyses": total_analyses,
        "total_oranges": total_oranges,
        "avg_diameter": round(avg_diameter_row, 2) if avg_diameter_row else 0
    }

def cleanup_expired_sessions():
    from datetime import datetime
    now = datetime.now().isoformat(sep=" ", timespec="seconds")
    conn = get_db()
    deleted = conn.execute("DELETE FROM sessions WHERE expires_at < ?", (now,)).rowcount
    conn.commit()
    conn.close()
    if deleted > 0:
        logger.info(f"Cleaned up {deleted} expired session(s)")
    return deleted

def cleanup_old_login_attempts(hours: int = 24):
    from datetime import datetime, timedelta
    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat(sep=" ", timespec="seconds")
    conn = get_db()
    deleted = conn.execute("DELETE FROM login_attempts WHERE timestamp < ?", (cutoff,)).rowcount
    conn.commit()
    conn.close()
    if deleted > 0:
        logger.info(f"Cleaned up {deleted} old login attempt(s)")
    return deleted
