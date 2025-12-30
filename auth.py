import os
import json
import hmac
import hashlib
import sqlite3
from typing import Optional

from config import APP_SECRET, SESSION_FILE
from db import get_conn
from utils import utc_now_iso


def hash_password(password: str, salt: Optional[bytes] = None):
    if salt is None:
        salt = os.urandom(16)
    ph = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 150_000)
    return ph.hex(), salt.hex()


def verify_password(password: str, pwd_hex: str, salt_hex: str) -> bool:
    nh = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), 150_000
    ).hex()
    return hmac.compare_digest(nh, pwd_hex)


def create_user(username: str, email: str, password: str):
    try:
        conn = get_conn()
        cur = conn.cursor()
        ph, sh = hash_password(password)
        cur.execute(
            "INSERT INTO users (username, email, pwd_hash, salt, created_at) VALUES (?,?,?,?,?)",
            (username, email, ph, sh, utc_now_iso()),
        )
        conn.commit()
        conn.close()
        return True, "Account created."
    except sqlite3.IntegrityError as e:
        msg = (
            "Username already exists."
            if "username" in str(e).lower()
            else "Email already exists."
        )
        return False, str(msg)
    except Exception as e:
        return False, f"Error creating user: {e}"


def get_user_by_login(identifier: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM users WHERE username=? OR email=? LIMIT 1",
        (identifier, identifier),
    )
    row = cur.fetchone()
    conn.close()
    return row


def get_user_by_id(user_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id=?", (user_id,))
    row = cur.fetchone()
    conn.close()
    return row


def save_persistent_session(user_id: int, pwd_hex: str):
    token = hashlib.sha256(f"{user_id}:{pwd_hex}:{APP_SECRET}".encode()).hexdigest()
    with open(SESSION_FILE, "w") as f:
        json.dump({"user_id": user_id, "token": token}, f)


def load_persistent_session():
    if not os.path.exists(SESSION_FILE):
        return None
    try:
        with open(SESSION_FILE, "r") as f:
            data = json.load(f)
        user_id = data.get("user_id")
        token = data.get("token")

        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT id, pwd_hash FROM users WHERE id=?", (user_id,))
        row = cur.fetchone()
        conn.close()

        if not row:
            return None

        expected = hashlib.sha256(
            f"{row['id']}:{row['pwd_hash']}:{APP_SECRET}".encode()
        ).hexdigest()
        if hmac.compare_digest(token, expected):
            return row["id"]
        return None
    except Exception:
        return None


def clear_persistent_session():
    try:
        if os.path.exists(SESSION_FILE):
            os.remove(SESSION_FILE)
    except Exception:
        pass