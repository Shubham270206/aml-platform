# Handles all database stuff using SQLite
# SQLite is file-based so no server setup needed - perfect for this project

import sqlite3
import os

DB_PATH = "models/alerts.db"


def init_db():
    # Creates the alerts table on first run
    os.makedirs("models", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id INTEGER,
            probability    REAL,
            step           INTEGER,
            type           TEXT,
            amount         REAL,
            created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_alert(transaction_id, probability, step, tx_type, amount):
    # Called every time we detect a fraud transaction
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO alerts (transaction_id, probability, step, type, amount)
        VALUES (?, ?, ?, ?, ?)
    """, (transaction_id, probability, step, tx_type, amount))
    conn.commit()
    conn.close()


def get_alerts(limit=50):
    # Returns the 50 most recent fraud alerts, newest first
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, transaction_id, probability, step, type, amount
        FROM alerts
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows