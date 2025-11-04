# database.py
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any

class DatabaseManager:
    def __init__(self, db_path="misconceptions.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table for analysis history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                explanation TEXT NOT NULL,
                detected_misconception TEXT,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Table for user feedback
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER,
                rating INTEGER,
                comments TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (analysis_id) REFERENCES analyses (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_analysis(self, explanation: str, result: Dict[str, Any]):
        """Save analysis result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analyses (explanation, detected_misconception, confidence)
            VALUES (?, ?, ?)
        ''', (explanation, result.get('detected_misconception'), result.get('confidence')))
        
        conn.commit()
        conn.close()
    
    def get_analysis_history(self, limit: int = 10):
        """Get recent analysis history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM analyses 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                "id": row[0],
                "explanation": row[1],
                "detected_misconception": row[2],
                "confidence": row[3],
                "timestamp": row[4]
            }
            for row in results
        ]

# Initialize database
db = DatabaseManager()