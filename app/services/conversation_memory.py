"""
Conversation Memory Service: Stores and manages chat conversation history.

Features:
- In-memory storage of chat conversations
- Per-user conversation sessions (by session ID)
- Conversation retrieval and history management
- Optional persistence to SQLite for production
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import os

DB_PATH = os.getenv("CONVERSATION_DB", "conversations.db")
USE_PERSISTENCE = os.getenv("USE_CONVERSATION_PERSISTENCE", "0").strip() == "1"


class ConversationMemory:
    """
    Manages conversation history for chat sessions.
    """

    def __init__(self, use_persistence: bool = USE_PERSISTENCE, db_path: str = DB_PATH):
        self.use_persistence = use_persistence
        self.db_path = db_path
        self._memory: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._max_turns_per_session = 100
        self._session_timeout_hours = 24

        if self.use_persistence:
            self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database for persistence."""
        if not self.use_persistence:
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    turns_count INTEGER DEFAULT 0,
                    summary TEXT,
                    data TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_id ON conversations(session_id);
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            # Silently fail if DB is not available (e.g., read-only filesystem)
            pass

    def add_turn(
        self,
        session_id: str,
        role: str,
        text: str,
        data: Optional[Dict[str, Any]] = None,
        disease: str = "",
        age: int = 0,
    ) -> Dict[str, Any]:
        """
        Add a turn (user or assistant message) to conversation history.

        Args:
            session_id: Unique identifier for the conversation session
            role: "user" or "assistant"
            text: The message text
            data: Optional metadata or response data
            disease: Patient's disease context
            age: Patient's age context

        Returns:
            The added turn record
        """
        with self._lock:
            if session_id not in self._memory:
                self._memory[session_id] = []

            turn = {
                "timestamp": datetime.utcnow().isoformat(),
                "role": role,
                "text": text,
                "data": data or {},
                "context": {"disease": disease, "age": age},
            }

            self._memory[session_id].append(turn)

            # Trim if too many turns
            if len(self._memory[session_id]) > self._max_turns_per_session:
                self._memory[session_id] = self._memory[session_id][-self._max_turns_per_session :]

            # Persist if enabled
            if self.use_persistence:
                self._persist_session(session_id)

            return turn

    def get_history(self, session_id: str, max_turns: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session identifier
            max_turns: Maximum number of recent turns to return

        Returns:
            List of conversation turns
        """
        with self._lock:
            history = self._memory.get(session_id, [])

            if max_turns and len(history) > max_turns:
                history = history[-max_turns:]

            return list(history)

    def get_context_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get a summary of the conversation context.

        Returns dict with:
        - turn_count: Number of turns in conversation
        - disease: Last known disease context
        - age: Last known age context
        - started_at: When conversation started
        - duration_seconds: How long conversation has been running
        """
        with self._lock:
            history = self._memory.get(session_id, [])

            if not history:
                return {
                    "turn_count": 0,
                    "disease": "",
                    "age": 0,
                    "started_at": None,
                    "duration_seconds": 0,
                }

            first_turn = history[0]
            last_turn = history[-1]

            started_at = datetime.fromisoformat(first_turn["timestamp"])
            updated_at = datetime.fromisoformat(last_turn["timestamp"])
            duration = (updated_at - started_at).total_seconds()

            context = last_turn.get("context", {})

            return {
                "turn_count": len(history),
                "disease": context.get("disease", ""),
                "age": context.get("age", 0),
                "started_at": started_at.isoformat(),
                "updated_at": updated_at.isoformat(),
                "duration_seconds": int(duration),
            }

    def clear_session(self, session_id: str) -> bool:
        """Clear all conversation history for a session."""
        with self._lock:
            if session_id in self._memory:
                del self._memory[session_id]
                if self.use_persistence:
                    self._clear_persisted_session(session_id)
                return True
        return False

    def _persist_session(self, session_id: str) -> None:
        """Persist conversation to database."""
        if not self.use_persistence or session_id not in self._memory:
            return

        try:
            history = self._memory[session_id]
            data_json = json.dumps(history)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO conversations 
                (session_id, created_at, updated_at, turns_count, data)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    datetime.utcnow().isoformat(),
                    datetime.utcnow().isoformat(),
                    len(history),
                    data_json,
                ),
            )

            conn.commit()
            conn.close()
        except Exception:
            # Silently fail if DB is not available
            pass

    def _clear_persisted_session(self, session_id: str) -> None:
        """Clear persisted conversation from database."""
        if not self.use_persistence:
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
            conn.commit()
            conn.close()
        except Exception:
            # Silently fail
            pass

    def load_persisted_session(self, session_id: str) -> bool:
        """Load conversation from database into memory."""
        if not self.use_persistence:
            return False

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT data FROM conversations WHERE session_id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                data_json = row[0]
                self._memory[session_id] = json.loads(data_json)
                return True

            return False
        except Exception:
            return False

    def get_all_sessions(self) -> List[str]:
        """Get list of all active session IDs."""
        with self._lock:
            return list(self._memory.keys())

    def stats(self) -> Dict[str, Any]:
        """Get statistics about conversation memory."""
        with self._lock:
            total_turns = sum(len(h) for h in self._memory.values())
            return {
                "active_sessions": len(self._memory),
                "total_turns": total_turns,
                "persistence_enabled": self.use_persistence,
                "db_path": self.db_path if self.use_persistence else None,
            }


# Global instance
_conversation_memory: Optional[ConversationMemory] = None


def get_conversation_memory() -> ConversationMemory:
    """Get or create the global conversation memory instance."""
    global _conversation_memory
    if _conversation_memory is None:
        _conversation_memory = ConversationMemory(use_persistence=USE_PERSISTENCE, db_path=DB_PATH)
    return _conversation_memory


def add_turn_to_memory(
    session_id: str,
    role: str,
    text: str,
    data: Optional[Dict[str, Any]] = None,
    disease: str = "",
    age: int = 0,
) -> Dict[str, Any]:
    """Helper function to add a turn."""
    memory = get_conversation_memory()
    return memory.add_turn(session_id, role, text, data, disease, age)


def get_conversation_history(session_id: str, max_turns: Optional[int] = None) -> List[Dict[str, Any]]:
    """Helper function to get history."""
    memory = get_conversation_memory()
    return memory.get_history(session_id, max_turns)


def get_context_summary(session_id: str) -> Dict[str, Any]:
    """Helper function to get context summary."""
    memory = get_conversation_memory()
    return memory.get_context_summary(session_id)
