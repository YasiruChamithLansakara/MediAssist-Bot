"""Activity metrics for lightweight backend dashboarding.

Tracks the recent usage of core user-facing flows without introducing a
persistent analytics subsystem. The goal is to surface live operational state
for the UI: lookup, chat, and OCR activity plus basic success/failure counts.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from threading import Lock
from typing import Any, Deque, Dict, List, Optional


@dataclass
class ActivityEvent:
    timestamp: str
    kind: str
    success: bool
    status_code: Optional[int]
    detail: str
    context: Dict[str, Any]


class ActivityMetrics:
    def __init__(self, max_events: int = 24):
        self._lock = Lock()
        self._events: Deque[ActivityEvent] = deque(maxlen=max_events)
        self._counts: Dict[str, int] = {
            "lookup": 0,
            "chat": 0,
                "ocr": 0,
                "ocr_tesseract": 0,
                "ocr_easyocr": 0,
                "ocr_other": 0,
            "ocr_text": 0,
            "errors": 0,
            "successes": 0,
        }

    def record(
        self,
        kind: str,
        *,
        success: bool,
        status_code: Optional[int] = None,
        detail: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        event_kind = (kind or "other").strip() or "other"
        payload = ActivityEvent(
            timestamp=datetime.utcnow().isoformat(),
            kind=event_kind,
            success=bool(success),
            status_code=status_code,
            detail=(detail or "").strip(),
            context=context or {},
        )

        with self._lock:
            self._events.appendleft(payload)
            if event_kind in self._counts:
                self._counts[event_kind] += 1
            # If this is an OCR event, also increment engine-specific counters
            if event_kind == "ocr":
                ctx = payload.context or {}
                engine = (ctx.get("engine") or "").strip().lower()
                if engine in ("easyocr", "easy_ocr"):
                    self._counts["ocr_easyocr"] += 1
                elif engine in ("tesseract", "pytesseract", "tesseract-ocr"):
                    self._counts["ocr_tesseract"] += 1
                else:
                    self._counts["ocr_other"] += 1
            if success:
                self._counts["successes"] += 1
            else:
                self._counts["errors"] += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            events = [asdict(event) for event in list(self._events)]
            counts = dict(self._counts)

        return {
            "counts": counts,
            "recent_events": events,
        }


_activity_metrics: Optional[ActivityMetrics] = None


def get_activity_metrics() -> ActivityMetrics:
    global _activity_metrics
    if _activity_metrics is None:
        _activity_metrics = ActivityMetrics()
    return _activity_metrics


def record_activity(
    kind: str,
    *,
    success: bool,
    status_code: Optional[int] = None,
    detail: str = "",
    context: Optional[Dict[str, Any]] = None,
) -> None:
    get_activity_metrics().record(
        kind,
        success=success,
        status_code=status_code,
        detail=detail,
        context=context,
    )


def get_activity_snapshot() -> Dict[str, Any]:
    return get_activity_metrics().snapshot()
