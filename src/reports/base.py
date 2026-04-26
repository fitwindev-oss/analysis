"""
Report section abstractions.

A ``ReportSection`` can render itself to both HTML (for the embedded
report viewer) and a list of reportlab flowables (for PDF export) from
the same input context. This keeps layout logic in one place per
section rather than duplicated between the two outputs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class SessionMetrics:
    """Cached per-session key metrics, sourced from result.json.

    Populated into the DB `session_metrics` table on analysis completion
    so the history section can query many sessions at once cheaply.
    """
    session_id:   str
    session_date: str                       # ISO
    metrics:      dict[str, float | int | str | None]


@dataclass
class ReportContext:
    """Everything a section needs to render one session's report."""
    session_dir:  Path                      # absolute path
    session_meta: dict                      # session.json
    result:       dict                      # result.json["result"] (or {})
    test_type:    str                       # e.g. "balance_eo"
    subject:      Optional[Any] = None      # src.db.models.Subject
    history:      list[SessionMetrics] = field(default_factory=list)
    audience:     str = "trainer"           # "trainer" | "subject"


class ReportSection(ABC):
    """Base class for every composable section (header / summary / ...).

    Subclasses override ``to_html`` and/or ``to_pdf_flowables``. The
    default ``enabled_for`` returns True for all audiences; override for
    sections that are trainer-only or subject-only.
    """

    @abstractmethod
    def to_html(self, ctx: ReportContext) -> str:
        ...

    def to_pdf_flowables(self, ctx: ReportContext) -> list:
        """Return a list of reportlab Flowables. Default: empty (skipped in PDF)."""
        return []

    def enabled_for(self, audience: str) -> bool:
        return True
