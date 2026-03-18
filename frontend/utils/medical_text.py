"""Medical text templates for report generation."""
from __future__ import annotations

from typing import Any

from frontend.utils.clinical_report_builder import build_clinical_report, build_pdf_clinical_report


def get_detailed_report(result: dict[str, Any]) -> str:
    """Return a safe report text without placeholder clinical measurements."""
    return build_clinical_report(result)


def get_pdf_report_text(result: dict[str, Any]) -> str:
    """Returns the detailed report text formatted with HTML tags for ReportLab."""
    return build_pdf_clinical_report(result)
