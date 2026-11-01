"""Seniority level detection for IT job listings based on title and description patterns."""

from __future__ import annotations

import re

JUNIOR_TITLE_PATTERN = re.compile(
    r"\b(junior\w*|trainee\w*|praktik\w*|intern\b|internship\w*|werkstudent\w*|working student|"
    r"absolvent\w*|nachwuchs\w*|entry[- ]level|berufseinsteiger\w*|berufsanf[äa]nger\w*|einsteiger\w*|quereinsteiger\w*)\b",
    re.IGNORECASE,
)

JUNIOR_DESC_PATTERN = re.compile(
    r"\b(berufseinsteiger\w*|berufsanf[äa]nger\w*|einsteiger\w*|quereinsteiger\w*|entry[- ]level|"
    r"erste[nr]? berufserfahrung|ohne berufserfahrung|keine berufserfahrung|geringe[nr]? berufserfahrung|"
    r"absolvent\w*|hochschulabsolvent\w*|karrierestart\w*|starterposition\w*|nachwuchskraft\w*)\b",
    re.IGNORECASE,
)

LEAD_TITLE_PATTERN = re.compile(
    r"\b(lead\b|principal|staff\b|architekt\w*|architect\w*|head of|teamlead\w*|director|engineering manager)\b",
    re.IGNORECASE,
)

SENIOR_TITLE_PATTERN = re.compile(
    r"\b(senior|sr\b|expert\w*|leiter\w*|spezialist\w*|specialist\w*)\b",
    re.IGNORECASE,
)


def detect_seniority(title: str, description: str = "") -> str:
    """
    Classify job seniority into 'junior', 'mid', 'senior', or 'lead'.

    Detects German recruitment terms such as 'Berufseinsteiger' or 'Absolvent'
    even when 'Junior' is omitted from the title. Avoids false positives when
    senior postings mention mentoring junior employees.
    """
    clean_title = title or ""
    clean_desc = description or ""

    # 1. Lead and architecture leadership roles in title take precedence
    if LEAD_TITLE_PATTERN.search(clean_title):
        return "lead"

    # 2. Senior roles in title
    if SENIOR_TITLE_PATTERN.search(clean_title):
        return "senior"

    # 3. Junior in title or explicit Berufseinsteiger / entry-level in description
    if JUNIOR_TITLE_PATTERN.search(clean_title) or JUNIOR_DESC_PATTERN.search(clean_desc):
        return "junior"

    # 4. Default to standard mid-level
    return "mid"


def get_seniority_badge(seniority: str) -> str:
    """Return a human-readable badge label for a seniority code."""
    badges = {
        "junior": "🟢 Junior / Berufseinsteiger",
        "mid": "🔵 Mid-Level",
        "senior": "🟣 Senior",
        "lead": "🟠 Lead / Architect",
    }
    return badges.get(seniority.lower(), "🔵 Mid-Level")
