"""Runtime configuration. Everything tunable lives here or in environment variables."""

import os
from datetime import date
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

FIXTURES_DIR = Path(
    os.getenv("FIXTURES_DIR", ROOT_DIR / "apps" / "insurance_claims" / "fixtures")
)


def today() -> date:
    """The real current date. TODAY_OVERRIDE (YYYY-MM-DD) exists only so tests are stable."""
    override = os.getenv("TODAY_OVERRIDE")
    return date.fromisoformat(override) if override else date.today()
