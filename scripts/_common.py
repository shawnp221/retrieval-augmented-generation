"""Tiny helpers shared by the step scripts."""
from __future__ import annotations

import logging
from pathlib import Path

from rag.access import User

DATA = Path(__file__).resolve().parent.parent / "data" / "sample_docs.jsonl"
DEFAULT_QUESTION = "What is the refund limit for invoice INV-12345?"

logging.basicConfig(level=logging.INFO, format="%(message)s")


def user_from_team(team: str) -> User:
    return User.of(f"demo-{team}", team)


def snippet(text: str, n: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"
