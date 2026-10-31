"""Step 4 - access control.

Every chunk carries `allowed_teams`. A user may see a chunk if one of their teams (or the
shared "all" label) is in that list. The filter is applied INSIDE both databases
(pre-filtering), so restricted text never even enters the candidate set. `can_see` is a
second, defence-in-depth check applied after hydration.

In production the user's teams must come from a trusted source (IdP / JWT claims),
never from client-supplied request parameters.
"""
from __future__ import annotations

from dataclasses import dataclass

from rag.models import Chunk

PUBLIC_LABEL = "all"


@dataclass(frozen=True)
class User:
    user_id: str
    teams: frozenset[str]

    @classmethod
    def of(cls, user_id: str, *teams: str) -> "User":
        return cls(user_id, frozenset(t.strip().lower() for t in teams))


def visible_teams(user: User) -> list[str]:
    return sorted(user.teams | {PUBLIC_LABEL})


def can_see(chunk: Chunk, user: User) -> bool:
    return bool(set(chunk.allowed_teams) & set(visible_teams(user)))
