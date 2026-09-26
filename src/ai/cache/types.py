from dataclasses import dataclass
from typing import Literal


CacheStrategy = Literal[
    "none",
    "automatic",
    "explicit",
]


@dataclass(frozen=True)
class CacheConfig:
    strategy: CacheStrategy = "automatic"