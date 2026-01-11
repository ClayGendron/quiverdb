from __future__ import annotations

from sqlalchemy import Engine
from sqlmodel import SQLModel


class GraphSchema:
    def __init__(
        self,
        model: type[SQLModel] | None = None,
        engine: Engine | None = None,
    ) -> None:
        self.model = model
        self.engine = engine
