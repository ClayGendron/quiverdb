"""GraphSchema - Container for graph models with database support."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlmodel import SQLModel

if TYPE_CHECKING:
    from sqlalchemy import Engine

    from .graph_model import GraphModel


class GraphSchema:
    """Schema container for graph models with database support.

    Auto-detects node vs edge models from their model_type attribute.
    Manages the SQLAlchemy engine and provides table creation.

    Usage:
        # Create schema with models
        schema = GraphSchema(Node, Edge, engine=engine)
        schema.create_tables()

        # Use with Graph
        G = Graph(schema)

        # Access type registries
        schema.node_types  # {'Document': Document, 'Wiki': Wiki, ...}
        schema.edge_types  # {'References': References, ...}
    """

    def __init__(
        self,
        *models: type[GraphModel],
        engine: Engine | None = None,
    ) -> None:
        """Initialize schema with models.

        Args:
            *models: GraphModel subclasses (node and/or edge models).
                     Auto-detected based on model_type attribute.
            engine: Optional SQLAlchemy engine for database operations.

        Raises:
            TypeError: If a model is not a valid GraphModel.
            ValueError: If multiple node or edge models are provided.
        """
        self._node_model: type[GraphModel] | None = None
        self._edge_model: type[GraphModel] | None = None
        self.engine = engine

        for model in models:
            if not hasattr(model, "model_type"):
                raise TypeError(
                    f"'{model.__name__}' is not a valid GraphModel. "
                    f"Ensure it inherits from GraphModel and has node/edge fields."
                )

            if model.model_type == "node":
                if self._node_model is not None:
                    raise ValueError(
                        f"Multiple node models provided: '{self._node_model.__name__}' "
                        f"and '{model.__name__}'. Only one node model is allowed."
                    )
                self._node_model = model

            elif model.model_type == "edge":
                if self._edge_model is not None:
                    raise ValueError(
                        f"Multiple edge models provided: '{self._edge_model.__name__}' "
                        f"and '{model.__name__}'. Only one edge model is allowed."
                    )
                self._edge_model = model

    @property
    def node_model(self) -> type[GraphModel]:
        """Get the node model.

        Raises:
            ValueError: If no node model is configured.
        """
        if self._node_model is None:
            raise ValueError("No node model configured in this schema")
        return self._node_model

    @property
    def edge_model(self) -> type[GraphModel]:
        """Get the edge model.

        Raises:
            ValueError: If no edge model is configured.
        """
        if self._edge_model is None:
            raise ValueError("No edge model configured in this schema")
        return self._edge_model

    @property
    def has_node_model(self) -> bool:
        """Check if a node model is configured."""
        return self._node_model is not None

    @property
    def has_edge_model(self) -> bool:
        """Check if an edge model is configured."""
        return self._edge_model is not None

    def create_tables(self) -> None:
        """Create database tables for all configured models.

        Raises:
            ValueError: If no engine is configured.
        """
        if self.engine is None:
            raise ValueError(
                "Cannot create tables without an engine. "
                "Initialize GraphSchema with engine=create_engine(...)."
            )
        SQLModel.metadata.create_all(self.engine)

    @property
    def node_types(self) -> dict[str, type[GraphModel]]:
        """Get all registered node type subclasses.

        Returns:
            Dictionary mapping type names to their classes.
            Empty dict if no node model is configured.
        """
        if self._node_model is None:
            return {}
        return self._node_model.get_registered_types()

    @property
    def edge_types(self) -> dict[str, type[GraphModel]]:
        """Get all registered edge type subclasses.

        Returns:
            Dictionary mapping type names to their classes.
            Empty dict if no edge model is configured.
        """
        if self._edge_model is None:
            return {}
        return self._edge_model.get_registered_types()

    def __repr__(self) -> str:
        node = self._node_model.__name__ if self._node_model else None
        edge = self._edge_model.__name__ if self._edge_model else None
        engine = "configured" if self.engine else None
        return f"GraphSchema(node_model={node}, edge_model={edge}, engine={engine})"
