"""GraphSchema - Container for graph models with database support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlmodel import SQLModel

from .graph_model import GraphModel

if TYPE_CHECKING:
    from sqlalchemy import Engine


class GraphSchema:
    """Schema container for graph models with database support.

    Auto-detects node vs edge models from their model_type attribute.
    Manages the SQLAlchemy engine and provides table creation.

    Each schema maintains its own type registry, scoped to the models
    it was initialized with. This prevents cross-contamination between
    different schemas.

    Usage:
        # Create schema with models
        schema = GraphSchema(Node, Edge, engine=engine)
        schema.create_tables()

        # Use with Graph
        G = Graph(schema)

        # Access type registries (scoped to this schema)
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
        self._node_type_registry: dict[str, type[GraphModel]] = {}
        self._edge_type_registry: dict[str, type[GraphModel]] = {}
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
                self._build_type_registry(model, self._node_type_registry)

            elif model.model_type == "edge":
                if self._edge_model is not None:
                    raise ValueError(
                        f"Multiple edge models provided: '{self._edge_model.__name__}' "
                        f"and '{model.__name__}'. Only one edge model is allowed."
                    )
                self._edge_model = model
                self._build_type_registry(model, self._edge_type_registry)

    def _build_type_registry(
        self, model: type[GraphModel], registry: dict[str, type[GraphModel]]
    ) -> None:
        """Build a scoped type registry for a model and its subtypes.

        Collects all registered subtypes that share the same table base
        as the given model.
        """
        table_base = getattr(model, "_table_base", None)
        if table_base is None:
            return

        # Collect all types from global registry that belong to this model's hierarchy
        for type_name, type_cls in GraphModel._type_registry.items():
            cls_table_base = getattr(type_cls, "_table_base", None)
            if cls_table_base is table_base:
                registry[type_name] = type_cls

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
        """Get all registered node type subclasses scoped to this schema.

        Returns:
            Dictionary mapping type names to their classes.
            Empty dict if no node model is configured.
        """
        return dict(self._node_type_registry)

    @property
    def edge_types(self) -> dict[str, type[GraphModel]]:
        """Get all registered edge type subclasses scoped to this schema.

        Returns:
            Dictionary mapping type names to their classes.
            Empty dict if no edge model is configured.
        """
        return dict(self._edge_type_registry)

    def validate_node(self, data: dict[str, Any]) -> GraphModel:
        """Validate node data using this schema's scoped type registry.

        Dispatches to the appropriate type subclass based on the 'type' field,
        using only types registered with this schema's node model.

        Args:
            data: Node data dictionary to validate.

        Returns:
            Validated GraphModel instance.

        Raises:
            ValueError: If no node model is configured.
        """
        if self._node_model is None:
            raise ValueError("No node model configured in this schema")
        return self._validate_with_registry(
            data, self._node_model, self._node_type_registry
        )

    def validate_edge(self, data: dict[str, Any]) -> GraphModel:
        """Validate edge data using this schema's scoped type registry.

        Dispatches to the appropriate type subclass based on the 'type' field,
        using only types registered with this schema's edge model.

        Args:
            data: Edge data dictionary to validate.

        Returns:
            Validated GraphModel instance.

        Raises:
            ValueError: If no edge model is configured.
        """
        if self._edge_model is None:
            raise ValueError("No edge model configured in this schema")
        return self._validate_with_registry(
            data, self._edge_model, self._edge_type_registry
        )

    def _validate_with_registry(
        self,
        data: dict[str, Any],
        base_model: type[GraphModel],
        registry: dict[str, type[GraphModel]],
    ) -> GraphModel:
        """Validate data using a scoped type registry.

        Args:
            data: Data dictionary to validate.
            base_model: The base model class to use as fallback.
            registry: The scoped type registry for dispatch.

        Returns:
            Validated GraphModel instance.
        """
        type_str = data.get("type")

        # Find target class in the scoped registry
        if type_str and type_str in registry:
            target_cls = registry[type_str]
        else:
            target_cls = base_model

        return target_cls.model_validate(data)

    def __repr__(self) -> str:
        node = self._node_model.__name__ if self._node_model else None
        edge = self._edge_model.__name__ if self._edge_model else None
        engine = "configured" if self.engine else None
        return f"GraphSchema(node_model={node}, edge_model={edge}, engine={engine})"
