"""Tests for quiverdb.graph_schema module."""

import pytest
from sqlalchemy import create_engine

from quiverdb import GraphModel, GraphSchema

from .conftest import make_unique_table_name


class TestGraphSchemaInit:
    """Tests for GraphSchema initialization and model detection."""

    def test_init_with_node_model(self, node_model) -> None:
        """Single node model is auto-detected."""
        schema = GraphSchema(node_model)
        assert schema.has_node_model
        assert not schema.has_edge_model
        assert schema._node_model is node_model

    def test_init_with_edge_model(self, edge_model) -> None:
        """Single edge model is auto-detected."""
        schema = GraphSchema(edge_model)
        assert schema.has_edge_model
        assert not schema.has_node_model
        assert schema._edge_model is edge_model

    def test_init_with_both_models(self, node_model, edge_model) -> None:
        """Node and edge models are detected regardless of order."""
        # Node first
        schema1 = GraphSchema(node_model, edge_model)
        assert schema1.has_node_model
        assert schema1.has_edge_model

        # Edge first
        schema2 = GraphSchema(edge_model, node_model)
        assert schema2.has_node_model
        assert schema2.has_edge_model

    def test_init_empty(self) -> None:
        """Empty schema (no models) is valid."""
        schema = GraphSchema()
        assert not schema.has_node_model
        assert not schema.has_edge_model

    def test_init_with_engine(self, node_model) -> None:
        """Engine is stored correctly."""
        engine = create_engine("sqlite:///:memory:")
        schema = GraphSchema(node_model, engine=engine)
        assert schema.engine is engine


class TestGraphSchemaErrors:
    """Tests for GraphSchema error handling."""

    def test_init_invalid_model_raises_typeerror(self) -> None:
        """Non-GraphModel raises TypeError."""

        class NotAGraphModel:
            pass

        with pytest.raises(TypeError, match="not a valid GraphModel"):
            GraphSchema(NotAGraphModel)

    def test_init_multiple_node_models_raises(self, node_model) -> None:
        """Two node models raises ValueError."""
        from sqlmodel import Field

        table_name = make_unique_table_name("test_nodes2")

        class AnotherNode(GraphModel, table=True, table_name=table_name):
            node: str = Field(primary_key=True)
            type: str

        with pytest.raises(ValueError, match="Multiple node models"):
            GraphSchema(node_model, AnotherNode)

    def test_init_multiple_edge_models_raises(self, edge_model) -> None:
        """Two edge models raises ValueError."""
        from sqlmodel import Field

        table_name = make_unique_table_name("test_edges2")

        class AnotherEdge(GraphModel, table=True, table_name=table_name):
            source: str = Field(primary_key=True)
            target: str = Field(primary_key=True)
            type: str

        with pytest.raises(ValueError, match="Multiple edge models"):
            GraphSchema(edge_model, AnotherEdge)


class TestGraphSchemaProperties:
    """Tests for GraphSchema properties."""

    def test_node_model_property(self, node_model) -> None:
        """Returns node model when configured."""
        schema = GraphSchema(node_model)
        assert schema.node_model is node_model

    def test_node_model_raises_when_not_configured(self) -> None:
        """Raises ValueError if no node model."""
        schema = GraphSchema()
        with pytest.raises(ValueError, match="No node model configured"):
            _ = schema.node_model

    def test_edge_model_property(self, edge_model) -> None:
        """Returns edge model when configured."""
        schema = GraphSchema(edge_model)
        assert schema.edge_model is edge_model

    def test_edge_model_raises_when_not_configured(self) -> None:
        """Raises ValueError if no edge model."""
        schema = GraphSchema()
        with pytest.raises(ValueError, match="No edge model configured"):
            _ = schema.edge_model

    def test_has_node_model_true(self, node_model) -> None:
        """has_node_model returns True when configured."""
        schema = GraphSchema(node_model)
        assert schema.has_node_model is True

    def test_has_node_model_false(self) -> None:
        """has_node_model returns False when not configured."""
        schema = GraphSchema()
        assert schema.has_node_model is False

    def test_has_edge_model_true(self, edge_model) -> None:
        """has_edge_model returns True when configured."""
        schema = GraphSchema(edge_model)
        assert schema.has_edge_model is True

    def test_has_edge_model_false(self) -> None:
        """has_edge_model returns False when not configured."""
        schema = GraphSchema()
        assert schema.has_edge_model is False


class TestGraphSchemaDatabaseOps:
    """Tests for GraphSchema database operations."""

    def test_create_tables_without_engine_raises(self, node_model) -> None:
        """Raises ValueError when no engine configured."""
        schema = GraphSchema(node_model)
        with pytest.raises(ValueError, match="Cannot create tables without an engine"):
            schema.create_tables()

    def test_create_tables_with_engine(self, schema_with_engine) -> None:
        """Creates tables successfully with engine."""
        # Should not raise
        schema_with_engine.create_tables()


class TestGraphSchemaTypeRegistry:
    """Tests for GraphSchema type registry access."""

    def test_node_types_returns_registry(self, node_model) -> None:
        """Returns dict of registered node types."""
        schema = GraphSchema(node_model)
        types = schema.node_types
        assert isinstance(types, dict)

    def test_edge_types_returns_registry(self, edge_model) -> None:
        """Returns dict of registered edge types."""
        schema = GraphSchema(edge_model)
        types = schema.edge_types
        assert isinstance(types, dict)

    def test_node_types_empty_when_no_model(self) -> None:
        """Returns empty dict when no node model."""
        schema = GraphSchema()
        assert schema.node_types == {}

    def test_edge_types_empty_when_no_model(self) -> None:
        """Returns empty dict when no edge model."""
        schema = GraphSchema()
        assert schema.edge_types == {}


class TestGraphSchemaRepr:
    """Tests for GraphSchema string representation."""

    def test_repr_with_both_models(self, node_model, edge_model) -> None:
        """Repr shows both model names."""
        schema = GraphSchema(node_model, edge_model)
        repr_str = repr(schema)
        assert "node_model=TestNode" in repr_str
        assert "edge_model=TestEdge" in repr_str
        assert "engine=None" in repr_str

    def test_repr_with_engine(self, node_model) -> None:
        """Repr shows engine as configured."""
        engine = create_engine("sqlite:///:memory:")
        schema = GraphSchema(node_model, engine=engine)
        repr_str = repr(schema)
        assert "engine=configured" in repr_str

    def test_repr_empty(self) -> None:
        """Repr shows None for missing models."""
        schema = GraphSchema()
        repr_str = repr(schema)
        assert "node_model=None" in repr_str
        assert "edge_model=None" in repr_str
