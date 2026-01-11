"""Shared test fixtures for quiverdb tests."""

import uuid

import pytest
from sqlalchemy import create_engine
from sqlmodel import Field

from quiverdb import Graph, GraphModel, GraphSchema


def make_unique_table_name(prefix: str) -> str:
    """Generate a unique table name to avoid SQLAlchemy conflicts."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def node_model():
    """Create a test node model with unique table name."""
    table_name = make_unique_table_name("test_nodes")

    class TestNode(GraphModel, table=True, table_name=table_name):
        node: str = Field(primary_key=True)
        type: str = Field(index=True)
        name: str | None = None

    return TestNode


@pytest.fixture
def edge_model():
    """Create a test edge model with unique table name."""
    table_name = make_unique_table_name("test_edges")

    class TestEdge(GraphModel, table=True, table_name=table_name):
        source: str = Field(primary_key=True)
        target: str = Field(primary_key=True)
        type: str = Field(default="RELATED_TO")

    return TestEdge


@pytest.fixture
def schema(node_model, edge_model):
    """Create a test schema with node and edge models."""
    return GraphSchema(node_model, edge_model)


@pytest.fixture
def schema_with_engine(node_model, edge_model):
    """Create a test schema with an in-memory SQLite engine."""
    engine = create_engine("sqlite:///:memory:")
    return GraphSchema(node_model, edge_model, engine=engine)


@pytest.fixture
def graph(schema):
    """Create a validated graph with schema."""
    return Graph(schema)


@pytest.fixture
def untyped_graph():
    """Create an untyped graph without schema."""
    return Graph()
