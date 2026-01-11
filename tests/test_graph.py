"""Tests for quiverdb.graph module."""

import pytest

from quiverdb import Graph, GraphSchema


class TestGraphInit:
    """Tests for Graph initialization."""

    def test_init_untyped(self) -> None:
        """Graph() creates untyped graph."""
        g = Graph()
        assert g.schema is None
        assert not g.has_validation

    def test_init_with_schema(self, schema) -> None:
        """Graph(schema) enables validation."""
        g = Graph(schema)
        assert g.schema is schema
        assert g.has_validation

    def test_has_validation_property(self, schema) -> None:
        """has_validation returns correct boolean."""
        assert Graph().has_validation is False
        assert Graph(schema).has_validation is True

    def test_schema_property(self, schema) -> None:
        """schema property returns schema or None."""
        assert Graph().schema is None
        assert Graph(schema).schema is schema


class TestGraphDunderMethods:
    """Tests for Graph dunder methods (__contains__, __getitem__, etc.)."""

    def test_contains_node(self, untyped_graph) -> None:
        """'node' in G works for existing node."""
        untyped_graph.add_node("a")
        assert "a" in untyped_graph

    def test_contains_node_missing(self, untyped_graph) -> None:
        """'node' in G returns False for missing node."""
        assert "nonexistent" not in untyped_graph

    def test_contains_edge(self, untyped_graph) -> None:
        """('a', 'b') in G works for existing edge."""
        untyped_graph.add_edge("a", "b")
        assert ("a", "b") in untyped_graph

    def test_contains_edge_missing(self, untyped_graph) -> None:
        """('a', 'b') in G returns False for missing edge."""
        untyped_graph.add_node("a")
        untyped_graph.add_node("b")
        assert ("a", "b") not in untyped_graph

    def test_getitem_node(self, untyped_graph) -> None:
        """G['node'] returns node attributes."""
        untyped_graph.add_node("a", foo="bar")
        assert untyped_graph["a"]["foo"] == "bar"

    def test_getitem_edge(self, untyped_graph) -> None:
        """G['a', 'b'] returns edge attributes."""
        untyped_graph.add_edge("a", "b", weight=1.5)
        assert untyped_graph["a", "b"]["weight"] == 1.5

    def test_setitem_node(self, untyped_graph) -> None:
        """G['node'] = {...} adds/updates node."""
        untyped_graph["a"] = {"foo": "bar"}
        assert "a" in untyped_graph
        assert untyped_graph["a"]["foo"] == "bar"

    def test_setitem_edge(self, untyped_graph) -> None:
        """G['a', 'b'] = {...} adds/updates edge."""
        untyped_graph["a", "b"] = {"weight": 2.0}
        assert ("a", "b") in untyped_graph
        assert untyped_graph["a", "b"]["weight"] == 2.0

    def test_len(self, untyped_graph) -> None:
        """len(G) returns node count."""
        assert len(untyped_graph) == 0
        untyped_graph.add_node("a")
        assert len(untyped_graph) == 1
        untyped_graph.add_node("b")
        assert len(untyped_graph) == 2

    def test_repr(self, untyped_graph, graph) -> None:
        """Repr shows node/edge counts and mode."""
        assert "nodes=0" in repr(untyped_graph)
        assert "untyped" in repr(untyped_graph)
        assert "validated" in repr(graph)


class TestGraphNodeOperations:
    """Tests for Graph node operations."""

    def test_add_node(self, untyped_graph) -> None:
        """Adds node with attributes."""
        untyped_graph.add_node("a", foo="bar", count=42)
        node = untyped_graph.get_node("a")
        assert node["node"] == "a"
        assert node["foo"] == "bar"
        assert node["count"] == 42

    def test_add_node_upsert(self, untyped_graph) -> None:
        """Merges attributes on existing node."""
        untyped_graph.add_node("a", foo="bar")
        untyped_graph.add_node("a", baz="qux")
        node = untyped_graph.get_node("a")
        assert node["foo"] == "bar"  # Preserved
        assert node["baz"] == "qux"  # Added

    def test_add_node_from_dict(self, untyped_graph) -> None:
        """Adds node from dict with 'node' key."""
        untyped_graph.add_node_from_dict({"node": "a", "foo": "bar"})
        assert "a" in untyped_graph
        assert untyped_graph["a"]["foo"] == "bar"

    def test_add_node_from_dict_missing_key_raises(self, untyped_graph) -> None:
        """Raises ValueError if 'node' key missing."""
        with pytest.raises(ValueError, match="must contain 'node' key"):
            untyped_graph.add_node_from_dict({"foo": "bar"})

    def test_get_node(self, untyped_graph) -> None:
        """Returns node attributes."""
        untyped_graph.add_node("a", foo="bar")
        node = untyped_graph.get_node("a")
        assert isinstance(node, dict)
        assert node["foo"] == "bar"

    def test_get_node_missing_raises(self, untyped_graph) -> None:
        """Raises KeyError for missing node."""
        with pytest.raises(KeyError, match="not found"):
            untyped_graph.get_node("nonexistent")

    def test_get_node_as_model(self, graph) -> None:
        """Returns GraphModel instance."""
        graph.add_node("a", type="TestType", name="Test")
        model = graph.get_node_as_model("a")
        assert model.node == "a"
        assert model.type == "TestType"
        assert model.name == "Test"

    def test_get_node_as_model_no_schema_raises(self, untyped_graph) -> None:
        """Raises ValueError if no node model configured."""
        untyped_graph.add_node("a")
        with pytest.raises(ValueError, match="No node model configured"):
            untyped_graph.get_node_as_model("a")

    def test_remove_node(self, untyped_graph) -> None:
        """Removes node and its edges."""
        untyped_graph.add_edge("a", "b")
        untyped_graph.add_edge("b", "c")
        untyped_graph.remove_node("b")
        assert "b" not in untyped_graph
        assert ("a", "b") not in untyped_graph
        assert ("b", "c") not in untyped_graph

    def test_remove_node_missing_raises(self, untyped_graph) -> None:
        """Raises KeyError for missing node."""
        with pytest.raises(KeyError, match="not found"):
            untyped_graph.remove_node("nonexistent")

    def test_nodes(self, untyped_graph) -> None:
        """Returns list of node identifiers."""
        untyped_graph.add_node("a")
        untyped_graph.add_node("b")
        untyped_graph.add_node("c")
        nodes = untyped_graph.nodes()
        assert set(nodes) == {"a", "b", "c"}


class TestGraphEdgeOperations:
    """Tests for Graph edge operations."""

    def test_add_edge(self, untyped_graph) -> None:
        """Adds edge with attributes."""
        untyped_graph.add_node("a")
        untyped_graph.add_node("b")
        untyped_graph.add_edge("a", "b", weight=1.5)
        edge = untyped_graph.get_edge("a", "b")
        assert edge["source"] == "a"
        assert edge["target"] == "b"
        assert edge["weight"] == 1.5

    def test_add_edge_auto_creates_nodes(self, untyped_graph) -> None:
        """Creates missing source/target nodes."""
        untyped_graph.add_edge("a", "b")
        assert "a" in untyped_graph
        assert "b" in untyped_graph
        assert ("a", "b") in untyped_graph

    def test_add_edge_upsert(self, untyped_graph) -> None:
        """Merges attributes on existing edge."""
        untyped_graph.add_edge("a", "b", weight=1.0)
        untyped_graph.add_edge("a", "b", label="test")
        edge = untyped_graph.get_edge("a", "b")
        assert edge["weight"] == 1.0  # Preserved
        assert edge["label"] == "test"  # Added

    def test_add_edge_from_dict(self, untyped_graph) -> None:
        """Adds edge from dict with source/target keys."""
        untyped_graph.add_edge_from_dict({"source": "a", "target": "b", "weight": 2.0})
        assert ("a", "b") in untyped_graph
        assert untyped_graph["a", "b"]["weight"] == 2.0

    def test_add_edge_from_dict_missing_source_raises(self, untyped_graph) -> None:
        """Raises ValueError if 'source' key missing."""
        with pytest.raises(ValueError, match="must contain 'source' key"):
            untyped_graph.add_edge_from_dict({"target": "b"})

    def test_add_edge_from_dict_missing_target_raises(self, untyped_graph) -> None:
        """Raises ValueError if 'target' key missing."""
        with pytest.raises(ValueError, match="must contain 'target' key"):
            untyped_graph.add_edge_from_dict({"source": "a"})

    def test_get_edge(self, untyped_graph) -> None:
        """Returns edge attributes."""
        untyped_graph.add_edge("a", "b", weight=1.5)
        edge = untyped_graph.get_edge("a", "b")
        assert isinstance(edge, dict)
        assert edge["weight"] == 1.5

    def test_get_edge_missing_source_raises(self, untyped_graph) -> None:
        """Raises KeyError if source node missing."""
        untyped_graph.add_node("b")
        with pytest.raises(KeyError, match="Source node.*not found"):
            untyped_graph.get_edge("a", "b")

    def test_get_edge_missing_target_raises(self, untyped_graph) -> None:
        """Raises KeyError if target node missing."""
        untyped_graph.add_node("a")
        with pytest.raises(KeyError, match="Target node.*not found"):
            untyped_graph.get_edge("a", "b")

    def test_get_edge_missing_edge_raises(self, untyped_graph) -> None:
        """Raises KeyError if edge doesn't exist."""
        untyped_graph.add_node("a")
        untyped_graph.add_node("b")
        with pytest.raises(KeyError, match="Edge.*not found"):
            untyped_graph.get_edge("a", "b")

    def test_get_edge_as_model(self, graph) -> None:
        """Returns GraphModel instance."""
        graph.add_node("a", type="Node")
        graph.add_node("b", type="Node")
        graph.add_edge("a", "b", type="LINKS")
        model = graph.get_edge_as_model("a", "b")
        assert model.source == "a"
        assert model.target == "b"
        assert model.type == "LINKS"

    def test_get_edge_as_model_no_schema_raises(self, untyped_graph) -> None:
        """Raises ValueError if no edge model configured."""
        untyped_graph.add_edge("a", "b")
        with pytest.raises(ValueError, match="No edge model configured"):
            untyped_graph.get_edge_as_model("a", "b")

    def test_remove_edge(self, untyped_graph) -> None:
        """Removes edge."""
        untyped_graph.add_edge("a", "b")
        untyped_graph.remove_edge("a", "b")
        assert ("a", "b") not in untyped_graph
        # Nodes should still exist
        assert "a" in untyped_graph
        assert "b" in untyped_graph

    def test_remove_edge_missing_raises(self, untyped_graph) -> None:
        """Raises KeyError for missing edge."""
        untyped_graph.add_node("a")
        untyped_graph.add_node("b")
        with pytest.raises(KeyError, match="Edge.*not found"):
            untyped_graph.remove_edge("a", "b")

    def test_has_edge(self, untyped_graph) -> None:
        """Returns True for existing edge."""
        untyped_graph.add_edge("a", "b")
        assert untyped_graph.has_edge("a", "b") is True

    def test_has_edge_missing(self, untyped_graph) -> None:
        """Returns False for missing edge."""
        assert untyped_graph.has_edge("a", "b") is False
        untyped_graph.add_node("a")
        untyped_graph.add_node("b")
        assert untyped_graph.has_edge("a", "b") is False

    def test_edges(self, untyped_graph) -> None:
        """Returns list of (source, target) tuples."""
        untyped_graph.add_edge("a", "b")
        untyped_graph.add_edge("b", "c")
        edges = untyped_graph.edges()
        assert set(edges) == {("a", "b"), ("b", "c")}


class TestGraphValidation:
    """Tests for Graph validation with schema."""

    def test_add_node_validates_against_schema(self, graph) -> None:
        """Data is validated through model."""
        graph.add_node("a", type="TestType", name="Test")
        node = graph.get_node("a")
        # model_dump includes all fields
        assert node["node"] == "a"
        assert node["type"] == "TestType"
        assert node["name"] == "Test"

    def test_add_edge_validates_against_schema(self, graph) -> None:
        """Data is validated through model."""
        graph.add_node("a", type="Node")
        graph.add_node("b", type="Node")
        graph.add_edge("a", "b", type="LINKS")
        edge = graph.get_edge("a", "b")
        assert edge["source"] == "a"
        assert edge["target"] == "b"
        assert edge["type"] == "LINKS"

    def test_untyped_mode_no_validation(self, untyped_graph) -> None:
        """No validation when schema=None."""
        # Can add any arbitrary data
        untyped_graph.add_node("a", arbitrary_field="value", number=123)
        node = untyped_graph.get_node("a")
        assert node["arbitrary_field"] == "value"
        assert node["number"] == 123


class TestGraphChangeTracking:
    """Tests for Graph change tracking."""

    def test_has_pending_changes_initially_false(self, untyped_graph) -> None:
        """No changes at start."""
        assert untyped_graph.has_pending_changes is False

    def test_has_pending_changes_after_add_node(self, untyped_graph) -> None:
        """True after node add."""
        untyped_graph.add_node("a")
        assert untyped_graph.has_pending_changes is True

    def test_has_pending_changes_after_add_edge(self, untyped_graph) -> None:
        """True after edge add."""
        untyped_graph.add_edge("a", "b")
        assert untyped_graph.has_pending_changes is True

    def test_discard_changes(self, untyped_graph) -> None:
        """Clears pending changes."""
        untyped_graph.add_node("a")
        untyped_graph.add_edge("a", "b")
        assert untyped_graph.has_pending_changes is True
        untyped_graph.discard_changes()
        assert untyped_graph.has_pending_changes is False

    def test_tracker_tracks_node_upserts(self, untyped_graph) -> None:
        """Tracker has node data."""
        untyped_graph.add_node("a", foo="bar")
        assert "a" in untyped_graph.tracker.node_upserts
        assert untyped_graph.tracker.node_upserts["a"]["foo"] == "bar"

    def test_tracker_tracks_edge_upserts(self, untyped_graph) -> None:
        """Tracker has edge data."""
        untyped_graph.add_edge("a", "b", weight=1.0)
        assert ("a", "b") in untyped_graph.tracker.edge_upserts
        assert untyped_graph.tracker.edge_upserts[("a", "b")]["weight"] == 1.0

    def test_tracker_tracks_node_deletes(self, untyped_graph) -> None:
        """Tracker tracks node deletions."""
        untyped_graph.add_node("a")
        untyped_graph.discard_changes()
        untyped_graph.remove_node("a")
        assert "a" in untyped_graph.tracker.node_deletes

    def test_tracker_tracks_edge_deletes(self, untyped_graph) -> None:
        """Tracker tracks edge deletions."""
        untyped_graph.add_edge("a", "b")
        untyped_graph.discard_changes()
        untyped_graph.remove_edge("a", "b")
        assert ("a", "b") in untyped_graph.tracker.edge_deletes

    def test_remove_node_tracks_incident_edges(self, untyped_graph) -> None:
        """Incident edges are tracked for deletion when node removed."""
        untyped_graph.add_edge("a", "b")
        untyped_graph.add_edge("b", "c")
        untyped_graph.discard_changes()
        untyped_graph.remove_node("b")
        assert ("a", "b") in untyped_graph.tracker.edge_deletes
        assert ("b", "c") in untyped_graph.tracker.edge_deletes

    def test_tracker_repr(self, untyped_graph) -> None:
        """Tracker repr shows counts."""
        untyped_graph.add_node("a")
        untyped_graph.add_edge("b", "c")
        repr_str = repr(untyped_graph.tracker)
        assert "nodes_upserts=3" in repr_str
        assert "edges_upserts=1" in repr_str
        assert "nodes_deletes=0" in repr_str
        assert "edges_deletes=0" in repr_str


class TestGraphSyncOperations:
    """Tests for Graph sync operations."""

    def test_sync_no_changes_noop(self, untyped_graph) -> None:
        """Does nothing when no changes."""
        # Should not raise
        untyped_graph.sync()

    def test_sync_without_engine_raises(self, graph) -> None:
        """Raises ValueError when no engine configured."""
        graph.add_node("a", type="Test")
        with pytest.raises(ValueError, match="Cannot sync without database engine"):
            graph.sync()

    def test_sync_not_implemented(self, schema_with_engine) -> None:
        """Raises NotImplementedError."""
        g = Graph(schema_with_engine)
        g.add_node("a", type="Test")
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            g.sync()
