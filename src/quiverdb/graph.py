"""Graph - In-memory graph with optional schema validation and change tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rustworkx import PyDiGraph

from .tracker import ChangeTracker

if TYPE_CHECKING:
    from collections.abc import Hashable

    from .graph_model import GraphModel
    from .graph_schema import GraphSchema


class Graph:
    """In-memory directed graph with optional schema validation and change tracking.

    Wraps rustworkx.PyDiGraph with a dict-like interface for nodes and edges.
    When a schema is provided, all data is validated against the configured
    GraphModel classes.

    Usage:
        # Typed mode with validation
        schema = GraphSchema(Node, Edge)
        G = Graph(schema)
        G["wiki-1"] = {"type": "Wiki", "name": "Home"}  # Validates against Node

        # Untyped mode for prototyping
        G = Graph()
        G["wiki-1"] = {"name": "Home"}  # No validation

        # Dict-like access
        G["node-1"]                    # Get node attrs
        G["node-1", "node-2"]          # Get edge attrs
        G["node-1"] = {"key": "value"} # Upsert node
        G["node-1", "node-2"] = {...}  # Upsert edge
        "node-1" in G                  # Check node exists
        ("node-1", "node-2") in G      # Check edge exists
    """

    def __init__(self, schema: GraphSchema | None = None) -> None:
        """Initialize the graph.

        Args:
            schema: Optional GraphSchema for validation. If None, the graph
                    operates in untyped mode (no validation).
        """
        self._graph = PyDiGraph()
        self._node_to_idx: dict[Hashable, int] = {}
        self._schema = schema
        self.tracker = ChangeTracker()

    @property
    def schema(self) -> GraphSchema | None:
        """Get the schema, if configured."""
        return self._schema

    @property
    def has_validation(self) -> bool:
        """Check if schema validation is enabled."""
        return self._schema is not None

    # --- Dunder Methods ---

    def __contains__(self, key: Hashable | tuple[Hashable, Hashable]) -> bool:
        """Check if node or edge exists.

        Usage:
            if "wiki-1" in G:              # Check node
            if ("wiki-1", "wiki-2") in G:  # Check edge
        """
        if isinstance(key, tuple) and len(key) == 2:
            return self.has_edge(key[0], key[1])
        return key in self._node_to_idx

    def __getitem__(self, key: Hashable | tuple[Hashable, Hashable]) -> dict[str, Any]:
        """Get node or edge attributes.

        Usage:
            G["wiki-1"]              # Get node attrs
            G["wiki-1", "wiki-2"]    # Get edge attrs
        """
        if isinstance(key, tuple) and len(key) == 2:
            return self.get_edge(key[0], key[1])
        return self.get_node(key)

    def __setitem__(
        self, key: Hashable | tuple[Hashable, Hashable], attr: dict[str, Any]
    ) -> None:
        """Add or update node/edge attributes (upsert behavior).

        Note: This MERGES attributes with existing data, it does not replace.
        Existing attributes not in `attr` are preserved.

        Usage:
            # Node upsert
            G["wiki-1"] = {"type": "Wiki", "name": "Home"}
            G["wiki-1"] = {"url": "https://..."}  # Adds url, keeps type and name

            # Edge upsert
            G["wiki-1", "wiki-2"] = {"type": "links"}
            G["wiki-1", "wiki-2"] = {"weight": 2.0}  # Adds weight, keeps type
        """
        if isinstance(key, tuple) and len(key) == 2:
            self.add_edge(key[0], key[1], **attr)
        else:
            self.add_node(key, **attr)

    def __len__(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self._node_to_idx)

    def __repr__(self) -> str:
        n_nodes = len(self._node_to_idx)
        n_edges = self._graph.num_edges()
        schema = "validated" if self._schema else "untyped"
        return f"Graph(nodes={n_nodes}, edges={n_edges}, mode={schema})"

    # --- Validation Helpers ---

    def _validate_node(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate node data against schema if configured."""
        if self._schema is None or not self._schema.has_node_model:
            return data
        validated = self._schema.node_model.model_validate(data)
        return validated.model_dump()

    def _validate_edge(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate edge data against schema if configured."""
        if self._schema is None or not self._schema.has_edge_model:
            return data
        validated = self._schema.edge_model.model_validate(data)
        return validated.model_dump()

    # --- Node Operations ---

    def add_node(self, node: Hashable, **attr: Any) -> None:
        """Add or update a node with attributes.

        If the node exists, merges new attributes with existing ones.
        If schema is configured, validates the merged data.

        Args:
            node: The node identifier.
            **attr: Node attributes to set.
        """
        if node in self._node_to_idx:
            idx = self._node_to_idx[node]
            node_data = {**self._graph[idx], **attr}
        else:
            node_data = {"node": node, **attr}

        node_data = self._validate_node(node_data)

        if node in self._node_to_idx:
            self._graph[self._node_to_idx[node]] = node_data
        else:
            idx = self._graph.add_node(node_data)
            self._node_to_idx[node] = idx

        self.tracker.track_node_upsert(node, node_data)

    def add_node_from_dict(self, node_dict: dict[str, Any]) -> None:
        """Add a node from a dictionary containing 'node' key.

        Args:
            node_dict: Dictionary with 'node' key and optional attributes.

        Raises:
            ValueError: If 'node' key is missing.
        """
        if "node" not in node_dict:
            raise ValueError("Node dictionary must contain 'node' key")

        node = node_dict["node"]
        attr = {k: v for k, v in node_dict.items() if k != "node"}
        self.add_node(node, **attr)

    def get_node(self, node: Hashable) -> dict[str, Any]:
        """Get node attributes.

        Args:
            node: The node identifier.

        Returns:
            Dictionary of node attributes.

        Raises:
            KeyError: If node doesn't exist.
        """
        if node not in self._node_to_idx:
            raise KeyError(f"Node '{node}' not found in graph")

        idx = self._node_to_idx[node]
        return self._graph[idx]

    def get_node_as_model(self, node: Hashable) -> GraphModel:
        """Get node data as a validated GraphModel instance.

        Args:
            node: The node identifier.

        Returns:
            GraphModel instance with node data.

        Raises:
            ValueError: If no node model is configured.
            KeyError: If node doesn't exist.
        """
        if self._schema is None or not self._schema.has_node_model:
            raise ValueError("No node model configured")
        return self._schema.node_model.model_validate(self.get_node(node))

    def remove_node(self, node: Hashable) -> None:
        """Remove a node and all its edges.

        Args:
            node: The node identifier.

        Raises:
            KeyError: If node doesn't exist.
        """
        if node not in self._node_to_idx:
            raise KeyError(f"Node '{node}' not found in graph")

        idx = self._node_to_idx[node]

        # Track edge deletions for incident edges
        for predecessor_idx in self._graph.predecessor_indices(idx):
            predecessor_node = self._graph[predecessor_idx]["node"]
            self.tracker.track_edge_delete(predecessor_node, node)

        for successor_idx in self._graph.successor_indices(idx):
            successor_node = self._graph[successor_idx]["node"]
            self.tracker.track_edge_delete(node, successor_node)

        self._graph.remove_node(idx)
        del self._node_to_idx[node]
        self.tracker.track_node_delete(node)

    # --- Edge Operations ---

    def add_edge(self, source: Hashable, target: Hashable, **attr: Any) -> None:
        """Add or update an edge with attributes.

        Auto-creates source/target nodes if they don't exist.
        If the edge exists, merges new attributes with existing ones.
        If schema is configured, validates the merged data.

        Args:
            source: Source node identifier.
            target: Target node identifier.
            **attr: Edge attributes to set.
        """
        # Auto-create nodes if missing
        if source not in self._node_to_idx:
            self.add_node(source)
        if target not in self._node_to_idx:
            self.add_node(target)

        src_idx = self._node_to_idx[source]
        tgt_idx = self._node_to_idx[target]

        edge_data = {"source": source, "target": target, **attr}

        if self._graph.has_edge(src_idx, tgt_idx):
            current = self._graph.get_edge_data(src_idx, tgt_idx)
            edge_data = {**current, **attr, "source": source, "target": target}

        edge_data = self._validate_edge(edge_data)

        if self._graph.has_edge(src_idx, tgt_idx):
            self._graph.update_edge(src_idx, tgt_idx, edge_data)
        else:
            self._graph.add_edge(src_idx, tgt_idx, edge_data)

        self.tracker.track_edge_upsert(source, target, edge_data)

    def add_edge_from_dict(self, edge_dict: dict[str, Any]) -> None:
        """Add an edge from a dictionary containing 'source' and 'target' keys.

        Args:
            edge_dict: Dictionary with 'source', 'target' keys and optional attributes.

        Raises:
            ValueError: If 'source' or 'target' key is missing.
        """
        if "source" not in edge_dict:
            raise ValueError("Edge dictionary must contain 'source' key")
        if "target" not in edge_dict:
            raise ValueError("Edge dictionary must contain 'target' key")

        source = edge_dict["source"]
        target = edge_dict["target"]
        attr = {k: v for k, v in edge_dict.items() if k not in ("source", "target")}
        self.add_edge(source, target, **attr)

    def get_edge(self, source: Hashable, target: Hashable) -> dict[str, Any]:
        """Get edge attributes.

        Args:
            source: Source node identifier.
            target: Target node identifier.

        Returns:
            Dictionary of edge attributes.

        Raises:
            KeyError: If source/target node or edge doesn't exist.
        """
        if source not in self._node_to_idx:
            raise KeyError(f"Source node '{source}' not found in graph")
        if target not in self._node_to_idx:
            raise KeyError(f"Target node '{target}' not found in graph")

        src_idx = self._node_to_idx[source]
        tgt_idx = self._node_to_idx[target]

        if not self._graph.has_edge(src_idx, tgt_idx):
            raise KeyError(f"Edge '{source}' -> '{target}' not found in graph")

        return self._graph.get_edge_data(src_idx, tgt_idx)

    def get_edge_as_model(self, source: Hashable, target: Hashable) -> GraphModel:
        """Get edge data as a validated GraphModel instance.

        Args:
            source: Source node identifier.
            target: Target node identifier.

        Returns:
            GraphModel instance with edge data.

        Raises:
            ValueError: If no edge model is configured.
            KeyError: If source/target node or edge doesn't exist.
        """
        if self._schema is None or not self._schema.has_edge_model:
            raise ValueError("No edge model configured")
        return self._schema.edge_model.model_validate(self.get_edge(source, target))

    def remove_edge(self, source: Hashable, target: Hashable) -> None:
        """Remove an edge.

        Args:
            source: Source node identifier.
            target: Target node identifier.

        Raises:
            KeyError: If source/target node or edge doesn't exist.
        """
        if source not in self._node_to_idx:
            raise KeyError(f"Source node '{source}' not found in graph")
        if target not in self._node_to_idx:
            raise KeyError(f"Target node '{target}' not found in graph")

        src_idx = self._node_to_idx[source]
        tgt_idx = self._node_to_idx[target]

        if not self._graph.has_edge(src_idx, tgt_idx):
            raise KeyError(f"Edge '{source}' -> '{target}' not found in graph")

        self._graph.remove_edge(src_idx, tgt_idx)
        self.tracker.track_edge_delete(source, target)

    def has_edge(self, source: Hashable, target: Hashable) -> bool:
        """Check if an edge exists.

        Args:
            source: Source node identifier.
            target: Target node identifier.

        Returns:
            True if the edge exists, False otherwise.
        """
        if source not in self._node_to_idx or target not in self._node_to_idx:
            return False

        src_idx = self._node_to_idx[source]
        tgt_idx = self._node_to_idx[target]
        return self._graph.has_edge(src_idx, tgt_idx)

    # --- Sync Operations ---

    def sync(self) -> None:
        """Sync all changes to database.

        Performs in order:
        1. Delete edges from DB
        2. Delete nodes from DB
        3. Upsert nodes to DB
        4. Upsert edges to DB

        Changes are cleared after successful sync.

        Raises:
            ValueError: If no schema or engine is configured.
            NotImplementedError: Database sync is not yet implemented.
        """
        if not self.tracker.has_changes:
            return

        if self._schema is None or self._schema.engine is None:
            raise ValueError(
                "Cannot sync without database engine. "
                "Initialize Graph with GraphSchema(models, engine=engine)."
            )

        # TODO: Implement actual sync with SQLAlchemy session
        raise NotImplementedError(
            "Database sync not yet implemented. "
            "Use SQLAlchemy Session directly for database operations."
        )

    # --- Change Tracking ---

    @property
    def has_pending_changes(self) -> bool:
        """Check if there are unsaved changes."""
        return self.tracker.has_changes

    def discard_changes(self) -> None:
        """Discard all pending changes without syncing."""
        self.tracker.clear()

    # --- Iteration ---

    def nodes(self) -> list[Hashable]:
        """Get all node identifiers."""
        return list(self._node_to_idx.keys())

    def edges(self) -> list[tuple[Hashable, Hashable]]:
        """Get all edges as (source, target) tuples."""
        result = []
        for src_idx, tgt_idx in self._graph.edge_list():
            src_data = self._graph[src_idx]
            tgt_data = self._graph[tgt_idx]
            result.append((src_data["node"], tgt_data["node"]))
        return result
