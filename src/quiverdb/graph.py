"""Graph - In-memory graph with optional schema validation and change tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import rustworkx
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

    Thread Safety:
        This class is NOT thread-safe. Use external locking if sharing
        a Graph instance across threads.

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
        self._idx_to_node: dict[int, Hashable] = {}
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
        """Validate node data against schema if configured.

        Uses the schema's scoped type registry for type dispatch.
        """
        if self._schema is None or not self._schema.has_node_model:
            return data
        try:
            validated = self._schema.validate_node(data)
            return validated.model_dump()
        except Exception as e:
            node_id = data.get("node", "unknown")
            raise ValueError(f"Validation failed for node '{node_id}': {e}") from e

    def _validate_edge(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate edge data against schema if configured.

        Uses the schema's scoped type registry for type dispatch.
        """
        if self._schema is None or not self._schema.has_edge_model:
            return data
        try:
            validated = self._schema.validate_edge(data)
            return validated.model_dump()
        except Exception as e:
            source = data.get("source", "unknown")
            target = data.get("target", "unknown")
            raise ValueError(
                f"Validation failed for edge '{source}' -> '{target}': {e}"
            ) from e

    # --- Node Operations ---

    def add_node(self, node: Hashable, **attr: Any) -> None:
        """Add or update a node with attributes.

        If the node exists, merges new attributes with existing ones.
        If schema is configured, validates the merged data.

        Args:
            node: The node identifier.
            **attr: Node attributes to set.

        Raises:
            ValueError: If node ID is None or not hashable.
        """
        if node is None:
            raise ValueError("Node ID cannot be None")
        try:
            hash(node)
        except TypeError as e:
            raise ValueError(f"Node ID must be hashable: {node!r}") from e

        if node in self._node_to_idx:
            idx = self._node_to_idx[node]
            # Merge existing data with new attributes
            existing = self._graph[idx]
            node_data = {"node": node, **existing, **attr}
        else:
            node_data = {"node": node, **attr}

        # Validate with node ID included (for schema validation)
        node_data = self._validate_node(node_data)

        # Store without node ID (it's in the mapping)
        stored_data = {k: v for k, v in node_data.items() if k != "node"}

        if node in self._node_to_idx:
            self._graph[self._node_to_idx[node]] = stored_data
        else:
            idx = self._graph.add_node(stored_data)
            self._node_to_idx[node] = idx
            self._idx_to_node[idx] = node

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
            Dictionary of node attributes including the node ID.

        Raises:
            KeyError: If node doesn't exist.
        """
        if node not in self._node_to_idx:
            raise KeyError(f"Node '{node}' not found in graph")

        idx = self._node_to_idx[node]
        return {"node": node, **self._graph[idx]}

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
            predecessor_node = self._idx_to_node[predecessor_idx]
            self.tracker.track_edge_delete(predecessor_node, node)

        for successor_idx in self._graph.successor_indices(idx):
            successor_node = self._idx_to_node[successor_idx]
            self.tracker.track_edge_delete(node, successor_node)

        self._graph.remove_node(idx)
        del self._node_to_idx[node]
        del self._idx_to_node[idx]
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

    # --- Graph Traversal ---

    def successors(self, node: Hashable) -> list[Hashable]:
        """Get all direct successors (out-neighbors) of a node.

        Args:
            node: The node identifier.

        Returns:
            List of successor node identifiers.

        Raises:
            KeyError: If node doesn't exist.
        """
        if node not in self._node_to_idx:
            raise KeyError(f"Node '{node}' not found in graph")
        idx = self._node_to_idx[node]
        return [self._idx_to_node[i] for i in self._graph.successor_indices(idx)]

    def predecessors(self, node: Hashable) -> list[Hashable]:
        """Get all direct predecessors (in-neighbors) of a node.

        Args:
            node: The node identifier.

        Returns:
            List of predecessor node identifiers.

        Raises:
            KeyError: If node doesn't exist.
        """
        if node not in self._node_to_idx:
            raise KeyError(f"Node '{node}' not found in graph")
        idx = self._node_to_idx[node]
        return [self._idx_to_node[i] for i in self._graph.predecessor_indices(idx)]

    def neighbors(self, node: Hashable) -> list[Hashable]:
        """Get all neighbors (predecessors + successors) of a node.

        Args:
            node: The node identifier.

        Returns:
            List of unique neighbor node identifiers.

        Raises:
            KeyError: If node doesn't exist.
        """
        return list(set(self.predecessors(node)) | set(self.successors(node)))

    def out_degree(self, node: Hashable) -> int:
        """Get the number of outgoing edges from a node.

        Args:
            node: The node identifier.

        Returns:
            Number of outgoing edges.

        Raises:
            KeyError: If node doesn't exist.
        """
        if node not in self._node_to_idx:
            raise KeyError(f"Node '{node}' not found in graph")
        idx = self._node_to_idx[node]
        return len(list(self._graph.successor_indices(idx)))

    def in_degree(self, node: Hashable) -> int:
        """Get the number of incoming edges to a node.

        Args:
            node: The node identifier.

        Returns:
            Number of incoming edges.

        Raises:
            KeyError: If node doesn't exist.
        """
        if node not in self._node_to_idx:
            raise KeyError(f"Node '{node}' not found in graph")
        idx = self._node_to_idx[node]
        return len(list(self._graph.predecessor_indices(idx)))

    def degree(self, node: Hashable) -> int:
        """Get the total degree (in + out) of a node.

        Args:
            node: The node identifier.

        Returns:
            Total number of edges incident to the node.

        Raises:
            KeyError: If node doesn't exist.
        """
        return self.in_degree(node) + self.out_degree(node)

    # --- Graph Algorithms ---

    def is_dag(self) -> bool:
        """Check if the graph is a directed acyclic graph (DAG).

        Returns:
            True if the graph has no cycles, False otherwise.
        """
        return rustworkx.is_directed_acyclic_graph(self._graph)

    def topological_sort(self) -> list[Hashable]:
        """Return nodes in topological order.

        Only valid for directed acyclic graphs (DAGs).

        Returns:
            List of node identifiers in topological order.

        Raises:
            ValueError: If the graph contains cycles.
        """
        if not self.is_dag():
            raise ValueError("Cannot perform topological sort: graph contains cycles")
        indices = rustworkx.topological_sort(self._graph)
        return [self._idx_to_node[idx] for idx in indices]

    def has_path(self, source: Hashable, target: Hashable) -> bool:
        """Check if a path exists from source to target.

        Args:
            source: Source node identifier.
            target: Target node identifier.

        Returns:
            True if a path exists, False otherwise.
            Returns False if either node doesn't exist.
        """
        if source not in self._node_to_idx or target not in self._node_to_idx:
            return False
        return rustworkx.has_path(
            self._graph,
            self._node_to_idx[source],
            self._node_to_idx[target],
        )

    def shortest_path(
        self, source: Hashable, target: Hashable
    ) -> list[Hashable] | None:
        """Find the shortest path from source to target.

        Uses BFS to find the path with minimum number of edges.

        Args:
            source: Source node identifier.
            target: Target node identifier.

        Returns:
            List of node identifiers forming the path (including source and target),
            or None if no path exists.

        Raises:
            KeyError: If source or target node doesn't exist.
        """
        if source not in self._node_to_idx:
            raise KeyError(f"Source node '{source}' not found in graph")
        if target not in self._node_to_idx:
            raise KeyError(f"Target node '{target}' not found in graph")

        src_idx = self._node_to_idx[source]
        tgt_idx = self._node_to_idx[target]

        try:
            path_indices = rustworkx.dijkstra_shortest_paths(
                self._graph, src_idx, tgt_idx, weight_fn=lambda _: 1.0
            )
            if tgt_idx not in path_indices:
                return None
            return [self._idx_to_node[idx] for idx in path_indices[tgt_idx]]
        except Exception:
            return None

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
        return [
            (self._idx_to_node[src_idx], self._idx_to_node[tgt_idx])
            for src_idx, tgt_idx in self._graph.edge_list()
        ]
