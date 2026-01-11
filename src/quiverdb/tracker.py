from dataclasses import dataclass, field
from typing import Dict, Set, Hashable, Any


@dataclass
class ChangeTracker:
    """
    Unit of Work pattern for tracking graph changes.

    Tracks pending INSERT/UPDATE/DELETE operations for both nodes
    and edges until sync() is called on the graph.

    The tracker handles the following scenarios:
    - Adding a new node: tracked in _node_upserts
    - Updating an existing node: overwrites previous entry in _node_upserts
    - Deleting a node: removes from _node_upserts (if pending), adds to _node_deletes
    - Re-adding a deleted node: removes from _node_deletes, adds to _node_upserts
    """
    node_upserts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edge_upserts: Dict[tuple, Dict[str, Any]] = field(default_factory=dict)
    node_deletes: Set[str] = field(default_factory=set)
    edge_deletes: Set[tuple] = field(default_factory=set)

    def track_node_upsert(self, node: Hashable, data: Dict[str, Any]) -> None:
        """Track a node insert or update"""
        self.node_deletes.discard(str(node))
        self.node_upserts[str(node)] = data

    def track_node_delete(self, node: Hashable) -> None:
        """Track a node deletion"""
        self.node_upserts.pop(str(node), None)
        self.node_deletes.add(str(node))

    def track_edge_upsert(self, source: Hashable, target: Hashable, data: Dict[str, Any]) -> None:
        """Track an edge insert or update"""
        key = (str(source), str(target))
        self.edge_deletes.discard(key)
        self.edge_upserts[key] = data

    def track_edge_delete(self, source: Hashable, target: Hashable) -> None:
        """Track an edge deletion"""
        key = (str(source), str(target))
        self.edge_upserts.pop(key, None)
        self.edge_deletes.add(key)

    @property
    def has_changes(self) -> bool:
        """Check if there are any pending changes."""
        return bool(
            self.node_upserts or self.edge_upserts or self.node_deletes or self.edge_deletes
        )

    def clear(self) -> None:
        """Clear all tracked changes."""
        self.node_upserts.clear()
        self.edge_upserts.clear()
        self.node_deletes.clear()
        self.edge_deletes.clear()

    def __repr__(self) -> str:
        n_u = f'nodes_upserts={len(self.node_upserts)}'
        e_u = f'edges_upserts={len(self.edge_upserts)}'
        n_d = f'nodes_deletes={len(self.node_deletes)}'
        e_d = f'edges_deletes={len(self.edge_deletes)}'
        return f'ChangeTracker({n_u}, {e_u}, {n_d}, {e_d})'


__all__ = ["ChangeTracker"]
