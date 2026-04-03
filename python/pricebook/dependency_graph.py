"""
Dependency graph for incremental risk computation.

DAG of market data → curves → instruments → portfolio. When a market
data node changes, only downstream dependents are recomputed.

    from pricebook.dependency_graph import DependencyGraph, GraphNode

    g = DependencyGraph()
    rate = g.add_node("ois_5y", category="market_data")
    curve = g.add_node("ois_curve", category="curve", depends_on=[rate])
    swap = g.add_node("irs_5y", category="instrument", depends_on=[curve])

    g.mark_dirty(rate)
    dirty = g.get_dirty_nodes()  # [rate, curve, swap]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class NodeCategory(Enum):
    MARKET_DATA = "market_data"
    CURVE = "curve"
    INSTRUMENT = "instrument"
    AGGREGATION = "aggregation"


@dataclass
class GraphNode:
    """A node in the dependency graph."""

    name: str
    category: NodeCategory
    dirty: bool = False
    value: float | None = None
    _dependents: list[GraphNode] = field(default_factory=list, repr=False)
    _dependencies: list[GraphNode] = field(default_factory=list, repr=False)

    def add_dependent(self, node: GraphNode) -> None:
        if node not in self._dependents:
            self._dependents.append(node)

    def add_dependency(self, node: GraphNode) -> None:
        if node not in self._dependencies:
            self._dependencies.append(node)

    @property
    def dependents(self) -> list[GraphNode]:
        return list(self._dependents)

    @property
    def dependencies(self) -> list[GraphNode]:
        return list(self._dependencies)


class DependencyGraph:
    """DAG of computation dependencies.

    Supports dirty-flag propagation and topological ordering
    for incremental recomputation.
    """

    def __init__(self):
        self._nodes: dict[str, GraphNode] = {}

    def add_node(
        self,
        name: str,
        category: str | NodeCategory = NodeCategory.MARKET_DATA,
        depends_on: list[GraphNode] | None = None,
    ) -> GraphNode:
        """Add a node to the graph.

        Args:
            name: unique node identifier.
            category: "market_data", "curve", "instrument", or "aggregation".
            depends_on: list of upstream nodes this node depends on.
        """
        if isinstance(category, str):
            category = NodeCategory(category)

        if name in self._nodes:
            raise ValueError(f"Node '{name}' already exists")

        node = GraphNode(name=name, category=category)
        self._nodes[name] = node

        if depends_on:
            for dep in depends_on:
                node.add_dependency(dep)
                dep.add_dependent(node)

        return node

    def get_node(self, name: str) -> GraphNode:
        if name not in self._nodes:
            raise KeyError(f"Node '{name}' not found")
        return self._nodes[name]

    def remove_node(self, name: str) -> None:
        """Remove a node and all its edges."""
        if name not in self._nodes:
            raise KeyError(f"Node '{name}' not found")
        node = self._nodes[name]

        # Remove from dependents' dependency lists
        for dep in node._dependencies:
            dep._dependents = [d for d in dep._dependents if d is not node]

        # Remove from dependencies' dependent lists
        for dep in node._dependents:
            dep._dependencies = [d for d in dep._dependencies if d is not node]

        del self._nodes[name]

    def mark_dirty(self, node: GraphNode) -> None:
        """Mark a node and all downstream dependents as dirty."""
        if node.dirty:
            return
        node.dirty = True
        for dep in node._dependents:
            self.mark_dirty(dep)

    def mark_dirty_by_name(self, name: str) -> None:
        self.mark_dirty(self.get_node(name))

    def get_dirty_nodes(self) -> list[GraphNode]:
        """Return all dirty nodes in topological order (upstream first)."""
        dirty = [n for n in self._nodes.values() if n.dirty]
        return self._topological_sort(dirty)

    def clean_all(self) -> None:
        """Reset all dirty flags."""
        for node in self._nodes.values():
            node.dirty = False

    def _topological_sort(self, nodes: list[GraphNode]) -> list[GraphNode]:
        """Topological sort of a subset of nodes."""
        node_set = set(id(n) for n in nodes)
        visited = set()
        result = []

        def visit(n):
            if id(n) in visited or id(n) not in node_set:
                return
            visited.add(id(n))
            for dep in n._dependencies:
                visit(dep)
            result.append(n)

        for n in nodes:
            visit(n)

        return result

    def has_cycle(self) -> bool:
        """Check for circular dependencies."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {name: WHITE for name in self._nodes}

        def dfs(name):
            color[name] = GRAY
            node = self._nodes[name]
            for dep in node._dependents:
                if color[dep.name] == GRAY:
                    return True
                if color[dep.name] == WHITE and dfs(dep.name):
                    return True
            color[name] = BLACK
            return False

        for name in self._nodes:
            if color[name] == WHITE:
                if dfs(name):
                    return True
        return False

    @property
    def size(self) -> int:
        return len(self._nodes)

    def nodes_by_category(self, category: str | NodeCategory) -> list[GraphNode]:
        if isinstance(category, str):
            category = NodeCategory(category)
        return [n for n in self._nodes.values() if n.category == category]
