from typing import List, Dict, Any, Optional, Set, Tuple
import networkx as nx
from collections import defaultdict

from src.utils.logger import get_logger

logger = get_logger(__name__)


class GraphUtils:
    """Utility functions for graph operations"""

    @staticmethod
    def create_node(
        node_id: str,
        node_type: str,
        name: str,
        functionality: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        node = {
            "id": node_id,
            "type": node_type,
            "name": name,
            "functionality": functionality,
            "description": kwargs.get("description", ""),
            "file_path": kwargs.get("file_path", ""),
            "code_type": kwargs.get("code_type", ""),
            "signature": kwargs.get("signature", ""),
            "docstring": kwargs.get("docstring", ""),
            "dependencies": kwargs.get("dependencies", []),
            "domain": kwargs.get("domain", ""),
            "complexity": kwargs.get("complexity", "basic"),
            "status": kwargs.get("status", "planned"),
        }
        return node

    @staticmethod
    def create_edge(
        edge_type: str, from_node: str, to_node: str, **kwargs
    ) -> Dict[str, Any]:
        """
        Create an edge dictionary

        Args:
            edge_type: Type (hierarchy/data_flow/execution_order)
            from_node: Source node ID
            to_node: Target node ID
            **kwargs: Additional attributes

        Returns:
            Edge dictionary
        """
        edge = {
            "type": edge_type,
            "from": from_node,
            "to": to_node,
        }

        if edge_type == "data_flow":
            edge.update(
                {
                    "data_type": kwargs.get("data_type", ""),
                    "transformation": kwargs.get("transformation", ""),
                }
            )

        edge.update(kwargs)
        return edge

    @staticmethod
    def validate_graph(graph: nx.DiGraph) -> Tuple[bool, List[str]]:
        """
        Validate graph integrity

        Args:
            graph: NetworkX DiGraph

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check for cycles
        if not nx.is_directed_acyclic_graph(graph):
            cycles = list(nx.simple_cycles(graph))
            errors.append(f"Graph contains {len(cycles)} cycles")
            logger.error("Graph validation failed: cycles detected")

        # Check for orphan nodes
        orphans = [node for node in graph.nodes() if graph.degree(node) == 0]
        if orphans:
            errors.append(f"Found {len(orphans)} orphan nodes: {orphans[:5]}")
            logger.warning("Found orphan nodes: %d", len(orphans))

        # Check for missing required attributes
        for node_id, data in graph.nodes(data=True):
            required_attrs = ["type", "name"]
            missing = [attr for attr in required_attrs if attr not in data]
            if missing:
                errors.append(
                    f"Node {node_id} missing attributes: {missing}"
                )

        is_valid = len(errors) == 0
        return is_valid, errors

    @staticmethod
    def get_root_nodes(graph: nx.DiGraph) -> List[str]:
        """Get all root nodes (no incoming edges)"""
        return [node for node in graph.nodes() if graph.in_degree(node) == 0]

    @staticmethod
    def get_leaf_nodes(graph: nx.DiGraph) -> List[str]:
        """Get all leaf nodes (no outgoing edges)"""
        return [node for node in graph.nodes() if graph.out_degree(node) == 0]

    @staticmethod
    def get_subgraph(graph: nx.DiGraph, **filters) -> nx.DiGraph:
        """
        Extract subgraph based on filters

        Args:
            graph: NetworkX DiGraph
            **filters: Node attribute filters (e.g., domain="data_ops")

        Returns:
            Subgraph
        """
        matching_nodes = []

        for node_id, data in graph.nodes(data=True):
            match = True
            for key, value in filters.items():
                if data.get(key) != value:
                    match = False
                    break
            if match:
                matching_nodes.append(node_id)

        return graph.subgraph(matching_nodes).copy()

    @staticmethod
    def find_dependencies(graph: nx.DiGraph, node_id: str) -> List[str]:
        """Find all dependencies of a node (predecessors)"""
        return list(graph.predecessors(node_id))

    @staticmethod
    def find_dependents(graph: nx.DiGraph, node_id: str) -> List[str]:
        """Find all dependents of a node (successors)"""
        return list(graph.successors(node_id))

    @staticmethod
    def topological_sort(graph: nx.DiGraph) -> List[str]:
        """
        Get topological ordering of nodes

        Returns:
            List of node IDs in dependency order
        """
        try:
            return list(nx.topological_sort(graph))
        except nx.NetworkXError as e:
            logger.error("Topological sort failed: %s", str(e))
            return []

    @staticmethod
    def find_path(graph: nx.DiGraph, from_node: str, to_node: str) -> List[str]:
        """
        Find path between two nodes

        Args:
            graph: NetworkX DiGraph
            from_node: Start node ID
            to_node: End node ID

        Returns:
            List of node IDs in path (empty if no path)
        """
        try:
            return nx.shortest_path(graph, from_node, to_node)
        except nx.NetworkXNoPath:
            return []
        except nx.NodeNotFound:
            logger.error("Node not found in graph")
            return []

    @staticmethod
    def get_complexity_score(graph: nx.DiGraph) -> float:
        """
        Estimate implementation complexity

        Returns:
            Complexity score (0-100)
        """
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()

        # Simple heuristic
        complexity = (num_nodes * 0.5) + (num_edges * 0.3)

        # Adjust for leaf nodes (actual code)
        leaf_nodes = GraphUtils.get_leaf_nodes(graph)
        complexity += len(leaf_nodes) * 0.2

        return min(complexity, 100.0)

    @staticmethod
    def get_coverage(graph: nx.DiGraph, reference_domains: Set[str]) -> float:
        """
        Calculate domain coverage

        Args:
            graph: NetworkX DiGraph
            reference_domains: Set of reference domains

        Returns:
            Coverage percentage (0-100)
        """
        graph_domains = set()

        for node_id, data in graph.nodes(data=True):
            domain = data.get("domain", "")
            if domain:
                graph_domains.add(domain)

        if not reference_domains:
            return 0.0

        covered = graph_domains.intersection(reference_domains)
        coverage = (len(covered) / len(reference_domains)) * 100

        return coverage

    @staticmethod
    def detect_circular_dependencies(graph: nx.DiGraph) -> List[List[str]]:
        """
        Detect circular dependencies

        Returns:
            List of cycles (each cycle is a list of node IDs)
        """
        try:
            cycles = list(nx.simple_cycles(graph))
            return cycles
        except:
            return []

    @staticmethod
    def merge_graphs(graph1: nx.DiGraph, graph2: nx.DiGraph) -> nx.DiGraph:
        """
        Merge two graphs

        Args:
            graph1: First graph
            graph2: Second graph

        Returns:
            Merged graph
        """
        merged = graph1.copy()
        merged.add_nodes_from(graph2.nodes(data=True))
        merged.add_edges_from(graph2.edges(data=True))
        return merged

    @staticmethod
    def graph_to_dict(graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Convert graph to dictionary format

        Args:
            graph: NetworkX DiGraph

        Returns:
            Dictionary representation
        """
        return {
            "nodes": [
                {"id": node_id, **data} for node_id, data in graph.nodes(data=True)
            ],
            "edges": [
                {"from": u, "to": v, **data} for u, v, data in graph.edges(data=True)
            ],
        }

    @staticmethod
    def dict_to_graph(data: Dict[str, Any]) -> nx.DiGraph:
        """
        Convert dictionary to graph

        Args:
            data: Dictionary with nodes and edges

        Returns:
            NetworkX DiGraph
        """
        graph = nx.DiGraph()

        for node in data.get("nodes", []):
            node_id = node.pop("id")
            graph.add_node(node_id, **node)

        for edge in data.get("edges", []):
            from_node = edge.pop("from")
            to_node = edge.pop("to")
            graph.add_edge(from_node, to_node, **edge)

        return graph


# Convenience functions for backward compatibility
def get_all_leaves(rpg):
    """Get all leaf nodes from RPG (convenience function)"""
    from src.core.rpg import NodeType

    leaf_nodes = []
    for node_id, data in rpg.graph.nodes(data=True):
        if data.get("type") == NodeType.LEAF:
            leaf_nodes.append(node_id)
    return leaf_nodes


def get_all_roots(rpg):
    """Get all root nodes from RPG (convenience function)"""
    from src.core.rpg import NodeType

    root_nodes = []
    for node_id, data in rpg.graph.nodes(data=True):
        if data.get("type") == NodeType.ROOT:
            root_nodes.append(node_id)
    return root_nodes
