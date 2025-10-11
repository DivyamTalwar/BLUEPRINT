from typing import List, Set, Dict, Any, Optional
import networkx as nx
from src.core.rpg import RepositoryPlanningGraph, NodeType
from src.utils.logger import StructuredLogger

logger = StructuredLogger("topological_traversal")


class TopologicalTraversal:

    def __init__(self, rpg: RepositoryPlanningGraph):
        self.rpg = rpg
        self.logger = logger

        # Track node status
        self.status = {}  # node_id → status
        for node_id in rpg.graph.nodes():
            self.status[node_id] = "pending"

    def get_execution_order(self) -> List[str]:
        """
        Get topological ordering of leaf nodes (functions/classes to generate).

        Returns:
            List of leaf node IDs in dependency order
        """
        try:
            # Get all leaf nodes
            leaf_nodes = [
                n for n, d in self.rpg.graph.nodes(data=True)
                if d.get("type") == NodeType.LEAF.value
            ]

            # Create subgraph with only leaf nodes and their dependencies
            subgraph = self._create_dependency_subgraph(leaf_nodes)

            # Topological sort
            order = list(nx.topological_sort(subgraph))

            # Filter to only leaf nodes
            leaf_order = [n for n in order if n in leaf_nodes]

            self.logger.log("info", f"Execution order computed: {len(leaf_order)} nodes")
            return leaf_order

        except nx.NetworkXError as e:
            self.logger.log("error", "Failed to create topological order (graph has cycles)", error=str(e))
            # Fallback: BFS order
            return self._get_bfs_order()

    def _create_dependency_subgraph(self, leaf_nodes: List[str]) -> nx.DiGraph:
        """
        Create subgraph containing only dependency relationships.

        Uses data_flow and execution_order edges.
        """
        subgraph = nx.DiGraph()

        # Add all leaf nodes
        for node_id in leaf_nodes:
            subgraph.add_node(node_id)

        # Add dependency edges
        for u, v, edge_data in self.rpg.graph.edges(data=True):
            if u in leaf_nodes and v in leaf_nodes:
                edge_type = edge_data.get("type")
                if edge_type in ["data_flow", "execution_order"]:
                    subgraph.add_edge(u, v)

        return subgraph

    def _get_bfs_order(self) -> List[str]:
        """Fallback: BFS order if topological sort fails."""
        leaf_nodes = [
            n for n, d in self.rpg.graph.nodes(data=True)
            if d.get("type") == NodeType.LEAF.value
        ]

        if not leaf_nodes:
            return []

        # Start from first leaf
        root = leaf_nodes[0]
        bfs_order = list(nx.bfs_tree(self.rpg.graph, root))

        # Filter to leaf nodes only
        return [n for n in bfs_order if n in leaf_nodes]

    def group_by_level(self) -> List[List[str]]:
        """
        Group nodes by execution level (distance from root).

        Returns nodes grouped by their level in the dependency graph.
        Nodes at the same level can be executed in parallel.

        Returns:
            List of lists, where each inner list contains node IDs at that level
        """
        execution_order = self.get_execution_order()

        if not execution_order:
            return []

        # Get all leaf nodes
        leaf_nodes = [
            n for n, d in self.rpg.graph.nodes(data=True)
            if d.get("type") == NodeType.LEAF.value
        ]

        # Create dependency subgraph
        dep_graph = self._create_dependency_subgraph(leaf_nodes)

        # Find nodes with no incoming edges (level 0)
        levels = []
        processed = set()
        remaining = set(execution_order)

        while remaining:
            # Find nodes with all dependencies processed
            current_level = []
            for node in remaining:
                predecessors = set(dep_graph.predecessors(node))
                if predecessors.issubset(processed):
                    current_level.append(node)

            if not current_level:
                # No progress - shouldn't happen with valid DAG
                # Add remaining nodes to final level
                levels.append(list(remaining))
                break

            levels.append(current_level)
            processed.update(current_level)
            remaining -= set(current_level)

        self.logger.log("info", "Grouped execution into levels", num_levels=len(levels))

        return levels

    def get_next_batch(self, batch_size: int = 1) -> List[str]:
        """
        Get next batch of nodes ready for generation.

        A node is ready if:
        1. Status is "pending"
        2. All dependencies are "completed"

        Args:
            batch_size: Number of nodes to return

        Returns:
            List of node IDs ready for generation
        """
        ready_nodes = []

        for node_id in self.rpg.graph.nodes():
            # Skip non-leaf nodes
            if self.rpg.graph.nodes[node_id].get("type") != NodeType.LEAF.value:
                continue

            # Skip if not pending
            if self.status.get(node_id) != "pending":
                continue

            # Check if all dependencies are completed
            if self._are_dependencies_ready(node_id):
                ready_nodes.append(node_id)

                if len(ready_nodes) >= batch_size:
                    break

        return ready_nodes

    def _are_dependencies_ready(self, node_id: str) -> bool:
        """
        Check if all dependencies of a node are completed.

        A node depends on its predecessors (nodes with edges pointing to it).
        """
        for predecessor in self.rpg.graph.predecessors(node_id):
            pred_status = self.status.get(predecessor)

            # If predecessor is a leaf node, it must be completed
            pred_type = self.rpg.graph.nodes[predecessor].get("type")
            if pred_type == NodeType.LEAF.value:
                if pred_status != "completed":
                    return False

        return True

    def mark_in_progress(self, node_id: str):
        """Mark node as being generated."""
        self.status[node_id] = "in_progress"
        self.logger.log("debug", f"Node marked in_progress: {node_id}")

    def mark_completed(self, node_id: str):
        """Mark node as successfully generated."""
        self.status[node_id] = "completed"
        self.rpg.graph.nodes[node_id]["status"] = "completed"
        self.logger.log("debug", f"Node marked completed: {node_id}")

    def mark_failed(self, node_id: str, error: str):
        """Mark node as failed."""
        self.status[node_id] = "failed"
        self.rpg.graph.nodes[node_id]["status"] = "failed"
        self.rpg.graph.nodes[node_id]["error"] = error
        self.logger.log("warning", f"Node marked failed: {node_id}", error=error)

    def get_progress(self) -> Dict[str, Any]:
        """
        Get current progress statistics.

        Returns:
            Dict with counts of pending/in_progress/completed/failed nodes
        """
        leaf_nodes = [
            n for n, d in self.rpg.graph.nodes(data=True)
            if d.get("type") == NodeType.LEAF.value
        ]

        stats = {
            "total": len(leaf_nodes),
            "pending": 0,
            "in_progress": 0,
            "completed": 0,
            "failed": 0,
        }

        for node_id in leaf_nodes:
            status = self.status.get(node_id, "pending")
            stats[status] = stats.get(status, 0) + 1

        stats["progress_percent"] = (stats["completed"] / stats["total"] * 100) if stats["total"] > 0 else 0

        return stats

    def is_complete(self) -> bool:
        """Check if all nodes are completed or failed."""
        stats = self.get_progress()
        return stats["pending"] == 0 and stats["in_progress"] == 0

    def save_checkpoint(self, filepath: str):
        """Save current status to checkpoint file."""
        import json

        checkpoint = {
            "status": self.status,
            "progress": self.get_progress(),
        }

        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        self.logger.log("info", f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath: str):
        """Load status from checkpoint file."""
        import json

        try:
            with open(filepath, 'r') as f:
                checkpoint = json.load(f)

            self.status = checkpoint.get("status", {})
            self.logger.log("info", f"Checkpoint loaded: {filepath}")

        except Exception as e:
            self.logger.log("error", "Failed to load checkpoint", error=str(e))

    def get_failed_nodes(self) -> List[str]:
        """Get list of nodes that failed generation."""
        return [node_id for node_id, status in self.status.items() if status == "failed"]

    def retry_failed_nodes(self):
        """Reset failed nodes to pending for retry."""
        failed = self.get_failed_nodes()

        for node_id in failed:
            self.status[node_id] = "pending"
            if "error" in self.rpg.graph.nodes[node_id]:
                del self.rpg.graph.nodes[node_id]["error"]

        self.logger.log("info", f"Reset {len(failed)} failed nodes for retry")
