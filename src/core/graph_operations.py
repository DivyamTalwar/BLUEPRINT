from typing import List, Dict, Any, Optional, Set
import networkx as nx

from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType, NodeStatus
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GraphOperations:
    """Advanced graph operations for RPG manipulation"""

    @staticmethod
    def merge_nodes(
        rpg: RepositoryPlanningGraph, node1_id: str, node2_id: str, new_name: str
    ) -> Optional[str]:
        """
        Merge two nodes into one

        Args:
            rpg: Repository Planning Graph
            node1_id: First node ID
            node2_id: Second node ID
            new_name: Name for merged node

        Returns:
            New node ID or None if failed
        """
        node1 = rpg.get_node(node1_id)
        node2 = rpg.get_node(node2_id)

        if not node1 or not node2:
            logger.error("One or both nodes not found")
            return None

        # Create merged node
        merged_id = rpg.add_node(
            name=new_name,
            node_type=NodeType(node1["type"]),
            functionality=f"{node1['functionality']}; {node2['functionality']}",
            description=f"{node1['description']} | {node2['description']}",
            domain=node1.get("domain", ""),
        )

        # Redirect edges
        for pred in rpg.get_dependencies(node1_id):
            rpg.add_edge(pred, merged_id, EdgeType.HIERARCHY)

        for pred in rpg.get_dependencies(node2_id):
            rpg.add_edge(pred, merged_id, EdgeType.HIERARCHY)

        for succ in rpg.get_dependents(node1_id):
            rpg.add_edge(merged_id, succ, EdgeType.HIERARCHY)

        for succ in rpg.get_dependents(node2_id):
            rpg.add_edge(merged_id, succ, EdgeType.HIERARCHY)

        # Remove old nodes
        rpg.remove_node(node1_id)
        rpg.remove_node(node2_id)

        logger.info("Merged nodes %s and %s into %s", node1_id, node2_id, merged_id)
        return merged_id

    @staticmethod
    def split_node(
        rpg: RepositoryPlanningGraph,
        node_id: str,
        split_criteria: Dict[str, Any],
    ) -> List[str]:
        """
        Split node into multiple nodes

        Args:
            rpg: Repository Planning Graph
            node_id: Node to split
            split_criteria: Criteria for splitting

        Returns:
            List of new node IDs
        """
        node = rpg.get_node(node_id)
        if not node:
            return []

        # Implementation depends on criteria
        # For now, simple example: split by functionality keywords
        new_node_ids = []

        # Extract split parts (simplified)
        parts = split_criteria.get("parts", [])

        for i, part in enumerate(parts):
            new_id = rpg.add_node(
                name=f"{node['name']}_part{i+1}",
                node_type=NodeType(node["type"]),
                functionality=part.get("functionality", ""),
                domain=node.get("domain", ""),
            )
            new_node_ids.append(new_id)

        # Preserve edges (simplified)
        for pred in rpg.get_dependencies(node_id):
            for new_id in new_node_ids:
                rpg.add_edge(pred, new_id, EdgeType.HIERARCHY)

        for succ in rpg.get_dependents(node_id):
            for new_id in new_node_ids:
                rpg.add_edge(new_id, succ, EdgeType.HIERARCHY)

        # Remove original
        rpg.remove_node(node_id)

        logger.info("Split node %s into %d parts", node_id, len(new_node_ids))
        return new_node_ids

    @staticmethod
    def reorder_execution(
        rpg: RepositoryPlanningGraph, subgraph_nodes: List[str]
    ) -> bool:
        """
        Reorder execution edges in subgraph

        Args:
            rpg: Repository Planning Graph
            subgraph_nodes: Nodes to reorder

        Returns:
            True if successful
        """
        # Remove existing execution order edges
        edges_to_remove = []
        for u, v, data in rpg.graph.edges(data=True):
            if (
                u in subgraph_nodes
                and v in subgraph_nodes
                and data.get("type") == EdgeType.EXECUTION_ORDER.value
            ):
                edges_to_remove.append((u, v))

        for edge in edges_to_remove:
            rpg.graph.remove_edge(*edge)

        # Reorder based on dependencies
        subgraph = rpg.graph.subgraph(subgraph_nodes).copy()

        try:
            order = list(nx.topological_sort(subgraph))

            # Add new execution order edges
            for i in range(len(order) - 1):
                rpg.add_edge(order[i], order[i + 1], EdgeType.EXECUTION_ORDER)

            logger.info("Reordered execution for %d nodes", len(subgraph_nodes))
            return True

        except nx.NetworkXError as e:
            logger.error("Failed to reorder: %s", str(e))
            return False

    @staticmethod
    def find_similar_nodes(
        rpg: RepositoryPlanningGraph, node_id: str, threshold: float = 0.7
    ) -> List[str]:
        """
        Find nodes similar to given node

        Args:
            rpg: Repository Planning Graph
            node_id: Reference node
            threshold: Similarity threshold

        Returns:
            List of similar node IDs
        """
        node = rpg.get_node(node_id)
        if not node:
            return []

        similar = []
        ref_functionality = node.get("functionality", "").lower()
        ref_domain = node.get("domain", "")

        for other_id, other_data in rpg.graph.nodes(data=True):
            if other_id == node_id:
                continue

            # Simple similarity based on domain and functionality keywords
            other_functionality = other_data.get("functionality", "").lower()
            other_domain = other_data.get("domain", "")

            # Domain match
            if other_domain == ref_domain:
                # Keyword overlap
                ref_words = set(ref_functionality.split())
                other_words = set(other_functionality.split())

                if ref_words and other_words:
                    overlap = len(ref_words.intersection(other_words))
                    total = len(ref_words.union(other_words))
                    similarity = overlap / total if total > 0 else 0

                    if similarity >= threshold:
                        similar.append(other_id)

        return similar

    @staticmethod
    def extract_module(
        rpg: RepositoryPlanningGraph, module_name: str
    ) -> RepositoryPlanningGraph:
        """
        Extract a module as separate RPG

        Args:
            rpg: Repository Planning Graph
            module_name: Module name to extract

        Returns:
            New RPG with module only
        """
        # Find module root
        module_nodes = []
        for node_id, data in rpg.graph.nodes(data=True):
            if data.get("name") == module_name:
                module_nodes.append(node_id)
                # Get all descendants
                descendants = nx.descendants(rpg.graph, node_id)
                module_nodes.extend(descendants)
                break

        if not module_nodes:
            logger.warning("Module not found: %s", module_name)
            return RepositoryPlanningGraph()

        # Create subgraph
        subgraph = rpg.graph.subgraph(module_nodes).copy()

        # Create new RPG
        module_rpg = RepositoryPlanningGraph(f"Module: {module_name}")
        module_rpg.graph = subgraph

        logger.info("Extracted module %s with %d nodes", module_name, len(module_nodes))
        return module_rpg

    @staticmethod
    def add_module(
        rpg: RepositoryPlanningGraph,
        module_name: str,
        features: List[str],
        parent_id: Optional[str] = None,
    ) -> str:
        """
        Add a complete module to RPG

        Args:
            rpg: Repository Planning Graph
            module_name: Module name
            features: List of feature descriptions
            parent_id: Optional parent node

        Returns:
            Module root node ID
        """
        # Create module root
        module_id = rpg.add_node(
            name=module_name, node_type=NodeType.ROOT, functionality=f"{module_name} module"
        )

        # Connect to parent if provided
        if parent_id:
            rpg.add_edge(parent_id, module_id, EdgeType.HIERARCHY)

        # Add features
        for feature in features:
            feature_id = rpg.add_node(
                name=feature,
                node_type=NodeType.LEAF,
                functionality=feature,
            )
            rpg.add_edge(module_id, feature_id, EdgeType.HIERARCHY)

        logger.info("Added module %s with %d features", module_name, len(features))
        return module_id

    @staticmethod
    def propagate_status(
        rpg: RepositoryPlanningGraph, node_id: str, status: NodeStatus
    ) -> int:
        """
        Propagate status to all descendants

        Args:
            rpg: Repository Planning Graph
            node_id: Starting node
            status: Status to propagate

        Returns:
            Number of nodes updated
        """
        descendants = nx.descendants(rpg.graph, node_id)
        descendants.add(node_id)

        count = 0
        for desc_id in descendants:
            rpg.update_node(desc_id, status=status.value)
            count += 1

        logger.info("Propagated status %s to %d nodes", status.value, count)
        return count

    @staticmethod
    def detect_bottlenecks(rpg: RepositoryPlanningGraph) -> List[Dict[str, Any]]:
        """
        Detect potential bottlenecks in graph

        Args:
            rpg: Repository Planning Graph

        Returns:
            List of bottleneck nodes with metrics
        """
        bottlenecks = []

        for node_id in rpg.graph.nodes():
            in_degree = rpg.graph.in_degree(node_id)
            out_degree = rpg.graph.out_degree(node_id)

            # High fan-in or fan-out indicates bottleneck
            if in_degree > 5 or out_degree > 5:
                node_data = rpg.get_node(node_id)
                bottlenecks.append(
                    {
                        "node_id": node_id,
                        "name": node_data.get("name", ""),
                        "in_degree": in_degree,
                        "out_degree": out_degree,
                        "type": "fan-in" if in_degree > out_degree else "fan-out",
                    }
                )

        logger.info("Detected %d potential bottlenecks", len(bottlenecks))
        return bottlenecks

    @staticmethod
    def optimize_graph(rpg: RepositoryPlanningGraph) -> Dict[str, Any]:
        """
        Optimize graph structure

        Args:
            rpg: Repository Planning Graph

        Returns:
            Optimization report
        """
        report = {
            "nodes_removed": 0,
            "edges_removed": 0,
            "nodes_merged": 0,
        }

        # Remove orphan nodes
        orphans = [node for node in rpg.graph.nodes() if rpg.graph.degree(node) == 0]
        for orphan in orphans:
            rpg.remove_node(orphan)
            report["nodes_removed"] += 1

        # Remove redundant edges (transitive reduction)
        try:
            reduced = nx.transitive_reduction(rpg.graph)
            edges_to_remove = set(rpg.graph.edges()) - set(reduced.edges())
            for edge in edges_to_remove:
                rpg.graph.remove_edge(*edge)
                report["edges_removed"] += 1
        except:
            pass

        logger.info("Graph optimization: %s", report)
        return report

    @staticmethod
    def validate(rpg: RepositoryPlanningGraph) -> bool:
        """
        Validate RPG structure.

        Args:
            rpg: Repository Planning Graph

        Returns:
            True if valid
        """
        try:
            # Check for cycles
            if GraphOperations.has_cycles(rpg):
                logger.warning("Graph has cycles")
                return False

            # Check all nodes have required fields
            for node_id, data in rpg.graph.nodes(data=True):
                if not data.get("type"):
                    logger.warning(f"Node {node_id} missing type")
                    return False
                if not data.get("name"):
                    logger.warning(f"Node {node_id} missing name")
                    return False

            logger.info("Graph validation passed")
            return True

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False

    @staticmethod
    def has_cycles(rpg: RepositoryPlanningGraph) -> bool:
        """
        Check if graph has cycles.

        Args:
            rpg: Repository Planning Graph

        Returns:
            True if cycles exist
        """
        try:
            # NetworkX will raise exception if cycles exist
            list(nx.find_cycle(rpg.graph, orientation='original'))
            return True
        except nx.NetworkXNoCycle:
            return False
        except Exception:
            return False
