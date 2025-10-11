from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from datetime import datetime
import uuid

from src.utils.graph_utils import GraphUtils
from src.utils.logger import get_logger

logger = get_logger(__name__)


class NodeType(str, Enum):
    ROOT = "root"  # Top-level modules/folders
    INTERMEDIATE = "intermediate"  # Files
    LEAF = "leaf"  # Functions/classes


class CodeType(str, Enum):
    FOLDER = "folder"
    FILE = "file"
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"


class EdgeType(str, Enum):
    HIERARCHY = "hierarchy"  # Parent-child relationship
    DATA_FLOW = "data_flow"  # Data passing between nodes
    EXECUTION_ORDER = "execution_order"  # Temporal dependencies


class NodeStatus(str, Enum):
    PLANNED = "planned"  # Stage 1: Functionality defined
    DESIGNED = "designed"  # Stage 2: Structure defined
    GENERATED = "generated"  # Stage 3: Code created but not validated
    SYNTAX_VALID = "syntax_valid"  # Stage 3: Passed static validation
    VALIDATED = "validated"  # Stage 3: Tests passed in Docker
    FAILED = "failed"  # Stage 3: Failed to generate code


@dataclass
class RPGNode:
    """RPG Node representation"""

    id: str
    type: NodeType
    name: str

    # Functional semantics (Stage 1)
    functionality: str = ""
    description: str = ""

    # Structural semantics (Stage 2)
    file_path: str = ""
    code_type: Optional[CodeType] = None

    # Implementation details (Stage 2-3)
    signature: str = ""
    docstring: str = ""
    implementation: str = ""
    dependencies: List[str] = field(default_factory=list)

    # Metadata
    domain: str = ""
    subdomain: str = ""
    complexity: str = "basic"  # basic/intermediate/advanced
    status: NodeStatus = NodeStatus.PLANNED

    # Testing
    test_code: str = ""
    test_passed: bool = False
    validation_method: str = ""  # "docker", "static", or "none"
    validation_errors: List[str] = field(default_factory=list)
    generation_attempts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "type": self.type.value if isinstance(self.type, NodeType) else self.type,
            "name": self.name,
            "functionality": self.functionality,
            "description": self.description,
            "file_path": self.file_path,
            "code_type": self.code_type.value if isinstance(self.code_type, CodeType) else self.code_type,
            "signature": self.signature,
            "docstring": self.docstring,
            "implementation": self.implementation,
            "dependencies": self.dependencies,
            "domain": self.domain,
            "subdomain": self.subdomain,
            "complexity": self.complexity,
            "status": self.status.value if isinstance(self.status, NodeStatus) else self.status,
            "test_code": self.test_code,
            "test_passed": self.test_passed,
            "validation_method": self.validation_method,
            "validation_errors": self.validation_errors,
            "generation_attempts": self.generation_attempts,
        }


@dataclass
class RPGEdge:
    """RPG Edge representation"""

    type: EdgeType
    from_node: str
    to_node: str

    # Data flow specific
    data_type: str = ""
    transformation: str = ""

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "type": self.type.value if isinstance(self.type, EdgeType) else self.type,
            "from": self.from_node,
            "to": self.to_node,
            "data_type": self.data_type,
            "transformation": self.transformation,
            "metadata": self.metadata,
        }


class RepositoryPlanningGraph:
    """
    Repository Planning Graph (RPG)

    Core data structure that unifies functional planning and code structure.
    """

    def __init__(self, repository_goal: str = ""):
        """
        Initialize RPG

        Args:
            repository_goal: User's repository description
        """
        self.graph = nx.DiGraph()
        self.repository_goal = repository_goal
        self.metadata = {
            "created": datetime.now().isoformat(),
            "repository_goal": repository_goal,
            "version": "1.0",
        }

        logger.info("RPG initialized for goal: %s", repository_goal)

    def add_node(
        self,
        name: str,
        node_type: NodeType,
        functionality: str = "",
        node_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Add node to graph

        Args:
            name: Node name
            node_type: Type of node
            functionality: What it does
            node_id: Optional custom ID
            **kwargs: Additional node attributes

        Returns:
            Node ID
        """
        if node_id is None:
            node_id = str(uuid.uuid4())

        node = RPGNode(
            id=node_id,
            type=node_type,
            name=name,
            functionality=functionality,
            **kwargs,
        )

        self.graph.add_node(node_id, **node.to_dict())
        logger.debug("Added node: %s (%s)", name, node_type.value)

        return node_id

    def add_edge(
        self,
        from_node: str,
        to_node: str,
        edge_type: EdgeType,
        **kwargs,
    ) -> bool:
        """
        Add edge to graph

        Args:
            from_node: Source node ID
            to_node: Target node ID
            edge_type: Type of edge
            **kwargs: Additional edge attributes

        Returns:
            True if successful
        """
        if from_node not in self.graph:
            logger.error("Source node not found: %s", from_node)
            return False

        if to_node not in self.graph:
            logger.error("Target node not found: %s", to_node)
            return False

        edge = RPGEdge(type=edge_type, from_node=from_node, to_node=to_node, **kwargs)

        self.graph.add_edge(from_node, to_node, **edge.to_dict())
        logger.debug("Added edge: %s -> %s (%s)", from_node, to_node, edge_type.value)

        return True

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node data"""
        if node_id in self.graph:
            return self.graph.nodes[node_id]
        return None

    def update_node(self, node_id: str, **kwargs) -> bool:
        """Update node attributes"""
        if node_id not in self.graph:
            logger.error("Node not found: %s", node_id)
            return False

        self.graph.nodes[node_id].update(kwargs)
        logger.debug("Updated node: %s", node_id)
        return True

    def remove_node(self, node_id: str) -> bool:
        """Remove node from graph"""
        if node_id not in self.graph:
            return False

        self.graph.remove_node(node_id)
        logger.debug("Removed node: %s", node_id)
        return True

    def get_root_nodes(self) -> List[str]:
        """Get all root nodes"""
        return GraphUtils.get_root_nodes(self.graph)

    def get_leaf_nodes(self) -> List[str]:
        """Get all leaf nodes"""
        return GraphUtils.get_leaf_nodes(self.graph)

    def get_nodes_by_type(self, node_type: NodeType) -> List[str]:
        """Get all nodes of specific type"""
        return [
            node_id
            for node_id, data in self.graph.nodes(data=True)
            if data.get("type") == node_type.value
        ]

    def get_nodes_by_status(self, status: NodeStatus) -> List[str]:
        """Get all nodes with specific status"""
        return [
            node_id
            for node_id, data in self.graph.nodes(data=True)
            if data.get("status") == status.value
        ]

    def get_dependencies(self, node_id: str) -> List[str]:
        """Get dependencies of a node"""
        return GraphUtils.find_dependencies(self.graph, node_id)

    def get_dependents(self, node_id: str) -> List[str]:
        """Get dependents of a node"""
        return GraphUtils.find_dependents(self.graph, node_id)

    def get_topological_order(self) -> List[str]:
        """Get nodes in topological order"""
        return GraphUtils.topological_sort(self.graph)

    def get_subgraph(self, **filters) -> "RepositoryPlanningGraph":
        """Extract subgraph based on filters"""
        subgraph_nx = GraphUtils.get_subgraph(self.graph, **filters)

        # Create new RPG with subgraph
        subgraph_rpg = RepositoryPlanningGraph(self.repository_goal)
        subgraph_rpg.graph = subgraph_nx
        subgraph_rpg.metadata = self.metadata.copy()

        return subgraph_rpg

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate graph integrity"""
        return GraphUtils.validate_graph(self.graph)

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        stats = {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "root_nodes": len(self.get_root_nodes()),
            "leaf_nodes": len(self.get_leaf_nodes()),
            "is_dag": nx.is_directed_acyclic_graph(self.graph),
            "complexity_score": GraphUtils.get_complexity_score(self.graph),
        }

        # Count by type
        for node_type in NodeType:
            stats[f"{node_type.value}_nodes"] = len(
                self.get_nodes_by_type(node_type)
            )

        # Count by status
        for status in NodeStatus:
            stats[f"{status.value}_nodes"] = len(self.get_nodes_by_status(status))

        return stats

    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary"""
        return {
            "metadata": self.metadata,
            "graph": GraphUtils.graph_to_dict(self.graph),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RepositoryPlanningGraph":
        """Import from dictionary"""
        rpg = cls(repository_goal=data["metadata"].get("repository_goal", ""))
        rpg.metadata = data["metadata"]
        rpg.graph = GraphUtils.dict_to_graph(data["graph"])
        return rpg

    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"RPG(nodes={stats['total_nodes']}, edges={stats['total_edges']}, goal='{self.repository_goal[:50]}...')"
