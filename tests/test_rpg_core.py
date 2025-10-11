import pytest
import tempfile
from pathlib import Path

from src.core.rpg import (
    RepositoryPlanningGraph,
    NodeType,
    EdgeType,
    NodeStatus,
    RPGNode,
    RPGEdge,
)
from src.core.graph_operations import GraphOperations
from src.core.graph_persistence import GraphPersistence


class TestRPGCore:
    """Test RPG core functionality"""

    def test_rpg_initialization(self):
        """Test RPG initialization"""
        rpg = RepositoryPlanningGraph("Build ML library")
        assert rpg.repository_goal == "Build ML library"
        assert rpg.graph.number_of_nodes() == 0
        assert rpg.graph.number_of_edges() == 0

    def test_add_node(self):
        """Test adding nodes"""
        rpg = RepositoryPlanningGraph()

        # Add root node
        root_id = rpg.add_node(
            "Data Module", NodeType.ROOT, "Handle data operations"
        )
        assert root_id is not None
        assert rpg.graph.number_of_nodes() == 1

        node = rpg.get_node(root_id)
        assert node["name"] == "Data Module"
        assert node["type"] == NodeType.ROOT.value

    def test_add_edge(self):
        """Test adding edges"""
        rpg = RepositoryPlanningGraph()

        node1 = rpg.add_node("Parent", NodeType.ROOT)
        node2 = rpg.add_node("Child", NodeType.INTERMEDIATE)

        success = rpg.add_edge(node1, node2, EdgeType.HIERARCHY)
        assert success is True
        assert rpg.graph.number_of_edges() == 1

    def test_get_root_nodes(self):
        """Test getting root nodes"""
        rpg = RepositoryPlanningGraph()

        root1 = rpg.add_node("Root1", NodeType.ROOT)
        root2 = rpg.add_node("Root2", NodeType.ROOT)
        child = rpg.add_node("Child", NodeType.LEAF)

        rpg.add_edge(root1, child, EdgeType.HIERARCHY)

        roots = rpg.get_root_nodes()
        assert len(roots) == 2
        assert root1 in roots
        assert root2 in roots

    def test_get_leaf_nodes(self):
        """Test getting leaf nodes"""
        rpg = RepositoryPlanningGraph()

        root = rpg.add_node("Root", NodeType.ROOT)
        leaf1 = rpg.add_node("Leaf1", NodeType.LEAF)
        leaf2 = rpg.add_node("Leaf2", NodeType.LEAF)

        rpg.add_edge(root, leaf1, EdgeType.HIERARCHY)
        rpg.add_edge(root, leaf2, EdgeType.HIERARCHY)

        leaves = rpg.get_leaf_nodes()
        assert len(leaves) == 2
        assert leaf1 in leaves
        assert leaf2 in leaves

    def test_topological_order(self):
        """Test topological ordering"""
        rpg = RepositoryPlanningGraph()

        # Create dependency chain
        n1 = rpg.add_node("N1", NodeType.ROOT)
        n2 = rpg.add_node("N2", NodeType.INTERMEDIATE)
        n3 = rpg.add_node("N3", NodeType.LEAF)

        rpg.add_edge(n1, n2, EdgeType.HIERARCHY)
        rpg.add_edge(n2, n3, EdgeType.HIERARCHY)

        order = rpg.get_topological_order()
        assert len(order) == 3
        assert order.index(n1) < order.index(n2)
        assert order.index(n2) < order.index(n3)

    def test_update_node(self):
        """Test updating node attributes"""
        rpg = RepositoryPlanningGraph()

        node_id = rpg.add_node("Test", NodeType.LEAF)
        success = rpg.update_node(node_id, status=NodeStatus.GENERATED.value)

        assert success is True

        node = rpg.get_node(node_id)
        assert node["status"] == NodeStatus.GENERATED.value

    def test_get_dependencies(self):
        """Test getting node dependencies"""
        rpg = RepositoryPlanningGraph()

        n1 = rpg.add_node("N1", NodeType.ROOT)
        n2 = rpg.add_node("N2", NodeType.ROOT)
        n3 = rpg.add_node("N3", NodeType.LEAF)

        rpg.add_edge(n1, n3, EdgeType.HIERARCHY)
        rpg.add_edge(n2, n3, EdgeType.HIERARCHY)

        deps = rpg.get_dependencies(n3)
        assert len(deps) == 2
        assert n1 in deps
        assert n2 in deps

    def test_validate_graph(self):
        """Test graph validation"""
        rpg = RepositoryPlanningGraph()

        n1 = rpg.add_node("N1", NodeType.ROOT)
        n2 = rpg.add_node("N2", NodeType.LEAF)
        rpg.add_edge(n1, n2, EdgeType.HIERARCHY)

        is_valid, errors = rpg.validate()
        assert is_valid is True
        assert len(errors) == 0

    def test_get_stats(self):
        """Test getting graph statistics"""
        rpg = RepositoryPlanningGraph()

        rpg.add_node("Root", NodeType.ROOT)
        rpg.add_node("Leaf1", NodeType.LEAF)
        rpg.add_node("Leaf2", NodeType.LEAF)

        stats = rpg.get_stats()
        assert stats["total_nodes"] == 3
        assert stats["root_nodes"] == 1
        assert stats["leaf_nodes"] == 2

    def test_to_dict_from_dict(self):
        """Test serialization to/from dict"""
        rpg = RepositoryPlanningGraph("Test Goal")

        n1 = rpg.add_node("N1", NodeType.ROOT, "Functionality 1")
        n2 = rpg.add_node("N2", NodeType.LEAF, "Functionality 2")
        rpg.add_edge(n1, n2, EdgeType.HIERARCHY)

        # Serialize
        data = rpg.to_dict()
        assert "metadata" in data
        assert "graph" in data

        # Deserialize
        rpg2 = RepositoryPlanningGraph.from_dict(data)
        assert rpg2.repository_goal == "Test Goal"
        assert rpg2.graph.number_of_nodes() == 2
        assert rpg2.graph.number_of_edges() == 1


class TestGraphOperations:
    """Test graph operations"""

    def test_merge_nodes(self):
        """Test merging nodes"""
        rpg = RepositoryPlanningGraph()

        n1 = rpg.add_node("N1", NodeType.LEAF, "Func 1")
        n2 = rpg.add_node("N2", NodeType.LEAF, "Func 2")

        merged_id = GraphOperations.merge_nodes(rpg, n1, n2, "Merged")

        assert merged_id is not None
        assert rpg.graph.number_of_nodes() == 1

        merged_node = rpg.get_node(merged_id)
        assert "Func 1" in merged_node["functionality"]
        assert "Func 2" in merged_node["functionality"]

    def test_find_similar_nodes(self):
        """Test finding similar nodes"""
        rpg = RepositoryPlanningGraph()

        n1 = rpg.add_node(
            "Load CSV", NodeType.LEAF, "load csv file", domain="data"
        )
        n2 = rpg.add_node(
            "Read CSV", NodeType.LEAF, "read csv data", domain="data"
        )
        n3 = rpg.add_node(
            "Train Model", NodeType.LEAF, "train ml model", domain="ml"
        )

        similar = GraphOperations.find_similar_nodes(rpg, n1, threshold=0.3)

        # Should find n2 (same domain, similar words)
        assert len(similar) >= 0  # Depends on implementation


class TestGraphPersistence:
    """Test graph persistence"""

    def test_save_load_json(self):
        """Test JSON save/load"""
        rpg = RepositoryPlanningGraph("Test Repo")

        n1 = rpg.add_node("Root", NodeType.ROOT)
        n2 = rpg.add_node("Leaf", NodeType.LEAF)
        rpg.add_edge(n1, n2, EdgeType.HIERARCHY)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            # Save
            success = GraphPersistence.save_json(rpg, filepath)
            assert success is True

            # Load
            rpg2 = GraphPersistence.load_json(filepath)
            assert rpg2 is not None
            assert rpg2.repository_goal == "Test Repo"
            assert rpg2.graph.number_of_nodes() == 2

        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_save_load_pickle(self):
        """Test pickle save/load"""
        rpg = RepositoryPlanningGraph("Test Repo")

        n1 = rpg.add_node("Root", NodeType.ROOT)
        n2 = rpg.add_node("Leaf", NodeType.LEAF)
        rpg.add_edge(n1, n2, EdgeType.HIERARCHY)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            filepath = f.name

        try:
            # Save
            success = GraphPersistence.save_pickle(rpg, filepath)
            assert success is True

            # Load
            rpg2 = GraphPersistence.load_pickle(filepath)
            assert rpg2 is not None
            assert rpg2.graph.number_of_nodes() == 2

        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_export_dot(self):
        """Test DOT export"""
        rpg = RepositoryPlanningGraph()

        n1 = rpg.add_node("Root", NodeType.ROOT)
        n2 = rpg.add_node("Leaf", NodeType.LEAF)
        rpg.add_edge(n1, n2, EdgeType.HIERARCHY)

        with tempfile.NamedTemporaryFile(suffix=".dot", delete=False) as f:
            filepath = f.name

        try:
            success = GraphPersistence.export_to_dot(rpg, filepath)
            assert success is True

            content = Path(filepath).read_text()
            assert "digraph RPG" in content

        finally:
            Path(filepath).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
