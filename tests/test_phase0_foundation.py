import sys
import os
import json
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.utils.config_loader import get_config
from src.utils.cost_tracker import CostTracker
from src.utils.file_ops import read_file, write_file
from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType
from src.core.graph_operations import GraphOperations
from src.core.graph_persistence import GraphPersistence
from src.utils.docker_runner import DockerRunner


class TestConfigLoader:
    """Test configuration loading and validation."""

    def test_load_config_success(self):
        """Test successful config loading."""
        config_loader = get_config()

        assert config_loader is not None
        config = config_loader.get_all()
        assert isinstance(config, dict)
        print("✓ Config loaded successfully")

    def test_config_has_required_keys(self):
        """Test config contains all required keys."""
        config = get_config().get_all()

        required_keys = ['stage1', 'stage2', 'stage3', 'llm', 'docker']
        for key in required_keys:
            assert key in config, f"Missing required config key: {key}"

        print("✓ All required config keys present")

    def test_stage1_config(self):
        """Test Stage 1 configuration."""
        config_loader = get_config()
        stage1 = config_loader.get_stage_config(1)

        assert 'iterations' in stage1
        assert isinstance(stage1['iterations'], int)
        assert stage1['iterations'] > 0

        print(f"✓ Stage 1 config valid (iterations: {stage1['iterations']})")

    def test_docker_config(self):
        """Test Docker configuration."""
        config = get_config().get_all()
        docker_cfg = config.get('docker', {})

        assert 'timeout' in docker_cfg
        assert 'base_image' in docker_cfg

        print("✓ Docker config valid")


class TestCostTracker:
    """Test cost tracking functionality."""

    def test_cost_tracker_initialization(self):
        """Test cost tracker initializes correctly."""
        tracker = CostTracker()

        assert tracker is not None
        assert hasattr(tracker, 'total_cost')

        print("✓ Cost tracker initialized")

    def test_track_llm_call(self):
        """Test tracking LLM API calls."""
        tracker = CostTracker()

        initial_cost = tracker.total_cost

        # Simulate tracking a call
        tracker.track_call(
            model="gemini-2.0-flash-exp",
            input_tokens=100,
            output_tokens=200
        )

        assert tracker.total_cost >= initial_cost

        print(f"✓ LLM call tracked (cost: ${tracker.total_cost:.4f})")

    def test_get_report(self):
        """Test generating cost report."""
        tracker = CostTracker()

        tracker.track_call("gemini-2.0-flash-exp", 100, 200)
        tracker.track_call("claude-3.5-sonnet", 50, 100)

        report = tracker.get_report()

        assert report is not None
        assert 'total_cost' in report
        assert 'calls' in report

        print(f"✓ Cost report generated: {report}")


class TestFileOperations:
    """Test file operation utilities."""

    def test_safe_write_and_read(self):
        """Test writing and reading files safely."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.txt")
            test_content = "Hello, BLUEPRINT!"

            # Write
            write_file(test_file, test_content)

            # Read
            content = read_file(test_file)
            assert content == test_content

            print("✓ Safe file write/read works")

    def test_handle_nonexistent_file(self):
        """Test reading nonexistent file."""
        try:
            content = read_file("/nonexistent/path/file.txt")
            # If it doesn't raise an error, content should be empty or None
            assert content is None or content == ""
        except FileNotFoundError:
            # This is also acceptable behavior
            pass

        print("✓ Nonexistent file handled gracefully")


class TestRPGCore:
    """Test Repository Planning Graph core functionality."""

    def test_rpg_initialization(self):
        """Test RPG initializes correctly."""
        rpg = RepositoryPlanningGraph("Test Repository")

        assert rpg is not None
        assert rpg.repository_goal == "Test Repository"
        assert rpg.graph.number_of_nodes() == 0

        print("✓ RPG initialized")

    def test_add_nodes(self):
        """Test adding nodes of different types."""
        rpg = RepositoryPlanningGraph("Test")

        # Add root node
        root_id = rpg.add_node("module1", NodeType.ROOT, "Main module")
        assert root_id in rpg.graph
        assert rpg.graph.nodes[root_id]["type"] == NodeType.ROOT.value

        # Add intermediate node
        inter_id = rpg.add_node("file1.py", NodeType.INTERMEDIATE, "Python file")
        assert inter_id in rpg.graph

        # Add leaf node
        leaf_id = rpg.add_node("function1", NodeType.LEAF, "A function")
        assert leaf_id in rpg.graph

        print(f"✓ Added nodes: {rpg.graph.number_of_nodes()} total")

    def test_add_edges(self):
        """Test adding edges between nodes."""
        rpg = RepositoryPlanningGraph("Test")

        node1 = rpg.add_node("node1", NodeType.ROOT, "Node 1")
        node2 = rpg.add_node("node2", NodeType.INTERMEDIATE, "Node 2")

        # Add hierarchy edge
        rpg.add_edge(node1, node2, EdgeType.HIERARCHY)

        assert rpg.graph.has_edge(node1, node2)

        print(f"✓ Added edges: {rpg.graph.number_of_edges()} total")

    def test_get_nodes_by_type(self):
        """Test filtering nodes by type."""
        rpg = RepositoryPlanningGraph("Test")

        rpg.add_node("root1", NodeType.ROOT, "Root 1")
        rpg.add_node("root2", NodeType.ROOT, "Root 2")
        rpg.add_node("file1", NodeType.INTERMEDIATE, "File 1")
        rpg.add_node("func1", NodeType.LEAF, "Function 1")

        root_nodes = [n for n, d in rpg.graph.nodes(data=True) if d.get("type") == NodeType.ROOT.value]

        assert len(root_nodes) == 2

        print(f"✓ Node filtering works: {len(root_nodes)} root nodes")


class TestGraphOperations:
    """Test graph operation utilities."""

    def test_validate_graph(self):
        """Test graph validation."""
        rpg = RepositoryPlanningGraph("Test")

        # Create valid graph
        root = rpg.add_node("root", NodeType.ROOT, "Root")
        leaf = rpg.add_node("leaf", NodeType.LEAF, "Leaf")
        rpg.add_edge(root, leaf, EdgeType.HIERARCHY)

        ops = GraphOperations(rpg)
        is_valid = ops.validate()

        assert is_valid

        print("✓ Graph validation works")

    def test_detect_cycles(self):
        """Test cycle detection."""
        rpg = RepositoryPlanningGraph("Test")

        # Create cycle
        node1 = rpg.add_node("node1", NodeType.LEAF, "Node 1")
        node2 = rpg.add_node("node2", NodeType.LEAF, "Node 2")
        node3 = rpg.add_node("node3", NodeType.LEAF, "Node 3")

        rpg.add_edge(node1, node2, EdgeType.DATA_FLOW)
        rpg.add_edge(node2, node3, EdgeType.DATA_FLOW)
        rpg.add_edge(node3, node1, EdgeType.DATA_FLOW)  # Creates cycle

        ops = GraphOperations(rpg)
        has_cycles = ops.has_cycles()

        assert has_cycles

        print("✓ Cycle detection works")

    def test_topological_sort(self):
        """Test topological sorting."""
        rpg = RepositoryPlanningGraph("Test")

        # Create DAG
        node1 = rpg.add_node("node1", NodeType.LEAF, "Node 1")
        node2 = rpg.add_node("node2", NodeType.LEAF, "Node 2")
        node3 = rpg.add_node("node3", NodeType.LEAF, "Node 3")

        rpg.add_edge(node1, node2, EdgeType.DATA_FLOW)
        rpg.add_edge(node2, node3, EdgeType.DATA_FLOW)

        ops = GraphOperations(rpg)
        order = ops.topological_sort()

        assert len(order) == 3
        assert order.index(node1) < order.index(node2)
        assert order.index(node2) < order.index(node3)

        print(f"✓ Topological sort works: {order}")


class TestGraphPersistence:
    """Test graph save/load functionality."""

    def test_save_and_load_json(self):
        """Test JSON persistence."""
        rpg = RepositoryPlanningGraph("Test Repository")

        # Add some nodes
        root = rpg.add_node("root", NodeType.ROOT, "Root node")
        leaf = rpg.add_node("leaf", NodeType.LEAF, "Leaf node")
        rpg.add_edge(root, leaf, EdgeType.HIERARCHY)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_graph.json")

            # Save
            persistence = GraphPersistence({})
            persistence.save_json(rpg, filepath)

            assert os.path.exists(filepath)

            # Load
            loaded_rpg = persistence.load_json(filepath)

            assert loaded_rpg is not None
            assert loaded_rpg.repository_goal == "Test Repository"
            assert loaded_rpg.graph.number_of_nodes() == 2
            assert loaded_rpg.graph.number_of_edges() == 1

            print("✓ JSON save/load works")

    def test_save_pickle(self):
        """Test pickle persistence."""
        rpg = RepositoryPlanningGraph("Test")
        rpg.add_node("node1", NodeType.ROOT, "Node 1")

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_graph.pkl")

            persistence = GraphPersistence({})
            persistence.save_pickle(rpg, filepath)

            assert os.path.exists(filepath)

            # Load
            loaded_rpg = persistence.load_pickle(filepath)

            assert loaded_rpg is not None
            assert loaded_rpg.graph.number_of_nodes() == 1

            print("✓ Pickle save/load works")


class TestDockerRunner:
    """Test Docker execution capabilities."""

    def test_docker_runner_initialization(self):
        """Test Docker runner initializes."""
        config = {"timeout": 300, "base_image": "python:3.11-slim"}

        try:
            runner = DockerRunner(config)
            assert runner is not None
            print("✓ Docker runner initialized")
        except Exception as e:
            print(f"⚠️  Docker not available: {e}")
            pytest.skip("Docker not available")

    def test_docker_availability(self):
        """Test Docker availability check."""
        config = {"timeout": 300, "base_image": "python:3.11-slim"}

        try:
            runner = DockerRunner(config)
            is_available = runner.is_docker_available()

            if is_available:
                print("✓ Docker is available")
            else:
                print("⚠️  Docker is not running")
                pytest.skip("Docker not running")
        except Exception as e:
            print(f"⚠️  Docker check failed: {e}")
            pytest.skip("Docker not available")

    @pytest.mark.skipif(not os.getenv("TEST_DOCKER"), reason="Docker tests disabled by default")
    def test_run_simple_code(self):
        """Test running simple Python code in Docker."""
        config = {"timeout": 60, "base_image": "python:3.11-slim"}

        runner = DockerRunner(config)

        if not runner.is_docker_available():
            pytest.skip("Docker not available")

        simple_code = """
print("Hello from Docker!")
result = 2 + 2
print(f"Result: {result}")
"""

        result = runner.run_code(simple_code, test_mode=False, timeout=30)

        assert result["success"] == True
        assert "Hello from Docker!" in result["output"]
        assert "Result: 4" in result["output"]

        print("✓ Docker code execution works")


def run_all_phase0_tests():
    """Run all Phase 0 tests and generate report."""
    print("\n" + "=" * 70)
    print("LEGENDARY TESTING - PHASE 0: FOUNDATION COMPONENTS")
    print("=" * 70 + "\n")

    test_classes = [
        TestConfigLoader,
        TestCostTracker,
        TestFileOperations,
        TestRPGCore,
        TestGraphOperations,
        TestGraphPersistence,
        TestDockerRunner,
    ]

    results = {
        "total_tests": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": []
    }

    for test_class in test_classes:
        print(f"\n📋 Testing: {test_class.__name__}")
        print("-" * 70)

        test_instance = test_class()

        # Get all test methods
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]

        for test_method_name in test_methods:
            results["total_tests"] += 1

            try:
                test_method = getattr(test_instance, test_method_name)
                test_method()
                results["passed"] += 1

            except pytest.skip.Exception as e:
                results["skipped"] += 1
                print(f"⊘ Skipped: {test_method_name} - {e}")

            except Exception as e:
                results["failed"] += 1
                results["errors"].append({
                    "test": f"{test_class.__name__}.{test_method_name}",
                    "error": str(e)
                })
                print(f"✗ Failed: {test_method_name}")
                print(f"  Error: {e}")

    # Print summary
    print("\n" + "=" * 70)
    print("PHASE 0 TEST RESULTS")
    print("=" * 70)
    print(f"Total Tests: {results['total_tests']}")
    print(f"✓ Passed: {results['passed']}")
    print(f"✗ Failed: {results['failed']}")
    print(f"⊘ Skipped: {results['skipped']}")

    success_rate = (results['passed'] / results['total_tests'] * 100) if results['total_tests'] > 0 else 0
    print(f"\nSuccess Rate: {success_rate:.1f}%")

    if results['errors']:
        print("\n❌ Failed Tests:")
        for error in results['errors']:
            print(f"  - {error['test']}: {error['error']}")

    print("=" * 70 + "\n")

    return results


if __name__ == "__main__":
    results = run_all_phase0_tests()

    # Exit with appropriate code
    if results['failed'] > 0:
        sys.exit(1)
    else:
        print("🎉 ALL PHASE 0 TESTS PASSED!")
        sys.exit(0)
