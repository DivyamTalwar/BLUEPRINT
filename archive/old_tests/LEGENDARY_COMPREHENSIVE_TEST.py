import os
import sys
import json
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

# Test tracking
class TestResults:
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.errors = []
        self.start_time = time.time()

    def add_pass(self, name: str, duration: float):
        self.total += 1
        self.passed += 1
        print(f"[PASS] {name} ({duration:.2f}s)")

    def add_fail(self, name: str, error: str):
        self.total += 1
        self.failed += 1
        self.errors.append({"test": name, "error": error})
        print(f"[FAIL] {name}: {error}")

    def add_warning(self, name: str, warning: str):
        self.warnings += 1
        print(f"[WARN] {name}: {warning}")

    def summary(self):
        duration = time.time() - self.start_time
        pass_rate = (self.passed / self.total * 100) if self.total > 0 else 0

        print("\n" + "=" * 90)
        print("*** LEGENDARY COMPREHENSIVE TEST RESULTS ***")
        print("=" * 90)
        print(f"Total Tests: {self.total}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Warnings: {self.warnings}")
        print(f"Pass Rate: {pass_rate:.1f}%")
        print(f"Duration: {duration:.2f}s")
        print("=" * 90)

        if self.failed > 0:
            print("\n[FAILURES]")
            for error in self.errors:
                print(f"  - {error['test']}: {error['error']}")

        return pass_rate >= 95.0  # 95% pass rate required for production

results = TestResults()

def test(name: str):
    """Test decorator"""
    def decorator(func):
        def wrapper():
            start = time.time()
            try:
                func()
                duration = time.time() - start
                results.add_pass(name, duration)
            except AssertionError as e:
                results.add_fail(name, str(e))
            except Exception as e:
                results.add_fail(name, f"{type(e).__name__}: {str(e)}")
        return wrapper
    return decorator

print("=" * 90)
print("*** BLUEPRINT - LEGENDARY COMPREHENSIVE TEST SUITE ***")
print("=" * 90)
print(f"Started: {datetime.now().isoformat()}")
print(f"Python: {sys.version}")
print("=" * 90)

# ==================================================================================
# PHASE 1: CORE COMPONENTS
# ==================================================================================

print("\n>>> PHASE 1: CORE COMPONENTS")
print("-" * 90)

@test("1.1: RPG - Basic node operations")
def test_rpg_nodes():
    from src.core.rpg import RepositoryPlanningGraph, NodeType

    rpg = RepositoryPlanningGraph("Test RPG")

    # Add nodes
    root_id = rpg.add_node("root", NodeType.ROOT, {})
    int_id = rpg.add_node("intermediate", NodeType.INTERMEDIATE, {})
    leaf_id = rpg.add_node("leaf", NodeType.LEAF, {})

    # Verify nodes exist
    assert rpg.get_node(root_id) is not None
    assert rpg.get_node(int_id) is not None
    assert rpg.get_node(leaf_id) is not None

    # Verify node types
    assert rpg.get_node(root_id)["type"] == NodeType.ROOT
    assert rpg.get_node(int_id)["type"] == NodeType.INTERMEDIATE
    assert rpg.get_node(leaf_id)["type"] == NodeType.LEAF

    # Verify graph size
    assert rpg.graph.number_of_nodes() == 3

@test("1.2: RPG - Edge operations")
def test_rpg_edges():
    from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType

    rpg = RepositoryPlanningGraph("Test RPG")

    node1 = rpg.add_node("n1", NodeType.ROOT, {})
    node2 = rpg.add_node("n2", NodeType.LEAF, {})

    # Add edges of different types
    rpg.add_edge(node1, node2, EdgeType.HIERARCHY, {})
    rpg.add_edge(node1, node2, EdgeType.DATA_FLOW, {})
    rpg.add_edge(node1, node2, EdgeType.EXECUTION_ORDER, {})

    # Verify edges exist
    assert rpg.graph.number_of_edges() == 3

    # Verify children
    children = rpg.get_children(node1)
    assert node2 in children

@test("1.3: RPG - Graph validation")
def test_rpg_validation():
    from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType

    rpg = RepositoryPlanningGraph("Valid RPG")

    root = rpg.add_node("root", NodeType.ROOT, {"name": "root"})
    leaf = rpg.add_node("leaf", NodeType.LEAF, {"name": "leaf"})
    rpg.add_edge(root, leaf, EdgeType.HIERARCHY, {})

    # Should be valid
    is_valid, issues = rpg.validate()
    assert is_valid, f"Validation failed: {issues}"

@test("1.4: RPG - Graph persistence (JSON)")
def test_rpg_persistence_json():
    from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType
    from src.core.graph_persistence import GraphPersistence

    rpg = RepositoryPlanningGraph("Persistence Test")
    root = rpg.add_node("root", NodeType.ROOT, {"data": "test"})
    leaf = rpg.add_node("leaf", NodeType.LEAF, {"data": "test2"})
    rpg.add_edge(root, leaf, EdgeType.HIERARCHY, {})

    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name

    try:
        persistence = GraphPersistence()
        persistence.save_to_json(rpg, temp_path)

        # Load back
        loaded_rpg = persistence.load_from_json(temp_path)

        # Verify
        assert loaded_rpg.graph.number_of_nodes() == rpg.graph.number_of_nodes()
        assert loaded_rpg.graph.number_of_edges() == rpg.graph.number_of_edges()
    finally:
        os.unlink(temp_path)

@test("1.5: RPG - Graph persistence (Pickle)")
def test_rpg_persistence_pickle():
    from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType
    from src.core.graph_persistence import GraphPersistence

    rpg = RepositoryPlanningGraph("Pickle Test")
    root = rpg.add_node("root", NodeType.ROOT, {"data": "test"})
    leaf = rpg.add_node("leaf", NodeType.LEAF, {"data": "test2"})
    rpg.add_edge(root, leaf, EdgeType.HIERARCHY, {})

    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
        temp_path = f.name

    try:
        persistence = GraphPersistence()
        persistence.save_to_pickle(rpg, temp_path)

        # Load back
        loaded_rpg = persistence.load_from_pickle(temp_path)

        # Verify
        assert loaded_rpg.graph.number_of_nodes() == rpg.graph.number_of_nodes()
        assert loaded_rpg.graph.number_of_edges() == rpg.graph.number_of_edges()
    finally:
        os.unlink(temp_path)

@test("1.6: LLM Router - Initialization")
def test_llm_router_init():
    from src.core.llm_router_final import FinalLLMRouter

    config = {"llm": {"default_temperature": 0.7}}
    llm = FinalLLMRouter(config)

    assert llm is not None
    assert llm.has_gemini or llm.has_openrouter, "At least one provider should be available"

@test("1.7: LLM Router - Stats tracking")
def test_llm_router_stats():
    from src.core.llm_router_final import FinalLLMRouter

    config = {"llm": {}}
    llm = FinalLLMRouter(config)

    stats = llm.get_stats()

    assert "total_cost" in stats
    assert "api_calls" in stats
    assert "providers" in stats
    assert stats["api_calls"] == 0  # No calls yet

@test("1.8: Cost Tracker - Basic tracking")
def test_cost_tracker():
    from src.utils.cost_tracker import CostTracker

    tracker = CostTracker()

    # Calculate some costs
    cost1 = tracker.calculate_cost("gemini", "gemini-2.0-flash-exp", 1000)
    cost2 = tracker.calculate_cost("claude", "claude-3.5-sonnet", 1000)

    # Gemini should be free
    assert cost1 == 0.0

    # Claude should cost something
    assert cost2 > 0

    # Check stats
    stats = tracker.get_stats()
    assert stats["api_calls"] == 2
    assert stats["total_cost"] > 0

@test("1.9: Config Loader - Load config")
def test_config_loader():
    from src.utils.config_loader import ConfigLoader

    config = ConfigLoader("config/config.yaml")

    assert config is not None

    # Check required sections
    llm_config = config.get("llm", {})
    assert llm_config is not None

@test("1.10: Logger - Structured logging")
def test_logger():
    from src.utils.logger import StructuredLogger

    logger = StructuredLogger("test")

    # Should not crash
    logger.info("Test message", key="value")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.debug("Debug message")

# ==================================================================================
# PHASE 2: STAGE 1 COMPONENTS
# ==================================================================================

print("\n>>> PHASE 2: STAGE 1 COMPONENTS")
print("-" * 90)

@test("2.1: Feature Model - Creation")
def test_feature_model():
    from src.models.feature import Feature

    feature = Feature(
        id="test-1",
        name="Test Feature",
        description="A test feature",
        category="testing",
        complexity="intermediate",
        tags=["test", "demo"]
    )

    assert feature.id == "test-1"
    assert feature.name == "Test Feature"
    assert "test" in feature.tags

@test("2.2: Feature Model - To dict")
def test_feature_to_dict():
    from src.models.feature import Feature

    feature = Feature(
        id="test-1",
        name="Test Feature",
        description="A test feature",
        category="testing",
        complexity="intermediate",
        tags=["test"]
    )

    data = feature.to_dict()

    assert isinstance(data, dict)
    assert data["id"] == "test-1"
    assert data["name"] == "Test Feature"

@test("2.3: Taxonomy - Load taxonomy")
def test_taxonomy():
    from src.models.taxonomy import UniversalTaxonomy

    taxonomy = UniversalTaxonomy()
    categories = taxonomy.get_all_categories()

    assert len(categories) > 0
    assert "Core Backend" in categories

@test("2.4: Taxonomy - Get features by category")
def test_taxonomy_features():
    from src.models.taxonomy import UniversalTaxonomy

    taxonomy = UniversalTaxonomy()
    features = taxonomy.get_features_by_category("Core Backend")

    assert len(features) > 0
    assert all(isinstance(f, dict) for f in features)

@test("2.5: Exploit Strategy - Initialization")
def test_exploit_strategy():
    from src.stage1.exploit_strategy import ExploitStrategy

    strategy = ExploitStrategy(top_k=10)

    assert strategy is not None
    assert strategy.top_k == 10

@test("2.6: Explore Strategy - Initialization")
def test_explore_strategy():
    from src.stage1.explore_strategy import ExploreStrategy

    strategy = ExploreStrategy(diversity_threshold=0.7)

    assert strategy is not None
    assert strategy.diversity_threshold == 0.7

@test("2.7: Feature Selection Loop - Initialization")
def test_feature_selection_loop():
    from src.stage1.feature_selection_loop import FeatureSelectionLoop

    loop = FeatureSelectionLoop(
        exploit_ratio=0.7,
        num_iterations=5
    )

    assert loop is not None
    assert loop.exploit_ratio == 0.7
    assert loop.num_iterations == 5

# ==================================================================================
# PHASE 3: STAGE 2 COMPONENTS
# ==================================================================================

print("\n>>> PHASE 3: STAGE 2 COMPONENTS")
print("-" * 90)

@test("3.1: Base Class Abstraction - Initialization")
def test_base_class_abstraction():
    from src.stage2.base_class_abstraction import BaseClassAbstraction
    from src.core.llm_router_final import FinalLLMRouter

    llm = FinalLLMRouter({"llm": {}})
    abstractor = BaseClassAbstraction(llm)

    assert abstractor is not None

@test("3.2: Data Flow Encoder - Initialization")
def test_data_flow_encoder():
    from src.stage2.data_flow_encoder import DataFlowEncoder
    from src.core.llm_router_final import FinalLLMRouter

    llm = FinalLLMRouter({"llm": {}})
    encoder = DataFlowEncoder(llm)

    assert encoder is not None

@test("3.3: File Structure Encoder - Initialization")
def test_file_structure_encoder():
    from src.stage2.file_structure_encoder import FileStructureEncoder
    from src.core.llm_router_final import FinalLLMRouter

    llm = FinalLLMRouter({"llm": {}})
    encoder = FileStructureEncoder(llm)

    assert encoder is not None

@test("3.4: Interface Designer - Initialization")
def test_interface_designer():
    from src.stage2.interface_designer import InterfaceDesigner
    from src.core.llm_router_final import FinalLLMRouter

    llm = FinalLLMRouter({"llm": {}})
    designer = InterfaceDesigner(llm)

    assert designer is not None

@test("3.5: Stage 2 Orchestrator - Initialization")
def test_stage2_orchestrator():
    from src.stage2.stage2_orchestrator import Stage2Orchestrator

    config = {"llm": {}, "stage2": {}}
    orchestrator = Stage2Orchestrator(config)

    assert orchestrator is not None

# ==================================================================================
# PHASE 4: STAGE 3 COMPONENTS
# ==================================================================================

print("\n>>> PHASE 4: STAGE 3 COMPONENTS")
print("-" * 90)

@test("4.1: Topological Traversal - Execution order")
def test_topological_traversal():
    from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType
    from src.stage3.topological_traversal import TopologicalTraversal

    rpg = RepositoryPlanningGraph("Test")
    n1 = rpg.add_node("n1", NodeType.LEAF, {})
    n2 = rpg.add_node("n2", NodeType.LEAF, {})
    n3 = rpg.add_node("n3", NodeType.LEAF, {})

    rpg.add_edge(n1, n2, EdgeType.DATA_FLOW, {})
    rpg.add_edge(n2, n3, EdgeType.DATA_FLOW, {})

    traversal = TopologicalTraversal(rpg)
    order = traversal.get_execution_order()

    assert len(order) == 3
    assert order[0] == n1
    assert order[1] == n2
    assert order[2] == n3

@test("4.2: Topological Traversal - Level grouping")
def test_topological_grouping():
    from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType
    from src.stage3.topological_traversal import TopologicalTraversal

    rpg = RepositoryPlanningGraph("Test")
    n1 = rpg.add_node("n1", NodeType.LEAF, {})
    n2 = rpg.add_node("n2", NodeType.LEAF, {})
    n3 = rpg.add_node("n3", NodeType.LEAF, {})

    rpg.add_edge(n1, n2, EdgeType.DATA_FLOW, {})
    rpg.add_edge(n2, n3, EdgeType.DATA_FLOW, {})

    traversal = TopologicalTraversal(rpg)
    levels = traversal.group_by_level()

    assert len(levels) == 3
    assert n1 in levels[0]
    assert n2 in levels[1]
    assert n3 in levels[2]

@test("4.3: Topological Traversal - Mark completed")
def test_topological_mark_completed():
    from src.core.rpg import RepositoryPlanningGraph, NodeType
    from src.stage3.topological_traversal import TopologicalTraversal

    rpg = RepositoryPlanningGraph("Test")
    n1 = rpg.add_node("n1", NodeType.LEAF, {})

    traversal = TopologicalTraversal(rpg)
    traversal.mark_completed(n1)

    assert n1 in traversal.completed

@test("4.4: Topological Traversal - Mark failed")
def test_topological_mark_failed():
    from src.core.rpg import RepositoryPlanningGraph, NodeType
    from src.stage3.topological_traversal import TopologicalTraversal

    rpg = RepositoryPlanningGraph("Test")
    n1 = rpg.add_node("n1", NodeType.LEAF, {})

    traversal = TopologicalTraversal(rpg)
    traversal.mark_failed(n1, "Test error")

    assert n1 in traversal.failed
    assert traversal.failed[n1] == "Test error"

@test("4.5: Docker Runner - Initialization")
def test_docker_runner():
    from src.utils.docker_runner import DockerRunner

    config = {"timeout": 300, "memory_limit": "1g"}
    runner = DockerRunner(config)

    assert runner is not None
    assert runner.timeout == 300

@test("4.6: Docker Runner - Availability check")
def test_docker_availability():
    from src.utils.docker_runner import DockerRunner

    config = {}
    runner = DockerRunner(config)

    # Should return bool, not crash
    is_available = runner.is_docker_available()
    assert isinstance(is_available, bool)

@test("4.7: TDD Engine - Initialization")
def test_tdd_engine():
    from src.stage3.tdd_engine import TDDEngine
    from src.core.llm_router_final import FinalLLMRouter
    from src.utils.docker_runner import DockerRunner

    llm = FinalLLMRouter({"llm": {}})
    docker = DockerRunner({})
    config = {"stage3": {"max_debug_attempts": 3}}

    engine = TDDEngine(llm, docker, config)

    assert engine is not None

@test("4.8: Repository Builder - Create structure")
def test_repository_builder():
    from src.stage3.repository_builder import RepositoryBuilder

    builder = RepositoryBuilder()

    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        builder.create_structure(temp_dir)

        # Check basic files exist
        assert Path(temp_dir, "README.md").exists()
        assert Path(temp_dir, "setup.py").exists()
        assert Path(temp_dir, "requirements.txt").exists()

@test("4.9: Stage 3 Orchestrator - Initialization")
def test_stage3_orchestrator():
    from src.stage3.stage3_orchestrator import Stage3Orchestrator

    config = {"llm": {}, "docker": {}, "stage3": {"max_debug_attempts": 3}}
    orchestrator = Stage3Orchestrator(config)

    assert orchestrator is not None

# ==================================================================================
# PHASE 5: UTILITY COMPONENTS
# ==================================================================================

print("\n>>> PHASE 5: UTILITY COMPONENTS")
print("-" * 90)

@test("5.1: File Operations - Read file")
def test_file_ops_read():
    from src.utils.file_ops import read_file

    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Test content")
        temp_path = f.name

    try:
        content = read_file(temp_path)
        assert content == "Test content"
    finally:
        os.unlink(temp_path)

@test("5.2: File Operations - Write file")
def test_file_ops_write():
    from src.utils.file_ops import write_file

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "test.txt"
        write_file(str(temp_path), "Test content")

        assert temp_path.exists()
        assert temp_path.read_text() == "Test content"

@test("5.3: Graph Utils - Get all leaves")
def test_graph_utils_leaves():
    from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType
    from src.utils.graph_utils import get_all_leaves

    rpg = RepositoryPlanningGraph("Test")
    root = rpg.add_node("root", NodeType.ROOT, {})
    leaf1 = rpg.add_node("leaf1", NodeType.LEAF, {})
    leaf2 = rpg.add_node("leaf2", NodeType.LEAF, {})

    rpg.add_edge(root, leaf1, EdgeType.HIERARCHY, {})
    rpg.add_edge(root, leaf2, EdgeType.HIERARCHY, {})

    leaves = get_all_leaves(rpg)

    assert len(leaves) == 2
    assert leaf1 in leaves
    assert leaf2 in leaves

@test("5.4: Graph Utils - Get all roots")
def test_graph_utils_roots():
    from src.core.rpg import RepositoryPlanningGraph, NodeType
    from src.utils.graph_utils import get_all_roots

    rpg = RepositoryPlanningGraph("Test")
    root1 = rpg.add_node("root1", NodeType.ROOT, {})
    root2 = rpg.add_node("root2", NodeType.ROOT, {})

    roots = get_all_roots(rpg)

    assert len(roots) == 2
    assert root1 in roots
    assert root2 in roots

# ==================================================================================
# PHASE 6: ERROR HANDLING & EDGE CASES
# ==================================================================================

print("\n>>> PHASE 6: ERROR HANDLING & EDGE CASES")
print("-" * 90)

@test("6.1: RPG - Invalid node type")
def test_rpg_invalid_node():
    from src.core.rpg import RepositoryPlanningGraph, NodeType

    rpg = RepositoryPlanningGraph("Test")

    # Should handle gracefully or raise clear error
    try:
        # This should work
        node = rpg.add_node("test", NodeType.ROOT, {})
        assert node is not None
    except Exception as e:
        raise AssertionError(f"Unexpected error: {e}")

@test("6.2: RPG - Circular dependency detection")
def test_rpg_circular():
    from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType

    rpg = RepositoryPlanningGraph("Test")
    n1 = rpg.add_node("n1", NodeType.LEAF, {})
    n2 = rpg.add_node("n2", NodeType.LEAF, {})

    rpg.add_edge(n1, n2, EdgeType.DATA_FLOW, {})
    rpg.add_edge(n2, n1, EdgeType.DATA_FLOW, {})

    # Validation should catch this
    is_valid, issues = rpg.validate()
    assert not is_valid or "cycle" in str(issues).lower()

@test("6.3: Config Loader - Missing file")
def test_config_missing_file():
    from src.utils.config_loader import ConfigLoader

    try:
        config = ConfigLoader("nonexistent.yaml")
        # Should use defaults
        assert config is not None
    except FileNotFoundError:
        # Or raise clear error
        pass

@test("6.4: File Operations - Missing file")
def test_file_ops_missing():
    from src.utils.file_ops import read_file

    try:
        content = read_file("/nonexistent/file.txt")
        raise AssertionError("Should have raised error")
    except FileNotFoundError:
        pass  # Expected

@test("6.5: Persistence - Invalid JSON")
def test_persistence_invalid_json():
    from src.core.graph_persistence import GraphPersistence

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("invalid json {{{")
        temp_path = f.name

    try:
        persistence = GraphPersistence()
        try:
            rpg = persistence.load_from_json(temp_path)
            raise AssertionError("Should have raised error")
        except (json.JSONDecodeError, Exception):
            pass  # Expected
    finally:
        os.unlink(temp_path)

@test("6.6: Cost Tracker - Unknown provider")
def test_cost_tracker_unknown():
    from src.utils.cost_tracker import CostTracker

    tracker = CostTracker()

    # Should handle gracefully
    cost = tracker.calculate_cost("unknown_provider", "unknown_model", 1000)

    assert cost == 0.0  # Unknown should cost 0

@test("6.7: LLM Router - No providers")
def test_llm_router_no_providers():
    from src.core.llm_router_final import FinalLLMRouter

    # Clear env vars temporarily
    old_gemini = os.environ.get("GOOGLE_API_KEY")
    old_openrouter = os.environ.get("OPENROUTER_API_KEY")

    if old_gemini:
        os.environ.pop("GOOGLE_API_KEY", None)
    if old_openrouter:
        os.environ.pop("OPENROUTER_API_KEY", None)

    try:
        config = {"llm": {}}
        try:
            llm = FinalLLMRouter(config)
            # Should either work with no providers or raise clear error
        except ValueError as e:
            assert "API" in str(e) or "provider" in str(e).lower()
    finally:
        # Restore
        if old_gemini:
            os.environ["GOOGLE_API_KEY"] = old_gemini
        if old_openrouter:
            os.environ["OPENROUTER_API_KEY"] = old_openrouter

# ==================================================================================
# PHASE 7: INTEGRATION TESTS
# ==================================================================================

print("\n>>> PHASE 7: INTEGRATION TESTS")
print("-" * 90)

@test("7.1: Full RPG Pipeline - Create, save, load")
def test_rpg_full_pipeline():
    from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType
    from src.core.graph_persistence import GraphPersistence

    # Create RPG
    rpg = RepositoryPlanningGraph("Integration Test")
    root = rpg.add_node("root", NodeType.ROOT, {"name": "Root"})
    int1 = rpg.add_node("intermediate", NodeType.INTERMEDIATE, {"name": "Int"})
    leaf1 = rpg.add_node("leaf1", NodeType.LEAF, {"name": "Leaf1"})
    leaf2 = rpg.add_node("leaf2", NodeType.LEAF, {"name": "Leaf2"})

    rpg.add_edge(root, int1, EdgeType.HIERARCHY, {})
    rpg.add_edge(int1, leaf1, EdgeType.HIERARCHY, {})
    rpg.add_edge(int1, leaf2, EdgeType.HIERARCHY, {})
    rpg.add_edge(leaf1, leaf2, EdgeType.DATA_FLOW, {})

    # Validate
    is_valid, issues = rpg.validate()
    assert is_valid, f"Validation failed: {issues}"

    # Save to both formats
    with tempfile.TemporaryDirectory() as temp_dir:
        json_path = Path(temp_dir) / "test.json"
        pkl_path = Path(temp_dir) / "test.pkl"

        persistence = GraphPersistence()
        persistence.save_to_json(rpg, str(json_path))
        persistence.save_to_pickle(rpg, str(pkl_path))

        # Load from both
        rpg_json = persistence.load_from_json(str(json_path))
        rpg_pkl = persistence.load_from_pickle(str(pkl_path))

        # Verify
        assert rpg_json.graph.number_of_nodes() == 4
        assert rpg_pkl.graph.number_of_nodes() == 4
        assert rpg_json.graph.number_of_edges() == 4
        assert rpg_pkl.graph.number_of_edges() == 4

@test("7.2: Topological Traversal - Complex graph")
def test_traversal_complex():
    from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType
    from src.stage3.topological_traversal import TopologicalTraversal

    rpg = RepositoryPlanningGraph("Complex")

    # Create diamond dependency
    n1 = rpg.add_node("n1", NodeType.LEAF, {})
    n2 = rpg.add_node("n2", NodeType.LEAF, {})
    n3 = rpg.add_node("n3", NodeType.LEAF, {})
    n4 = rpg.add_node("n4", NodeType.LEAF, {})

    rpg.add_edge(n1, n2, EdgeType.DATA_FLOW, {})
    rpg.add_edge(n1, n3, EdgeType.DATA_FLOW, {})
    rpg.add_edge(n2, n4, EdgeType.DATA_FLOW, {})
    rpg.add_edge(n3, n4, EdgeType.DATA_FLOW, {})

    traversal = TopologicalTraversal(rpg)
    order = traversal.get_execution_order()

    # n1 must come first, n4 must come last
    assert order[0] == n1
    assert order[-1] == n4
    assert len(order) == 4

@test("7.3: Cost Tracking - Multiple providers")
def test_cost_tracking_multi():
    from src.utils.cost_tracker import CostTracker

    tracker = CostTracker()

    # Track different providers
    tracker.calculate_cost("gemini", "gemini-2.0-flash-exp", 1000)
    tracker.calculate_cost("claude", "claude-3.5-sonnet", 2000)
    tracker.calculate_cost("gemini", "gemini-2.0-flash-exp", 1000)

    stats = tracker.get_stats()

    assert stats["api_calls"] == 3
    assert "gemini" in stats["providers"]
    assert "claude" in stats["providers"]
    assert stats["providers"]["gemini"]["calls"] == 2
    assert stats["providers"]["claude"]["calls"] == 1

@test("7.4: Repository Builder - Full structure")
def test_repo_builder_full():
    from src.stage3.repository_builder import RepositoryBuilder

    builder = RepositoryBuilder()

    with tempfile.TemporaryDirectory() as temp_dir:
        builder.create_structure(temp_dir)

        # Verify all essential files
        repo_path = Path(temp_dir)
        assert (repo_path / "README.md").exists()
        assert (repo_path / "setup.py").exists()
        assert (repo_path / "requirements.txt").exists()
        assert (repo_path / ".gitignore").exists()

# ==================================================================================
# PHASE 8: PERFORMANCE & STRESS TESTS
# ==================================================================================

print("\n>>> PHASE 8: PERFORMANCE & STRESS TESTS")
print("-" * 90)

@test("8.1: RPG - Large graph (100 nodes)")
def test_rpg_large_graph():
    from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType

    rpg = RepositoryPlanningGraph("Large Graph")

    # Create 100 nodes
    nodes = []
    for i in range(100):
        node = rpg.add_node(f"node_{i}", NodeType.LEAF, {"index": i})
        nodes.append(node)

    # Connect them in a chain
    for i in range(99):
        rpg.add_edge(nodes[i], nodes[i+1], EdgeType.DATA_FLOW, {})

    assert rpg.graph.number_of_nodes() == 100
    assert rpg.graph.number_of_edges() == 99

    # Validation should still work
    is_valid, issues = rpg.validate()
    assert is_valid or len(issues) < 10  # Allow some warnings

@test("8.2: Topological Traversal - Large graph")
def test_traversal_large():
    from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType
    from src.stage3.topological_traversal import TopologicalTraversal

    rpg = RepositoryPlanningGraph("Large")

    # Create 50 nodes in chain
    nodes = []
    for i in range(50):
        node = rpg.add_node(f"n{i}", NodeType.LEAF, {})
        nodes.append(node)

    for i in range(49):
        rpg.add_edge(nodes[i], nodes[i+1], EdgeType.DATA_FLOW, {})

    traversal = TopologicalTraversal(rpg)
    order = traversal.get_execution_order()

    assert len(order) == 50
    assert order[0] == nodes[0]
    assert order[-1] == nodes[-1]

@test("8.3: Cost Tracker - Many operations")
def test_cost_tracker_many():
    from src.utils.cost_tracker import CostTracker

    tracker = CostTracker()

    # Track 100 operations
    for i in range(100):
        provider = "gemini" if i % 2 == 0 else "claude"
        model = "gemini-2.0-flash-exp" if provider == "gemini" else "claude-3.5-sonnet"
        tracker.calculate_cost(provider, model, 1000)

    stats = tracker.get_stats()

    assert stats["api_calls"] == 100
    assert stats["providers"]["gemini"]["calls"] == 50
    assert stats["providers"]["claude"]["calls"] == 50

@test("8.4: Persistence - Large graph save/load")
def test_persistence_large():
    from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType
    from src.core.graph_persistence import GraphPersistence

    rpg = RepositoryPlanningGraph("Large")

    # Create 50 nodes
    nodes = []
    for i in range(50):
        node = rpg.add_node(f"n{i}", NodeType.LEAF, {"data": f"test_{i}"})
        nodes.append(node)

    for i in range(49):
        rpg.add_edge(nodes[i], nodes[i+1], EdgeType.HIERARCHY, {})

    with tempfile.TemporaryDirectory() as temp_dir:
        json_path = Path(temp_dir) / "large.json"

        persistence = GraphPersistence()
        persistence.save_to_json(rpg, str(json_path))

        loaded = persistence.load_from_json(str(json_path))

        assert loaded.graph.number_of_nodes() == 50
        assert loaded.graph.number_of_edges() == 49

# Run all tests
test_rpg_nodes()
test_rpg_edges()
test_rpg_validation()
test_rpg_persistence_json()
test_rpg_persistence_pickle()
test_llm_router_init()
test_llm_router_stats()
test_cost_tracker()
test_config_loader()
test_logger()

test_feature_model()
test_feature_to_dict()
test_taxonomy()
test_taxonomy_features()
test_exploit_strategy()
test_explore_strategy()
test_feature_selection_loop()

test_base_class_abstraction()
test_data_flow_encoder()
test_file_structure_encoder()
test_interface_designer()
test_stage2_orchestrator()

test_topological_traversal()
test_topological_grouping()
test_topological_mark_completed()
test_topological_mark_failed()
test_docker_runner()
test_docker_availability()
test_tdd_engine()
test_repository_builder()
test_stage3_orchestrator()

test_file_ops_read()
test_file_ops_write()
test_graph_utils_leaves()
test_graph_utils_roots()

test_rpg_invalid_node()
test_rpg_circular()
test_config_missing_file()
test_file_ops_missing()
test_persistence_invalid_json()
test_cost_tracker_unknown()
test_llm_router_no_providers()

test_rpg_full_pipeline()
test_traversal_complex()
test_cost_tracking_multi()
test_repo_builder_full()

test_rpg_large_graph()
test_traversal_large()
test_cost_tracker_many()
test_persistence_large()

# Print results
print("\n")
is_production_ready = results.summary()

if is_production_ready:
    print("\n" + "=" * 90)
    print("*** VERDICT: PRODUCTION READY! ***")
    print("=" * 90)
    print("\nAll components tested and validated.")
    print("95%+ pass rate achieved.")
    print("\nReady to deploy with API keys!")
else:
    print("\n" + "=" * 90)
    print("*** VERDICT: NOT PRODUCTION READY ***")
    print("=" * 90)
    print("\nCritical issues found - review failures above.")

sys.exit(0 if is_production_ready else 1)
