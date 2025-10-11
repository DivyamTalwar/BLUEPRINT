import os
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

# Test results
results = {
    "total": 0,
    "passed": 0,
    "failed": 0,
    "warnings": 0,
    "errors": [],
    "critical_failures": []
}

def test(name: str, critical: bool = False):
    def decorator(func):
        def wrapper():
            global results
            results["total"] += 1
            try:
                start = time.time()
                func()
                duration = time.time() - start
                results["passed"] += 1
                print(f"[PASS] {name} ({duration:.2f}s)")
                return True
            except AssertionError as e:
                results["failed"] += 1
                error_msg = f"{name}: {str(e)}"
                results["errors"].append(error_msg)
                if critical:
                    results["critical_failures"].append(error_msg)
                print(f"[FAIL] {name}: {str(e)}")
                return False
            except Exception as e:
                results["failed"] += 1
                error_msg = f"{name}: {type(e).__name__}: {str(e)}"
                results["errors"].append(error_msg)
                if critical:
                    results["critical_failures"].append(error_msg)
                print(f"[FAIL] {name}: {type(e).__name__}: {str(e)}")
                return False
        return wrapper
    return decorator


@test("1. Environment file exists", critical=True)
def test_env_exists():
    env_path = Path(".env")
    assert env_path.exists(), ".env file not found - copy .env.example to .env"

@test("2. All API keys configured", critical=True)
def test_api_keys():
    from dotenv import load_dotenv
    load_dotenv()

    required_keys = {
        "GOOGLE_API_KEY": "Gemini API",
        "OPENROUTER_API_KEY": "OpenRouter (Claude)",
        "COHERE_API_KEY": "Cohere Embeddings",
        "PINECONE_API_KEY": "Pinecone Vector DB",
    }

    missing = []
    for key, service in required_keys.items():
        value = os.getenv(key)
        if not value or value.startswith("ADD-YOUR-"):
            missing.append(f"{key} ({service})")

    assert len(missing) == 0, f"Missing API keys: {', '.join(missing)}"

@test("3. Config file valid", critical=True)
def test_config():
    from src.utils.config_loader import ConfigLoader

    config = ConfigLoader("config.yaml")
    all_config = config.get_all()

    required_sections = ["stage1", "stage2", "stage3", "llm", "docker"]
    for section in required_sections:
        assert section in all_config, f"Missing config section: {section}"

@test("4. Pinecone index exists with features", critical=True)
def test_pinecone_features():
    from dotenv import load_dotenv
    load_dotenv()

    try:
        from pinecone import Pinecone

        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        index_name = "blueprint-features"
        existing_indexes = [idx.name for idx in pc.list_indexes()]

        assert index_name in existing_indexes, \
            f"Pinecone index '{index_name}' not found - run: python scripts/generate_feature_tree.py"

        index = pc.Index(index_name)
        stats = index.describe_index_stats()

        assert stats['total_vector_count'] > 0, \
            f"Pinecone index empty - run: python scripts/generate_feature_tree.py"

        print(f"    -> {stats['total_vector_count']} features loaded")

    except ImportError:
        raise AssertionError("Pinecone library not installed - run: pip install pinecone-client")

@test("5. LLM router can initialize", critical=True)
def test_llm_router():
    from src.core.llm_router_final import FinalLLMRouter

    config = {"llm": {"default_temperature": 0.7, "max_tokens": 2000}}
    llm = FinalLLMRouter(config)

    assert llm is not None
    assert llm.has_gemini or llm.has_openrouter, "At least one LLM provider must be available"

@test("6. Feature tree JSON exists", critical=True)
def test_feature_tree_json():
    json_path = Path("data/feature_tree.json")

    if not json_path.exists():
        raise AssertionError(
            "Feature tree JSON not found - run: python scripts/generate_feature_tree.py"
        )

    import json
    with open(json_path) as f:
        data = json.load(f)

    assert "features" in data, "Feature tree JSON missing 'features' key"
    assert len(data["features"]) > 0, "Feature tree JSON has no features"

    print(f"    -> {len(data['features'])} features in JSON")

# Run critical tests
test_env_exists()
test_api_keys()
test_config()
test_pinecone_features()
test_llm_router()
test_feature_tree_json()

print()


# ==================================================================================
# COMPONENT TESTS - VALIDATE ALL COMPONENTS
# ==================================================================================

print(">>> COMPONENT TESTS")
print("-" * 90)

@test("7. User input processor works")
def test_user_input_processor():
    from src.stage1.user_input_processor import UserInputProcessor
    from src.core.llm_router_final import FinalLLMRouter

    llm = FinalLLMRouter({"llm": {}})
    processor = UserInputProcessor(llm)

    assert processor is not None
    assert processor.llm_router is not None

@test("8. Feature selection loop initializes")
def test_feature_selection_loop():
    from src.stage1.feature_selection_loop import FeatureSelectionLoop
    from unittest.mock import Mock

    exploit = Mock()
    explore = Mock()

    loop = FeatureSelectionLoop(exploit, explore, num_iterations=30)

    assert loop is not None
    assert loop.num_iterations == 30

@test("9. Stage 2 orchestrator ready")
def test_stage2_orchestrator():
    from src.stage2.stage2_orchestrator import Stage2Orchestrator

    config = {"llm": {}, "stage2": {"signature_batch_size": 5}}
    orchestrator = Stage2Orchestrator(config)

    assert orchestrator is not None
    assert orchestrator.llm is not None

@test("10. Stage 3 orchestrator ready")
def test_stage3_orchestrator():
    from src.stage3.stage3_orchestrator import Stage3Orchestrator

    config = {"llm": {}, "docker": {"timeout": 300}, "stage3": {"max_debug_attempts": 3}}

    try:
        orchestrator = Stage3Orchestrator(config)
        assert orchestrator is not None
    except Exception as e:
        if "Docker" in str(e):
            results["warnings"] += 1
            print(f"    [WARN] Docker not available: {e}")
        else:
            raise

@test("11. RPG creation and validation")
def test_rpg_complete():
    from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType
    from src.core.graph_operations import GraphOperations

    rpg = RepositoryPlanningGraph("Production Test")

    root = rpg.add_node("root", NodeType.ROOT, "Root", file_path="src/")
    leaf = rpg.add_node("leaf", NodeType.LEAF, "Leaf", signature="def test():")
    rpg.add_edge(root, leaf, EdgeType.HIERARCHY)

    is_valid = GraphOperations.validate(rpg)
    assert is_valid, "RPG validation failed"

@test("12. Graph persistence (JSON + Pickle)")
def test_graph_persistence_full():
    from src.core.rpg import RepositoryPlanningGraph, NodeType
    from src.core.graph_persistence import GraphPersistence
    import tempfile

    rpg = RepositoryPlanningGraph("Test")
    rpg.add_node("test", NodeType.ROOT, "Test")

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "test.json")
        pkl_path = os.path.join(tmpdir, "test.pkl")

        GraphPersistence.save_json(rpg, json_path)
        GraphPersistence.save_pickle(rpg, pkl_path)

        loaded_json = GraphPersistence.load_json(json_path)
        loaded_pkl = GraphPersistence.load_pickle(pkl_path)

        assert loaded_json.graph.number_of_nodes() == 1
        assert loaded_pkl.graph.number_of_nodes() == 1

@test("13. Topological traversal with grouping")
def test_topological_full():
    from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType
    from src.stage3.topological_traversal import TopologicalTraversal

    rpg = RepositoryPlanningGraph("Test")

    n1 = rpg.add_node("n1", NodeType.LEAF, "N1")
    n2 = rpg.add_node("n2", NodeType.LEAF, "N2")
    n3 = rpg.add_node("n3", NodeType.LEAF, "N3")

    rpg.add_edge(n1, n2, EdgeType.DATA_FLOW)
    rpg.add_edge(n2, n3, EdgeType.DATA_FLOW)

    traversal = TopologicalTraversal(rpg)
    order = traversal.get_execution_order()
    levels = traversal.group_by_level()

    assert len(order) == 3
    assert len(levels) == 3

# Run component tests
test_user_input_processor()
test_feature_selection_loop()
test_stage2_orchestrator()
test_stage3_orchestrator()
test_rpg_complete()
test_graph_persistence_full()
test_topological_full()

print()


# ==================================================================================
# INTEGRATION TESTS
# ==================================================================================

print(">>> INTEGRATION TESTS")
print("-" * 90)

@test("14. Main pipeline can initialize")
def test_main_pipeline():
    from main import BLUEPRINTPipeline

    pipeline = BLUEPRINTPipeline(config_path="config.yaml")

    assert pipeline is not None
    assert pipeline.config is not None
    assert pipeline.llm is not None
    assert pipeline.cost_tracker is not None

@test("15. Prerequisites check passes")
def test_prerequisites():
    from main import BLUEPRINTPipeline

    pipeline = BLUEPRINTPipeline()
    result = pipeline.check_prerequisites()

    assert result, "Prerequisites check failed - see errors above"

# Run integration tests
test_main_pipeline()
test_prerequisites()

print()


# ==================================================================================
# DOCKER TESTS (NON-CRITICAL)
# ==================================================================================

print(">>> DOCKER TESTS (Optional)")
print("-" * 90)

@test("16. Docker available")
def test_docker():
    try:
        import docker
        client = docker.from_env()
        client.ping()
        print("    -> Docker available")
    except Exception as e:
        results["warnings"] += 1
        raise AssertionError(f"Docker not available (non-critical): {e}")

test_docker()

print()


# ==================================================================================
# FINAL RESULTS
# ==================================================================================

print("=" * 90)
print("*** PRODUCTION READINESS RESULTS ***")
print("=" * 90)
print()

print(f"Total Tests: {results['total']}")
print(f"Passed: {results['passed']}")
print(f"Failed: {results['failed']}")
print(f"Warnings: {results['warnings']}")
print()

success_rate = (results['passed'] / results['total'] * 100) if results['total'] > 0 else 0

print(f"Success Rate: {success_rate:.1f}%")
print()

if results["critical_failures"]:
    print("=" * 90)
    print("[CRITICAL] PRODUCTION BLOCKERS:")
    print("=" * 90)
    for error in results["critical_failures"]:
        print(f"  [X] {error}")
    print()
    print("VERDICT: NOT READY FOR PRODUCTION")
    print()
    print("Fix critical issues above before deploying!")
    print("=" * 90)
    sys.exit(1)

elif results["failed"] > 0:
    print("[ERRORS] Non-Critical Failures:")
    for error in results["errors"]:
        if error not in results["critical_failures"]:
            print(f"  [X] {error}")
    print()

if results["warnings"] > 0:
    print("[WARNINGS] Optional Features:")
    print("  Docker not available - TDD functionality limited")
    print("  This is OK for testing, recommended for production")
    print()

if len(results["critical_failures"]) == 0 and success_rate >= 85:
    print("=" * 90)
    print("*** PRODUCTION READY! ***")
    print("=" * 90)
    print()
    print("All critical tests passed!")
    print("BLUEPRINT is ready for production deployment.")
    print()
    print("To generate code:")
    print('  python main.py "Build a REST API for blog management"')
    print()
    print("=" * 90)
    sys.exit(0)
else:
    print("=" * 90)
    print("NOT READY - Review failures above")
    print("=" * 90)
    sys.exit(1)
