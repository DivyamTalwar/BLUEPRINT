import os
import sys
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Test results
test_results = {"total": 0, "passed": 0, "failed": 0, "errors": []}

def test(name: str):
    """Test decorator"""
    def decorator(func):
        def wrapper():
            global test_results
            test_results["total"] += 1
            try:
                start = time.time()
                func()
                duration = time.time() - start
                test_results["passed"] += 1
                print(f"[PASS] {name} ({duration:.2f}s)")
                return True
            except AssertionError as e:
                test_results["failed"] += 1
                test_results["errors"].append(f"{name}: {str(e)}")
                print(f"[FAIL] {name}: {str(e)}")
                return False
            except Exception as e:
                test_results["failed"] += 1
                test_results["errors"].append(f"{name}: {str(e)}")
                print(f"[FAIL] {name}: {type(e).__name__}: {str(e)}")
                return False
        return wrapper
    return decorator


print("=" * 90)
print("*** BLUEPRINT PIPELINE INTEGRATION TEST ***")
print("=" * 90)
print()


# =================================================================================
# TEST 1: RPG CORE FUNCTIONALITY
# =================================================================================

print(">>> TEST GROUP 1: RPG Core & Graph Operations")
print("-" * 90)

@test("1.1: Create RPG with multiple node types")
def test_rpg_creation():
    from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType

    rpg = RepositoryPlanningGraph("Test Microservice")

    # Create structure
    root = rpg.add_node("service", NodeType.ROOT, "Main service", file_path="src/")
    api_module = rpg.add_node("api/", NodeType.INTERMEDIATE, "API module", file_path="src/api/")
    handler_file = rpg.add_node("handlers.py", NodeType.INTERMEDIATE, "Request handlers", file_path="src/api/handlers.py")

    get_user = rpg.add_node("get_user", NodeType.LEAF, "Get user endpoint", signature="async def get_user(user_id: int):")
    create_user = rpg.add_node("create_user", NodeType.LEAF, "Create user endpoint", signature="async def create_user(data: dict):")

    # Build hierarchy
    rpg.add_edge(root, api_module, EdgeType.HIERARCHY)
    rpg.add_edge(api_module, handler_file, EdgeType.HIERARCHY)
    rpg.add_edge(handler_file, get_user, EdgeType.HIERARCHY)
    rpg.add_edge(handler_file, create_user, EdgeType.HIERARCHY)

    # Add data flow
    rpg.add_edge(create_user, get_user, EdgeType.DATA_FLOW, data_type="User")

    assert rpg.graph.number_of_nodes() == 5
    assert rpg.graph.number_of_edges() == 5
    print(f"  -> Created RPG: {rpg.graph.number_of_nodes()} nodes, {rpg.graph.number_of_edges()} edges")

@test("1.2: Graph validation - valid graph")
def test_graph_validation():
    from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType
    from src.core.graph_operations import GraphOperations

    rpg = RepositoryPlanningGraph("Valid Graph")
    n1 = rpg.add_node("n1", NodeType.ROOT, "Root")
    n2 = rpg.add_node("n2", NodeType.LEAF, "Leaf")
    rpg.add_edge(n1, n2, EdgeType.HIERARCHY)

    is_valid = GraphOperations.validate(rpg)
    assert is_valid, "Valid graph marked as invalid"
    print(f"  -> Graph validation passed")

@test("1.3: Cycle detection")
def test_cycle_detection():
    from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType
    from src.core.graph_operations import GraphOperations

    # Create graph with cycle
    rpg = RepositoryPlanningGraph("Cyclic Graph")
    a = rpg.add_node("a", NodeType.LEAF, "Node A")
    b = rpg.add_node("b", NodeType.LEAF, "Node B")
    c = rpg.add_node("c", NodeType.LEAF, "Node C")

    rpg.add_edge(a, b, EdgeType.DATA_FLOW)
    rpg.add_edge(b, c, EdgeType.DATA_FLOW)
    rpg.add_edge(c, a, EdgeType.DATA_FLOW)

    has_cycles = GraphOperations.has_cycles(rpg)
    assert has_cycles, "Cycle not detected"
    print(f"  -> Cycle correctly detected")

@test("1.4: Graph persistence - JSON")
def test_graph_persistence_json():
    from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType
    from src.core.graph_persistence import GraphPersistence
    import tempfile

    rpg = RepositoryPlanningGraph("Persistence Test")
    root = rpg.add_node("root", NodeType.ROOT, "Root node", file_path="src/")
    leaf = rpg.add_node("leaf", NodeType.LEAF, "Leaf node", signature="def func():")
    rpg.add_edge(root, leaf, EdgeType.HIERARCHY)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.json")
        GraphPersistence.save_json(rpg, path)

        loaded = GraphPersistence.load_json(path)

        assert loaded.repository_goal == rpg.repository_goal
        assert loaded.graph.number_of_nodes() == rpg.graph.number_of_nodes()
        assert loaded.graph.number_of_edges() == rpg.graph.number_of_edges()

    print(f"  -> JSON persistence verified")

@test("1.5: Graph persistence - Pickle")
def test_graph_persistence_pickle():
    from src.core.rpg import RepositoryPlanningGraph, NodeType
    from src.core.graph_persistence import GraphPersistence
    import tempfile

    rpg = RepositoryPlanningGraph("Pickle Test")
    rpg.add_node("test", NodeType.ROOT, "Test")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.pkl")
        GraphPersistence.save_pickle(rpg, path)

        loaded = GraphPersistence.load_pickle(path)

        assert loaded.graph.number_of_nodes() == 1

    print(f"  -> Pickle persistence verified")

# Run Group 1 tests
test_rpg_creation()
test_graph_validation()
test_cycle_detection()
test_graph_persistence_json()
test_graph_persistence_pickle()

print()


# =================================================================================
# TEST 2: TOPOLOGICAL ORDERING
# =================================================================================

print(">>> TEST GROUP 2: Execution Order & Traversal")
print("-" * 90)

@test("2.1: Topological sort - linear dependencies")
def test_topological_linear():
    from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType
    from src.stage3.topological_traversal import TopologicalTraversal

    rpg = RepositoryPlanningGraph("Linear Pipeline")

    step1 = rpg.add_node("extract", NodeType.LEAF, "Extract data")
    step2 = rpg.add_node("transform", NodeType.LEAF, "Transform data")
    step3 = rpg.add_node("load", NodeType.LEAF, "Load data")

    rpg.add_edge(step1, step2, EdgeType.DATA_FLOW)
    rpg.add_edge(step2, step3, EdgeType.DATA_FLOW)

    traversal = TopologicalTraversal(rpg)
    order = traversal.get_execution_order()

    assert len(order) == 3
    assert order.index(step1) < order.index(step2) < order.index(step3)

    print(f"  -> Linear ordering correct: {order}")

@test("2.2: Topological sort - complex DAG")
def test_topological_complex():
    from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType
    from src.stage3.topological_traversal import TopologicalTraversal

    rpg = RepositoryPlanningGraph("Complex DAG")

    # Diamond pattern
    root = rpg.add_node("root", NodeType.LEAF, "Root")
    left = rpg.add_node("left", NodeType.LEAF, "Left branch")
    right = rpg.add_node("right", NodeType.LEAF, "Right branch")
    merge = rpg.add_node("merge", NodeType.LEAF, "Merge")

    rpg.add_edge(root, left, EdgeType.DATA_FLOW)
    rpg.add_edge(root, right, EdgeType.DATA_FLOW)
    rpg.add_edge(left, merge, EdgeType.DATA_FLOW)
    rpg.add_edge(right, merge, EdgeType.DATA_FLOW)

    traversal = TopologicalTraversal(rpg)
    order = traversal.get_execution_order()

    assert len(order) == 4
    # Root must come first
    assert order[0] == root
    # Merge must come last
    assert order[3] == merge
    # Left and right can be in any order but after root
    assert order.index(left) > order.index(root)
    assert order.index(right) > order.index(root)

    print(f"  -> Complex DAG ordering correct")

@test("2.3: Grouping by level")
def test_grouping():
    from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType
    from src.stage3.topological_traversal import TopologicalTraversal

    rpg = RepositoryPlanningGraph("Leveled Graph")

    l1 = rpg.add_node("level1", NodeType.LEAF, "Level 1")
    l2a = rpg.add_node("level2a", NodeType.LEAF, "Level 2A")
    l2b = rpg.add_node("level2b", NodeType.LEAF, "Level 2B")
    l3 = rpg.add_node("level3", NodeType.LEAF, "Level 3")

    rpg.add_edge(l1, l2a, EdgeType.DATA_FLOW)
    rpg.add_edge(l1, l2b, EdgeType.DATA_FLOW)
    rpg.add_edge(l2a, l3, EdgeType.DATA_FLOW)
    rpg.add_edge(l2b, l3, EdgeType.DATA_FLOW)

    traversal = TopologicalTraversal(rpg)
    grouped = traversal.group_by_level()

    assert len(grouped) == 3
    assert len(grouped[0]) == 1  # l1
    assert len(grouped[1]) == 2  # l2a, l2b
    assert len(grouped[2]) == 1  # l3

    print(f"  -> Grouping by level correct: {len(grouped)} levels")

# Run Group 2 tests
test_topological_linear()
test_topological_complex()
test_grouping()

print()


# =================================================================================
# TEST 3: END-TO-END INTEGRATION
# =================================================================================

print(">>> TEST GROUP 3: End-to-End Integration (No API Calls)")
print("-" * 90)

@test("3.1: Complete CLI tool pipeline")
def test_e2e_cli():
    from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType
    from src.core.graph_operations import GraphOperations
    from src.stage3.topological_traversal import TopologicalTraversal

    # Simulate Stage 1 output (normally from LLM)
    requirements = ["Parse arguments", "Execute commands", "Display output"]

    # Stage 2: Build RPG
    rpg = RepositoryPlanningGraph("File Organization CLI")

    cli_root = rpg.add_node("file_organizer", NodeType.ROOT, "CLI Tool", file_path="src/")
    main = rpg.add_node("main.py", NodeType.INTERMEDIATE, "Main entry", file_path="src/main.py")
    utils = rpg.add_node("utils.py", NodeType.INTERMEDIATE, "Utilities", file_path="src/utils.py")

    parse_args = rpg.add_node("parse_args", NodeType.LEAF, "Argument parser", signature="def parse_args():")
    organize = rpg.add_node("organize_files", NodeType.LEAF, "File organizer", signature="def organize_files(path, mode):")

    rpg.add_edge(cli_root, main, EdgeType.HIERARCHY)
    rpg.add_edge(cli_root, utils, EdgeType.HIERARCHY)
    rpg.add_edge(main, parse_args, EdgeType.HIERARCHY)
    rpg.add_edge(utils, organize, EdgeType.HIERARCHY)
    rpg.add_edge(parse_args, organize, EdgeType.DATA_FLOW, data_type="Args")

    # Validate
    assert GraphOperations.validate(rpg)

    # Stage 3: Get execution order
    traversal = TopologicalTraversal(rpg)
    order = traversal.get_execution_order()

    assert len(order) > 0
    assert order.index(parse_args) < order.index(organize)

    print(f"  -> CLI pipeline: {rpg.graph.number_of_nodes()} nodes, execution order computed")

@test("3.2: Complete Web API pipeline")
def test_e2e_web_api():
    from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType
    from src.core.graph_persistence import GraphPersistence
    import tempfile

    # Build Web API structure
    rpg = RepositoryPlanningGraph("Blog API")

    api = rpg.add_node("blog_api", NodeType.ROOT, "Blog API", file_path="src/")
    routes = rpg.add_node("routes/", NodeType.INTERMEDIATE, "API routes", file_path="src/routes/")
    models = rpg.add_node("models/", NodeType.INTERMEDIATE, "Data models", file_path="src/models/")

    posts_route = rpg.add_node("posts.py", NodeType.INTERMEDIATE, "Posts routes", file_path="src/routes/posts.py")
    users_route = rpg.add_node("users.py", NodeType.INTERMEDIATE, "Users routes", file_path="src/routes/users.py")

    get_posts = rpg.add_node("get_posts", NodeType.LEAF, "Get posts", signature="async def get_posts():")
    create_post = rpg.add_node("create_post", NodeType.LEAF, "Create post", signature="async def create_post(data):")

    # Build graph
    rpg.add_edge(api, routes, EdgeType.HIERARCHY)
    rpg.add_edge(api, models, EdgeType.HIERARCHY)
    rpg.add_edge(routes, posts_route, EdgeType.HIERARCHY)
    rpg.add_edge(routes, users_route, EdgeType.HIERARCHY)
    rpg.add_edge(posts_route, get_posts, EdgeType.HIERARCHY)
    rpg.add_edge(posts_route, create_post, EdgeType.HIERARCHY)

    assert rpg.graph.number_of_nodes() == 7

    # Test persistence
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "api_rpg.json")
        GraphPersistence.save_json(rpg, path)
        loaded = GraphPersistence.load_json(path)
        assert loaded.graph.number_of_nodes() == 7

    print(f"  -> Web API pipeline: {rpg.graph.number_of_nodes()} nodes, persisted & loaded")

@test("3.3: Complete ML library pipeline")
def test_e2e_ml():
    from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType
    from src.stage3.topological_traversal import TopologicalTraversal

    rpg = RepositoryPlanningGraph("Classification Library")

    lib = rpg.add_node("ml_lib", NodeType.ROOT, "ML Library", file_path="src/")
    models_pkg = rpg.add_node("models/", NodeType.INTERMEDIATE, "Models", file_path="src/models/")
    classifier = rpg.add_node("classifier.py", NodeType.INTERMEDIATE, "Classifier", file_path="src/models/classifier.py")

    fit = rpg.add_node("fit", NodeType.LEAF, "Fit model", signature="def fit(self, X, y):")
    predict = rpg.add_node("predict", NodeType.LEAF, "Predict", signature="def predict(self, X):")
    score = rpg.add_node("score", NodeType.LEAF, "Score", signature="def score(self, X, y):")

    rpg.add_edge(lib, models_pkg, EdgeType.HIERARCHY)
    rpg.add_edge(models_pkg, classifier, EdgeType.HIERARCHY)
    rpg.add_edge(classifier, fit, EdgeType.HIERARCHY)
    rpg.add_edge(classifier, predict, EdgeType.HIERARCHY)
    rpg.add_edge(classifier, score, EdgeType.HIERARCHY)

    # Data flow: must fit before predict/score
    rpg.add_edge(fit, predict, EdgeType.DATA_FLOW, data_type="trained_model")
    rpg.add_edge(fit, score, EdgeType.DATA_FLOW, data_type="trained_model")

    # Get execution order
    traversal = TopologicalTraversal(rpg)
    order = traversal.get_execution_order()

    # Fit must come before predict and score
    assert order.index(fit) < order.index(predict)
    assert order.index(fit) < order.index(score)

    print(f"  -> ML library pipeline: dependencies validated, execution order correct")

@test("3.4: ETL data pipeline")
def test_e2e_etl():
    from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType
    from src.core.graph_operations import GraphOperations

    rpg = RepositoryPlanningGraph("ETL Pipeline")

    pipeline = rpg.add_node("etl", NodeType.ROOT, "ETL Pipeline", file_path="src/")
    extract_mod = rpg.add_node("extract.py", NodeType.INTERMEDIATE, "Extract", file_path="src/extract.py")
    transform_mod = rpg.add_node("transform.py", NodeType.INTERMEDIATE, "Transform", file_path="src/transform.py")
    load_mod = rpg.add_node("load.py", NodeType.INTERMEDIATE, "Load", file_path="src/load.py")

    extract_fn = rpg.add_node("extract_data", NodeType.LEAF, "Extract", signature="def extract_data(source):")
    clean_fn = rpg.add_node("clean_data", NodeType.LEAF, "Clean", signature="def clean_data(data):")
    transform_fn = rpg.add_node("transform_data", NodeType.LEAF, "Transform", signature="def transform_data(data):")
    load_fn = rpg.add_node("load_data", NodeType.LEAF, "Load", signature="def load_data(data, dest):")

    # Hierarchy
    rpg.add_edge(pipeline, extract_mod, EdgeType.HIERARCHY)
    rpg.add_edge(pipeline, transform_mod, EdgeType.HIERARCHY)
    rpg.add_edge(pipeline, load_mod, EdgeType.HIERARCHY)
    rpg.add_edge(extract_mod, extract_fn, EdgeType.HIERARCHY)
    rpg.add_edge(transform_mod, clean_fn, EdgeType.HIERARCHY)
    rpg.add_edge(transform_mod, transform_fn, EdgeType.HIERARCHY)
    rpg.add_edge(load_mod, load_fn, EdgeType.HIERARCHY)

    # Data flow: extract -> clean -> transform -> load
    rpg.add_edge(extract_fn, clean_fn, EdgeType.DATA_FLOW, data_type="raw_data")
    rpg.add_edge(clean_fn, transform_fn, EdgeType.DATA_FLOW, data_type="clean_data")
    rpg.add_edge(transform_fn, load_fn, EdgeType.DATA_FLOW, data_type="processed_data")

    # Must not have cycles
    has_cycles = GraphOperations.has_cycles(rpg)
    assert not has_cycles, "ETL pipeline has cycles!"

    print(f"  -> ETL pipeline: {rpg.graph.number_of_nodes()} nodes, no cycles, valid")

@test("3.5: Microservice architecture")
def test_e2e_microservice():
    from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType
    from src.core.graph_persistence import GraphPersistence
    from src.stage3.topological_traversal import TopologicalTraversal
    import tempfile

    rpg = RepositoryPlanningGraph("Notification Microservice")

    service = rpg.add_node("notification_svc", NodeType.ROOT, "Notification Service", file_path="src/")
    handlers = rpg.add_node("handlers/", NodeType.INTERMEDIATE, "Handlers", file_path="src/handlers/")
    models = rpg.add_node("models/", NodeType.INTERMEDIATE, "Models", file_path="src/models/")

    email_handler = rpg.add_node("email.py", NodeType.INTERMEDIATE, "Email handler", file_path="src/handlers/email.py")
    sms_handler = rpg.add_node("sms.py", NodeType.INTERMEDIATE, "SMS handler", file_path="src/handlers/sms.py")

    send_email = rpg.add_node("send_email", NodeType.LEAF, "Send email", signature="async def send_email(to, msg):")
    send_sms = rpg.add_node("send_sms", NodeType.LEAF, "Send SMS", signature="async def send_sms(phone, msg):")

    # Build structure
    rpg.add_edge(service, handlers, EdgeType.HIERARCHY)
    rpg.add_edge(service, models, EdgeType.HIERARCHY)
    rpg.add_edge(handlers, email_handler, EdgeType.HIERARCHY)
    rpg.add_edge(handlers, sms_handler, EdgeType.HIERARCHY)
    rpg.add_edge(email_handler, send_email, EdgeType.HIERARCHY)
    rpg.add_edge(sms_handler, send_sms, EdgeType.HIERARCHY)

    # Verify structure
    assert rpg.graph.number_of_nodes() == 7
    assert rpg.graph.number_of_edges() == 6

    # Test full pipeline
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        json_path = os.path.join(tmpdir, "microservice.json")
        pkl_path = os.path.join(tmpdir, "microservice.pkl")

        GraphPersistence.save_json(rpg, json_path)
        GraphPersistence.save_pickle(rpg, pkl_path)

        # Load
        loaded_json = GraphPersistence.load_json(json_path)
        loaded_pkl = GraphPersistence.load_pickle(pkl_path)

        assert loaded_json.graph.number_of_nodes() == 7
        assert loaded_pkl.graph.number_of_nodes() == 7

        # Compute execution order
        traversal = TopologicalTraversal(rpg)
        order = traversal.get_execution_order()

        assert len(order) > 0

    print(f"  -> Microservice: complete pipeline verified (save/load/execute)")

# Run Group 3 tests
test_e2e_cli()
test_e2e_web_api()
test_e2e_ml()
test_e2e_etl()
test_e2e_microservice()

print()


# =================================================================================
# FINAL RESULTS
# =================================================================================

print("=" * 90)
print("[RESULTS] PIPELINE INTEGRATION TEST RESULTS")
print("=" * 90)
print()

print(f"[OK] Total Tests Run: {test_results['total']}")
print(f"[OK] Passed: {test_results['passed']}")
print(f"[X] Failed: {test_results['failed']}")
print()

success_rate = (test_results["passed"] / test_results["total"] * 100) if test_results["total"] > 0 else 0
print(f"[STATS] Success Rate: {success_rate:.1f}%")
print()

if test_results["failed"] > 0:
    print("[X] Failed Tests:")
    for error in test_results["errors"]:
        print(f"   - {error}")
    print()

print("=" * 90)
if test_results["failed"] == 0:
    print("*** 100% SUCCESS! PIPELINE INTEGRATION IS LEGENDARY! ***")
else:
    print(f"[OK] {test_results['passed']}/{test_results['total']} tests passed ({success_rate:.1f}%)")
print("=" * 90)
