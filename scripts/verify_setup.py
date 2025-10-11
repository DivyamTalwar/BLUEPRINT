import sys
import os
from pathlib import Path

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    print("Testing imports...")

    try:
        from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType
        print("✓ RPG core imported successfully")

        from src.core.llm_router import LLMRouter
        print("✓ LLM Router imported successfully")

        from src.core.graph_operations import GraphOperations
        print("✓ Graph Operations imported successfully")

        from src.core.graph_persistence import GraphPersistence
        print("✓ Graph Persistence imported successfully")

        from src.core.graph_visualization import GraphVisualization
        print("✓ Graph Visualization imported successfully")

        from src.utils.config_loader import ConfigLoader
        print("✓ Config Loader imported successfully")

        from src.utils.cost_tracker import CostTracker
        print("✓ Cost Tracker imported successfully")

        from src.utils.file_ops import FileOperations
        print("✓ File Operations imported successfully")

        return True

    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")

    try:
        from src.utils.config_loader import get_config

        config = get_config()
        print("✓ Configuration loaded successfully")

        # Validate config structure
        assert config.get("api") is not None
        assert config.get("stage1") is not None
        assert config.get("stage2") is not None
        assert config.get("stage3") is not None

        print("✓ Configuration structure valid")

        # Check environment variables
        env_validation = config.validate_env_vars()
        print(f"\nEnvironment Variables Status:")
        for var, status in env_validation.items():
            status_icon = "✓" if status else "✗"
            print(f"  {status_icon} {var}: {'Set' if status else 'Not Set'}")

        return True

    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_rpg_basic():
    """Test basic RPG functionality"""
    print("\nTesting RPG basic functionality...")

    try:
        from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType

        # Create RPG
        rpg = RepositoryPlanningGraph("Test Repository")
        print("✓ RPG created successfully")

        # Add nodes
        root = rpg.add_node("Root Module", NodeType.ROOT, "Main module")
        leaf = rpg.add_node("Feature 1", NodeType.LEAF, "Test feature")
        print("✓ Nodes added successfully")

        # Add edge
        rpg.add_edge(root, leaf, EdgeType.HIERARCHY)
        print("✓ Edge added successfully")

        # Validate
        is_valid, errors = rpg.validate()
        assert is_valid, f"Validation failed: {errors}"
        print("✓ Graph validation passed")

        # Get stats
        stats = rpg.get_stats()
        assert stats["total_nodes"] == 2
        assert stats["total_edges"] == 1
        print("✓ Statistics correct")

        return True

    except Exception as e:
        print(f"✗ RPG test failed: {e}")
        return False


def test_persistence():
    """Test graph persistence"""
    print("\nTesting graph persistence...")

    try:
        from src.core.rpg import RepositoryPlanningGraph, NodeType
        from src.core.graph_persistence import GraphPersistence
        import tempfile

        # Create test RPG
        rpg = RepositoryPlanningGraph("Persistence Test")
        rpg.add_node("Node 1", NodeType.ROOT, "Test")

        # Test JSON save/load
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = f.name

        GraphPersistence.save_json(rpg, json_path)
        loaded_rpg = GraphPersistence.load_json(json_path)

        assert loaded_rpg is not None
        assert loaded_rpg.repository_goal == "Persistence Test"
        print("✓ JSON persistence working")

        # Cleanup
        Path(json_path).unlink()

        return True

    except Exception as e:
        print(f"✗ Persistence test failed: {e}")
        return False


def test_cost_tracker():
    """Test cost tracking"""
    print("\nTesting cost tracker...")

    try:
        from src.utils.cost_tracker import CostTracker

        tracker = CostTracker()

        # Simulate API call
        cost = tracker.calculate_cost("openai", "gpt-4o", 1000)
        assert cost > 0
        print(f"✓ Cost calculation working (1000 tokens = ${cost:.4f})")

        # Get stats
        stats = tracker.get_stats()
        assert stats["total_tokens"] == 1000
        assert stats["api_calls"] == 1
        print("✓ Cost tracking working")

        return True

    except Exception as e:
        print(f"✗ Cost tracker test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("BLUEPRINT SETUP VERIFICATION")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("RPG Basic", test_rpg_basic),
        ("Persistence", test_persistence),
        ("Cost Tracker", test_cost_tracker),
    ]

    results = []

    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print()
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! BLUEPRINT is ready to use.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
