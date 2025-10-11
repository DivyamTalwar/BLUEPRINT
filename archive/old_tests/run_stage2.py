import sys
import os

# Windows UTF-8 fix
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from src.utils.config_loader import load_config
from src.core.graph_persistence import GraphPersistence
from src.stage2.stage2_orchestrator import Stage2Orchestrator
from src.utils.logger import StructuredLogger

logger = StructuredLogger("run_stage2")


def main():
    print("\n" + "=" * 60)
    print("BLUEPRINT - STAGE 2: IMPLEMENTATION-LEVEL CONSTRUCTION")
    print("=" * 60 + "\n")

    # Load configuration
    config = load_config()

    # Load functionality graph from Stage 1
    graph_path = input("Path to Stage 1 functionality graph [output/stage1_functionality_graph.json]: ").strip()
    if not graph_path:
        graph_path = "output/stage1_functionality_graph.json"

    if not os.path.exists(graph_path):
        print(f"❌ Error: Graph file not found: {graph_path}")
        print("\nPlease run Stage 1 first (run_stage1.py)")
        return

    print(f"\n📂 Loading functionality graph from: {graph_path}")

    persistence = GraphPersistence(config)
    functionality_graph = persistence.load_json(graph_path)

    if not functionality_graph:
        print("❌ Error: Failed to load graph")
        return

    # Get repository type
    print("\n📋 Repository type:")
    print("  1. library (Python package/library)")
    print("  2. web (Web application)")
    print("  3. api (REST API)")
    print("  4. cli (Command-line tool)")
    print("  5. ml (Machine learning project)")

    repo_type_map = {
        "1": "library",
        "2": "web",
        "3": "api",
        "4": "cli",
        "5": "ml"
    }

    choice = input("\nSelect type [1]: ").strip() or "1"
    repo_type = repo_type_map.get(choice, "library")

    print(f"\n✅ Repository type: {repo_type}")

    # Initialize Stage 2 orchestrator
    print("\n🚀 Starting Stage 2...\n")
    orchestrator = Stage2Orchestrator(config)

    # Run Stage 2
    try:
        complete_rpg = orchestrator.run(
            functionality_graph=functionality_graph,
            repo_type=repo_type,
            save_checkpoints=True
        )

        # Validate result
        print("\n🔍 Validating RPG...")
        validation_report = orchestrator.validate_rpg(complete_rpg)

        if validation_report["is_valid"]:
            print("✅ RPG validation passed!")

            if validation_report["warnings"]:
                print(f"\n⚠️  {len(validation_report['warnings'])} warnings:")
                for warning in validation_report["warnings"]:
                    print(f"   - {warning}")

            print("\n" + "=" * 60)
            print("STAGE 2 COMPLETE!")
            print("=" * 60)
            print("\n📊 Results:")
            print(f"   - Nodes: {validation_report['total_nodes']}")
            print(f"   - Edges: {validation_report['total_edges']}")
            print("\n📁 Output files:")
            print("   - output/stage2_complete_rpg.json (complete RPG)")
            print("   - output/stage2_checkpoint_*.json (checkpoints)")
            print("\n▶️  Next step: Run Stage 3 (run_stage3.py)")

        else:
            print("❌ RPG validation failed!")
            print("\n🐛 Issues:")
            for issue in validation_report["issues"]:
                print(f"   - {issue}")

            print("\nPlease review and fix issues before proceeding to Stage 3.")

    except Exception as e:
        print(f"\n❌ Error during Stage 2: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
