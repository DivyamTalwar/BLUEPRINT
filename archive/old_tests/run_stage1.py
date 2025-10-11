import sys
import os
from pathlib import Path

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))

from src.stage1.stage1_orchestrator import Stage1Orchestrator
from src.utils.logger import setup_logging, get_logger
from src.utils.config_loader import get_config

setup_logging(log_level="INFO", log_file="logs/stage1.log")
logger = get_logger(__name__)


def main():
    """Run Stage 1"""
    print("=" * 70)
    print("BLUEPRINT STAGE 1 - PROPOSAL-LEVEL CONSTRUCTION")
    print("=" * 70)

    # Check API keys
    config = get_config()
    env_check = config.validate_env_vars()

    missing = [k for k, v in env_check.items() if not v]
    if missing:
        print("\n❌ Missing API keys:")
        for key in missing:
            print(f"   - {key}")
        print("\nPlease add your API keys to .env file")
        return 1

    print("\n✓ API keys validated")

    # Check if feature tree exists
    features_path = Path("data/features_with_embeddings.json")
    if not features_path.exists():
        print("\n❌ Feature tree not found!")
        print("Run: python build_feature_tree.py")
        return 1

    print("✓ Feature tree found")

    # Get user input
    print("\n" + "=" * 70)
    print("REPOSITORY DESCRIPTION")
    print("=" * 70)
    print("\nDescribe the repository you want to create.")
    print("Be specific about features, functionality, and requirements.")
    print("\nExamples:")
    print("  - Build a REST API for blog management with authentication")
    print("  - Create a CLI tool for data processing and visualization")
    print("  - Machine learning library for time series forecasting")
    print()

    user_description = input("Your repository: ").strip()

    if not user_description:
        print("❌ No description provided")
        return 1

    print(f"\n✓ Got description: {user_description[:100]}...")

    # Configuration
    print("\n" + "=" * 70)
    print("CONFIGURATION")
    print("=" * 70)

    stage1_config = config.get_stage_config(1)

    iterations = stage1_config.get("iterations", 30)
    print(f"Iterations: {iterations}")

    response = input(f"\nUse default settings? (yes/no) [yes]: ").strip().lower()

    if response == "no":
        try:
            iterations = int(input(f"Number of iterations [30]: ") or "30")
        except ValueError:
            iterations = 30

    print(f"\n✓ Configuration set: {iterations} iterations")

    # Confirm
    print("\n" + "=" * 70)
    print("READY TO RUN")
    print("=" * 70)
    print(f"\nThis will:")
    print(f"  1. Process your repository description")
    print(f"  2. Run {iterations} iterations of feature selection")
    print(f"  3. Build functionality graph (RPG)")
    print(f"\nEstimated cost: $2-5")
    print(f"Estimated time: 5-15 minutes")

    response = input("\nContinue? (yes/no): ").strip().lower()

    if response != "yes":
        print("Aborted.")
        return 0

    # Run Stage 1
    print("\n" + "=" * 70)
    print("RUNNING STAGE 1")
    print("=" * 70)

    try:
        orchestrator = Stage1Orchestrator(output_dir="output/stage1")

        rpg = orchestrator.run(
            user_description=user_description,
            iterations=iterations,
            save_intermediate=True,
        )

        # Visualize
        print("\nGenerating visualization...")
        orchestrator.visualize_graph(rpg)

        print("\n" + "=" * 70)
        print("✅ STAGE 1 COMPLETE!")
        print("=" * 70)

        print(f"\nOutput directory: output/stage1")
        print(f"  - functionality_graph.json")
        print(f"  - functionality_graph.png")
        print(f"  - selected_features.json")
        print(f"  - selection_log.json")
        print(f"  - graph_summary.txt")

        print(f"\n🎯 Ready for Stage 2!")

        return 0

    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
        return 1

    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        logger.exception("Stage 1 failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
