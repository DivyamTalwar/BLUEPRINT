import sys
import os

# Windows UTF-8 fix
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from src.utils.config_loader import load_config
from src.core.graph_persistence import GraphPersistence
from src.stage3.stage3_orchestrator import Stage3Orchestrator
from src.utils.logger import StructuredLogger

logger = StructuredLogger("run_stage3")


def main():
    """Run Stage 3 pipeline."""
    print("\n" + "=" * 60)
    print("BLUEPRINT - STAGE 3: CODE GENERATION")
    print("=" * 60 + "\n")

    # Load configuration
    config = load_config()

    # Load complete RPG from Stage 2
    graph_path = input("Path to Stage 2 complete RPG [output/stage2_complete_rpg.json]: ").strip()
    if not graph_path:
        graph_path = "output/stage2_complete_rpg.json"

    if not os.path.exists(graph_path):
        print(f"❌ Error: RPG file not found: {graph_path}")
        print("\nPlease run Stage 2 first (run_stage2.py)")
        return

    print(f"\n📂 Loading complete RPG from: {graph_path}")

    persistence = GraphPersistence(config)
    complete_rpg = persistence.load_json(graph_path)

    if not complete_rpg:
        print("❌ Error: Failed to load RPG")
        return

    # Get output directory
    output_dir = input("\nOutput directory for generated repository [./generated_repo]: ").strip()
    if not output_dir:
        output_dir = "./generated_repo"

    print(f"\n📁 Will generate repository at: {output_dir}")

    # Warning about Docker requirement
    print("\n⚠️  IMPORTANT: Stage 3 requires Docker for code testing")
    print("   Please ensure Docker Desktop is running")

    proceed = input("\nContinue? [y/N]: ").strip().lower()
    if proceed != 'y':
        print("Cancelled.")
        return

    # Initialize Stage 3 orchestrator
    print("\n🚀 Starting Stage 3...\n")
    print("This may take 30-60 minutes depending on repository size...")
    print("Progress will be displayed in real-time.\n")

    orchestrator = Stage3Orchestrator(config)

    # Run Stage 3
    try:
        repo_path = orchestrator.run(
            complete_rpg=complete_rpg,
            output_dir=output_dir,
            checkpoint_interval=10
        )

        if repo_path:
            print("\n" + "=" * 60)
            print("STAGE 3 COMPLETE!")
            print("=" * 60)
            print(f"\n✅ Repository generated at: {repo_path}")
            print("\n📋 Next steps:")
            print(f"   1. cd {repo_path}")
            print("   2. pip install -r requirements.txt")
            print("   3. pytest tests/")
            print("   4. Review and customize the code")
            print("\n🎉 Your repository is ready!")

        else:
            print("\n❌ Code generation failed. Check logs for details.")

    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted!")
        print("You can resume from checkpoint using:")
        print("   orchestrator.resume_from_checkpoint('output/stage3_checkpoint.json', ...)")

    except Exception as e:
        print(f"\n❌ Error during Stage 3: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
