import sys
import os
import time

# Windows UTF-8 fix
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from src.utils.config_loader import load_config
from src.stage1.stage1_orchestrator import Stage1Orchestrator
from src.stage2.stage2_orchestrator import Stage2Orchestrator
from src.stage3.stage3_orchestrator import Stage3Orchestrator
from src.utils.logger import StructuredLogger

logger = StructuredLogger("full_pipeline")


def print_banner():
    """Print BLUEPRINT banner."""
    print("\n" + "=" * 70)
    print(" ██████╗ ██╗     ██╗   ██╗███████╗██████╗ ██████╗ ██╗███╗   ██╗████████╗")
    print(" ██╔══██╗██║     ██║   ██║██╔════╝██╔══██╗██╔══██╗██║████╗  ██║╚══██╔══╝")
    print(" ██████╔╝██║     ██║   ██║█████╗  ██████╔╝██████╔╝██║██╔██╗ ██║   ██║   ")
    print(" ██╔══██╗██║     ██║   ██║██╔══╝  ██╔═══╝ ██╔══██╗██║██║╚██╗██║   ██║   ")
    print(" ██████╔╝███████╗╚██████╔╝███████╗██║     ██║  ██║██║██║ ╚████║   ██║   ")
    print(" ╚═════╝ ╚══════╝ ╚═════╝ ╚══════╝╚═╝     ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝   ╚═╝   ")
    print("=" * 70)
    print("   Universal Repository Generation System - Full Pipeline")
    print("=" * 70 + "\n")


def main():
    """Run complete 3-stage pipeline."""
    print_banner()

    # Load configuration
    config = load_config()

    # Get user input
    print("📝 Describe the repository you want to build:")
    print("   Example: 'Build a machine learning library with regression and classification algorithms'\n")

    user_description = input("Your description: ").strip()

    if not user_description:
        print("❌ Error: Description cannot be empty")
        return

    print(f"\n✅ Description: {user_description}\n")

    # Get output directory
    output_dir = input("Output directory for generated repository [./generated_repo]: ").strip()
    if not output_dir:
        output_dir = "./generated_repo"

    # Configuration summary
    print("\n" + "=" * 70)
    print("PIPELINE CONFIGURATION")
    print("=" * 70)
    print(f"📋 Description: {user_description}")
    print(f"📁 Output: {output_dir}")
    print(f"🔧 Stage 1 Iterations: {config.get('stage1', {}).get('iterations', 30)}")
    print(f"🔧 Max TDD Attempts: {config.get('max_debug_attempts', 8)}")
    print("=" * 70 + "\n")

    # Warning about Docker
    print("⚠️  REQUIREMENTS:")
    print("   ✓ Docker Desktop must be running (for Stage 3)")
    print("   ✓ API keys configured in .env file")
    print("   ✓ Estimated time: 45-90 minutes")
    print("   ✓ Estimated cost: $5-15 (depending on complexity)\n")

    proceed = input("Continue with full pipeline? [y/N]: ").strip().lower()
    if proceed != 'y':
        print("Cancelled.")
        return

    # Start timer
    pipeline_start = time.time()

    # =========================================================================
    # STAGE 1: Proposal-level Construction (Feature Selection)
    # =========================================================================

    print("\n" + "=" * 70)
    print("STAGE 1: PROPOSAL-LEVEL CONSTRUCTION")
    print("=" * 70 + "\n")

    try:
        stage1 = Stage1Orchestrator(config)

        functionality_graph = stage1.run(
            user_description=user_description,
            iterations=config.get('stage1', {}).get('iterations', 30),
            save_checkpoints=True
        )

        print("\n✅ Stage 1 complete!")

    except Exception as e:
        print(f"\n❌ Stage 1 failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # =========================================================================
    # STAGE 2: Implementation-level Construction (RPG Design)
    # =========================================================================

    print("\n" + "=" * 70)
    print("STAGE 2: IMPLEMENTATION-LEVEL CONSTRUCTION")
    print("=" * 70 + "\n")

    try:
        stage2 = Stage2Orchestrator(config)

        complete_rpg = stage2.run(
            functionality_graph=functionality_graph,
            repo_type="library",  # Auto-detect or default
            save_checkpoints=True
        )

        # Validate RPG
        validation = stage2.validate_rpg(complete_rpg)

        if not validation["is_valid"]:
            print("\n❌ RPG validation failed:")
            for issue in validation["issues"]:
                print(f"   - {issue}")
            return

        print("\n✅ Stage 2 complete!")

    except Exception as e:
        print(f"\n❌ Stage 2 failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # =========================================================================
    # STAGE 3: Code Generation (TDD-based Implementation)
    # =========================================================================

    print("\n" + "=" * 70)
    print("STAGE 3: CODE GENERATION")
    print("=" * 70 + "\n")

    try:
        stage3 = Stage3Orchestrator(config)

        repo_path = stage3.run(
            complete_rpg=complete_rpg,
            output_dir=output_dir,
            checkpoint_interval=10
        )

        print("\n✅ Stage 3 complete!")

    except Exception as e:
        print(f"\n❌ Stage 3 failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # =========================================================================
    # PIPELINE COMPLETE
    # =========================================================================

    pipeline_elapsed = time.time() - pipeline_start

    print("\n" + "=" * 70)
    print("🎉 PIPELINE COMPLETE! 🎉")
    print("=" * 70)
    print(f"\n⏱️  Total time: {pipeline_elapsed/60:.1f} minutes")
    print(f"\n📂 Generated repository: {repo_path}")
    print("\n📋 What was generated:")
    print("   ✓ Complete folder/file structure")
    print("   ✓ All function/class implementations")
    print("   ✓ Comprehensive test suite")
    print("   ✓ Documentation (README.md)")
    print("   ✓ Package setup (setup.py, requirements.txt)")
    print("\n📋 Next steps:")
    print(f"   1. cd {repo_path}")
    print("   2. pip install -r requirements.txt")
    print("   3. pytest tests/")
    print("   4. Review and customize the code")
    print("\n🚀 Your repository is ready to use!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        print("Check output/ directory for checkpoints to resume")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
