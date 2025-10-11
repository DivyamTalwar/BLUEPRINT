import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from src.utils.logger import get_logger
from src.utils.config_loader import ConfigLoader
from src.utils.cost_tracker import CostTracker
from src.core.rpg import RepositoryPlanningGraph
from src.core.graph_persistence import GraphPersistence

# Stage imports
from src.stage1.user_input_processor import UserInputProcessor, RepositoryRequest
from src.stage1.feature_selection_loop import FeatureSelectionLoop
from src.stage1.exploit_strategy import ExploitStrategy
from src.stage1.explore_strategy import ExploreStrategy
from src.stage1.cohere_embeddings import CohereEmbeddings
from src.stage2.stage2_orchestrator import Stage2Orchestrator
from src.stage3.stage3_orchestrator import Stage3Orchestrator
from src.core.llm_router_final import FinalLLMRouter

# Load environment
load_dotenv()

logger = get_logger(__name__)


@dataclass
class PipelineCheckpoint:
    """Pipeline execution checkpoint"""
    timestamp: str
    stage: int
    user_request: Dict[str, Any]
    selected_features: Optional[list] = None
    rpg_path: Optional[str] = None
    output_dir: Optional[str] = None
    cost_stats: Optional[Dict[str, Any]] = None


class BLUEPRINTPipeline:
    """
    Main BLUEPRINT Pipeline Orchestrator

    Executes complete 3-stage code generation:
    1. Stage 1: Feature Selection (exploit-explore)
    2. Stage 2: RPG Construction (implementation design)
    3. Stage 3: Code Generation (TDD with Docker)
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize pipeline"""

        # Load configuration
        self.config = ConfigLoader(config_path or "config.yaml")

        # Initialize cost tracker
        self.cost_tracker = CostTracker()

        # Initialize LLM router
        self.llm = FinalLLMRouter(self.config.get_all())

        # Pipeline state
        self.current_stage = 0
        self.user_request: Optional[RepositoryRequest] = None
        self.selected_features = []
        self.rpg: Optional[RepositoryPlanningGraph] = None
        self.output_dir: Optional[Path] = None

        logger.info("BLUEPRINT Pipeline initialized",
                   config_path=config_path or "config.yaml")

    def check_prerequisites(self) -> bool:
        """
        Check all prerequisites before running pipeline

        Returns:
            True if all prerequisites met, False otherwise
        """
        logger.info("Checking prerequisites...")

        issues = []
        warnings = []

        # Check API keys
        required_keys = {
            "GOOGLE_API_KEY": "Gemini (FREE tier)",
            "OPENROUTER_API_KEY": "OpenRouter for Claude",
            "COHERE_API_KEY": "Cohere for embeddings",
            "PINECONE_API_KEY": "Pinecone vector DB",
        }

        for key, service in required_keys.items():
            value = os.getenv(key)
            if not value or value.startswith("ADD-YOUR-"):
                issues.append(f"{key} not set (needed for: {service})")
            else:
                logger.info(f"[OK] {key} found")

        # Check Docker (warning only)
        docker_available = False
        try:
            import docker
            client = docker.from_env()
            client.ping()
            docker_available = True
            logger.info("[OK] Docker available for test validation")
        except Exception as e:
            warnings.append(f"Docker not available - will use static validation instead")
            logger.warning("Docker not available - enabling skip_docker mode")
            # Auto-enable skip_docker mode
            if not self.config.get("stage3.skip_docker", False):
                self.config.config["stage3"]["skip_docker"] = True
                logger.info("Auto-enabled skip_docker mode in config")

        # Check Pinecone
        try:
            from pinecone import Pinecone
            api_key = os.getenv("PINECONE_API_KEY")
            if api_key and not api_key.startswith("ADD-YOUR-"):
                pc = Pinecone(api_key=api_key)

                index_name = "blueprint-features"
                existing_indexes = [idx.name for idx in pc.list_indexes()]
                if index_name in existing_indexes:
                    index = pc.Index(index_name)
                    stats = index.describe_index_stats()
                    logger.info(f"[OK] Pinecone connected - {stats['total_vector_count']} features")
                else:
                    issues.append(f"Pinecone index '{index_name}' not found - run: python scripts/generate_feature_tree.py")
        except Exception as e:
            issues.append(f"Pinecone check failed: {e}")

        # Print results
        print()
        if issues:
            print("[ERROR] Prerequisites NOT met:")
            for issue in issues:
                print(f"  [X] {issue}")
            print()
            print("To fix:")
            print("  1. Copy .env.example to .env")
            print("  2. Add your API keys to .env")
            print("  3. Run: python scripts/generate_feature_tree.py")
            print()
            return False

        if warnings:
            print("[WARNING] Some optional features unavailable:")
            for warning in warnings:
                print(f"  [!] {warning}")
            print()

        print("[SUCCESS] All prerequisites met! Ready to generate code!")
        print()
        return True

    def run(
        self,
        user_description: str,
        output_dir: Optional[str] = None,
        checkpoint_file: Optional[str] = None
    ) -> bool:
        """
        Run complete BLUEPRINT pipeline

        Args:
            user_description: Natural language description of repository
            output_dir: Output directory for generated code
            checkpoint_file: Path to save checkpoints

        Returns:
            True if successful, False otherwise
        """

        start_time = time.time()

        logger.info("=" * 80)
        logger.info("BLUEPRINT PIPELINE STARTED")
        logger.info("=" * 80)
        logger.info("User request", description=user_description[:100])

        try:
            # Check prerequisites
            if not self.check_prerequisites():
                logger.error("Prerequisites not met - aborting")
                return False

            # Set output directory
            if output_dir is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = f"output/generated_{timestamp}"

            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

            logger.info("Output directory", path=str(self.output_dir))

            # ================================================================
            # STAGE 1: FEATURE SELECTION
            # ================================================================

            logger.info("")
            logger.info("STAGE 1: FEATURE SELECTION")
            logger.info("-" * 80)

            self.current_stage = 1

            # Process user input
            logger.info("Processing user input...")
            processor = UserInputProcessor(self.llm)
            self.user_request = processor.process(user_description)

            logger.info("User request processed",
                      repo_type=self.user_request.repo_type,
                      domain=self.user_request.primary_domain,
                      complexity=self.user_request.complexity_estimate)

            # Validate request
            is_valid, errors = processor.validate_request(self.user_request)
            if not is_valid:
                logger.error("Request validation failed", errors=errors)
                return False

            # Get target feature count
            target_features = processor.get_target_feature_count(self.user_request)
            logger.info(f"Target features: {target_features}")

            # Initialize vector DB and strategies
            logger.info("Initializing feature selection strategies...")

            from src.stage1.vector_store import VectorStore
            from src.stage1.cohere_embeddings import CohereEmbeddings

            try:
                pinecone_api_key = os.getenv("PINECONE_API_KEY")
                if not pinecone_api_key:
                    logger.error("PINECONE_API_KEY not found in .env")
                    return False

                vector_store = VectorStore(
                    api_key=pinecone_api_key,
                    index_name="blueprint-features",
                    dimension=1024  # Cohere embed-english-v3.0
                )
                vector_store.index = vector_store.pc.Index(vector_store.index_name)
            except Exception as e:
                logger.error("Failed to initialize Vector Store", error=str(e))
                return False

            cohere_api_key = os.getenv("COHERE_API_KEY")
            if not cohere_api_key:
                logger.error("COHERE_API_KEY not found in .env")
                return False

            cohere_embeddings = CohereEmbeddings(api_key=cohere_api_key)
            exploit_strategy = ExploitStrategy(vector_store, cohere_embeddings)
            explore_strategy = ExploreStrategy(vector_store, cohere_embeddings, self.llm)

            # Run feature selection loop
            num_iterations = self.config.get("stage1.iterations", 30)
            logger.info(f"Running {num_iterations}-iteration feature selection...")

            selection_loop = FeatureSelectionLoop(
                exploit_strategy,
                explore_strategy,
                num_iterations
            )

            self.selected_features = selection_loop.run(
                self.user_request,
                target_features=target_features
            )

            logger.info("Feature selection complete",
                      features_selected=len(self.selected_features))

            # Save checkpoint
            if checkpoint_file:
                self._save_checkpoint(checkpoint_file, stage=1)

            # ================================================================
            # STAGE 2: IMPLEMENTATION DESIGN (RPG CONSTRUCTION)
            # ================================================================

            logger.info("")
            logger.info("STAGE 2: IMPLEMENTATION DESIGN")
            logger.info("-" * 80)

            self.current_stage = 2

            # Build functionality graph from features
            logger.info("Building functionality graph from features...")
            from src.stage1.functionality_graph_builder import FunctionalityGraphBuilder
            from src.core.llm_router_final import FinalLLMRouter

            llm_router = FinalLLMRouter(self.config.get_all())
            graph_builder = FunctionalityGraphBuilder(llm_router)

            functionality_graph = graph_builder.build(
                request=self.user_request,
                selected_features=self.selected_features
            )

            logger.info("Functionality graph built",
                      nodes=functionality_graph.graph.number_of_nodes(),
                      edges=functionality_graph.graph.number_of_edges())

            # Initialize Stage 2
            stage2 = Stage2Orchestrator(self.config.get_all())

            # Transform functionality graph into complete RPG
            logger.info("Building complete Repository Planning Graph...")

            self.rpg = stage2.run(
                functionality_graph=functionality_graph,
                repo_type=self.user_request.repo_type,
                save_checkpoints=True
            )

            logger.info("RPG construction complete",
                      nodes=self.rpg.graph.number_of_nodes(),
                      edges=self.rpg.graph.number_of_edges())

            # Save RPG
            rpg_path = self.output_dir / "rpg.json"
            GraphPersistence.save_json(self.rpg, str(rpg_path))
            logger.info("RPG saved", path=str(rpg_path))

            # Save checkpoint
            if checkpoint_file:
                self._save_checkpoint(checkpoint_file, stage=2)

            # ================================================================
            # STAGE 3: CODE GENERATION (TDD)
            # ================================================================

            logger.info("")
            logger.info("STAGE 3: CODE GENERATION")
            logger.info("-" * 80)

            self.current_stage = 3

            # Initialize Stage 3
            stage3 = Stage3Orchestrator(self.config.get_all())

            # Generate code
            logger.info("Generating code with TDD...")

            success = stage3.generate_repository(
                rpg=self.rpg,
                output_dir=str(self.output_dir)
            )

            if not success:
                logger.error("Code generation failed")
                return False

            logger.info("Code generation complete!")

            # Save checkpoint
            if checkpoint_file:
                self._save_checkpoint(checkpoint_file, stage=3)

            # ================================================================
            # FINALIZATION
            # ================================================================

            duration = time.time() - start_time

            # Get cost stats
            cost_stats = self.cost_tracker.get_stats()

            logger.info("")
            logger.info("=" * 80)
            logger.info("PIPELINE COMPLETE!")
            logger.info("=" * 80)
            logger.info(f"Duration: {duration:.2f}s")
            logger.info(f"Output: {self.output_dir}")
            logger.info(f"Total cost: ${cost_stats['total_cost']:.4f}")
            logger.info(f"Features: {len(self.selected_features)}")
            logger.info(f"RPG nodes: {self.rpg.graph.number_of_nodes()}")
            logger.info("=" * 80)

            # Save final stats
            self._save_final_report()

            return True

        except Exception as e:
            # Safely encode error message to avoid unicode issues
            error_msg = str(e).encode('ascii', errors='replace').decode('ascii')
            logger.error("Pipeline failed", error=error_msg)
            return False

    def _save_checkpoint(self, checkpoint_file: str, stage: int):
        """Save pipeline checkpoint"""

        checkpoint = PipelineCheckpoint(
            timestamp=datetime.now().isoformat(),
            stage=stage,
            user_request=self.user_request.to_dict() if self.user_request else {},
            selected_features=[f.id for f in self.selected_features] if self.selected_features else [],
            rpg_path=str(self.output_dir / "rpg.json") if self.rpg else None,
            output_dir=str(self.output_dir) if self.output_dir else None,
            cost_stats=self.cost_tracker.get_stats()
        )

        with open(checkpoint_file, 'w') as f:
            json.dump(asdict(checkpoint), f, indent=2)

        logger.info(f"Checkpoint saved: {checkpoint_file}")

    def _save_final_report(self):
        """Save final execution report"""

        report_path = self.output_dir / "BLUEPRINT_REPORT.md"

        cost_stats = self.cost_tracker.get_stats()

        report = f"""# BLUEPRINT Generation Report

## Repository Information
- **Description**: {self.user_request.raw_description}
- **Type**: {self.user_request.repo_type}
- **Domain**: {self.user_request.primary_domain}
- **Complexity**: {self.user_request.complexity_estimate}

## Generation Stats
- **Features Selected**: {len(self.selected_features)}
- **RPG Nodes**: {self.rpg.graph.number_of_nodes()}
- **RPG Edges**: {self.rpg.graph.number_of_edges()}

## Cost Breakdown
- **Total Cost**: ${cost_stats['total_cost']:.4f}
- **API Calls**: {cost_stats['api_calls']}
- **Total Tokens**: {cost_stats['total_tokens']:,}

## Stage Breakdown
### Stage 1: Feature Selection
- Features: {len(self.selected_features)}

### Stage 2: Implementation Design
- Nodes: {self.rpg.graph.number_of_nodes()}
- Edges: {self.rpg.graph.number_of_edges()}

### Stage 3: Code Generation
- Output: {self.output_dir}

## Next Steps
1. Review generated code in `{self.output_dir}`
2. Run tests: `pytest`
3. Start development server
4. Customize as needed

---
Generated by BLUEPRINT - The Legendary Code Generation System
"""

        with open(report_path, 'w') as f:
            f.write(report)

        logger.info(f"Report saved: {report_path}")


def main():
    """Main CLI entry point"""

    parser = argparse.ArgumentParser(
        description="BLUEPRINT - Legendary Code Generation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "Build a REST API for blog management"
  python main.py --interactive
  python main.py --check-only
  python main.py "Create ML library" --output my_ml_lib

For more info: https://github.com/blueprint/blueprint
        """
    )

    parser.add_argument(
        "description",
        nargs="?",
        help="Natural language description of repository to generate"
    )

    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Interactive mode - prompts for input"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output directory for generated code"
    )

    parser.add_argument(
        "-c", "--config",
        help="Path to config.yaml file"
    )

    parser.add_argument(
        "--checkpoint",
        help="Path to save/load checkpoint"
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check prerequisites, don't generate"
    )

    args = parser.parse_args()

    # Print banner
    print()
    print("=" * 80)
    print("BLUEPRINT - THE LEGENDARY CODE GENERATION SYSTEM")
    print("=" * 80)
    print()

    # Initialize pipeline
    pipeline = BLUEPRINTPipeline(config_path=args.config)

    # Check-only mode
    if args.check_only:
        success = pipeline.check_prerequisites()
        sys.exit(0 if success else 1)

    # Get user description
    if args.interactive:
        print("Enter your repository description:")
        print("(Press Ctrl+D or Ctrl+Z when done)\n")
        description = sys.stdin.read().strip()
    elif args.description:
        description = args.description
    else:
        parser.print_help()
        sys.exit(1)

    # Run pipeline
    success = pipeline.run(
        user_description=description,
        output_dir=args.output,
        checkpoint_file=args.checkpoint
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
