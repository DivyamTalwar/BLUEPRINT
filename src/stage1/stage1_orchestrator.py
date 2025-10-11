import os
from typing import Dict, Any, Optional
from pathlib import Path

from src.core.llm_router import LLMRouter
from src.core.rpg import RepositoryPlanningGraph
from src.core.graph_persistence import GraphPersistence
from src.stage1.user_input_processor import UserInputProcessor, RepositoryRequest
from src.stage1.embedding_generator import EmbeddingGenerator
from src.stage1.vector_store import VectorStore
from src.stage1.exploit_strategy import ExploitStrategy
from src.stage1.explore_strategy import ExploreStrategy
from src.stage1.feature_selection_loop import FeatureSelectionLoop
from src.stage1.functionality_graph_builder import FunctionalityGraphBuilder
from src.utils.config_loader import get_config
from src.utils.logger import get_logger
from src.utils.file_ops import FileOperations

logger = get_logger(__name__)


class Stage1Orchestrator:
    def __init__(
        self,
        config_path: Optional[str] = None,
        output_dir: str = "output/stage1",
    ):
        """
        Initialize Stage 1 orchestrator

        Args:
            config_path: Path to config file
            output_dir: Output directory
        """
        # Load config
        self.config = get_config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        logger.info("Initializing Stage 1 components")

        # LLM Router
        self.llm_router = LLMRouter(self.config.get_all())

        # User input processor
        self.input_processor = UserInputProcessor(self.llm_router)

        # Embedding generator
        self.embedding_generator = EmbeddingGenerator(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-3-large",
        )

        # Vector store
        vector_config = self.config.get_vector_db_config()
        self.vector_store = VectorStore(
            api_key=os.getenv("PINECONE_API_KEY"),
            index_name=vector_config["index_name"],
            dimension=vector_config["dimension"],
            metric=vector_config["metric"],
        )

        # Strategies
        self.exploit_strategy = ExploitStrategy(
            vector_store=self.vector_store,
            embedding_generator=self.embedding_generator,
        )

        self.explore_strategy = ExploreStrategy(
            vector_store=self.vector_store,
            embedding_generator=self.embedding_generator,
            llm_router=self.llm_router,
        )

        # Functionality graph builder
        self.graph_builder = FunctionalityGraphBuilder(self.llm_router)

        logger.info("Stage 1 initialized")

    def run(
        self,
        user_description: str,
        iterations: int = 30,
        target_features: Optional[int] = None,
        save_intermediate: bool = True,
    ) -> RepositoryPlanningGraph:
        """
        Run complete Stage 1 pipeline

        Args:
            user_description: User's repository description
            iterations: Number of selection iterations
            target_features: Target number of features (auto if None)
            save_intermediate: Save intermediate results

        Returns:
            Functionality graph (RPG)
        """
        logger.info("=" * 70)
        logger.info("STAGE 1: PROPOSAL-LEVEL CONSTRUCTION")
        logger.info("=" * 70)

        # Step 1: Process user input
        logger.info("\nSTEP 1: PROCESSING USER INPUT")
        logger.info("-" * 70)

        request = self.input_processor.process(user_description)

        # Validate
        is_valid, errors = self.input_processor.validate_request(request)
        if not is_valid:
            logger.error("Request validation failed: %s", errors)
            raise ValueError(f"Invalid request: {errors}")

        # Enhance
        request = self.input_processor.enhance_request(request)

        logger.info("Request processed:")
        logger.info("  Type: %s", request.repo_type)
        logger.info("  Domain: %s", request.primary_domain)
        logger.info("  Requirements: %d explicit, %d implicit",
                   len(request.explicit_requirements),
                   len(request.implicit_requirements))

        if save_intermediate:
            FileOperations.write_json(
                str(self.output_dir / "request.json"),
                request.to_dict(),
            )

        # Determine target features
        if target_features is None:
            target_features = self.input_processor.get_target_feature_count(request)

        logger.info("  Target features: %d", target_features)

        # Step 2: Feature selection loop
        logger.info("\nSTEP 2: FEATURE SELECTION (%d iterations)", iterations)
        logger.info("-" * 70)

        selection_loop = FeatureSelectionLoop(
            exploit_strategy=self.exploit_strategy,
            explore_strategy=self.explore_strategy,
            num_iterations=iterations,
        )

        selected_features = selection_loop.run(
            request=request,
            target_features=target_features,
        )

        # Log stats
        stats = selection_loop.get_stats()
        logger.info("Selection complete:")
        logger.info("  Features: %d", stats["total_features"])
        logger.info("  Domains: %d", stats["domains_visited"])
        logger.info("  Exploit: %d, Explore: %d", stats["exploit_total"], stats["explore_total"])

        if save_intermediate:
            # Save features
            features_data = [f.to_dict() for f in selected_features]
            FileOperations.write_json(
                str(self.output_dir / "selected_features.json"),
                {"features": features_data, "count": len(features_data)},
            )

            # Save selection log
            selection_log = selection_loop.export_selection_log()
            FileOperations.write_json(
                str(self.output_dir / "selection_log.json"),
                selection_log,
            )

        # Step 3: Build functionality graph
        logger.info("\nSTEP 3: BUILDING FUNCTIONALITY GRAPH")
        logger.info("-" * 70)

        rpg = self.graph_builder.build(
            request=request,
            selected_features=selected_features,
        )

        # Refine graph
        rpg = self.graph_builder.refine_graph(rpg)

        rpg_stats = rpg.get_stats()
        logger.info("Functionality graph built:")
        logger.info("  Nodes: %d", rpg_stats["total_nodes"])
        logger.info("  Edges: %d", rpg_stats["total_edges"])
        logger.info("  Root modules: %d", rpg_stats["root_nodes"])

        # Save RPG
        if save_intermediate:
            GraphPersistence.save_json(rpg, str(self.output_dir / "functionality_graph.json"))
            GraphPersistence.export_summary(rpg, str(self.output_dir / "graph_summary.txt"))

        # Create checkpoint
        GraphPersistence.create_checkpoint(
            rpg,
            stage="stage1",
            output_dir=str(self.output_dir / "checkpoints"),
        )

        logger.info("\n" + "=" * 70)
        logger.info("STAGE 1 COMPLETE")
        logger.info("=" * 70)

        # Final stats
        self._log_final_stats(request, stats, rpg_stats)

        return rpg

    def _log_final_stats(
        self,
        request: RepositoryRequest,
        selection_stats: Dict[str, Any],
        rpg_stats: Dict[str, Any],
    ):
        """Log final statistics"""

        logger.info("\nFINAL STATISTICS:")
        logger.info("-" * 70)

        # Request
        logger.info("Repository:")
        logger.info("  Type: %s", request.repo_type)
        logger.info("  Complexity: %s", request.complexity_estimate)
        logger.info("  Primary domain: %s", request.primary_domain)

        # Features
        logger.info("\nFeatures:")
        logger.info("  Total selected: %d", selection_stats["total_features"])
        logger.info("  Domains covered: %d", selection_stats["domains_visited"])
        logger.info("  By complexity: %s", selection_stats.get("features_by_complexity", {}))

        # Graph
        logger.info("\nFunctionality Graph:")
        logger.info("  Nodes: %d", rpg_stats["total_nodes"])
        logger.info("  Edges: %d", rpg_stats["total_edges"])
        logger.info("  Modules: %d", rpg_stats["root_nodes"])
        logger.info("  Is DAG: %s", rpg_stats["is_dag"])

        # Costs
        llm_stats = self.llm_router.get_stats()
        emb_stats = self.embedding_generator.get_stats()

        logger.info("\nCosts:")
        logger.info("  LLM: $%.2f (%d calls)", llm_stats["total_cost"], llm_stats["api_calls"])
        logger.info("  Embeddings: $%.2f (%d tokens)", emb_stats["total_cost"], emb_stats["total_tokens"])
        logger.info("  Total: $%.2f", llm_stats["total_cost"] + emb_stats["total_cost"])

        # Files
        logger.info("\nOutput files:")
        logger.info("  %s", self.output_dir)

    def visualize_graph(self, rpg: RepositoryPlanningGraph, output_path: Optional[str] = None):
        """Visualize functionality graph"""

        from src.core.graph_visualization import GraphVisualization

        if output_path is None:
            output_path = str(self.output_dir / "functionality_graph.png")

        logger.info("Visualizing functionality graph: %s", output_path)

        GraphVisualization.visualize_full_graph(
            rpg=rpg,
            output_path=output_path,
            layout="hierarchical",
        )

        logger.info("Visualization saved")
