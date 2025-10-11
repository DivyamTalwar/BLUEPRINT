import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from src.utils.logger import get_logger
from src.utils.cost_tracker import CostTracker
from src.stage1.user_input_processor import UserInputProcessor, RepositoryRequest
from src.stage1.feature_selection_loop import FeatureSelectionLoop
from src.stage2.stage2_orchestrator import Stage2Orchestrator
from src.stage3.stage3_orchestrator import Stage3Orchestrator
from src.core.llm_router_final import FinalLLMRouter
from src.utils.config_loader import get_config

load_dotenv()
logger = get_logger(__name__)

class LegendaryProductionTest:
    def __init__(self):
        self.start_time = time.time()
        self.cost_tracker = CostTracker()
        self.config = get_config()
        self.llm = FinalLLMRouter(self.config.get_all())
        self.results = {
            "stage1": None,
            "stage2": None,
            "stage3": None,
            "total_cost": 0.0,
            "total_time": 0.0,
            "success": False
        }

    def print_header(self, title: str):
        print("\n" + "=" * 100)
        print(f"  {title}")
        print("=" * 100 + "\n")

    def print_stage(self, stage: int, name: str):
        print("\n" + "-" * 100)
        print(f"  STAGE {stage}: {name}")
        print("-" * 100 + "\n")

    def test_stage1_feature_selection(self, project_description: str) -> dict:
        """Test Stage 1: Feature Selection"""
        self.print_stage(1, "FEATURE SELECTION (Exploit-Explore with Pinecone)")

        try:
            # Parse user input
            print("[1/4] Processing user input...")
            processor = UserInputProcessor(self.llm, self.config)
            user_request = processor.process_natural_language(project_description)
            print(f"      Project Type: {user_request.project_type}")
            print(f"      Domain: {user_request.domain}")
            print(f"      Core Features: {len(user_request.core_features)}")
            print()

            # Initialize feature selection
            print("[2/4] Initializing feature selection loop...")
            feature_loop = FeatureSelectionLoop(
                llm=self.llm,
                config=self.config,
                user_request=user_request
            )
            print("      Feature selection initialized")
            print()

            # Run feature selection (limit iterations for testing)
            print("[3/4] Running exploit-explore loop...")
            print("      (Using 5 iterations for testing)")
            selected_features = feature_loop.run(max_iterations=5)
            print(f"      Selected {len(selected_features)} features")
            print()

            # Show top features
            print("[4/4] Top selected features:")
            for i, feature in enumerate(selected_features[:5], 1):
                print(f"      {i}. {feature.name} ({feature.complexity.value})")
                print(f"         Domain: {feature.domain}/{feature.subdomain}")
            print()

            result = {
                "success": True,
                "user_request": user_request,
                "features_count": len(selected_features),
                "features": selected_features,
                "cost": self.llm.get_stats()["total_cost"]
            }

            print(f"[SUCCESS] Stage 1 completed!")
            print(f"          Features: {len(selected_features)}")
            print(f"          Cost: ${result['cost']:.4f}")
            print()

            return result

        except Exception as e:
            logger.error(f"Stage 1 failed: {e}", exc_info=True)
            print(f"[FAILED] Stage 1 error: {e}")
            return {"success": False, "error": str(e)}

    def test_stage2_architecture(self, user_request, selected_features) -> dict:
        """Test Stage 2: Architecture Design"""
        self.print_stage(2, "ARCHITECTURE DESIGN (RPG Construction)")

        try:
            print("[1/3] Initializing Stage 2 orchestrator...")
            stage2 = Stage2Orchestrator(
                llm=self.llm,
                config=self.config
            )
            print("      Orchestrator initialized")
            print()

            print("[2/3] Building Repository Planning Graph...")
            rpg = stage2.build_rpg(
                user_request=user_request,
                selected_features=selected_features
            )
            print(f"      Nodes: {len(rpg.graph.nodes)}")
            print(f"      Edges: {len(rpg.graph.edges)}")
            print()

            print("[3/3] Validating graph structure...")
            is_valid = rpg.validate()
            print(f"      Valid: {is_valid}")
            print()

            result = {
                "success": True,
                "rpg": rpg,
                "nodes_count": len(rpg.graph.nodes),
                "edges_count": len(rpg.graph.edges),
                "cost": self.llm.get_stats()["total_cost"]
            }

            print(f"[SUCCESS] Stage 2 completed!")
            print(f"          Nodes: {result['nodes_count']}")
            print(f"          Edges: {result['edges_count']}")
            print(f"          Cost: ${result['cost']:.4f}")
            print()

            return result

        except Exception as e:
            logger.error(f"Stage 2 failed: {e}", exc_info=True)
            print(f"[FAILED] Stage 2 error: {e}")
            return {"success": False, "error": str(e)}

    def test_stage3_code_generation(self, rpg, output_dir: str) -> dict:
        """Test Stage 3: Code Generation"""
        self.print_stage(3, "CODE GENERATION (TDD + Implementation)")

        try:
            print("[1/3] Initializing Stage 3 orchestrator...")
            stage3 = Stage3Orchestrator(
                llm=self.llm,
                config=self.config
            )
            print("      Orchestrator initialized")
            print()

            print("[2/3] Generating code repository...")
            print("      (This may take a few minutes)")
            output_path = stage3.generate_repository(
                rpg=rpg,
                output_dir=output_dir
            )
            print(f"      Generated: {output_path}")
            print()

            print("[3/3] Validating generated code...")
            # Check if files were created
            output_path_obj = Path(output_path)
            if output_path_obj.exists():
                files = list(output_path_obj.rglob("*.py"))
                print(f"      Python files: {len(files)}")
                print(f"      Output: {output_path}")
            else:
                print(f"      Warning: Output directory not found")
            print()

            result = {
                "success": True,
                "output_path": output_path,
                "files_count": len(files) if output_path_obj.exists() else 0,
                "cost": self.llm.get_stats()["total_cost"]
            }

            print(f"[SUCCESS] Stage 3 completed!")
            print(f"          Files: {result['files_count']}")
            print(f"          Output: {output_path}")
            print(f"          Cost: ${result['cost']:.4f}")
            print()

            return result

        except Exception as e:
            logger.error(f"Stage 3 failed: {e}", exc_info=True)
            print(f"[FAILED] Stage 3 error: {e}")
            return {"success": False, "error": str(e)}

    def run_full_pipeline_test(self, project_description: str, output_dir: str):
        """Run complete end-to-end test"""
        self.print_header("LEGENDARY PRODUCTION TEST - FULL PIPELINE")

        print(f"Project: {project_description}")
        print(f"Output: {output_dir}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Stage 1: Feature Selection
        stage1_result = self.test_stage1_feature_selection(project_description)
        self.results["stage1"] = stage1_result

        if not stage1_result["success"]:
            print("[FAILED] Pipeline stopped at Stage 1")
            return self.print_final_results()

        # Stage 2: Architecture Design
        stage2_result = self.test_stage2_architecture(
            stage1_result["user_request"],
            stage1_result["features"]
        )
        self.results["stage2"] = stage2_result

        if not stage2_result["success"]:
            print("[FAILED] Pipeline stopped at Stage 2")
            return self.print_final_results()

        # Stage 3: Code Generation
        stage3_result = self.test_stage3_code_generation(
            stage2_result["rpg"],
            output_dir
        )
        self.results["stage3"] = stage3_result

        if not stage3_result["success"]:
            print("[FAILED] Pipeline stopped at Stage 3")
            return self.print_final_results()

        # Success!
        self.results["success"] = True
        self.results["total_time"] = time.time() - self.start_time
        self.results["total_cost"] = self.llm.get_stats()["total_cost"]

        return self.print_final_results()

    def print_final_results(self):
        """Print final test results"""
        self.print_header("FINAL RESULTS")

        print("STAGE RESULTS:")
        print(f"  Stage 1 (Feature Selection): {'PASS' if self.results['stage1'] and self.results['stage1']['success'] else 'FAIL'}")
        print(f"  Stage 2 (Architecture):       {'PASS' if self.results['stage2'] and self.results['stage2']['success'] else 'FAIL'}")
        print(f"  Stage 3 (Code Generation):    {'PASS' if self.results['stage3'] and self.results['stage3']['success'] else 'FAIL'}")
        print()

        print("METRICS:")
        if self.results["stage1"] and self.results["stage1"]["success"]:
            print(f"  Features Selected: {self.results['stage1']['features_count']}")
        if self.results["stage2"] and self.results["stage2"]["success"]:
            print(f"  Graph Nodes: {self.results['stage2']['nodes_count']}")
            print(f"  Graph Edges: {self.results['stage2']['edges_count']}")
        if self.results["stage3"] and self.results["stage3"]["success"]:
            print(f"  Files Generated: {self.results['stage3']['files_count']}")
        print()

        print("COSTS:")
        total_cost = self.llm.get_stats()["total_cost"]
        api_calls = self.llm.get_stats()["api_calls"]
        print(f"  Total Cost: ${total_cost:.4f}")
        print(f"  API Calls: {api_calls}")
        print(f"  Cost per Call: ${total_cost/api_calls:.4f}" if api_calls > 0 else "  Cost per Call: $0.0000")
        print()

        print("TIME:")
        total_time = time.time() - self.start_time
        print(f"  Total Time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        print()

        if self.results["success"]:
            print("=" * 100)
            print("SUCCESS! BLUEPRINT IS PRODUCTION READY!")
            print("=" * 100)
        else:
            print("=" * 100)
            print("PIPELINE INCOMPLETE - Review errors above")
            print("=" * 100)

        return self.results


def main():
    """Run the legendary production test"""

    # Test project
    project = "Build a simple REST API for managing blog posts with CRUD operations"
    output_dir = "output/test_blog_api"

    # Run test
    tester = LegendaryProductionTest()
    results = tester.run_full_pipeline_test(project, output_dir)

    # Save results
    results_file = Path("test_results_legendary.json")
    with open(results_file, "w") as f:
        json.dump({
            "success": results["success"],
            "timestamp": datetime.now().isoformat(),
            "project": project,
            "stage1_features": results["stage1"]["features_count"] if results["stage1"] and results["stage1"]["success"] else 0,
            "stage2_nodes": results["stage2"]["nodes_count"] if results["stage2"] and results["stage2"]["success"] else 0,
            "stage3_files": results["stage3"]["files_count"] if results["stage3"] and results["stage3"]["success"] else 0,
            "total_cost": results["total_cost"],
            "total_time": results["total_time"]
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
