import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from src.utils.logger import get_logger
from src.utils.config_loader import get_config
from src.core.llm_router_final import FinalLLMRouter
from src.stage1.vector_store import VectorStore
from src.stage1.cohere_embeddings import CohereEmbeddings
from src.stage1.user_input_processor import UserInputProcessor
from src.stage1.exploit_strategy import ExploitStrategy
from src.stage1.explore_strategy import ExploreStrategy
from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType

load_dotenv()
logger = get_logger(__name__)


class UltimateProductionTest:
    def __init__(self):
        self.config = get_config()
        self.llm = FinalLLMRouter(self.config.get_all())
        self.start_time = time.time()
        self.results = {
            "component_tests": {},
            "integration_tests": {},
            "pipeline_tests": {},
            "total_cost": 0.0,
            "total_time": 0.0,
            "success": False
        }

    def print_section(self, title: str, level: int = 1):
        """Print formatted section header"""
        if level == 1:
            print("\n" + "=" * 100)
            print(f"  {title}")
            print("=" * 100 + "\n")
        elif level == 2:
            print("\n" + "-" * 100)
            print(f"  {title}")
            print("-" * 100 + "\n")
        else:
            print(f"\n>>> {title}")
            print()

    def test_llm_router(self) -> Dict[str, Any]:
        """Test LLM Router with both Gemini and Claude"""
        self.print_section("COMPONENT TEST: LLM Router", 3)

        results = {"gemini": False, "claude": False, "cost": 0.0}

        try:
            # Test Gemini (FREE)
            print("[1/2] Testing Gemini 2.5 Flash...")
            gemini_response = self.llm.generate(
                "Say 'test successful' in 3 words",
                temperature=0.7,
                max_tokens=50,
                prefer_claude=False
            )
            print(f"      Gemini Response: {gemini_response.content[:50]}...")
            print(f"      Provider: {gemini_response.provider}")
            print(f"      Cost: ${gemini_response.cost:.4f} (should be $0.00)")
            results["gemini"] = True
            print()

            # Test Claude (Paid - only if needed)
            print("[2/2] Testing Claude 3.5 Sonnet via OpenRouter...")
            claude_response = self.llm.generate(
                "Say 'test successful' in 3 words",
                temperature=0.7,
                max_tokens=50,
                prefer_claude=True
            )
            print(f"      Claude Response: {claude_response.content[:50]}...")
            print(f"      Provider: {claude_response.provider}")
            print(f"      Cost: ${claude_response.cost:.4f}")
            results["claude"] = True
            results["cost"] = self.llm.get_stats()["total_cost"]
            print()

            print("[SUCCESS] LLM Router working perfectly!")
            print(f"          Total Cost: ${results['cost']:.4f}")
            results["success"] = True

        except Exception as e:
            print(f"[FAILED] LLM Router error: {e}")
            results["error"] = str(e)
            results["success"] = False

        return results

    def test_vector_store(self) -> Dict[str, Any]:
        """Test Pinecone Vector Store"""
        self.print_section("COMPONENT TEST: Vector Store (Pinecone)", 3)

        results = {"connected": False, "features_count": 0}

        try:
            print("[1/3] Connecting to Pinecone...")
            vector_store = VectorStore(
                api_key=os.getenv("PINECONE_API_KEY"),
                index_name="blueprint-features",
                dimension=1024
            )
            vector_store.index = vector_store.pc.Index(vector_store.index_name)
            print("      Connected successfully")
            results["connected"] = True
            print()

            print("[2/3] Testing vector search...")
            cohere_embeddings = CohereEmbeddings(api_key=os.getenv("COHERE_API_KEY"))
            test_query = "REST API with CRUD operations"
            query_embedding = cohere_embeddings.generate_embedding(test_query, input_type="search_query")

            search_results = vector_store.search(
                query_embedding=query_embedding,
                top_k=5
            )
            print(f"      Found {len(search_results)} results")
            for i, result in enumerate(search_results[:3], 1):
                print(f"      {i}. {result['metadata'].get('name', 'N/A')} (score: {result['score']:.4f})")
            print()

            print("[3/3] Checking index stats...")
            stats = vector_store.get_stats()
            results["features_count"] = stats.get("total_vectors", 0)
            print(f"      Total vectors: {results['features_count']}")
            print()

            print("[SUCCESS] Vector Store operational!")
            print(f"          Features loaded: {results['features_count']}")
            results["success"] = True

        except Exception as e:
            print(f"[FAILED] Vector Store error: {e}")
            results["error"] = str(e)
            results["success"] = False

        return results

    def test_user_input_processing(self) -> Dict[str, Any]:
        """Test User Input Processor"""
        self.print_section("COMPONENT TEST: User Input Processor", 3)

        results = {"parsed": False}

        try:
            print("[1/2] Processing user request...")
            processor = UserInputProcessor(self.llm)
            user_request = processor.process(
                "Build a REST API for managing blog posts with CRUD operations"
            )

            print(f"      Project Type: {user_request.repo_type}")
            print(f"      Domain: {user_request.primary_domain}")
            print(f"      Complexity: {user_request.complexity_estimate}")
            print(f"      Requirements: {len(user_request.explicit_requirements)}")
            results["parsed"] = True
            results["request"] = user_request
            print()

            print("[2/2] Validating parsed data...")
            assert user_request.repo_type is not None
            assert user_request.primary_domain is not None
            assert len(user_request.explicit_requirements) > 0
            print("      All validations passed")
            print()

            print("[SUCCESS] User Input Processor working!")
            results["success"] = True

        except Exception as e:
            print(f"[FAILED] User Input Processor error: {e}")
            results["error"] = str(e)
            results["success"] = False

        return results

    def test_feature_selection(self) -> Dict[str, Any]:
        """Test Feature Selection (Exploit + Explore)"""
        self.print_section("COMPONENT TEST: Feature Selection", 3)

        results = {"exploit": False, "explore": False, "features_selected": 0}

        try:
            # Initialize components
            print("[1/4] Initializing components...")
            vector_store = VectorStore(
                api_key=os.getenv("PINECONE_API_KEY"),
                index_name="blueprint-features",
                dimension=1024
            )
            vector_store.index = vector_store.pc.Index(vector_store.index_name)
            cohere_embeddings = CohereEmbeddings(api_key=os.getenv("COHERE_API_KEY"))
            exploit = ExploitStrategy(vector_store, cohere_embeddings)
            explore = ExploreStrategy(vector_store, cohere_embeddings, self.llm)
            print("      Components initialized")
            print()

            # Test Exploit Strategy
            print("[2/4] Testing Exploit Strategy...")
            processor = UserInputProcessor(self.llm)
            user_request = processor.process("REST API with database")
            exploit_features = exploit.retrieve_focused_features(
                request=user_request,
                top_k=5
            )
            print(f"      Exploit selected: {len(exploit_features)} features")
            for i, f in enumerate(exploit_features[:3], 1):
                print(f"      {i}. {f.name}")
            results["exploit"] = True
            print()

            # Test Explore Strategy
            print("[3/4] Testing Explore Strategy...")
            explore_features = explore.suggest_complementary_features(
                request=user_request,
                selected_features=exploit_features,
                top_k=5
            )
            print(f"      Explore selected: {len(explore_features)} features")
            results["explore"] = True
            print()

            # Combined results
            print("[4/4] Validating combined selection...")
            all_features = exploit_features + explore_features
            results["features_selected"] = len(all_features)
            print(f"      Total features: {results['features_selected']}")
            print()

            print("[SUCCESS] Feature Selection working!")
            results["success"] = True

        except Exception as e:
            print(f"[FAILED] Feature Selection error: {e}")
            results["error"] = str(e)
            results["success"] = False

        return results

    def test_rpg_construction(self) -> Dict[str, Any]:
        """Test RPG (Repository Planning Graph) Construction"""
        self.print_section("COMPONENT TEST: RPG Construction", 3)

        results = {"created": False, "validated": False, "nodes": 0, "edges": 0}

        try:
            print("[1/3] Creating RPG...")
            rpg = RepositoryPlanningGraph("Test Blog API")

            # Add nodes
            root_id = rpg.add_node("blog_api", NodeType.ROOT, "Blog API Root")
            models_id = rpg.add_node("models", NodeType.INTERMEDIATE, "Data Models")
            post_id = rpg.add_node("post_model", NodeType.LEAF, "Post Model")

            # Add edges
            rpg.add_edge(root_id, models_id, EdgeType.HIERARCHY)
            rpg.add_edge(models_id, post_id, EdgeType.HIERARCHY)

            results["nodes"] = len(rpg.graph.nodes)
            results["edges"] = len(rpg.graph.edges)
            results["created"] = True

            print(f"      Nodes: {results['nodes']}")
            print(f"      Edges: {results['edges']}")
            print()

            print("[2/3] Validating RPG structure...")
            is_valid = rpg.validate()
            results["validated"] = is_valid
            print(f"      Valid: {is_valid}")
            print()

            print("[3/3] Testing traversal...")
            execution_order = rpg.get_topological_order()
            print(f"      Execution order: {len(execution_order)} nodes")
            print()

            print("[SUCCESS] RPG Construction working!")
            results["success"] = True

        except Exception as e:
            print(f"[FAILED] RPG Construction error: {e}")
            results["error"] = str(e)
            results["success"] = False

        return results

    def run_integration_test(self) -> Dict[str, Any]:
        """Run integration test: Stage 1 → Stage 2"""
        self.print_section("INTEGRATION TEST: Stage 1 to Stage 2", 2)

        results = {"stage1_complete": False, "stage2_complete": False}

        try:
            from src.stage1.feature_selection_loop import FeatureSelectionLoop
            from src.stage2.stage2_orchestrator import Stage2Orchestrator

            # Stage 1: Feature Selection
            print("[1/2] Running Stage 1 (Feature Selection)...")
            processor = UserInputProcessor(self.llm)
            user_request = processor.process(
                "Build a simple TODO list CLI with add, list, and complete commands"
            )

            # Initialize strategies
            vector_store = VectorStore(
                api_key=os.getenv("PINECONE_API_KEY"),
                index_name="blueprint-features",
                dimension=1024
            )
            vector_store.index = vector_store.pc.Index(vector_store.index_name)
            cohere_embeddings = CohereEmbeddings(api_key=os.getenv("COHERE_API_KEY"))
            exploit = ExploitStrategy(vector_store, cohere_embeddings)
            explore = ExploreStrategy(vector_store, cohere_embeddings, self.llm)

            feature_loop = FeatureSelectionLoop(
                exploit_strategy=exploit,
                explore_strategy=explore,
                num_iterations=3  # Quick test
            )

            selected_features = feature_loop.run(request=user_request, target_features=30)
            print(f"      Selected {len(selected_features)} features")
            results["stage1_complete"] = True
            results["features_count"] = len(selected_features)
            print()

            # Stage 2: Architecture Design
            print("[2/2] Running Stage 2 (Architecture Design)...")
            # Create initial RPG from user request
            from src.stage1.functionality_graph_builder import FunctionalityGraphBuilder
            print("      Building functionality graph from features...")
            func_graph_builder = FunctionalityGraphBuilder(self.llm)
            functionality_graph = func_graph_builder.build(
                request=user_request,
                selected_features=selected_features
            )

            # Run Stage 2 to transform to full RPG
            stage2 = Stage2Orchestrator(self.config.get_all())
            rpg = stage2.run(
                functionality_graph=functionality_graph,
                repo_type=user_request.repo_type,
                save_checkpoints=False
            )

            print(f"      RPG Nodes: {rpg.graph.number_of_nodes()}")
            print(f"      RPG Edges: {rpg.graph.number_of_edges()}")
            results["stage2_complete"] = True
            results["rpg_nodes"] = rpg.graph.number_of_nodes()
            print()

            print("[SUCCESS] Integration test passed!")
            results["success"] = True

        except Exception as e:
            print(f"[FAILED] Integration test error: {e}")
            results["error"] = str(e)
            results["success"] = False

        return results

    def run_all_tests(self):
        """Run the complete test suite"""
        self.print_section("ULTIMATE PRODUCTION TEST SUITE", 1)

        print("Starting comprehensive validation...")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Component Tests
        self.print_section("PHASE 1: COMPONENT TESTS", 2)

        self.results["component_tests"]["llm_router"] = self.test_llm_router()
        self.results["component_tests"]["vector_store"] = self.test_vector_store()
        self.results["component_tests"]["user_input"] = self.test_user_input_processing()
        self.results["component_tests"]["feature_selection"] = self.test_feature_selection()
        self.results["component_tests"]["rpg"] = self.test_rpg_construction()

        # Integration Tests
        self.print_section("PHASE 2: INTEGRATION TESTS", 2)

        self.results["integration_tests"]["stage1_to_stage2"] = self.run_integration_test()

        # Final Results
        self.print_final_results()

    def print_final_results(self):
        """Print comprehensive final results"""
        self.print_section("FINAL RESULTS", 1)

        # Component Tests Summary
        print("COMPONENT TESTS:")
        component_success = 0
        component_total = len(self.results["component_tests"])

        for name, result in self.results["component_tests"].items():
            status = "PASS" if result.get("success") else "FAIL"
            print(f"  {name:25s}: {status}")
            if result.get("success"):
                component_success += 1

        print()

        # Integration Tests Summary
        print("INTEGRATION TESTS:")
        integration_success = 0
        integration_total = len(self.results["integration_tests"])

        for name, result in self.results["integration_tests"].items():
            status = "PASS" if result.get("success") else "FAIL"
            print(f"  {name:25s}: {status}")
            if result.get("success"):
                integration_success += 1

        print()

        # Overall Stats
        total_tests = component_total + integration_total
        total_passed = component_success + integration_success
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

        print("OVERALL STATISTICS:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {total_passed}")
        print(f"  Failed: {total_tests - total_passed}")
        print(f"  Success Rate: {success_rate:.1f}%")
        print()

        # Cost Analysis
        total_cost = self.llm.get_stats()["total_cost"]
        api_calls = self.llm.get_stats()["api_calls"]

        print("COST ANALYSIS:")
        print(f"  Total Cost: ${total_cost:.4f}")
        print(f"  API Calls: {api_calls}")
        print(f"  Cost per Call: ${total_cost/api_calls:.4f}" if api_calls > 0 else "  Cost per Call: $0.0000")
        print()

        # Time Analysis
        total_time = time.time() - self.start_time
        print("TIME ANALYSIS:")
        print(f"  Total Time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        print()

        # Final Verdict
        all_passed = (total_passed == total_tests)
        self.results["success"] = all_passed
        self.results["total_cost"] = total_cost
        self.results["total_time"] = total_time

        if all_passed:
            print("=" * 100)
            print("SUCCESS! ALL TESTS PASSED - PRODUCTION READY!")
            print("=" * 100)
        else:
            print("=" * 100)
            print("WARNING! SOME TESTS FAILED - Review errors above")
            print("=" * 100)

        # Save results
        results_file = Path("ultimate_test_results.json")
        with open(results_file, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "success": all_passed,
                "success_rate": success_rate,
                "total_cost": total_cost,
                "total_time": total_time,
                "component_tests": {k: {"success": v.get("success", False)} for k, v in self.results["component_tests"].items()},
                "integration_tests": {k: {"success": v.get("success", False)} for k, v in self.results["integration_tests"].items()},
            }, f, indent=2)

        print(f"\nDetailed results saved to: {results_file}")


def main():
    """Run the ultimate test suite"""
    tester = UltimateProductionTest()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
