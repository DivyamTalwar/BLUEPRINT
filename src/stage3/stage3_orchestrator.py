import time
from typing import Dict, Any, Optional
from pathlib import Path

from src.core.rpg import RepositoryPlanningGraph
from src.core.llm_router_final import FinalLLMRouter
from src.stage3.topological_traversal import TopologicalTraversal
from src.stage3.tdd_engine import TDDEngine
from src.stage3.repository_builder import RepositoryBuilder
from src.utils.docker_runner import DockerRunner
from src.utils.logger import StructuredLogger
from src.utils.cost_tracker import CostTracker

logger = StructuredLogger("stage3_orchestrator")


class Stage3Orchestrator:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = FinalLLMRouter(config)
        self.docker = DockerRunner(config.get("docker", {}))
        self.cost_tracker = CostTracker()
        self.logger = logger

        # Initialize Stage 3 components with stage3 config
        stage3_config = config.get("stage3", {})
        stage3_config["docker"] = config.get("docker", {})  # Include docker config
        self.tdd_engine = TDDEngine(self.llm, self.docker, stage3_config)

    def generate_repository(
        self,
        rpg: RepositoryPlanningGraph,
        output_dir: str = "./generated_repo",
        checkpoint_interval: int = 10
    ) -> bool:
        start_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info("STAGE 3: CODE GENERATION")
        self.logger.info("=" * 60)

        # Check Docker availability (warning only)
        if not self.docker.is_docker_available():
            self.logger.warning("Docker is not available. TDD will be limited - tests won't be executed.")

        # Step 1: Initialize traversal engine
        self.logger.info("Initializing topological traversal...")
        traversal = TopologicalTraversal(rpg)
        execution_order = traversal.get_execution_order()

        self.logger.info(f"Generation order computed: {len(execution_order)} nodes")

        # Step 2: Generate code for all nodes
        self.logger.info("Starting code generation with TDD...")
        generated_code = self._generate_all_code(rpg, traversal, execution_order, checkpoint_interval)

        # Step 3: Build repository
        self.logger.info("Building repository structure...")
        builder = RepositoryBuilder(output_dir, self.config)
        repo_path = builder.build(rpg, generated_code)

        # Step 4: Run integration tests
        self.logger.info("Running integration tests...")
        integration_success = self._run_integration_tests(repo_path)

        # Summary
        elapsed = time.time() - start_time
        self._print_summary(traversal, generated_code, integration_success, elapsed, repo_path)

        # Return success based on generation and tests
        progress = traversal.get_progress()
        success = progress["failed"] == 0 and integration_success

        return success

    def _generate_all_code(
        self,
        rpg: RepositoryPlanningGraph,
        traversal: TopologicalTraversal,
        execution_order: list,
        checkpoint_interval: int
    ) -> Dict[str, Dict]:
        generated_code = {}
        checkpoint_count = 0

        for i, node_id in enumerate(execution_order):
            node_data = rpg.graph.nodes[node_id]
            node_name = node_data.get("name", node_id)

            # Progress
            progress = traversal.get_progress()
            self.logger.info(f"[{i+1}/{len(execution_order)}] Generating: {node_name} "
                          f"(Progress: {progress['progress_percent']:.1f}%)")

            # Mark in progress
            traversal.mark_in_progress(node_id)

            # Generate with TDD
            try:
                has_code, result = self.tdd_engine.generate(rpg, node_id)

                # CRITICAL FIX: Save code if it was generated, regardless of validation status
                if has_code and result.get("implementation"):
                    generated_code[node_id] = result

                    # Update node status in RPG
                    status = result.get("status", "generated")
                    validation_method = result.get("validation_method", "none")
                    errors = result.get("errors", [])

                    rpg.update_node(
                        node_id,
                        implementation=result.get("implementation"),
                        test_code=result.get("test_code"),
                        validation_method=validation_method,
                        validation_errors=errors,
                        generation_attempts=result.get("attempts", 0),
                        status=status
                    )

                    # Mark in traversal based on status
                    if status == "validated":
                        traversal.mark_completed(node_id)
                        self.logger.info(f"[✓] Validated successfully in {result.get('attempts', 1)} attempt(s)")
                    elif status in ["generated", "syntax_valid"]:
                        traversal.mark_completed(node_id)  # Still mark as completed since we have code
                        self.logger.info(f"[⚠] Code generated but not validated ({validation_method})")
                    else:
                        traversal.mark_failed(node_id, result.get("error", "Unknown error"))
                        self.logger.warning(f"[✗] Generation failed")
                else:
                    # No code was generated at all
                    traversal.mark_failed(node_id, result.get("error", "Unknown error"))
                    self.logger.error(f"[✗] Failed to generate code")

            except Exception as e:
                traversal.mark_failed(node_id, str(e))
                self.logger.error(f"[✗] Exception during generation", error=str(e))

            # Checkpoint
            checkpoint_count += 1
            if checkpoint_count >= checkpoint_interval:
                traversal.save_checkpoint("output/stage3_checkpoint.json")
                checkpoint_count = 0

        return generated_code

    def _run_integration_tests(self, repo_path: str) -> bool:
        try:
            self.logger.info("Running pytest on generated repository...")

            success, stdout, stderr = self.docker.execute_tests(
                code_dir=repo_path,
                test_command="pytest tests/ -v",
                requirements=self._read_requirements(repo_path)
            )

            if success:
                self.logger.info("[OK] All integration tests passed")
            else:
                self.logger.warning("[FAIL] Some integration tests failed")
                self.logger.debug("Test output:", output=stdout + stderr)

            return success

        except Exception as e:
            self.logger.error("Failed to run integration tests", error=str(e))
            return False

    def _read_requirements(self, repo_path: str) -> list:
        try:
            req_file = Path(repo_path) / "requirements.txt"
            if req_file.exists():
                return req_file.read_text().strip().split('\n')
        except:
            pass
        return []

    def _print_summary(self, traversal, generated_code: Dict, integration_success: bool,
                      elapsed_time: float, repo_path: str):
        """Print Stage 3 completion summary."""
        progress = traversal.get_progress()

        # Calculate statistics
        total = progress["total"]
        completed = progress["completed"]
        failed = progress["failed"]

        # Count by validation status
        validated = sum(1 for code in generated_code.values() if code.get("status") == "validated")
        syntax_valid = sum(1 for code in generated_code.values() if code.get("status") == "syntax_valid")
        generated_only = sum(1 for code in generated_code.values() if code.get("status") == "generated")

        # Success rate based on code generation (not just validation)
        code_generated = validated + syntax_valid + generated_only
        generation_rate = (code_generated / total * 100) if total > 0 else 0
        validation_rate = (validated / total * 100) if total > 0 else 0

        # Count lines of code
        total_lines = sum(
            len(code.get("implementation", "").split('\n'))
            for code in generated_code.values()
        )

        total_test_lines = sum(
            len(code.get("test_code", "").split('\n'))
            for code in generated_code.values()
        )

        # Count validation methods
        docker_validated = sum(1 for code in generated_code.values() if code.get("validation_method") == "docker")
        static_validated = sum(1 for code in generated_code.values() if code.get("validation_method") == "static")
        no_validation = sum(1 for code in generated_code.values() if code.get("validation_method") == "none")

        self.logger.info("=" * 70)
        self.logger.info("STAGE 3 COMPLETE")
        self.logger.info("=" * 70)
        self.logger.info(f"Execution time: {elapsed_time/60:.1f} minutes")
        self.logger.info("")
        self.logger.info("Code Generation:")
        self.logger.info(f"  - Total nodes: {total}")
        self.logger.info(f"  - Code generated: {code_generated} ({generation_rate:.1f}%)")
        self.logger.info(f"  - Failed: {failed}")
        self.logger.info("")
        self.logger.info("Validation Status:")
        self.logger.info(f"  - ✓ Validated (Docker tests passed): {validated} ({validation_rate:.1f}%)")
        self.logger.info(f"  - ✓ Syntax Valid (static analysis): {syntax_valid}")
        self.logger.info(f"  - ⚠ Generated (not validated): {generated_only}")
        self.logger.info("")
        self.logger.info("Validation Methods:")
        self.logger.info(f"  - Docker TDD: {docker_validated}")
        self.logger.info(f"  - Static Analysis: {static_validated}")
        self.logger.info(f"  - None (skipped): {no_validation}")
        self.logger.info("")
        self.logger.info("Code Metrics:")
        self.logger.info(f"  - Implementation LOC: {total_lines:,}")
        self.logger.info(f"  - Test LOC: {total_test_lines:,}")
        self.logger.info(f"  - Total LOC: {total_lines + total_test_lines:,}")
        self.logger.info("")
        self.logger.info("Repository:")
        self.logger.info(f"  - Location: {repo_path}")
        self.logger.info("")
        self.logger.info("Next steps:")
        self.logger.info(f"  1. cd {repo_path}")
        self.logger.info("  2. pip install -r requirements.txt")
        self.logger.info("  3. pytest tests/")
        self.logger.info("=" * 70)

    def resume_from_checkpoint(self, checkpoint_path: str, complete_rpg: RepositoryPlanningGraph,output_dir: str = "./generated_repo") -> str:

        self.logger.info("Resuming from checkpoint...")

        traversal = TopologicalTraversal(complete_rpg)
        traversal.load_checkpoint(checkpoint_path)

        # Get remaining nodes
        execution_order = traversal.get_execution_order()
        remaining = [n for n in execution_order if traversal.status.get(n) == "pending"]

        self.logger.info(f"Resuming with {len(remaining)} remaining nodes")

        # Continue generation
        generated_code = self._generate_all_code(complete_rpg, traversal, remaining, checkpoint_interval=10)

        # Build repository
        builder = RepositoryBuilder(output_dir, self.config)
        repo_path = builder.build(complete_rpg, generated_code)

        return repo_path
