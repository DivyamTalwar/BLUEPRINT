import time
from typing import Dict, Any, Optional
from src.core.rpg import RepositoryPlanningGraph
from src.core.llm_router_final import FinalLLMRouter
from src.core.graph_persistence import GraphPersistence
from src.stage2.file_structure_encoder import FileStructureEncoder
from src.stage2.data_flow_encoder import DataFlowEncoder
from src.stage2.base_class_abstraction import BaseClassAbstraction
from src.stage2.interface_designer import InterfaceDesigner
from src.utils.logger import StructuredLogger
from src.utils.cost_tracker import CostTracker

logger = StructuredLogger("stage2_orchestrator")


class Stage2Orchestrator:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = FinalLLMRouter(config)
        self.cost_tracker = CostTracker()
        self.logger = logger

        # Initialize Stage 2 components
        self.file_encoder = FileStructureEncoder(self.llm, config)
        self.data_flow_encoder = DataFlowEncoder(self.llm, config)
        self.base_abstractor = BaseClassAbstraction(self.llm, config)
        self.interface_designer = InterfaceDesigner(self.llm, config)

        self.persistence = GraphPersistence()

    def run(
        self,
        functionality_graph: RepositoryPlanningGraph,
        repo_type: str = "library",
        save_checkpoints: bool = True
    ) -> RepositoryPlanningGraph:
        start_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info("STAGE 2: IMPLEMENTATION-LEVEL CONSTRUCTION")
        self.logger.info("=" * 60)

        rpg = functionality_graph

        # Step 1: File Structure Encoding
        self.logger.info("Step 1/4: Encoding file structure...")
        rpg = self.file_encoder.encode(rpg, repo_type)

        if save_checkpoints:
            self.persistence.save_json(rpg, "output/stage2_checkpoint_1_file_structure.json")

        # Step 2: Data Flow Encoding
        self.logger.info("Step 2/4: Encoding data flows...")
        rpg = self.data_flow_encoder.encode(rpg)

        if save_checkpoints:
            self.persistence.save_json(rpg, "output/stage2_checkpoint_2_data_flows.json")

        # Step 3: Base Class Abstraction
        self.logger.info("Step 3/4: Abstracting base classes...")
        rpg = self.base_abstractor.abstract(rpg)

        if save_checkpoints:
            self.persistence.save_json(rpg, "output/stage2_checkpoint_3_base_classes.json")

        # Step 4: Interface Design
        self.logger.info("Step 4/4: Designing function signatures...")
        rpg = self.interface_designer.design(rpg)

        # Final save
        if save_checkpoints:
            self.persistence.save_json(rpg, "output/stage2_complete_rpg.json")

        # Summary
        elapsed = time.time() - start_time
        self._print_summary(rpg, elapsed)

        return rpg

    def _print_summary(self, rpg: RepositoryPlanningGraph, elapsed_time: float):
        # Count nodes by type
        root_count = sum(1 for _, d in rpg.graph.nodes(data=True) if d.get("type") == "root")
        intermediate_count = sum(1 for _, d in rpg.graph.nodes(data=True) if d.get("type") == "intermediate")
        leaf_count = sum(1 for _, d in rpg.graph.nodes(data=True) if d.get("type") == "leaf")

        # Count edges by type
        hierarchy_edges = sum(1 for _, _, d in rpg.graph.edges(data=True) if d.get("type") == "hierarchy")
        data_flow_edges = sum(1 for _, _, d in rpg.graph.edges(data=True) if d.get("type") == "data_flow")

        # Count signatures
        signatures_count = sum(1 for _, d in rpg.graph.nodes(data=True) if d.get("signature"))

        # Count base classes
        base_classes = sum(1 for _, d in rpg.graph.nodes(data=True) if d.get("is_base_class"))

        self.logger.info("=" * 60)
        self.logger.info("STAGE 2 COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"Execution time: {elapsed_time:.1f}s")
        self.logger.info("")
        self.logger.info("Graph Structure:")
        self.logger.info(f"  - Root nodes (modules): {root_count}")
        self.logger.info(f"  - Intermediate nodes (files): {intermediate_count}")
        self.logger.info(f"  - Leaf nodes (functions/classes): {leaf_count}")
        self.logger.info(f"  - Hierarchy edges: {hierarchy_edges}")
        self.logger.info(f"  - Data flow edges: {data_flow_edges}")
        self.logger.info("")
        self.logger.info("Implementation Details:")
        self.logger.info(f"  - Base classes created: {base_classes}")
        self.logger.info(f"  - Function signatures designed: {signatures_count}")
        self.logger.info("")
        self.logger.info("Outputs saved:")
        self.logger.info("  - output/stage2_complete_rpg.json")
        self.logger.info("  - output/stage2_checkpoint_*.json")
        self.logger.info("=" * 60)

    def validate_rpg(self, rpg: RepositoryPlanningGraph) -> Dict[str, Any]:
        issues = []
        warnings = []

        # Check 1: All leaf nodes have signatures
        leaves_without_sig = [
            n for n, d in rpg.graph.nodes(data=True)
            if d.get("type") == "leaf" and not d.get("signature") and not d.get("is_base_class")
        ]

        if leaves_without_sig:
            issues.append(f"{len(leaves_without_sig)} leaf nodes missing signatures")

        # Check 2: Orphan nodes
        orphans = [n for n in rpg.graph.nodes() if rpg.graph.degree(n) == 0]
        if orphans:
            warnings.append(f"{len(orphans)} orphan nodes (not connected)")

        # Check 3: Cycles
        import networkx as nx
        try:
            cycles = list(nx.simple_cycles(rpg.graph))
            if cycles:
                issues.append(f"Graph has {len(cycles)} cycles (should be DAG)")
        except:
            pass

        # Check 4: Data flow types
        flows_without_type = [
            (u, v) for u, v, d in rpg.graph.edges(data=True)
            if d.get("type") == "data_flow" and not d.get("data_type")
        ]

        if flows_without_type:
            warnings.append(f"{len(flows_without_type)} data flows missing type information")

        # Check 5: File paths
        nodes_without_path = [
            n for n, d in rpg.graph.nodes(data=True)
            if d.get("type") in ["root", "intermediate"] and not d.get("file_path")
        ]

        if nodes_without_path:
            issues.append(f"{len(nodes_without_path)} nodes missing file paths")

        # Summary
        is_valid = len(issues) == 0

        report = {
            "is_valid": is_valid,
            "issues": issues,
            "warnings": warnings,
            "total_nodes": rpg.graph.number_of_nodes(),
            "total_edges": rpg.graph.number_of_edges(),
        }

        if is_valid:
            self.logger.info("[OK] RPG validation passed", warnings=len(warnings))
        else:
            self.logger.error("[FAIL] RPG validation failed", issues=issues)

        return report
