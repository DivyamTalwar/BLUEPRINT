import json
from typing import Dict, List, Any, Optional, Set
from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType
from src.core.llm_router_final import FinalLLMRouter
from src.utils.logger import StructuredLogger
import networkx as nx

logger = StructuredLogger("data_flow_encoder")


class DataFlowEncoder:
    """
    Encodes data flow relationships between components.

    Creates two types of edges:
    1. DATA_FLOW: How data moves between nodes (with type information)
    2. EXECUTION_ORDER: Temporal dependencies (which runs before what)
    """

    def __init__(self, llm_router: FinalLLMRouter, config: Dict[str, Any]):
        self.llm = llm_router
        self.config = config
        self.logger = logger

    def encode(self, rpg: RepositoryPlanningGraph) -> RepositoryPlanningGraph:
        """
        Main encoding method - adds data flow edges to graph.

        Args:
            rpg: Graph with file structure already encoded

        Returns:
            Updated RPG with data flow and execution order edges
        """
        self.logger.log("info", "Starting data flow encoding")

        # Step 1: Inter-module data flows (between folders)
        self._encode_inter_module_flows(rpg)

        # Step 2: Intra-module data flows (within folders)
        self._encode_intra_module_flows(rpg)

        # Step 3: Execution order (topological dependencies)
        self._encode_execution_order(rpg)

        # Step 4: Validate DAG (no circular dependencies)
        self._validate_no_cycles(rpg)

        self.logger.log("info", "Data flow encoding complete")
        return rpg

    def _encode_inter_module_flows(self, rpg: RepositoryPlanningGraph):
        """
        Define data flows between top-level modules.

        Example:
        Data Processing -> Algorithms: training_data (DataFrame)
        Algorithms -> Evaluation: predictions (ndarray)
        """
        root_nodes = [n for n, d in rpg.graph.nodes(data=True) if d.get("type") == NodeType.ROOT.value]

        if len(root_nodes) < 2:
            self.logger.log("info", "Less than 2 modules, skipping inter-module flows")
            return

        # Extract module information
        modules_info = []
        for node_id in root_nodes:
            node_data = rpg.graph.nodes[node_id]
            modules_info.append({
                "id": node_id,
                "name": node_data.get("name"),
                "functionality": node_data.get("functionality", ""),
                "folder": node_data.get("file_path", ""),
            })

        # LLM determines data flows
        flows = self._generate_inter_module_flows(modules_info, rpg.repository_goal)

        # Add flow edges to graph
        for flow in flows:
            from_id = flow.get("from_module_id")
            to_id = flow.get("to_module_id")

            if from_id in rpg.graph and to_id in rpg.graph:
                rpg.add_edge(
                    from_id,
                    to_id,
                    EdgeType.DATA_FLOW,
                    data_type=flow.get("data_type"),
                    transformation=flow.get("description", "")
                )

                self.logger.log("debug", f"Inter-module flow: {flow.get('data_name')} ({flow.get('data_type')})",
                              from_module=rpg.graph.nodes[from_id].get("name"),
                              to_module=rpg.graph.nodes[to_id].get("name"))

    def _generate_inter_module_flows(self, modules: List[Dict], repo_goal: str) -> List[Dict]:
        """Use LLM to determine data flows between modules."""
        prompt = f"""You are designing data flow between modules in a Python repository.

Repository goal: {repo_goal}

Modules:
{json.dumps(modules, indent=2)}

Define how data flows between these modules. Specify:
1. Which module outputs data
2. Which module receives it
3. Data name (e.g., "training_data", "predictions")
4. Data type (e.g., "DataFrame", "ndarray", "List[dict]")
5. Brief description

Output JSON format:
{{
  "flows": [
    {{
      "from_module_id": "module_id_1",
      "to_module_id": "module_id_2",
      "data_name": "processed_data",
      "data_type": "pandas.DataFrame",
      "description": "Cleaned and transformed data ready for training"
    }},
    ...
  ]
}}

Guidelines:
- Create logical data pipelines
- Use Python type hints (DataFrame, List[str], dict, etc.)
- Ensure flows create DAG (no cycles)

IMPORTANT: Return ONLY valid JSON, no additional text."""

        try:
            response = self.llm.generate(
                prompt=prompt,
                json_mode=True,
                temperature=0.4,
                max_tokens=2000
            )

            result = json.loads(response.content)
            return result.get("flows", [])

        except Exception as e:
            self.logger.log("error", "Failed to generate inter-module flows", error=str(e))
            return []

    def _encode_intra_module_flows(self, rpg: RepositoryPlanningGraph):
        """
        Define data flows within each module (between files).

        Example within data/ module:
        loader.py -> transform.py -> validate.py
        """
        root_nodes = [n for n, d in rpg.graph.nodes(data=True) if d.get("type") == NodeType.ROOT.value]

        for root_id in root_nodes:
            # Get all file nodes within this module
            file_nodes = []
            for successor in rpg.graph.successors(root_id):
                if rpg.graph.nodes[successor].get("type") == NodeType.INTERMEDIATE.value:
                    file_nodes.append(successor)

            if len(file_nodes) < 2:
                continue

            # Extract file information
            files_info = []
            for file_id in file_nodes:
                file_data = rpg.graph.nodes[file_id]
                files_info.append({
                    "id": file_id,
                    "name": file_data.get("name"),
                    "functionality": file_data.get("functionality", ""),
                    "file_path": file_data.get("file_path", ""),
                })

            # LLM determines flows between files
            module_name = rpg.graph.nodes[root_id].get("name")
            flows = self._generate_intra_module_flows(files_info, module_name)

            # Add flow edges
            for flow in flows:
                from_id = flow.get("from_file_id")
                to_id = flow.get("to_file_id")

                if from_id in rpg.graph and to_id in rpg.graph:
                    rpg.add_edge(
                        from_id,
                        to_id,
                        EdgeType.DATA_FLOW,
                        data_type=flow.get("data_type"),
                        transformation=flow.get("description", "")
                    )

    def _generate_intra_module_flows(self, files: List[Dict], module_name: str) -> List[Dict]:
        """Use LLM to determine flows within a module."""
        prompt = f"""You are designing data flow between files within the "{module_name}" module.

Files:
{json.dumps(files, indent=2)}

Define the data flow pipeline within this module:
1. Which file outputs data
2. Which file receives it
3. Data name and type
4. Create a logical processing pipeline

Output JSON format:
{{
  "flows": [
    {{
      "from_file_id": "file_id_1",
      "to_file_id": "file_id_2",
      "data_name": "raw_data",
      "data_type": "DataFrame",
      "description": "Unprocessed data from loader"
    }},
    ...
  ]
}}

IMPORTANT: Return ONLY valid JSON, no additional text."""

        try:
            response = self.llm.generate(
                prompt=prompt,
                json_mode=True,
                temperature=0.3,
                max_tokens=1500
            )

            result = json.loads(response.content)
            return result.get("flows", [])

        except Exception as e:
            self.logger.log("error", f"Failed to generate intra-module flows for {module_name}", error=str(e))
            return []

    def _encode_execution_order(self, rpg: RepositoryPlanningGraph):
        """
        Create EXECUTION_ORDER edges based on data flow dependencies.

        If A flows data to B, then A must execute before B.
        """
        # Find all DATA_FLOW edges
        data_flow_edges = [
            (u, v, d) for u, v, d in rpg.graph.edges(data=True)
            if d.get("type") == EdgeType.DATA_FLOW.value
        ]

        # For each data flow, add execution order
        for from_node, to_node, edge_data in data_flow_edges:
            # Add execution order edge (if not exists)
            if not rpg.graph.has_edge(from_node, to_node):
                rpg.add_edge(from_node, to_node, EdgeType.EXECUTION_ORDER)
            else:
                # Update existing edge to include execution order
                rpg.graph.edges[from_node, to_node]["execution_order"] = True

        self.logger.log("info", f"Added {len(data_flow_edges)} execution order relationships")

    def _validate_no_cycles(self, rpg: RepositoryPlanningGraph):
        """
        Validate that the graph is a DAG (no circular dependencies).

        This is critical for code generation - can't execute in cycles.
        """
        try:
            # NetworkX will raise exception if cycles exist
            cycles = list(nx.simple_cycles(rpg.graph))

            if cycles:
                self.logger.log("error", "CIRCULAR DEPENDENCIES DETECTED", cycles=cycles[:5])

                # Try to break cycles by removing problematic edges
                for cycle in cycles[:3]:  # Fix first 3 cycles
                    if len(cycle) >= 2:
                        # Remove last edge in cycle
                        rpg.graph.remove_edge(cycle[-1], cycle[0])
                        self.logger.log("warning", f"Removed edge to break cycle: {cycle[-1]} -> {cycle[0]}")

                # Re-check
                remaining_cycles = list(nx.simple_cycles(rpg.graph))
                if remaining_cycles:
                    self.logger.log("error", "Unable to resolve all cycles", count=len(remaining_cycles))
                else:
                    self.logger.log("info", "All cycles resolved")
            else:
                self.logger.log("info", "Graph is a valid DAG (no cycles)")

        except Exception as e:
            self.logger.log("error", "Failed to validate DAG", error=str(e))

    def get_execution_order(self, rpg: RepositoryPlanningGraph) -> List[str]:
        """
        Get topological ordering of nodes for code generation.

        Returns:
            List of node IDs in execution order (dependencies first)
        """
        try:
            return list(nx.topological_sort(rpg.graph))
        except nx.NetworkXError as e:
            self.logger.log("error", "Cannot create topological order (graph has cycles)", error=str(e))
            # Fallback: BFS order
            if rpg.graph.nodes():
                root = next(iter(rpg.graph.nodes()))
                return list(nx.bfs_tree(rpg.graph, root))
            return []
