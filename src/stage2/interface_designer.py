import json
from typing import Dict, List, Any, Optional
from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType
from src.core.llm_router_final import FinalLLMRouter
from src.utils.logger import StructuredLogger

logger = StructuredLogger("interface_designer")


class InterfaceDesigner:

    def __init__(self, llm_router: FinalLLMRouter, config: Dict[str, Any]):
        self.llm = llm_router
        self.config = config
        self.logger = logger

    def design(self, rpg: RepositoryPlanningGraph) -> RepositoryPlanningGraph:
        """
        Main design method - creates signatures for all features.

        Args:
            rpg: Graph with file structure, data flows, and base classes

        Returns:
            Updated RPG with signatures for all leaf nodes
        """
        self.logger.log("info", "Starting interface design")

        leaf_nodes = [n for n, d in rpg.graph.nodes(data=True) if d.get("type") == NodeType.LEAF.value]

        self.logger.log("info", f"Designing signatures for {len(leaf_nodes)} features")

        # Process in batches for efficiency
        batch_size = self.config.get("signature_batch_size", 5)

        for i in range(0, len(leaf_nodes), batch_size):
            batch = leaf_nodes[i:i+batch_size]
            self._design_batch(rpg, batch)

        self.logger.log("info", "Interface design complete")
        return rpg

    def _design_batch(self, rpg: RepositoryPlanningGraph, node_ids: List[str]):
        # Extract context for each node
        nodes_context = []
        for node_id in node_ids:
            context = self._extract_node_context(rpg, node_id)
            if context:
                nodes_context.append(context)

        if not nodes_context:
            return

        # LLM designs signatures for batch
        signatures = self._design_signatures_llm(nodes_context)

        # Apply signatures to nodes
        for sig in signatures:
            node_id = sig.get("node_id")
            if node_id in rpg.graph:
                rpg.graph.nodes[node_id]["signature"] = sig.get("signature")
                rpg.graph.nodes[node_id]["docstring"] = sig.get("docstring")
                rpg.graph.nodes[node_id]["parameters"] = sig.get("parameters")
                rpg.graph.nodes[node_id]["return_type"] = sig.get("return_type")
                rpg.graph.nodes[node_id]["imports"] = sig.get("imports", [])

                self.logger.log("debug", f"Designed signature for {rpg.graph.nodes[node_id].get('name')}",
                              signature=sig.get("signature"))

    def _extract_node_context(self, rpg: RepositoryPlanningGraph, node_id: str) -> Optional[Dict]:
        node_data = rpg.graph.nodes[node_id]

        # Skip base classes (they're already designed)
        if node_data.get("is_base_class"):
            return None

        context = {
            "node_id": node_id,
            "name": node_data.get("name"),
            "functionality": node_data.get("functionality", ""),
            "code_type": node_data.get("code_type"),  # function or method
            "class_name": node_data.get("class_name"),
            "inherits_from": node_data.get("inherits_from"),
        }

        # Get input data types from incoming edges
        input_data = []
        for pred in rpg.graph.predecessors(node_id):
            edge_data = rpg.graph.edges.get((pred, node_id), {})
            if edge_data.get("type") == EdgeType.DATA_FLOW.value:
                input_data.append({
                    "name": edge_data.get("data_name"),
                    "type": edge_data.get("data_type"),
                    "description": edge_data.get("description", "")
                })

        context["input_data"] = input_data

        # Get output data types from outgoing edges
        output_data = []
        for succ in rpg.graph.successors(node_id):
            edge_data = rpg.graph.edges.get((node_id, succ), {})
            if edge_data.get("type") == EdgeType.DATA_FLOW.value:
                output_data.append({
                    "name": edge_data.get("data_name"),
                    "type": edge_data.get("data_type"),
                    "description": edge_data.get("description", "")
                })

        context["output_data"] = output_data

        # Get base class design if inherits
        if context["inherits_from"]:
            base_class = self._find_base_class(rpg, context["inherits_from"])
            if base_class:
                context["base_class_methods"] = base_class.get("base_class_design", {}).get("methods", [])

        return context

    def _find_base_class(self, rpg: RepositoryPlanningGraph, class_name: str) -> Optional[Dict]:
        """Find base class node by name."""
        for node_id, node_data in rpg.graph.nodes(data=True):
            if (node_data.get("is_base_class") and
                node_data.get("name") == class_name):
                return node_data
        return None

    def _design_signatures_llm(self, nodes_context: List[Dict]) -> List[Dict]:
        prompt = f"""Design Python function/method signatures for these features.

Features to design:
{json.dumps(nodes_context, indent=2)}

For each feature, design:
1. Complete signature with type hints (Python 3.11+)
2. Parameter names and types (based on input_data)
3. Return type (based on output_data)
4. Comprehensive docstring (Google style)
5. Required imports (typing, pandas, numpy, etc.)

Guidelines:
- Use modern type hints: List[str], Dict[str, Any], Optional[int]
- Parameter names should be descriptive
- Include default values where appropriate
- Docstring should have Args, Returns, Raises sections
- For methods, include 'self' as first parameter
- Match base class signatures if inherits_from is specified

Output JSON format:
{{
  "signatures": [
    {{
      "node_id": "node_123",
      "signature": "def load_csv(filepath: str, encoding: str = 'utf-8') -> pd.DataFrame",
      "parameters": [
        {{"name": "filepath", "type": "str", "description": "Path to CSV file"}},
        {{"name": "encoding", "type": "str", "default": "utf-8", "description": "File encoding"}}
      ],
      "return_type": "pd.DataFrame",
      "docstring": "Load CSV file into DataFrame.\\n\\nArgs:\\n    filepath: Path to CSV file\\n    encoding: File encoding (default: utf-8)\\n\\nReturns:\\n    DataFrame with loaded data\\n\\nRaises:\\n    FileNotFoundError: If file doesn't exist",
      "imports": ["import pandas as pd", "from typing import Optional"]
    }},
    ...
  ]
}}

IMPORTANT: Return ONLY valid JSON, no additional text."""

        try:
            response = self.llm.generate(
                prompt=prompt,
                json_mode=True,
                temperature=0.2,
                max_tokens=4000
            )

            result = json.loads(response.content)
            return result.get("signatures", [])

        except Exception as e:
            self.logger.log("error", "Failed to design signatures", error=str(e))
            # Fallback: basic signatures
            return self._create_fallback_signatures(nodes_context)

    def _create_fallback_signatures(self, nodes_context: List[Dict]) -> List[Dict]:
        """Create basic signatures if LLM fails."""
        signatures = []

        for ctx in nodes_context:
            name = ctx.get("name", "feature").lower().replace(" ", "_")
            code_type = ctx.get("code_type", "function")

            # Basic signature
            if code_type == "method":
                signature = f"def {name}(self, *args, **kwargs)"
            else:
                signature = f"def {name}(*args, **kwargs)"

            signatures.append({
                "node_id": ctx.get("node_id"),
                "signature": signature,
                "parameters": [],
                "return_type": "Any",
                "docstring": ctx.get("functionality", ""),
                "imports": ["from typing import Any"]
            })

        return signatures

    def get_all_imports(self, rpg: RepositoryPlanningGraph) -> List[str]:
        all_imports = set()

        for node_id, node_data in rpg.graph.nodes(data=True):
            imports = node_data.get("imports", [])
            all_imports.update(imports)

        return sorted(list(all_imports))
