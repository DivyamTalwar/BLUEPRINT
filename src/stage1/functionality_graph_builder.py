from typing import List, Dict, Any
import json

from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType, NodeStatus
from src.models.feature import Feature
from src.core.llm_router import LLMRouter
from src.stage1.user_input_processor import RepositoryRequest
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FunctionalityGraphBuilder:
    """Build functionality graph (RPG) from selected features"""

    def __init__(self, llm_router: LLMRouter):
        """
        Initialize builder

        Args:
            llm_router: LLM router for refactoring
        """
        self.llm_router = llm_router

    def build(
        self,
        request: RepositoryRequest,
        selected_features: List[Feature],
    ) -> RepositoryPlanningGraph:
        """
        Build functionality graph from features

        Args:
            request: Repository request
            selected_features: Selected features

        Returns:
            Functionality graph (RPG)
        """
        logger.info(
            "Building functionality graph from %d features",
            len(selected_features),
        )

        # Create RPG
        rpg = RepositoryPlanningGraph(repository_goal=request.raw_description)

        # Step 1: Organize features into modules
        module_structure = self._organize_into_modules(request, selected_features)

        # Step 2: Build RPG hierarchy
        self._build_hierarchy(rpg, module_structure, selected_features)

        # Step 3: Add data flow edges
        self._add_data_flows(rpg, module_structure)

        # Validate
        is_valid, errors = rpg.validate()
        if not is_valid:
            logger.warning("RPG validation warnings: %s", errors)

        stats = rpg.get_stats()
        logger.info(
            "Functionality graph built: %d nodes, %d edges",
            stats["total_nodes"],
            stats["total_edges"],
        )

        return rpg

    def _organize_into_modules(
        self,
        request: RepositoryRequest,
        features: List[Feature],
    ) -> Dict[str, Any]:
        """Organize features into modular structure using LLM"""

        logger.info("Organizing %d features into modules", len(features))

        # Group features by domain
        by_domain = {}
        for feature in features:
            domain = feature.domain
            if domain not in by_domain:
                by_domain[domain] = []
            by_domain[domain].append(feature)

        # Build prompt
        feature_list = []
        for i, feature in enumerate(features[:100], 1):  # Limit to 100 for prompt
            feature_list.append(
                f"{i}. {feature.name} ({feature.domain}/{feature.subdomain}) - {feature.description[:100]}"
            )

        prompt = f"""Organize these features into a hierarchical module structure for a {request.repo_type} repository.

Repository Goal: {request.raw_description}

Features to Organize ({len(features)} total, showing first 100):
{chr(10).join(feature_list)}

Task: Create a modular hierarchy with:
- 4-8 top-level modules (main functionality areas)
- 2-5 sub-modules within each top-level module
- Assign features to appropriate modules

Principles:
- High cohesion (related features together)
- Low coupling (minimal cross-module dependencies)
- Clear, descriptive module names
- Logical grouping by functionality, not just domain

Output as JSON:
{{
  "modules": [
    {{
      "name": "Module Name",
      "description": "What this module does",
      "sub_modules": [
        {{
          "name": "Sub-module Name",
          "description": "What this sub-module does",
          "feature_names": ["Feature 1", "Feature 2", ...]
        }}
      ]
    }}
  ]
}}"""

        try:
            response = self.llm_router.generate(
                prompt=prompt,
                temperature=0.4,
                max_tokens=4000,
                json_mode=True,
            )

            data = json.loads(response.content)
            logger.info("Organized into %d modules", len(data.get("modules", [])))

            return data

        except Exception as e:
            logger.error("Module organization failed: %s", str(e))
            # Fallback: Simple organization by domain
            return self._fallback_organization(features)

    def _fallback_organization(self, features: List[Feature]) -> Dict[str, Any]:
        """Fallback: Organize by domain"""

        logger.info("Using fallback organization by domain")

        by_domain = {}
        for feature in features:
            domain = feature.domain
            if domain not in by_domain:
                by_domain[domain] = []
            by_domain[domain].append(feature)

        modules = []
        for domain, domain_features in by_domain.items():
            # Group by subdomain
            by_subdomain = {}
            for feature in domain_features:
                subdomain = feature.subdomain
                if subdomain not in by_subdomain:
                    by_subdomain[subdomain] = []
                by_subdomain[subdomain].append(feature.name)

            sub_modules = [
                {
                    "name": subdomain,
                    "description": f"{subdomain} functionality",
                    "feature_names": feature_names,
                }
                for subdomain, feature_names in by_subdomain.items()
            ]

            modules.append({
                "name": domain,
                "description": f"{domain} operations",
                "sub_modules": sub_modules,
            })

        return {"modules": modules}

    def _build_hierarchy(
        self,
        rpg: RepositoryPlanningGraph,
        module_structure: Dict[str, Any],
        features: List[Feature],
    ):
        """Build RPG hierarchy from module structure"""

        logger.info("Building RPG hierarchy")

        # Create feature name to feature mapping
        feature_map = {f.name: f for f in features}

        # Add top-level modules
        for module in module_structure.get("modules", []):
            # Create module node (ROOT)
            module_id = rpg.add_node(
                name=module["name"],
                node_type=NodeType.ROOT,
                functionality=module.get("description", ""),
                domain=module.get("domain", ""),
                status=NodeStatus.PLANNED,
            )

            # Add sub-modules
            for sub_module in module.get("sub_modules", []):
                # Create sub-module node (INTERMEDIATE)
                sub_module_id = rpg.add_node(
                    name=sub_module["name"],
                    node_type=NodeType.INTERMEDIATE,
                    functionality=sub_module.get("description", ""),
                    domain=module.get("domain", ""),
                    status=NodeStatus.PLANNED,
                )

                # Link to parent module
                rpg.add_edge(module_id, sub_module_id, EdgeType.HIERARCHY)

                # Add features
                for feature_name in sub_module.get("feature_names", []):
                    feature = feature_map.get(feature_name)

                    if not feature:
                        # Try partial match
                        for fname, fobj in feature_map.items():
                            if feature_name.lower() in fname.lower():
                                feature = fobj
                                break

                    if feature:
                        # Create feature node (LEAF)
                        feature_id = rpg.add_node(
                            name=feature.name,
                            node_type=NodeType.LEAF,
                            functionality=feature.description,
                            domain=feature.domain,
                            subdomain=feature.subdomain,
                            complexity=feature.complexity.value,
                            status=NodeStatus.PLANNED,
                        )

                        # Link to sub-module
                        rpg.add_edge(sub_module_id, feature_id, EdgeType.HIERARCHY)

        logger.info("RPG hierarchy built")

    def _add_data_flows(
        self,
        rpg: RepositoryPlanningGraph,
        module_structure: Dict[str, Any],
    ):
        """Add data flow edges between modules"""

        logger.info("Adding data flow edges")

        # Get all root nodes (modules)
        root_nodes = rpg.get_root_nodes()

        if len(root_nodes) < 2:
            logger.info("Not enough modules for data flows")
            return

        # Common data flow patterns
        patterns = [
            ("data_operations", "api_web", "processed_data"),
            ("data_operations", "machine_learning", "training_data"),
            ("api_web", "authentication_security", "user_credentials"),
            ("machine_learning", "data_visualization", "predictions"),
            ("database_orm", "api_web", "query_results"),
        ]

        # Add data flows based on patterns
        added = 0
        for from_domain, to_domain, data_type in patterns:
            from_nodes = [
                node_id for node_id in root_nodes
                if from_domain in rpg.get_node(node_id).get("name", "").lower()
            ]

            to_nodes = [
                node_id for node_id in root_nodes
                if to_domain in rpg.get_node(node_id).get("name", "").lower()
            ]

            if from_nodes and to_nodes:
                rpg.add_edge(
                    from_nodes[0],
                    to_nodes[0],
                    EdgeType.DATA_FLOW,
                    data_type=data_type,
                )
                added += 1

        logger.info("Added %d data flow edges", added)

    def refine_graph(self, rpg: RepositoryPlanningGraph) -> RepositoryPlanningGraph:
        """Refine graph structure"""

        logger.info("Refining graph structure")

        from src.core.graph_operations import GraphOperations

        # Optimize graph
        report = GraphOperations.optimize_graph(rpg)
        logger.info("Optimization: %s", report)

        # Detect bottlenecks
        bottlenecks = GraphOperations.detect_bottlenecks(rpg)
        if bottlenecks:
            logger.warning("Found %d bottlenecks", len(bottlenecks))
            for bottleneck in bottlenecks[:3]:
                logger.warning("Bottleneck: %s", bottleneck)

        return rpg
