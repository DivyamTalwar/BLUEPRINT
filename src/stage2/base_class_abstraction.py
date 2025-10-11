import json
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict
from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType
from src.core.llm_router_final import FinalLLMRouter
from src.utils.logger import StructuredLogger

logger = StructuredLogger("base_class_abstraction")


class BaseClassAbstraction:
    def __init__(self, llm_router: FinalLLMRouter, config: Dict[str, Any]):
        self.llm = llm_router
        self.config = config
        self.logger = logger

    def abstract(self, rpg: RepositoryPlanningGraph) -> RepositoryPlanningGraph:
        """
        Main abstraction method - creates base classes.

        Args:
            rpg: Graph with file structure and code types

        Returns:
            Updated RPG with base class nodes added
        """
        self.logger.log("info", "Starting base class abstraction")

        # Step 1: Group features by class
        class_groups = self._group_features_by_class(rpg)

        # Step 2: Identify patterns across classes
        patterns = self._identify_patterns(class_groups)

        # Step 3: Design base classes
        base_classes = self._design_base_classes(patterns, rpg.repository_goal)

        # Step 4: Add base class nodes to graph
        self._add_base_classes_to_graph(rpg, base_classes)

        self.logger.log("info", "Base class abstraction complete",
                       base_classes_created=len(base_classes))
        return rpg

    def _group_features_by_class(self, rpg: RepositoryPlanningGraph) -> Dict[str, List[str]]:
        """
        Group features that belong to the same class.

        Returns:
            Dict mapping class_name -> list of feature IDs
        """
        class_groups = defaultdict(list)

        for node_id, node_data in rpg.graph.nodes(data=True):
            if node_data.get("type") == NodeType.LEAF.value:
                code_type = node_data.get("code_type")
                class_name = node_data.get("class_name")

                if code_type == "method" and class_name:
                    class_groups[class_name].append(node_id)

        self.logger.log("debug", f"Found {len(class_groups)} classes", classes=list(class_groups.keys()))
        return dict(class_groups)

    def _identify_patterns(self, class_groups: Dict[str, List[str]]) -> List[Dict]:
        """
        Identify common patterns across classes.

        Example:
        - Multiple classes with fit() and predict() -> BaseEstimator
        - Multiple classes with transform() -> BaseTransformer
        """
        if len(class_groups) < 2:
            self.logger.log("info", "Less than 2 classes, skipping pattern detection")
            return []

        # Analyze each class
        class_analyses = []
        for class_name, feature_ids in class_groups.items():
            class_analyses.append({
                "class_name": class_name,
                "method_count": len(feature_ids),
                "feature_ids": feature_ids
            })

        # Use LLM to find patterns
        patterns = self._detect_patterns_llm(class_analyses)
        return patterns

    def _detect_patterns_llm(self, class_analyses: List[Dict]) -> List[Dict]:
        """Use LLM to detect patterns and recommend base classes."""
        prompt = f"""You are analyzing classes to identify common patterns and design base classes.

Classes:
{json.dumps(class_analyses, indent=2)}

Identify common patterns and recommend base classes. Consider:
1. Common method names (e.g., fit, predict, transform)
2. Similar responsibilities
3. Shared interfaces

Output JSON format:
{{
  "patterns": [
    {{
      "pattern_name": "Estimator Pattern",
      "description": "Classes that fit and predict",
      "matching_classes": ["LinearRegression", "DecisionTree", "KMeans"],
      "common_methods": ["fit", "predict"],
      "recommended_base_class": "BaseEstimator"
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
                max_tokens=2000
            )

            result = json.loads(response.content)
            return result.get("patterns", [])

        except Exception as e:
            self.logger.log("error", "Failed to detect patterns", error=str(e))
            return []

    def _design_base_classes(self, patterns: List[Dict], repo_goal: str) -> List[Dict]:
        """
        Design base classes based on identified patterns.

        Returns:
            List of base class specifications
        """
        if not patterns:
            return []

        base_classes = []

        for pattern in patterns:
            base_class_name = pattern.get("recommended_base_class")
            common_methods = pattern.get("common_methods", [])

            # LLM designs the base class interface
            design = self._design_base_class_llm(
                base_class_name=base_class_name,
                pattern_description=pattern.get("description"),
                common_methods=common_methods,
                repo_goal=repo_goal
            )

            if design:
                design["pattern_name"] = pattern.get("pattern_name")
                design["matching_classes"] = pattern.get("matching_classes", [])
                base_classes.append(design)

        return base_classes

    def _design_base_class_llm(self, base_class_name: str, pattern_description: str,
                               common_methods: List[str], repo_goal: str) -> Optional[Dict]:
        """Use LLM to design a base class interface."""
        prompt = f"""Design a Python base class for this pattern.

Repository goal: {repo_goal}
Base class name: {base_class_name}
Pattern: {pattern_description}
Common methods: {', '.join(common_methods)}

Design:
1. Class docstring explaining purpose
2. Method signatures (can be abstract or with default implementation)
3. Type hints
4. Which methods should be abstract (raise NotImplementedError)

Output JSON format:
{{
  "class_name": "BaseEstimator",
  "docstring": "Base class for all estimators...",
  "methods": [
    {{
      "name": "fit",
      "signature": "def fit(self, X, y=None) -> 'BaseEstimator'",
      "docstring": "Fit the model to data",
      "is_abstract": true
    }},
    ...
  ],
  "attributes": ["is_fitted", "n_features_"],
  "imports": ["from abc import ABC, abstractmethod"]
}}

IMPORTANT: Return ONLY valid JSON, no additional text."""

        try:
            response = self.llm.generate(
                prompt=prompt,
                json_mode=True,
                temperature=0.3,
                max_tokens=2000
            )

            return json.loads(response.content)

        except Exception as e:
            self.logger.log("error", f"Failed to design base class {base_class_name}", error=str(e))
            return None

    def _add_base_classes_to_graph(self, rpg: RepositoryPlanningGraph, base_classes: List[Dict]):
        """
        Add base class nodes to the graph.

        Base classes are added as special leaf nodes in a "base/" folder.
        """
        if not base_classes:
            return

        # Create base classes folder node
        base_folder_id = rpg.add_node(
            name="base",
            node_type=NodeType.ROOT,
            functionality="Base classes and abstract interfaces",
            file_path="src/base/",
            code_type="folder"
        )

        # Add each base class
        for base_class in base_classes:
            class_name = base_class.get("class_name")
            file_name = f"{self._to_snake_case(class_name)}.py"

            # Create file node for base class
            file_node_id = rpg.add_node(
                name=file_name,
                node_type=NodeType.INTERMEDIATE,
                functionality=base_class.get("docstring", ""),
                file_path=f"src/base/{file_name}",
                code_type="file"
            )

            # Link to base folder
            rpg.add_edge(base_folder_id, file_node_id, EdgeType.HIERARCHY)

            # Create leaf node for the class itself
            class_node_id = rpg.add_node(
                name=class_name,
                node_type=NodeType.LEAF,
                functionality=base_class.get("docstring", ""),
                file_path=f"src/base/{file_name}",
                code_type="class"
            )

            # Add custom fields via update_node
            rpg.update_node(
                class_node_id,
                is_base_class=True,
                base_class_design=base_class  # Store full design
            )

            # Link to file
            rpg.add_edge(file_node_id, class_node_id, EdgeType.HIERARCHY)

            # Update child classes to reference base class
            for child_class_name in base_class.get("matching_classes", []):
                # Find child class nodes
                for node_id, node_data in rpg.graph.nodes(data=True):
                    if (node_data.get("code_type") == "method" and
                        node_data.get("class_name") == child_class_name):
                        # Add inheritance reference
                        rpg.graph.nodes[node_id]["inherits_from"] = class_name

            self.logger.log("debug", f"Added base class: {class_name}",
                          methods=len(base_class.get("methods", [])),
                          children=len(base_class.get("matching_classes", [])))

    def _to_snake_case(self, name: str) -> str:
        """Convert CamelCase to snake_case."""
        import re
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
