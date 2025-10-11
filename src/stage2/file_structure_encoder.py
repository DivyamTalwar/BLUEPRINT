import json
from typing import Dict, List, Any, Tuple, Optional
from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType
from src.core.llm_router_final import FinalLLMRouter
from src.utils.logger import StructuredLogger

logger = StructuredLogger("file_structure_encoder")


class FileStructureEncoder:
    def __init__(self, llm_router: FinalLLMRouter, config: Dict[str, Any]):
        self.llm = llm_router
        self.config = config
        self.logger = logger

    def encode(self, rpg: RepositoryPlanningGraph, repo_type: str = "library") -> RepositoryPlanningGraph:
        self.logger.log("info", "Starting file structure encoding",
                       repo_type=repo_type, node_count=rpg.graph.number_of_nodes())

        # Step 1: Assign folders to root nodes (modules)
        self._assign_folders_to_modules(rpg, repo_type)

        # Step 2: Group features into files
        self._group_features_into_files(rpg)

        # Step 3: Determine function vs class implementation
        self._assign_code_types(rpg)

        self.logger.log("info", "File structure encoding complete")
        return rpg

    def _assign_folders_to_modules(self, rpg: RepositoryPlanningGraph, repo_type: str):
        root_nodes = [n for n, d in rpg.graph.nodes(data=True) if d.get("type") == NodeType.ROOT.value]

        if not root_nodes:
            self.logger.log("warning", "No root nodes found in graph")
            return

        # Extract module information
        modules_info = []
        for node_id in root_nodes:
            node_data = rpg.graph.nodes[node_id]
            modules_info.append({
                "name": node_data.get("name"),
                "functionality": node_data.get("functionality", ""),
                "domain": node_data.get("domain", ""),
            })

        # LLM prompt for folder assignment
        prompt = self._create_folder_assignment_prompt(modules_info, repo_type)

        try:
            response = self.llm.generate(
                prompt=prompt,
                json_mode=True,
                temperature=0.3,
                max_tokens=2000
            )

            folder_assignments = json.loads(response.content)

            # Apply folder assignments to nodes
            for node_id in root_nodes:
                node_data = rpg.graph.nodes[node_id]
                module_name = node_data.get("name")

                # Find assignment for this module
                assignment = next((a for a in folder_assignments.get("assignments", [])
                                 if a.get("module_name") == module_name), None)

                if assignment:
                    rpg.graph.nodes[node_id]["file_path"] = assignment.get("folder_path")
                    rpg.graph.nodes[node_id]["code_type"] = "folder"
                    self.logger.log("debug", f"Assigned folder: {module_name} -> {assignment.get('folder_path')}")

        except Exception as e:
            self.logger.log("error", "Failed to assign folders", error=str(e))
            # Fallback: simple naming
            for i, node_id in enumerate(root_nodes):
                node_data = rpg.graph.nodes[node_id]
                folder_name = node_data.get("name", "").lower().replace(" ", "_")
                rpg.graph.nodes[node_id]["file_path"] = f"src/{folder_name}/"
                rpg.graph.nodes[node_id]["code_type"] = "folder"

    def _create_folder_assignment_prompt(self, modules: List[Dict], repo_type: str) -> str:
        return f"""You are designing the folder structure for a Python {repo_type}.

Given these modules, assign appropriate folder paths following Python best practices.

Modules:
{json.dumps(modules, indent=2)}

Consider:
1. Standard Python package layout (src/, lib/, tests/)
2. Domain conventions (e.g., web apps use app/, APIs use api/)
3. Scalability (easy to add more modules)
4. Clear separation of concerns

Output JSON format:
{{
  "assignments": [
    {{
      "module_name": "Data Processing",
      "folder_path": "src/data/",
      "reasoning": "Handles data-related operations"
    }},
    ...
  ]
}}

IMPORTANT: Return ONLY valid JSON, no additional text."""

    def _group_features_into_files(self, rpg: RepositoryPlanningGraph):
        """
        Group related features into files within each folder.

        Process:
        1. For each root node (folder), get its child features
        2. Use LLM to group features into logical files
        3. Create intermediate nodes for files
        4. Re-link features to file nodes
        """
        root_nodes = [n for n, d in rpg.graph.nodes(data=True) if d.get("type") == NodeType.ROOT.value]

        for root_id in root_nodes:
            # Get all features under this module
            child_features = list(rpg.graph.successors(root_id))

            if not child_features:
                continue

            # Extract feature information
            features_info = []
            for feat_id in child_features:
                feat_data = rpg.graph.nodes[feat_id]
                features_info.append({
                    "id": feat_id,
                    "name": feat_data.get("name"),
                    "functionality": feat_data.get("functionality", ""),
                    "domain": feat_data.get("domain", ""),
                })

            # LLM groups features into files
            module_name = rpg.graph.nodes[root_id].get("name")
            folder_path = rpg.graph.nodes[root_id].get("file_path", "src/")

            file_groupings = self._generate_file_groupings(features_info, module_name, folder_path)

            # Create intermediate nodes for files and re-link features
            self._create_file_nodes(rpg, root_id, file_groupings, folder_path)

    def _generate_file_groupings(self, features: List[Dict], module_name: str,
                                folder_path: str) -> List[Dict]:
        """Use LLM to group features into files."""
        prompt = f"""You are organizing features into Python files for the "{module_name}" module.

Features to organize:
{json.dumps(features, indent=2)}

Folder: {folder_path}

Group related features into files. Guidelines:
1. High cohesion - related features together
2. Target ~5-15 features per file
3. Clear file responsibilities
4. Logical naming (e.g., loader.py, transform.py, validators.py)

Output JSON format:
{{
  "files": [
    {{
      "file_name": "loader.py",
      "purpose": "Data loading operations",
      "feature_ids": ["feat1", "feat2", ...]
    }},
    ...
  ]
}}

IMPORTANT: Return ONLY valid JSON, no additional text."""

        try:
            response = self.llm.generate(
                prompt=prompt,
                json_mode=True,
                temperature=0.4,
                max_tokens=3000
            )

            result = json.loads(response.content)
            return result.get("files", [])

        except Exception as e:
            self.logger.log("error", "Failed to group features into files", error=str(e))
            # Fallback: one file per module
            return [{
                "file_name": f"{module_name.lower().replace(' ', '_')}.py",
                "purpose": f"Main {module_name} functionality",
                "feature_ids": [f["id"] for f in features]
            }]

    def _create_file_nodes(self, rpg: RepositoryPlanningGraph, root_id: str,
                          file_groupings: List[Dict], folder_path: str):
        """
        Create intermediate nodes for files and re-link features.

        Original structure: ROOT -> LEAF (features)
        New structure: ROOT -> INTERMEDIATE (files) -> LEAF (features)
        """
        for file_info in file_groupings:
            file_name = file_info.get("file_name")
            feature_ids = file_info.get("feature_ids", [])

            # Create intermediate node for file
            file_path = folder_path + file_name
            file_node_id = rpg.add_node(
                name=file_name,
                node_type=NodeType.INTERMEDIATE,
                functionality=file_info.get("purpose", ""),
                file_path=file_path,
                code_type="file"
            )

            # Link root -> file
            rpg.add_edge(root_id, file_node_id, EdgeType.HIERARCHY)

            # Re-link features: remove root -> feature, add file -> feature
            for feat_id in feature_ids:
                if feat_id in rpg.graph:
                    # Remove old edge (root -> feature)
                    if rpg.graph.has_edge(root_id, feat_id):
                        rpg.graph.remove_edge(root_id, feat_id)

                    # Add new edge (file -> feature)
                    rpg.add_edge(file_node_id, feat_id, EdgeType.HIERARCHY)

                    # Update feature's parent reference
                    rpg.graph.nodes[feat_id]["parent_file"] = file_path

    def _assign_code_types(self, rpg: RepositoryPlanningGraph):
        leaf_nodes = [n for n, d in rpg.graph.nodes(data=True) if d.get("type") == NodeType.LEAF.value]

        # Group by file for context
        files_to_features = {}
        for leaf_id in leaf_nodes:
            leaf_data = rpg.graph.nodes[leaf_id]
            parent_file = leaf_data.get("parent_file", "unknown")

            if parent_file not in files_to_features:
                files_to_features[parent_file] = []
            files_to_features[parent_file].append(leaf_id)

        # Decide function vs class for each file's features
        for file_path, feature_ids in files_to_features.items():
            features_info = []
            for feat_id in feature_ids:
                feat_data = rpg.graph.nodes[feat_id]
                features_info.append({
                    "id": feat_id,
                    "name": feat_data.get("name"),
                    "functionality": feat_data.get("functionality", ""),
                })

            assignments = self._decide_function_vs_class(features_info, file_path)

            # Apply assignments
            for assignment in assignments:
                feat_id = assignment.get("feature_id")
                if feat_id in rpg.graph:
                    rpg.graph.nodes[feat_id]["code_type"] = assignment.get("code_type")
                    rpg.graph.nodes[feat_id]["class_name"] = assignment.get("class_name")

    def _decide_function_vs_class(self, features: List[Dict], file_path: str) -> List[Dict]:
        """Use LLM to decide function vs class implementation."""
        prompt = f"""Decide whether each feature should be implemented as a function or class method.

File: {file_path}
Features:
{json.dumps(features, indent=2)}

Guidelines:
- Stateful operations (e.g., models, transformers) -> class
- Stateless utilities (e.g., load file, calculate metric) -> function
- Related features (e.g., fit, predict) -> methods of same class
- Standalone features -> individual functions

Output JSON format:
{{
  "assignments": [
    {{
      "feature_id": "feat1",
      "code_type": "function",
      "class_name": null
    }},
    {{
      "feature_id": "feat2",
      "code_type": "method",
      "class_name": "LinearRegression"
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
            return result.get("assignments", [])

        except Exception as e:
            self.logger.log("error", "Failed to assign code types", error=str(e))
            # Fallback: all functions
            return [{"feature_id": f["id"], "code_type": "function", "class_name": None}
                   for f in features]
