import json
import pickle
from pathlib import Path
from typing import Optional
import networkx as nx

from src.core.rpg import RepositoryPlanningGraph
from src.utils.file_ops import FileOperations
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GraphPersistence:

    @staticmethod
    def save_json(rpg: RepositoryPlanningGraph, filepath: str) -> bool:

        if not rpg:
            logger.error("Cannot save None RPG")
            raise ValueError("RPG cannot be None")

        if not filepath or not filepath.strip():
            logger.error("Invalid filepath")
            raise ValueError("Filepath cannot be empty")

        try:
            data = rpg.to_dict()
            success = FileOperations.write_json(filepath, data, indent=2)

            if success:
                logger.info("Saved RPG to JSON: %s (nodes: %d, edges: %d)",
                        filepath, rpg.graph.number_of_nodes(), rpg.graph.number_of_edges())
            return success

        except (IOError, OSError) as e:
            logger.error("File system error saving RPG", error=str(e), filepath=filepath)
            raise IOError(f"Failed to write RPG file: {str(e)}")
        except Exception as e:
            logger.error("Error saving RPG to JSON", error=str(e), filepath=filepath)
            return False

    @staticmethod
    def load_json(filepath: str) -> Optional[RepositoryPlanningGraph]:
        """
        Load RPG from JSON file

        Args:
            filepath: Input file path

        Returns:
            RepositoryPlanningGraph or None if failed
        """
        try:
            data = FileOperations.read_json(filepath)
            if data is None:
                return None

            rpg = RepositoryPlanningGraph.from_dict(data)
            logger.info("Loaded RPG from JSON: %s", filepath)
            return rpg

        except Exception as e:
            logger.error("Error loading RPG from JSON: %s", str(e))
            return None

    @staticmethod
    def save_pickle(rpg: RepositoryPlanningGraph, filepath: str) -> bool:
        """
        Save RPG to pickle file (faster, binary)

        Args:
            rpg: Repository Planning Graph
            filepath: Output file path

        Returns:
            True if successful
        """
        try:
            success = FileOperations.write_pickle(filepath, rpg)

            if success:
                logger.info("Saved RPG to pickle: %s", filepath)
            return success

        except Exception as e:
            logger.error("Error saving RPG to pickle: %s", str(e))
            return False

    @staticmethod
    def load_pickle(filepath: str) -> Optional[RepositoryPlanningGraph]:
        """
        Load RPG from pickle file

        Args:
            filepath: Input file path

        Returns:
            RepositoryPlanningGraph or None if failed
        """
        try:
            rpg = FileOperations.read_pickle(filepath)

            if rpg:
                logger.info("Loaded RPG from pickle: %s", filepath)
            return rpg

        except Exception as e:
            logger.error("Error loading RPG from pickle: %s", str(e))
            return None

    @staticmethod
    def save_graphml(rpg: RepositoryPlanningGraph, filepath: str) -> bool:
        """
        Save RPG to GraphML format (for visualization tools)

        Args:
            rpg: Repository Planning Graph
            filepath: Output file path

        Returns:
            True if successful
        """
        try:
            nx.write_graphml(rpg.graph, filepath)
            logger.info("Saved RPG to GraphML: %s", filepath)
            return True

        except Exception as e:
            logger.error("Error saving RPG to GraphML: %s", str(e))
            return False

    @staticmethod
    def load_graphml(filepath: str, repository_goal: str = "") -> Optional[RepositoryPlanningGraph]:
        """
        Load RPG from GraphML format

        Args:
            filepath: Input file path
            repository_goal: Repository goal description

        Returns:
            RepositoryPlanningGraph or None if failed
        """
        try:
            graph = nx.read_graphml(filepath)

            rpg = RepositoryPlanningGraph(repository_goal)
            rpg.graph = graph

            logger.info("Loaded RPG from GraphML: %s", filepath)
            return rpg

        except Exception as e:
            logger.error("Error loading RPG from GraphML: %s", str(e))
            return None

    @staticmethod
    def export_to_neo4j_cypher(rpg: RepositoryPlanningGraph, filepath: str) -> bool:
        """
        Export RPG to Neo4j Cypher statements

        Args:
            rpg: Repository Planning Graph
            filepath: Output file path

        Returns:
            True if successful
        """
        try:
            cypher_statements = []

            # Create nodes
            for node_id, data in rpg.graph.nodes(data=True):
                properties = ", ".join(
                    [
                        f"{key}: '{value}'" if isinstance(value, str) else f"{key}: {value}"
                        for key, value in data.items()
                        if value and key != "id"
                    ]
                )
                cypher = f"CREATE (n{node_id}:Node {{id: '{node_id}', {properties}}})"
                cypher_statements.append(cypher)

            # Create relationships
            for u, v, data in rpg.graph.edges(data=True):
                edge_type = data.get("type", "RELATED_TO").upper()
                properties = ", ".join(
                    [
                        f"{key}: '{value}'" if isinstance(value, str) else f"{key}: {value}"
                        for key, value in data.items()
                        if value and key not in ["from", "to", "type"]
                    ]
                )

                if properties:
                    cypher = f"MATCH (a:Node {{id: '{u}'}}), (b:Node {{id: '{v}'}}) CREATE (a)-[r:{edge_type} {{{properties}}}]->(b)"
                else:
                    cypher = f"MATCH (a:Node {{id: '{u}'}}), (b:Node {{id: '{v}'}}) CREATE (a)-[:{edge_type}]->(b)"

                cypher_statements.append(cypher)

            # Write to file
            content = ";\n".join(cypher_statements) + ";"
            success = FileOperations.write_text(filepath, content)

            if success:
                logger.info("Exported RPG to Cypher: %s", filepath)
            return success

        except Exception as e:
            logger.error("Error exporting to Cypher: %s", str(e))
            return False

    @staticmethod
    def export_to_dot(rpg: RepositoryPlanningGraph, filepath: str) -> bool:
        """
        Export RPG to DOT format (Graphviz)

        Args:
            rpg: Repository Planning Graph
            filepath: Output file path

        Returns:
            True if successful
        """
        try:
            # Convert to DOT format
            dot_lines = ["digraph RPG {"]
            dot_lines.append('  rankdir=TB;')
            dot_lines.append('  node [shape=box, style=rounded];')

            # Add nodes
            for node_id, data in rpg.graph.nodes(data=True):
                label = data.get("name", node_id)
                node_type = data.get("type", "")
                color = {
                    "root": "lightblue",
                    "intermediate": "lightgreen",
                    "leaf": "lightyellow"
                }.get(node_type, "white")

                dot_lines.append(
                    f'  "{node_id}" [label="{label}", fillcolor="{color}", style="filled,rounded"];'
                )

            # Add edges
            for u, v, data in rpg.graph.edges(data=True):
                edge_type = data.get("type", "")
                style = {
                    "hierarchy": "solid",
                    "data_flow": "dashed",
                    "execution_order": "dotted"
                }.get(edge_type, "solid")

                dot_lines.append(f'  "{u}" -> "{v}" [style={style}];')

            dot_lines.append("}")

            content = "\n".join(dot_lines)
            success = FileOperations.write_text(filepath, content)

            if success:
                logger.info("Exported RPG to DOT: %s", filepath)
            return success

        except Exception as e:
            logger.error("Error exporting to DOT: %s", str(e))
            return False

    @staticmethod
    def create_checkpoint(
        rpg: RepositoryPlanningGraph, stage: str, output_dir: str = "output/checkpoints"
    ) -> bool:
        """
        Create checkpoint of current RPG state

        Args:
            rpg: Repository Planning Graph
            stage: Stage name (e.g., "stage1", "stage2")
            output_dir: Checkpoint directory

        Returns:
            True if successful
        """
        try:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"{stage}_{timestamp}"

            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Save JSON (human-readable)
            json_path = Path(output_dir) / f"{checkpoint_name}.json"
            GraphPersistence.save_json(rpg, str(json_path))

            # Save pickle (fast restore)
            pickle_path = Path(output_dir) / f"{checkpoint_name}.pkl"
            GraphPersistence.save_pickle(rpg, str(pickle_path))

            logger.info("Created checkpoint: %s", checkpoint_name)
            return True

        except Exception as e:
            logger.error("Error creating checkpoint: %s", str(e))
            return False

    @staticmethod
    def load_checkpoint(checkpoint_path: str) -> Optional[RepositoryPlanningGraph]:
        """
        Load RPG from checkpoint

        Args:
            checkpoint_path: Path to checkpoint file (.json or .pkl)

        Returns:
            RepositoryPlanningGraph or None if failed
        """
        path = Path(checkpoint_path)

        if path.suffix == ".json":
            return GraphPersistence.load_json(str(path))
        elif path.suffix == ".pkl":
            return GraphPersistence.load_pickle(str(path))
        else:
            logger.error("Unknown checkpoint format: %s", path.suffix)
            return None

    @staticmethod
    def export_summary(rpg: RepositoryPlanningGraph, filepath: str) -> bool:
        """
        Export human-readable summary of RPG

        Args:
            rpg: Repository Planning Graph
            filepath: Output file path

        Returns:
            True if successful
        """
        try:
            lines = []
            lines.append("=" * 80)
            lines.append("REPOSITORY PLANNING GRAPH SUMMARY")
            lines.append("=" * 80)
            lines.append(f"\nGoal: {rpg.repository_goal}")
            lines.append(f"\nCreated: {rpg.metadata.get('created', 'Unknown')}")

            # Statistics
            stats = rpg.get_stats()
            lines.append("\n" + "-" * 80)
            lines.append("STATISTICS")
            lines.append("-" * 80)
            for key, value in stats.items():
                lines.append(f"{key}: {value}")

            # Modules (root nodes)
            lines.append("\n" + "-" * 80)
            lines.append("MODULES")
            lines.append("-" * 80)
            for node_id in rpg.get_root_nodes():
                node = rpg.get_node(node_id)
                lines.append(f"\n{node['name']}:")
                lines.append(f"  Functionality: {node.get('functionality', 'N/A')}")
                lines.append(f"  Status: {node.get('status', 'N/A')}")

                # List children
                children = rpg.get_dependents(node_id)
                if children:
                    lines.append(f"  Children ({len(children)}):")
                    for child_id in children[:5]:  # Show first 5
                        child = rpg.get_node(child_id)
                        lines.append(f"    - {child['name']}")
                    if len(children) > 5:
                        lines.append(f"    ... and {len(children) - 5} more")

            lines.append("\n" + "=" * 80)

            content = "\n".join(lines)
            success = FileOperations.write_text(filepath, content)

            if success:
                logger.info("Exported RPG summary: %s", filepath)
            return success

        except Exception as e:
            logger.error("Error exporting summary: %s", str(e))
            return False
