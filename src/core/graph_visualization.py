from typing import Optional, Dict, Any, List
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

from src.core.rpg import RepositoryPlanningGraph, NodeType, EdgeType
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GraphVisualization:
    """Visualize Repository Planning Graph"""

    def __init__(self, rpg: Optional[RepositoryPlanningGraph] = None):
        """
        Initialize visualization

        Args:
            rpg: Optional RPG to visualize
        """
        self.rpg = rpg

    @staticmethod
    def visualize_full_graph(
        rpg: RepositoryPlanningGraph,
        output_path: str,
        figsize: tuple = (20, 15),
        layout: str = "hierarchical",
    ) -> bool:
        """
        Visualize entire RPG

        Args:
            rpg: Repository Planning Graph
            output_path: Output image path
            figsize: Figure size
            layout: Layout algorithm (hierarchical, spring, circular)

        Returns:
            True if successful
        """
        try:
            plt.figure(figsize=figsize)

            # Choose layout
            if layout == "hierarchical":
                pos = nx.spring_layout(rpg.graph, k=2, iterations=50)
            elif layout == "spring":
                pos = nx.spring_layout(rpg.graph)
            elif layout == "circular":
                pos = nx.circular_layout(rpg.graph)
            else:
                pos = nx.spring_layout(rpg.graph)

            # Color nodes by type
            node_colors = []
            for node_id in rpg.graph.nodes():
                node_data = rpg.get_node(node_id)
                node_type = node_data.get("type", "")

                if node_type == NodeType.ROOT.value:
                    node_colors.append("lightblue")
                elif node_type == NodeType.INTERMEDIATE.value:
                    node_colors.append("lightgreen")
                elif node_type == NodeType.LEAF.value:
                    node_colors.append("lightyellow")
                else:
                    node_colors.append("lightgray")

            # Draw nodes
            nx.draw_networkx_nodes(
                rpg.graph, pos, node_color=node_colors, node_size=300, alpha=0.9
            )

            # Draw edges with different styles
            hierarchy_edges = [
                (u, v)
                for u, v, d in rpg.graph.edges(data=True)
                if d.get("type") == EdgeType.HIERARCHY.value
            ]
            data_flow_edges = [
                (u, v)
                for u, v, d in rpg.graph.edges(data=True)
                if d.get("type") == EdgeType.DATA_FLOW.value
            ]
            exec_order_edges = [
                (u, v)
                for u, v, d in rpg.graph.edges(data=True)
                if d.get("type") == EdgeType.EXECUTION_ORDER.value
            ]

            if hierarchy_edges:
                nx.draw_networkx_edges(
                    rpg.graph, pos, edgelist=hierarchy_edges, style="solid", alpha=0.5
                )

            if data_flow_edges:
                nx.draw_networkx_edges(
                    rpg.graph,
                    pos,
                    edgelist=data_flow_edges,
                    style="dashed",
                    edge_color="blue",
                    alpha=0.5,
                )

            if exec_order_edges:
                nx.draw_networkx_edges(
                    rpg.graph,
                    pos,
                    edgelist=exec_order_edges,
                    style="dotted",
                    edge_color="red",
                    alpha=0.5,
                )

            # Draw labels
            labels = {
                node_id: rpg.get_node(node_id).get("name", "")[:20]
                for node_id in rpg.graph.nodes()
            }
            nx.draw_networkx_labels(rpg.graph, pos, labels, font_size=8)

            # Add legend
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D([0], [0], color="lightblue", marker="o", linestyle="", label="Root Nodes"),
                Line2D([0], [0], color="lightgreen", marker="o", linestyle="", label="Intermediate"),
                Line2D([0], [0], color="lightyellow", marker="o", linestyle="", label="Leaf Nodes"),
                Line2D([0], [0], color="black", linestyle="-", label="Hierarchy"),
                Line2D([0], [0], color="blue", linestyle="--", label="Data Flow"),
                Line2D([0], [0], color="red", linestyle=":", label="Execution Order"),
            ]
            plt.legend(handles=legend_elements, loc="upper right")

            plt.title(f"RPG: {rpg.repository_goal[:60]}", fontsize=14)
            plt.axis("off")
            plt.tight_layout()

            # Save
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            logger.info("Saved full graph visualization: %s", output_path)
            return True

        except Exception as e:
            logger.error("Error visualizing graph: %s", str(e))
            return False

    @staticmethod
    def visualize_subgraph(
        rpg: RepositoryPlanningGraph,
        node_ids: List[str],
        output_path: str,
        figsize: tuple = (12, 8),
    ) -> bool:
        """
        Visualize subgraph

        Args:
            rpg: Repository Planning Graph
            node_ids: Nodes to include
            output_path: Output image path
            figsize: Figure size

        Returns:
            True if successful
        """
        try:
            subgraph = rpg.graph.subgraph(node_ids).copy()

            plt.figure(figsize=figsize)
            pos = nx.spring_layout(subgraph, k=1.5)

            # Color by status
            node_colors = []
            for node_id in subgraph.nodes():
                status = rpg.get_node(node_id).get("status", "planned")

                if status == "planned":
                    node_colors.append("lightgray")
                elif status == "designed":
                    node_colors.append("lightblue")
                elif status == "implemented":
                    node_colors.append("lightgreen")
                elif status == "tested":
                    node_colors.append("green")
                elif status == "failed":
                    node_colors.append("red")
                else:
                    node_colors.append("white")

            nx.draw_networkx_nodes(
                subgraph, pos, node_color=node_colors, node_size=500, alpha=0.9
            )

            nx.draw_networkx_edges(subgraph, pos, alpha=0.5)

            labels = {
                node_id: rpg.get_node(node_id).get("name", "") for node_id in subgraph.nodes()
            }
            nx.draw_networkx_labels(subgraph, pos, labels, font_size=10)

            plt.title("Subgraph Visualization", fontsize=14)
            plt.axis("off")
            plt.tight_layout()

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            logger.info("Saved subgraph visualization: %s", output_path)
            return True

        except Exception as e:
            logger.error("Error visualizing subgraph: %s", str(e))
            return False

    @staticmethod
    def visualize_data_flow(
        rpg: RepositoryPlanningGraph, output_path: str, figsize: tuple = (16, 12)
    ) -> bool:
        """
        Visualize data flow only

        Args:
            rpg: Repository Planning Graph
            output_path: Output image path
            figsize: Figure size

        Returns:
            True if successful
        """
        try:
            # Extract data flow edges
            data_flow_graph = nx.DiGraph()

            for u, v, data in rpg.graph.edges(data=True):
                if data.get("type") == EdgeType.DATA_FLOW.value:
                    data_flow_graph.add_edge(u, v, **data)

            # Add nodes
            for node_id in data_flow_graph.nodes():
                node_data = rpg.get_node(node_id)
                data_flow_graph.nodes[node_id].update(node_data)

            plt.figure(figsize=figsize)
            pos = nx.spring_layout(data_flow_graph, k=2)

            # Draw
            nx.draw_networkx_nodes(
                data_flow_graph, pos, node_color="lightblue", node_size=800, alpha=0.9
            )

            nx.draw_networkx_edges(
                data_flow_graph, pos, edge_color="blue", arrows=True, arrowsize=20
            )

            labels = {
                node_id: data_flow_graph.nodes[node_id].get("name", "")
                for node_id in data_flow_graph.nodes()
            }
            nx.draw_networkx_labels(data_flow_graph, pos, labels, font_size=10)

            # Edge labels (data types)
            edge_labels = {}
            for u, v, data in data_flow_graph.edges(data=True):
                data_type = data.get("data_type", "")
                if data_type:
                    edge_labels[(u, v)] = data_type[:15]

            nx.draw_networkx_edge_labels(
                data_flow_graph, pos, edge_labels, font_size=8
            )

            plt.title("Data Flow Visualization", fontsize=14)
            plt.axis("off")
            plt.tight_layout()

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            logger.info("Saved data flow visualization: %s", output_path)
            return True

        except Exception as e:
            logger.error("Error visualizing data flow: %s", str(e))
            return False

    @staticmethod
    def visualize_implementation_status(
        rpg: RepositoryPlanningGraph, output_path: str, figsize: tuple = (18, 12)
    ) -> bool:
        """
        Visualize implementation status

        Args:
            rpg: Repository Planning Graph
            output_path: Output image path
            figsize: Figure size

        Returns:
            True if successful
        """
        try:
            plt.figure(figsize=figsize)
            pos = nx.spring_layout(rpg.graph, k=2)

            # Color by status
            status_colors = {
                "planned": "lightgray",
                "designed": "lightblue",
                "implemented": "lightgreen",
                "tested": "green",
                "failed": "red",
            }

            node_colors = []
            for node_id in rpg.graph.nodes():
                status = rpg.get_node(node_id).get("status", "planned")
                node_colors.append(status_colors.get(status, "white"))

            nx.draw_networkx_nodes(
                rpg.graph, pos, node_color=node_colors, node_size=400, alpha=0.9
            )

            nx.draw_networkx_edges(rpg.graph, pos, alpha=0.3)

            labels = {
                node_id: rpg.get_node(node_id).get("name", "")[:15]
                for node_id in rpg.graph.nodes()
            }
            nx.draw_networkx_labels(rpg.graph, pos, labels, font_size=8)

            # Legend
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D(
                    [0], [0], color=color, marker="o", linestyle="", markersize=10, label=status.capitalize()
                )
                for status, color in status_colors.items()
            ]
            plt.legend(handles=legend_elements, loc="upper right")

            # Status summary
            stats = rpg.get_stats()
            summary_text = f"Planned: {stats.get('planned_nodes', 0)} | "
            summary_text += f"Designed: {stats.get('designed_nodes', 0)} | "
            summary_text += f"Implemented: {stats.get('implemented_nodes', 0)} | "
            summary_text += f"Tested: {stats.get('tested_nodes', 0)} | "
            summary_text += f"Failed: {stats.get('failed_nodes', 0)}"

            plt.title(f"Implementation Status\n{summary_text}", fontsize=14)
            plt.axis("off")
            plt.tight_layout()

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            logger.info("Saved status visualization: %s", output_path)
            return True

        except Exception as e:
            logger.error("Error visualizing status: %s", str(e))
            return False

    @staticmethod
    def create_html_interactive(
        rpg: RepositoryPlanningGraph, output_path: str
    ) -> bool:
        """
        Create interactive HTML visualization (requires pyvis)

        Args:
            rpg: Repository Planning Graph
            output_path: Output HTML path

        Returns:
            True if successful
        """
        try:
            from pyvis.network import Network

            net = Network(height="800px", width="100%", directed=True)

            # Add nodes
            for node_id, data in rpg.graph.nodes(data=True):
                node_type = data.get("type", "")
                status = data.get("status", "planned")

                color = {
                    "root": "#ADD8E6",
                    "intermediate": "#90EE90",
                    "leaf": "#FFFFE0",
                }.get(node_type, "#D3D3D3")

                title = f"""
                <b>{data.get('name', '')}</b><br>
                Type: {node_type}<br>
                Status: {status}<br>
                Functionality: {data.get('functionality', 'N/A')[:100]}
                """

                net.add_node(
                    node_id,
                    label=data.get("name", "")[:20],
                    color=color,
                    title=title,
                )

            # Add edges
            for u, v, data in rpg.graph.edges(data=True):
                edge_type = data.get("type", "")
                net.add_edge(u, v, title=edge_type)

            # Configure physics
            net.set_options("""
            {
                "physics": {
                    "enabled": true,
                    "barnesHut": {
                        "gravitationalConstant": -30000,
                        "centralGravity": 0.3,
                        "springLength": 95,
                        "springConstant": 0.04
                    }
                }
            }
            """)

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            net.save_graph(output_path)

            logger.info("Saved interactive HTML visualization: %s", output_path)
            return True

        except ImportError:
            logger.warning("pyvis not installed. Skipping HTML visualization.")
            return False
        except Exception as e:
            logger.error("Error creating HTML visualization: %s", str(e))
            return False
