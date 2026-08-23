"""Microbenchmark for dependency-aware Stage 3 wavefront scheduling."""

import threading
import time

from src.core.rpg import NodeType, RepositoryPlanningGraph
from src.stage3.stage3_orchestrator import Stage3Orchestrator
from src.stage3.topological_traversal import TopologicalTraversal


class SilentLogger:
    def info(self, *args, **kwargs):
        pass

    warning = error = info


class SimulatedTDD:
    def __init__(self, delay=0.05):
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def generate(self, _rpg, node_id):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(self.delay)
        with self.lock:
            self.active -= 1
        return True, {
            "implementation": f"def {node_id}(): pass",
            "status": "validated",
            "validation_method": "docker",
        }


def run(node_count=50, workers=5):
    rpg = RepositoryPlanningGraph("wavefront benchmark")
    order = [
        rpg.add_node(f"task-{index}", NodeType.LEAF, node_id=f"task_{index}")
        for index in range(node_count)
    ]
    traversal = TopologicalTraversal(rpg)
    tdd = SimulatedTDD()
    orchestrator = Stage3Orchestrator.__new__(Stage3Orchestrator)
    orchestrator.tdd_engine = tdd
    orchestrator.parallel_requests = workers
    orchestrator.logger = SilentLogger()

    started = time.perf_counter()
    orchestrator._generate_all_code(rpg, traversal, order, checkpoint_interval=999)
    elapsed = time.perf_counter() - started
    serial_estimate = node_count * tdd.delay
    print(
        f"nodes={node_count} workers={workers} elapsed={elapsed:.3f}s "
        f"serial_estimate={serial_estimate:.3f}s "
        f"speedup={serial_estimate / elapsed:.2f}x max_active={tdd.max_active}"
    )


if __name__ == "__main__":
    run()
