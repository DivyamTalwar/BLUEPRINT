import threading
import time

from src.core.rpg import EdgeType, NodeType, RepositoryPlanningGraph
from src.stage3.stage3_orchestrator import Stage3Orchestrator
from src.stage3.topological_traversal import TopologicalTraversal


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class _ConcurrentTDD:
    def __init__(self, delay=0.05):
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self.started_at = {}
        self.finished_at = {}
        self.lock = threading.Lock()

    def generate(self, _rpg, node_id):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.started_at[node_id] = time.perf_counter()
        time.sleep(self.delay)
        with self.lock:
            self.finished_at[node_id] = time.perf_counter()
            self.active -= 1
        return True, {
            "implementation": f"def {node_id}(): pass",
            "test_code": "",
            "status": "validated",
            "validation_method": "docker",
            "errors": [],
            "attempts": 1,
        }


def _orchestrator(tdd, workers=4):
    orchestrator = Stage3Orchestrator.__new__(Stage3Orchestrator)
    orchestrator.tdd_engine = tdd
    orchestrator.parallel_requests = workers
    orchestrator.logger = _Logger()
    return orchestrator


def test_independent_nodes_run_concurrently_and_commit_deterministically():
    rpg = RepositoryPlanningGraph("parallel")
    node_ids = [
        rpg.add_node(f"task-{index}", NodeType.LEAF, node_id=f"task_{index}")
        for index in range(4)
    ]
    traversal = TopologicalTraversal(rpg)
    tdd = _ConcurrentTDD()

    started = time.perf_counter()
    generated = _orchestrator(tdd)._generate_all_code(
        rpg, traversal, node_ids, checkpoint_interval=99
    )
    elapsed = time.perf_counter() - started

    assert tdd.max_active == 4
    assert elapsed < 0.15
    assert list(generated) == node_ids


def test_dependency_wave_waits_for_predecessors():
    rpg = RepositoryPlanningGraph("dependencies")
    first = rpg.add_node("first", NodeType.LEAF, node_id="first")
    second = rpg.add_node("second", NodeType.LEAF, node_id="second")
    final = rpg.add_node("final", NodeType.LEAF, node_id="final")
    rpg.add_edge(first, final, EdgeType.DATA_FLOW)
    rpg.add_edge(second, final, EdgeType.DATA_FLOW)
    traversal = TopologicalTraversal(rpg)
    tdd = _ConcurrentTDD(delay=0.02)

    generated = _orchestrator(tdd)._generate_all_code(
        rpg, traversal, [first, second, final], checkpoint_interval=99
    )

    assert list(generated) == [first, second, final]
    assert tdd.started_at[final] >= max(
        tdd.finished_at[first], tdd.finished_at[second]
    )
