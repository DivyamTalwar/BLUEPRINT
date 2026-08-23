# Dependency-aware wavefront generation

Stage 3 schedules independent RPG leaf nodes concurrently while preserving the
graph's dependency order. Nodes in the same topological level share a bounded
worker pool; BLUEPRINT waits for the entire level before starting dependents.
Worker results are committed to the RPG in deterministic topological order.

Configure the upper bound in `config.yaml`:

```yaml
performance:
  parallel_requests: 5
```

Set the value to `1` for serial generation. The router protects aggregate cost,
token, and API-call counters, so `get_stats()` remains exact under concurrency.

Run the deterministic benchmark with:

```bash
python benchmarks/benchmark_wavefront.py
```

The benchmark simulates 50 independent 50 ms requests. On a five-worker pool,
the acceptance target is at least a 4x speed-up over its 2.5 second serial
estimate, with `max_active=5`. Real-world throughput remains subject to provider
rate limits and task latency.
