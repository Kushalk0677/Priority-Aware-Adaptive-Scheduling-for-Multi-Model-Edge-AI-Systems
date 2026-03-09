"""
scheduler.py — All seven scheduler implementations + Task definition.

Schedulers:
  - FIFO            : First in, first out
  - Round Robin     : Equal time slices
  - Static Priority : Fixed priority ordering
  - EDF             : Earliest Deadline First (classical real-time)
  - PQ_Deadline     : Priority Queue with deadline urgency weighting
  - QoS             : Quality-of-Service scheduler (priority tiers + deadlines)
  - PAES            : Priority-Aware Edge Scheduler (adaptive, ours)
"""

import heapq
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np


# ── Task ─────────────────────────────────────────────────────────────────────

@dataclass
class Task:
    model_name:           str
    priority:             float
    expected_latency_ms:  float
    expected_energy_mj:   float
    deadline_ms:          float
    run_fn:               Callable       # returns (actual_latency_ms, actual_energy_mj)
    input_data:           object = None
    arrival_time:         float = field(default_factory=time.perf_counter)
    task_id:              str   = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def paes_score(self, alpha=1.0, beta=1.0, gamma=1.0) -> float:
        """
        Higher score = higher scheduling priority.

        Score = α·P  +  β·(1/latency)  +  γ·(1/energy)

        Intuition:
          - α·P          : reward important tasks
          - β/latency    : reward fast tasks (get them done quickly)
          - γ/energy     : reward cheap tasks (save power for expensive ones)
        """
        return (
            alpha * self.priority
            + beta  * (1.0 / max(self.expected_latency_ms, 1e-6))
            + gamma * (1.0 / max(self.expected_energy_mj,  1e-6))
        )


# ── Result record ─────────────────────────────────────────────────────────────

@dataclass
class TaskResult:
    task_id:          str
    model_name:       str
    scheduler_mode:   str
    actual_latency_ms:float
    actual_energy_mj: float
    deadline_ms:      float
    deadline_missed:  bool
    queue_wait_ms:    float   # time spent waiting before execution


# ── Base Scheduler ────────────────────────────────────────────────────────────

VALID_MODES = (
    "fifo", "round_robin", "static_priority",
    "edf", "pq_deadline", "qos", "paes"
)

# QoS tier thresholds — tasks with priority >= HIGH_TIER get guaranteed slots
QOS_HIGH_TIER  = 2.5
QOS_MED_TIER   = 1.5


class Scheduler:
    def __init__(self, mode: str, alpha=1.0, beta=1.0, gamma=1.0):
        assert mode in VALID_MODES, f"Unknown mode: {mode}"
        self.mode    = mode
        self.alpha   = alpha
        self.beta    = beta
        self.gamma   = gamma
        self.queue:   list             = []
        self.results: list[TaskResult] = []
        self.counter: int              = 0
        self._rr_index: int            = 0

    def submit(self, task: Task):
        """Push a task onto the priority queue."""
        score = self._score(task)
        heapq.heappush(self.queue, (score, self.counter, task))
        self.counter += 1

    def _score(self, task: Task) -> float:
        """Lower heap score = executed first (Python heapq is a min-heap)."""

        if self.mode == "fifo":
            # Arrival order — no priority awareness
            return task.arrival_time

        elif self.mode == "round_robin":
            score = self._rr_index
            self._rr_index += 1
            return float(score)

        elif self.mode == "static_priority":
            # Fixed priority only — no deadline or latency awareness
            return -task.priority

        elif self.mode == "edf":
            # Earliest Deadline First — always run the task whose
            # absolute deadline expires soonest.
            # Absolute deadline = arrival_time + deadline_ms (converted to s)
            abs_deadline = task.arrival_time + task.deadline_ms / 1000.0
            return abs_deadline                             # min-heap → earliest first

        elif self.mode == "pq_deadline":
            # Priority Queue with Deadline Urgency:
            # Combines static priority with deadline proximity.
            # urgency = how close the task is to missing its deadline (0→1)
            # score   = priority + urgency bonus
            now           = time.perf_counter()
            time_to_ddl   = max((task.arrival_time + task.deadline_ms/1000.0) - now, 1e-6)
            urgency       = 1.0 / time_to_ddl              # higher = more urgent
            score         = task.priority + urgency
            return -score                                   # negate for min-heap

        elif self.mode == "qos":
            # Quality-of-Service Scheduler:
            # Tasks are partitioned into three tiers by priority.
            # Within each tier, ordered by deadline urgency.
            # High-tier tasks always preempt lower tiers.
            now         = time.perf_counter()
            time_to_ddl = max((task.arrival_time + task.deadline_ms/1000.0) - now, 1e-6)
            urgency     = 1.0 / time_to_ddl

            if task.priority >= QOS_HIGH_TIER:
                tier = 0                                    # highest tier
            elif task.priority >= QOS_MED_TIER:
                tier = 1
            else:
                tier = 2                                    # lowest tier

            # Primary sort: tier (ascending). Secondary: urgency (descending).
            return (tier, -urgency)

        elif self.mode == "paes":
            return -task.paes_score(self.alpha, self.beta, self.gamma)

    def run_next(self) -> Optional[TaskResult]:
        if not self.queue:
            return None

        _, _, task = heapq.heappop(self.queue)
        enqueue_time = task.arrival_time
        exec_start   = time.perf_counter()
        queue_wait   = (exec_start - enqueue_time) * 1000   # ms

        actual_latency_ms, actual_energy_mj = task.run_fn(task.input_data)

        result = TaskResult(
            task_id           = task.task_id,
            model_name        = task.model_name,
            scheduler_mode    = self.mode,
            actual_latency_ms = actual_latency_ms,
            actual_energy_mj  = actual_energy_mj,
            deadline_ms       = task.deadline_ms,
            deadline_missed   = actual_latency_ms > task.deadline_ms,
            queue_wait_ms     = queue_wait,
        )
        self.results.append(result)
        return result

    def run_all(self):
        while self.queue:
            self.run_next()

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        if not self.results:
            return {}
        latencies = [r.actual_latency_ms for r in self.results]
        energies  = [r.actual_energy_mj  for r in self.results]
        waits     = [r.queue_wait_ms      for r in self.results]
        misses    = [r.deadline_missed    for r in self.results]

        return {
            "scheduler":        self.mode,
            "n_tasks":          len(self.results),
            "avg_latency_ms":   round(float(np.mean(latencies)),  2),
            "p50_latency_ms":   round(float(np.percentile(latencies, 50)), 2),
            "p95_latency_ms":   round(float(np.percentile(latencies, 95)), 2),
            "p99_latency_ms":   round(float(np.percentile(latencies, 99)), 2),
            "avg_wait_ms":      round(float(np.mean(waits)),      2),
            "miss_rate":        round(float(np.mean(misses)),     4),
            "missed_count":     int(sum(misses)),
            "throughput_tps":   round(len(self.results) /
                                      max(sum(latencies)/1000, 1e-6), 2),
            "avg_energy_mj":    round(float(np.mean(energies)),   4),
            "total_energy_mj":  round(float(np.sum(energies)),    2),
        }

    def per_model_stats(self) -> dict:
        """Break down latency and miss rate per model."""
        from collections import defaultdict
        buckets = defaultdict(list)
        for r in self.results:
            buckets[r.model_name].append(r)

        out = {}
        for model, records in buckets.items():
            lats   = [r.actual_latency_ms for r in records]
            misses = [r.deadline_missed    for r in records]
            out[model] = {
                "avg_latency_ms": round(float(np.mean(lats)), 2),
                "p95_latency_ms": round(float(np.percentile(lats, 95)), 2),
                "miss_rate":      round(float(np.mean(misses)), 4),
                "n":              len(records),
            }
        return out