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

Fixes in this revision (v2):
  [1] deadline_missed now uses total response time (queue_wait + inference)
      vs the deadline, not inference time alone. Previously a task waiting
      9900ms then running in 100ms would be marked as "met" — incorrect.

  [2] stats() now reports priority_weighted_avg_wait_ms: each task's wait
      is weighted by its priority before averaging. High-priority tasks
      (e.g. YOLOv5 at 3.0) contribute more to the metric than low-priority
      tasks (e.g. MiDaS at 1.0). This surfaces whether the scheduler is
      actually serving important tasks faster — the unweighted average
      previously hid per-model behaviour.

  [3] PAES scoring adds a deadline-proximity bonus: when a task is within
      PAES_DEADLINE_BONUS_THETA ms of missing its deadline, an urgency term
      is added to its score so it jumps the queue. This addresses the known
      low-load miss rate problem (25.3% at 30 tasks) where the energy term
      occasionally deferred urgent-but-expensive tasks in sparse queues.

  [4] per_model_stats() now also returns avg_wait_ms and p95_wait_ms per
      model, so reactive vs proactive wait can be inspected separately
      rather than averaged into a single number that hides per-model
      behaviour (the Prime Minister vs servant problem).
"""

import heapq
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np


# ── PAES deadline proximity threshold ────────────────────────────────────────
# When time remaining before deadline drops below this value (ms), PAES injects
# an urgency bonus. Tune per deployment.
PAES_DEADLINE_BONUS_THETA  = 150.0   # ms
PAES_DEADLINE_BONUS_WEIGHT = 2.0     # score bonus when threshold triggered


# ── Task ─────────────────────────────────────────────────────────────────────

@dataclass
class Task:
    model_name:           str
    priority:             float
    expected_latency_ms:  float
    expected_energy_mj:   float
    deadline_ms:          float
    run_fn:               Callable
    input_data:           object = None
    arrival_time:         float  = field(default_factory=time.perf_counter)
    task_id:              str    = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def paes_score(self, alpha=1.0, beta=1.0, gamma=1.0) -> float:
        """
        Higher score = higher scheduling priority.

        Score = α·P  +  β·(1/L)  +  γ·(1/E)  [+ deadline bonus if near deadline]

          α·P        : prevent starvation of high-priority tasks (~100x dominant)
          β/L        : SJF logic — short tasks reduce average queue wait
          γ/E        : tie-breaker among same-priority tasks
          bonus      : urgency spike when task nears its deadline [Fix 3]
        """
        base = (
            alpha * self.priority
            + beta  * (1.0 / max(self.expected_latency_ms, 1e-6))
            + gamma * (1.0 / max(self.expected_energy_mj,  1e-6))
        )
        # Fix [3]: deadline proximity bonus
        now               = time.perf_counter()
        abs_deadline      = self.arrival_time + self.deadline_ms / 1000.0
        time_remaining_ms = (abs_deadline - now) * 1000.0
        if time_remaining_ms < PAES_DEADLINE_BONUS_THETA:
            base += PAES_DEADLINE_BONUS_WEIGHT
        return base

    def normalized_paes_score(self,
                               alpha=1.0, beta=1.0, gamma=1.0,
                               p_max=3.0, l_min=35.0, e_min=400.0) -> float:
        """
        Normalized variant: each term in (0,1], making α/β/γ directly
        interpretable as relative importance weights independent of workload
        magnitudes.

        score = α·(P/P_max) + β·(L_min/L) + γ·(E_min/E)
        """
        return (
            alpha * (self.priority / max(p_max, 1e-6))
            + beta  * (l_min / max(self.expected_latency_ms, 1e-6))
            + gamma * (e_min / max(self.expected_energy_mj,  1e-6))
        )


# ── Result record ─────────────────────────────────────────────────────────────

@dataclass
class TaskResult:
    task_id:              str
    model_name:           str
    scheduler_mode:       str
    actual_latency_ms:    float
    actual_energy_mj:     float
    deadline_ms:          float
    deadline_missed:      bool    # Fix [1]: total_response_ms > deadline_ms
    queue_wait_ms:        float
    total_response_ms:    float   # queue_wait_ms + actual_latency_ms
    priority:             float   # stored for priority-weighted wait [Fix 2]


# ── Base Scheduler ────────────────────────────────────────────────────────────

VALID_MODES = (
    "fifo", "round_robin", "static_priority",
    "edf", "pq_deadline", "qos", "paes", "estimated_sjf"
)

QOS_HIGH_TIER = 2.5
QOS_MED_TIER  = 1.5


class Scheduler:
    def __init__(self, mode: str, alpha=1.0, beta=1.0, gamma=1.0):
        assert mode in VALID_MODES, f"Unknown mode: {mode}"
        self.mode       = mode
        self.alpha      = alpha
        self.beta       = beta
        self.gamma      = gamma
        self.queue:     list             = []
        self.results:   list[TaskResult] = []
        self.counter:   int              = 0
        self._rr_index: int              = 0

    def submit(self, task: Task):
        score = self._score(task)
        heapq.heappush(self.queue, (score, self.counter, task))
        self.counter += 1

    def _score(self, task: Task) -> float:
        """Lower heap score = executed first (min-heap)."""

        if self.mode == "fifo":
            return task.arrival_time

        elif self.mode == "round_robin":
            s = self._rr_index
            self._rr_index += 1
            return float(s)

        elif self.mode == "static_priority":
            return -task.priority

        elif self.mode == "edf":
            return task.arrival_time + task.deadline_ms / 1000.0

        elif self.mode == "pq_deadline":
            now         = time.perf_counter()
            time_to_ddl = max((task.arrival_time + task.deadline_ms/1000.0) - now, 1e-6)
            return -(task.priority + 1.0 / time_to_ddl)

        elif self.mode == "qos":
            now         = time.perf_counter()
            time_to_ddl = max((task.arrival_time + task.deadline_ms/1000.0) - now, 1e-6)
            urgency     = 1.0 / time_to_ddl
            tier = 0 if task.priority >= QOS_HIGH_TIER else (1 if task.priority >= QOS_MED_TIER else 2)
            return (tier, -urgency)

        elif self.mode == "estimated_sjf":
            return -task.paes_score(0.0, 1.0, 0.0)

        elif self.mode == "paes":
            return -task.paes_score(self.alpha, self.beta, self.gamma)

    def run_next(self) -> Optional[TaskResult]:
        if not self.queue:
            return None

        _, _, task    = heapq.heappop(self.queue)
        exec_start    = time.perf_counter()
        queue_wait    = (exec_start - task.arrival_time) * 1000   # ms

        actual_latency_ms, actual_energy_mj = task.run_fn(task.input_data)

        total_response_ms = queue_wait + actual_latency_ms  # Fix [1]

        result = TaskResult(
            task_id           = task.task_id,
            model_name        = task.model_name,
            scheduler_mode    = self.mode,
            actual_latency_ms = actual_latency_ms,
            actual_energy_mj  = actual_energy_mj,
            deadline_ms       = task.deadline_ms,
            deadline_missed   = total_response_ms > task.deadline_ms,  # Fix [1]
            queue_wait_ms     = queue_wait,
            total_response_ms = total_response_ms,
            priority          = task.priority,                          # Fix [2]
        )
        self.results.append(result)
        return result

    def run_all(self):
        while self.queue:
            self.run_next()

    # ── Stats ──────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        if not self.results:
            return {}

        latencies  = [r.actual_latency_ms  for r in self.results]
        energies   = [r.actual_energy_mj   for r in self.results]
        waits      = [r.queue_wait_ms       for r in self.results]
        totals     = [r.total_response_ms   for r in self.results]
        misses     = [r.deadline_missed     for r in self.results]
        priorities = [r.priority            for r in self.results]

        # Fix [2]: priority-weighted average queue wait
        total_priority = sum(priorities)
        pw_avg_wait    = sum(w * p for w, p in zip(waits, priorities)) / max(total_priority, 1e-6)

        return {
            "scheduler":                     self.mode,
            "n_tasks":                       len(self.results),
            "avg_latency_ms":                round(float(np.mean(latencies)),  2),
            "p50_latency_ms":                round(float(np.percentile(latencies, 50)), 2),
            "p95_latency_ms":                round(float(np.percentile(latencies, 95)), 2),
            "p99_latency_ms":                round(float(np.percentile(latencies, 99)), 2),
            "avg_wait_ms":                   round(float(np.mean(waits)),      2),
            "priority_weighted_avg_wait_ms": round(pw_avg_wait,                2),  # Fix [2]
            "avg_total_response_ms":         round(float(np.mean(totals)),     2),
            "miss_rate":                     round(float(np.mean(misses)),     4),
            "missed_count":                  int(sum(misses)),
            "throughput_tps":                round(len(self.results) /
                                                   max(sum(latencies)/1000, 1e-6), 2),
            "avg_energy_mj":                 round(float(np.mean(energies)),   4),
            "total_energy_mj":               round(float(np.sum(energies)),    2),
        }

    def per_model_stats(self) -> dict:
        """
        Per-model breakdown of latency, wait, total response, and miss rate.

        Fix [4]: now includes avg_wait_ms and p95_wait_ms per model.
        Models sorted by priority descending (reactive first, proactive last)
        so the PM vs servant split is immediately visible.
        """
        from collections import defaultdict
        buckets = defaultdict(list)
        for r in self.results:
            buckets[r.model_name].append(r)

        out = {}
        for model, records in sorted(buckets.items(),
                                     key=lambda x: -x[1][0].priority):
            lats   = [r.actual_latency_ms  for r in records]
            waits  = [r.queue_wait_ms       for r in records]
            totals = [r.total_response_ms   for r in records]
            misses = [r.deadline_missed     for r in records]

            out[model] = {
                "priority":              records[0].priority,
                "n":                     len(records),
                "avg_latency_ms":        round(float(np.mean(lats)),   2),
                "p95_latency_ms":        round(float(np.percentile(lats, 95)), 2),
                "avg_wait_ms":           round(float(np.mean(waits)),  2),   # Fix [4]
                "p95_wait_ms":           round(float(np.percentile(waits, 95)), 2),
                "avg_total_response_ms": round(float(np.mean(totals)), 2),
                "miss_rate":             round(float(np.mean(misses)), 4),
            }
        return out
