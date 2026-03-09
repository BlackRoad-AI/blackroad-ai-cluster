"""
BlackRoad AI Cluster — Orchestration system for distributed AI inference.
Node health monitoring, priority job scheduling, and automatic load balancing.

Usage:
    python src/ai_cluster.py node --name "A100-Node-01" --host 10.0.0.1 --gpus 8 --gpu-mem 80.0
    python src/ai_cluster.py nodes
    python src/ai_cluster.py job --model-id llama3-70b --type inference --priority 8 --gpus 2
    python src/ai_cluster.py schedule
    python src/ai_cluster.py health
    python src/ai_cluster.py balance
    python src/ai_cluster.py complete <job_id> --tokens 1024
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ── ANSI colours ─────────────────────────────────────────────────────────────
R = "\033[0;31m"; G = "\033[0;32m"; Y = "\033[1;33m"
C = "\033[0;36m"; B = "\033[0;34m"; NC = "\033[0m"
BOLD = "\033[1m"

DB_PATH = Path.home() / ".blackroad" / "ai_cluster.db"


# ── Data models ───────────────────────────────────────────────────────────────
@dataclass
class ClusterNode:
    """A GPU compute node in the BlackRoad AI cluster."""
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    host: str = "localhost"
    port: int = 8080
    gpu_count: int = 1
    gpu_type: str = "unknown"
    gpu_memory_gb: float = 24.0
    cpu_cores: int = 8
    ram_gb: float = 32.0
    status: str = "offline"          # online | offline | degraded | draining
    current_load: float = 0.0        # 0.0–1.0 utilisation
    max_concurrent_jobs: int = 4
    registered_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_heartbeat: Optional[str] = None


@dataclass
class ClusterJob:
    """A unit of work to be executed on a cluster node."""
    job_id: str = field(default_factory=lambda: f"job-{str(uuid.uuid4())[:8]}")
    model_id: str = ""
    node_id: Optional[str] = None
    job_type: str = "inference"      # inference | training | embedding | benchmark
    priority: int = 5                # 1 (low) – 10 (critical)
    gpu_required: int = 1
    status: str = "queued"           # queued | running | done | failed | cancelled
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


@dataclass
class ClusterHealth:
    """Point-in-time snapshot of cluster health."""
    total_nodes: int = 0
    online_nodes: int = 0
    offline_nodes: int = 0
    total_gpus: int = 0
    avg_load: float = 0.0
    queued_jobs: int = 0
    running_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    snapshot_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ── Orchestrator ──────────────────────────────────────────────────────────────
class AIClusterOrchestrator:
    """
    BlackRoad AI Cluster orchestrator.

    Manages compute nodes, schedules jobs with least-loaded routing,
    rebalances under high skew, and tracks cluster health — all persisted
    in SQLite for zero-config offline operation.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    # ── Schema ────────────────────────────────────────────────────────────────
    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                node_id              TEXT PRIMARY KEY,
                name                 TEXT DEFAULT '',
                host                 TEXT NOT NULL,
                port                 INTEGER DEFAULT 8080,
                gpu_count            INTEGER DEFAULT 1,
                gpu_type             TEXT DEFAULT 'unknown',
                gpu_memory_gb        REAL DEFAULT 24.0,
                cpu_cores            INTEGER DEFAULT 8,
                ram_gb               REAL DEFAULT 32.0,
                status               TEXT DEFAULT 'offline',
                current_load         REAL DEFAULT 0.0,
                max_concurrent_jobs  INTEGER DEFAULT 4,
                registered_at        TEXT,
                last_heartbeat       TEXT
            );
            CREATE TABLE IF NOT EXISTS jobs (
                job_id        TEXT PRIMARY KEY,
                model_id      TEXT DEFAULT '',
                node_id       TEXT,
                job_type      TEXT DEFAULT 'inference',
                priority      INTEGER DEFAULT 5,
                gpu_required  INTEGER DEFAULT 1,
                status        TEXT DEFAULT 'queued',
                input_tokens  INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                latency_ms    REAL DEFAULT 0.0,
                created_at    TEXT,
                started_at    TEXT,
                completed_at  TEXT
            );
            CREATE TABLE IF NOT EXISTS health_snapshots (
                snap_id       TEXT PRIMARY KEY,
                snapshot_json TEXT,
                created_at    TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
            CREATE INDEX IF NOT EXISTS idx_nodes_load ON nodes(status, current_load);
        """)
        self._conn.commit()

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _node_from_row(self, row) -> ClusterNode:
        return ClusterNode(
            node_id=row["node_id"],
            name=row["name"] or "",
            host=row["host"],
            port=row["port"],
            gpu_count=row["gpu_count"],
            gpu_type=row["gpu_type"],
            gpu_memory_gb=row["gpu_memory_gb"],
            cpu_cores=row["cpu_cores"],
            ram_gb=row["ram_gb"],
            status=row["status"],
            current_load=row["current_load"],
            max_concurrent_jobs=row["max_concurrent_jobs"],
            registered_at=row["registered_at"],
            last_heartbeat=row["last_heartbeat"],
        )

    def _job_from_row(self, row) -> ClusterJob:
        return ClusterJob(
            job_id=row["job_id"],
            model_id=row["model_id"],
            node_id=row["node_id"],
            job_type=row["job_type"],
            priority=row["priority"],
            gpu_required=row["gpu_required"],
            status=row["status"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            latency_ms=row["latency_ms"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
        )

    def _running_job_count(self, node_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE node_id=? AND status='running'",
            (node_id,)
        ).fetchone()
        return row[0] if row else 0

    # ── Public API ────────────────────────────────────────────────────────────
    def register_node(self, node: ClusterNode) -> ClusterNode:
        """Register a compute node and bring it online.

        Args:
            node: ClusterNode instance with hardware specs.

        Returns:
            The registered node with status set to 'online'.
        """
        node.status = "online"
        node.last_heartbeat = datetime.utcnow().isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (node.node_id, node.name, node.host, node.port, node.gpu_count,
             node.gpu_type, node.gpu_memory_gb, node.cpu_cores, node.ram_gb,
             node.status, node.current_load, node.max_concurrent_jobs,
             node.registered_at, node.last_heartbeat)
        )
        self._conn.commit()
        print(f"{G}✓{NC} Node {BOLD}{node.node_id}{NC} {node.name} online — "
              f"{C}{node.gpu_count}×GPU {node.gpu_memory_gb}GB{NC} @ {node.host}:{node.port}")
        return node

    def list_nodes(self) -> List[ClusterNode]:
        """Return all registered nodes sorted by current load."""
        rows = self._conn.execute(
            "SELECT * FROM nodes ORDER BY current_load ASC"
        ).fetchall()
        return [self._node_from_row(r) for r in rows]

    def submit_job(self, job: ClusterJob) -> ClusterJob:
        """Queue a new inference/training job.

        Args:
            job: ClusterJob instance describing the work.

        Returns:
            The job with status set to 'queued'.
        """
        job.status = "queued"
        self._conn.execute(
            "INSERT OR REPLACE INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (job.job_id, job.model_id, job.node_id, job.job_type, job.priority,
             job.gpu_required, job.status, job.input_tokens, job.output_tokens,
             job.latency_ms, job.created_at, job.started_at, job.completed_at)
        )
        self._conn.commit()
        print(f"{C}→{NC} Job {BOLD}{job.job_id}{NC} queued "
              f"[{job.job_type}, model={job.model_id}, priority={job.priority}]")
        return job

    def schedule_jobs(self) -> int:
        """Assign queued jobs to available nodes using least-loaded routing.

        Returns:
            Number of jobs successfully scheduled.
        """
        queued = self._conn.execute(
            "SELECT * FROM jobs WHERE status='queued' "
            "ORDER BY priority DESC, created_at ASC"
        ).fetchall()

        scheduled = 0
        for row in queued:
            node_row = self._conn.execute(
                """SELECT n.node_id, n.gpu_count, n.max_concurrent_jobs, n.current_load
                   FROM nodes n
                   WHERE n.status='online'
                     AND (SELECT COUNT(*) FROM jobs j
                          WHERE j.node_id=n.node_id AND j.status='running')
                          < n.max_concurrent_jobs
                   ORDER BY n.current_load ASC
                   LIMIT 1"""
            ).fetchone()
            if not node_row:
                break

            nid = node_row["node_id"]
            gpu_count = node_row["gpu_count"]
            gpu_req = row["gpu_required"]
            delta = round(gpu_req / max(gpu_count, 1) * 0.25, 4)
            new_load = min(1.0, node_row["current_load"] + delta)
            now = datetime.utcnow().isoformat()

            self._conn.execute(
                "UPDATE jobs SET status='running', node_id=?, started_at=? WHERE job_id=?",
                (nid, now, row["job_id"])
            )
            self._conn.execute(
                "UPDATE nodes SET current_load=?, last_heartbeat=? WHERE node_id=?",
                (new_load, now, nid)
            )
            scheduled += 1

        self._conn.commit()
        print(f"{G}✓{NC} Scheduled {scheduled}/{len(queued)} jobs")
        return scheduled

    def complete_job(self, job_id: str, output_tokens: int = 0,
                     latency_ms: float = 0.0) -> None:
        """Mark a job as done and release its node load.

        Args:
            job_id:        Job to complete.
            output_tokens: Tokens generated.
            latency_ms:    End-to-end latency.
        """
        row = self._conn.execute(
            "SELECT node_id, gpu_required FROM jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        if row:
            nid, gpu_req = row["node_id"], row["gpu_required"]
            self._conn.execute(
                "UPDATE jobs SET status='done', completed_at=?, "
                "output_tokens=?, latency_ms=? WHERE job_id=?",
                (datetime.utcnow().isoformat(), output_tokens, latency_ms, job_id)
            )
            if nid:
                node_row = self._conn.execute(
                    "SELECT gpu_count FROM nodes WHERE node_id=?", (nid,)
                ).fetchone()
                if node_row:
                    delta = round(gpu_req / max(node_row["gpu_count"], 1) * 0.25, 4)
                    self._conn.execute(
                        "UPDATE nodes SET current_load=MAX(0,current_load-?) WHERE node_id=?",
                        (delta, nid)
                    )
            self._conn.commit()
            print(f"{G}✓{NC} Job {C}{job_id}{NC} completed "
                  f"({output_tokens} tokens, {latency_ms:.1f}ms)")

    def fail_job(self, job_id: str, error: str = "") -> None:
        """Mark a job as failed and release its node load."""
        row = self._conn.execute(
            "SELECT node_id, gpu_required FROM jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        if row:
            nid, gpu_req = row["node_id"], row["gpu_required"]
            self._conn.execute(
                "UPDATE jobs SET status='failed', completed_at=? WHERE job_id=?",
                (datetime.utcnow().isoformat(), job_id)
            )
            if nid:
                self._conn.execute(
                    "UPDATE nodes SET current_load=MAX(0,current_load-0.25) WHERE node_id=?",
                    (nid,)
                )
            self._conn.commit()
            print(f"{R}✗{NC} Job {C}{job_id}{NC} failed: {error[:80]}")

    def get_cluster_health(self) -> ClusterHealth:
        """Return a snapshot of cluster health and utilisation."""
        health = ClusterHealth()
        loads: List[float] = []

        for row in self._conn.execute("SELECT * FROM nodes").fetchall():
            health.total_nodes += 1
            health.total_gpus += row["gpu_count"]
            loads.append(row["current_load"])
            if row["status"] == "online":
                health.online_nodes += 1
            else:
                health.offline_nodes += 1

        health.avg_load = round(sum(loads) / max(len(loads), 1), 4)

        for status, cnt in self._conn.execute(
            "SELECT status, COUNT(*) FROM jobs GROUP BY status"
        ).fetchall():
            if status == "queued":
                health.queued_jobs = cnt
            elif status == "running":
                health.running_jobs = cnt
            elif status == "done":
                health.completed_jobs = cnt
            elif status == "failed":
                health.failed_jobs = cnt

        # Persist snapshot
        snap = {
            "total_nodes": health.total_nodes,
            "online_nodes": health.online_nodes,
            "total_gpus": health.total_gpus,
            "avg_load": health.avg_load,
            "queued_jobs": health.queued_jobs,
            "running_jobs": health.running_jobs,
        }
        self._conn.execute(
            "INSERT INTO health_snapshots VALUES (?,?,?)",
            (str(uuid.uuid4())[:8], json.dumps(snap), health.snapshot_at)
        )
        self._conn.commit()
        return health

    def balance_load(self, skew_threshold: float = 0.35) -> int:
        """Migrate running jobs from overloaded nodes to idle ones.

        A migration triggers when the load difference between any pair of
        online nodes exceeds skew_threshold (default 0.35).

        Returns:
            Number of jobs migrated.
        """
        rows = self._conn.execute(
            "SELECT * FROM nodes WHERE status='online' ORDER BY current_load ASC"
        ).fetchall()
        nodes = [self._node_from_row(r) for r in rows]
        if len(nodes) < 2:
            print(f"{Y}⚠{NC}  Need ≥2 online nodes to balance load.")
            return 0

        migrated = 0
        idle_nodes = [n for n in nodes if n.current_load < 0.3]
        hot_nodes = [n for n in nodes if n.current_load > 0.7]

        for hot in hot_nodes:
            if not idle_nodes:
                break
            target = min(idle_nodes, key=lambda n: n.current_load)
            if hot.current_load - target.current_load < skew_threshold:
                continue
            job_row = self._conn.execute(
                "SELECT job_id FROM jobs WHERE node_id=? AND status='running' "
                "ORDER BY priority ASC LIMIT 1",
                (hot.node_id,)
            ).fetchone()
            if not job_row:
                continue
            delta = 0.15
            self._conn.execute(
                "UPDATE jobs SET node_id=? WHERE job_id=?",
                (target.node_id, job_row["job_id"])
            )
            self._conn.execute(
                "UPDATE nodes SET current_load=MAX(0,current_load-?) WHERE node_id=?",
                (delta, hot.node_id)
            )
            self._conn.execute(
                "UPDATE nodes SET current_load=MIN(1,current_load+?) WHERE node_id=?",
                (delta, target.node_id)
            )
            migrated += 1
            target.current_load = min(1.0, target.current_load + delta)
            if target.current_load >= 0.6:
                idle_nodes.remove(target)

        self._conn.commit()
        print(f"{G}✓{NC} Load balanced: {migrated} job(s) migrated, "
              f"{len(hot_nodes)} overloaded, {len(idle_nodes)} idle nodes")
        return migrated

    def drain_node(self, node_id: str) -> int:
        """Mark node as draining; returns count of running jobs still on it."""
        self._conn.execute(
            "UPDATE nodes SET status='draining' WHERE node_id=?", (node_id,)
        )
        self._conn.commit()
        count = self._running_job_count(node_id)
        print(f"{Y}⚠{NC}  Node {C}{node_id}{NC} draining ({count} jobs still running)")
        return count

    def remove_node(self, node_id: str) -> bool:
        """Remove a node (must have no running jobs)."""
        if self._running_job_count(node_id) > 0:
            print(f"{R}✗{NC} Cannot remove node {node_id}: jobs still running.")
            return False
        self._conn.execute("DELETE FROM nodes WHERE node_id=?", (node_id,))
        self._conn.commit()
        print(f"{G}✓{NC} Node {C}{node_id}{NC} removed.")
        return True

    def list_jobs(self, status: Optional[str] = None,
                  limit: int = 50) -> List[ClusterJob]:
        """List jobs optionally filtered by status."""
        q = "SELECT * FROM jobs"
        params: List = []
        if status:
            q += " WHERE status=?"
            params.append(status)
        q += " ORDER BY priority DESC, created_at ASC LIMIT ?"
        params.append(limit)
        return [self._job_from_row(r)
                for r in self._conn.execute(q, params).fetchall()]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _bar(val: float, width: int = 20) -> str:
    """Coloured utilisation bar."""
    filled = int(val * width)
    color = G if val < 0.5 else (Y if val < 0.8 else R)
    return color + "█" * filled + NC + "░" * (width - filled) + f" {val:.0%}"


# ── CLI ───────────────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ai_cluster",
        description="BlackRoad AI Cluster — distributed inference orchestrator",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # node: register a GPU node
    n = sub.add_parser("node", help="Register a GPU node")
    n.add_argument("--name", default="")
    n.add_argument("--host", required=True)
    n.add_argument("--port", type=int, default=8080)
    n.add_argument("--gpus", type=int, default=1)
    n.add_argument("--gpu-type", default="unknown")
    n.add_argument("--gpu-mem", type=float, default=24.0)
    n.add_argument("--max-jobs", type=int, default=4)

    # nodes: list all nodes
    sub.add_parser("nodes", help="List all nodes")

    # job: submit a job
    j = sub.add_parser("job", help="Submit a job")
    j.add_argument("--model-id", default="")
    j.add_argument("--type", default="inference")
    j.add_argument("--priority", type=int, default=5)
    j.add_argument("--gpus", type=int, default=1)

    # schedule: assign queued jobs to nodes
    sub.add_parser("schedule", help="Schedule queued jobs across nodes")

    # health: cluster health snapshot
    sub.add_parser("health", help="Show cluster health")

    # balance: rebalance load
    bal = sub.add_parser("balance", help="Rebalance job load across nodes")
    bal.add_argument("--skew", type=float, default=0.35)

    # complete: mark a job done
    c = sub.add_parser("complete", help="Mark a job as complete")
    c.add_argument("job_id")
    c.add_argument("--tokens", type=int, default=0)
    c.add_argument("--latency-ms", type=float, default=0.0)

    # fail: mark a job failed
    f = sub.add_parser("fail", help="Mark a job as failed")
    f.add_argument("job_id")
    f.add_argument("--error", default="")

    # drain: drain a node
    d = sub.add_parser("drain", help="Drain a node")
    d.add_argument("node_id")

    # remove: remove a node
    r = sub.add_parser("remove", help="Remove a node")
    r.add_argument("node_id")

    # jobs: list jobs
    jl = sub.add_parser("jobs", help="List jobs")
    jl.add_argument("--status", default=None)

    return p


def main() -> None:
    args = _build_parser().parse_args()
    orch = AIClusterOrchestrator()

    try:
        if args.cmd == "node":
            orch.register_node(ClusterNode(
                name=args.name, host=args.host, port=args.port,
                gpu_count=args.gpus, gpu_type=args.gpu_type,
                gpu_memory_gb=args.gpu_mem, max_concurrent_jobs=args.max_jobs,
            ))

        elif args.cmd == "nodes":
            nodes = orch.list_nodes()
            if not nodes:
                print(f"{Y}No nodes registered.{NC}")
                return
            print(f"\n{BOLD}{B}── Cluster Nodes ──────────────────────{NC}")
            for n in nodes:
                dot = (G + "●") if n.status == "online" else (R + "●")
                running = orch._running_job_count(n.node_id)
                print(f"  {dot}{NC} {C}{n.node_id}{NC} {BOLD}{n.name}{NC}  "
                      f"{n.host}:{n.port}  {n.gpu_count}×GPU  "
                      f"load={_bar(n.current_load, 12)}  "
                      f"jobs={running}/{n.max_concurrent_jobs}")

        elif args.cmd == "job":
            orch.submit_job(ClusterJob(
                model_id=args.model_id, job_type=args.type,
                priority=args.priority, gpu_required=args.gpus,
            ))

        elif args.cmd == "schedule":
            orch.schedule_jobs()

        elif args.cmd == "health":
            h = orch.get_cluster_health()
            print(f"\n{BOLD}{B}── Cluster Health ─────────────────────{NC}")
            print(f"  {C}Nodes{NC}    {h.online_nodes}/{h.total_nodes} online "
                  f"({h.offline_nodes} offline)")
            print(f"  {C}GPUs{NC}     {h.total_gpus} total | avg load {_bar(h.avg_load)}")
            print(f"  {C}Jobs{NC}     queued={h.queued_jobs} "
                  f"running={h.running_jobs} done={h.completed_jobs}")
            print()

        elif args.cmd == "balance":
            orch.balance_load(skew_threshold=args.skew)

        elif args.cmd == "complete":
            orch.complete_job(args.job_id, args.tokens, args.latency_ms)

        elif args.cmd == "fail":
            orch.fail_job(args.job_id, args.error)

        elif args.cmd == "drain":
            orch.drain_node(args.node_id)

        elif args.cmd == "remove":
            orch.remove_node(args.node_id)

        elif args.cmd == "jobs":
            jobs = orch.list_jobs(status=args.status)
            if not jobs:
                print(f"{Y}No jobs found.{NC}")
                return
            print(f"\n{BOLD}{B}── Jobs ────────────────────────────────{NC}")
            for j in jobs:
                sc = {"queued": Y, "running": C, "done": G, "failed": R}.get(j.status, NC)
                print(f"  {C}{j.job_id}{NC}  [{sc}{j.status}{NC}]  "
                      f"model={j.model_id or '—'}  "
                      f"p={j.priority}  node={j.node_id or '—'}")

    finally:
        orch.close()


if __name__ == "__main__":
    main()
