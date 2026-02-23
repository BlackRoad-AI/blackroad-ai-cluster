"""
BlackRoad AI Cluster — Orchestration system for distributed AI inference.
Node health monitoring, load balancing, and job scheduling.
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# ── ANSI colours ─────────────────────────────────────────────────────────────
R = "\033[0;31m"; G = "\033[0;32m"; Y = "\033[1;33m"
C = "\033[0;36m"; B = "\033[0;34m"; M = "\033[0;35m"; NC = "\033[0m"
BOLD = "\033[1m"

DB_PATH = Path.home() / ".blackroad" / "ai_cluster.db"


# ── Data models ───────────────────────────────────────────────────────────────
@dataclass
class ClusterNode:
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    host: str = "localhost"
    port: int = 8080
    gpu_count: int = 1
    gpu_memory_gb: float = 24.0
    cpu_cores: int = 16
    ram_gb: float = 64.0
    status: str = "offline"   # online, offline, degraded
    current_load: float = 0.0
    max_concurrent_jobs: int = 4
    registered_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_heartbeat: Optional[str] = None


@dataclass
class ClusterJob:
    job_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    model_id: str = ""
    node_id: Optional[str] = None
    job_type: str = "inference"   # inference, training, benchmark
    priority: int = 5
    gpu_required: int = 1
    status: str = "queued"   # queued, running, done, failed
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


@dataclass
class LoadBalancerPolicy:
    policy_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "round_robin"   # round_robin, least_loaded, gpu_affinity
    max_queue_depth: int = 100
    health_check_interval_s: int = 30
    failover_enabled: bool = True
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ClusterHealth:
    total_nodes: int
    online_nodes: int
    degraded_nodes: int
    total_gpus: int
    available_gpus: int
    queued_jobs: int
    running_jobs: int
    completed_jobs: int
    avg_load: float
    checked_at: str


# ── Core class ────────────────────────────────────────────────────────────────
class AIClusterOrchestrator:
    """Distributed AI cluster orchestration with health monitoring and scheduling."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._init_db()
        self._rr_index = 0

    def _init_db(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                name TEXT, host TEXT, port INTEGER,
                gpu_count INTEGER, gpu_memory_gb REAL,
                cpu_cores INTEGER, ram_gb REAL,
                status TEXT, current_load REAL,
                max_concurrent_jobs INTEGER,
                registered_at TEXT, last_heartbeat TEXT
            );
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                model_id TEXT, node_id TEXT,
                job_type TEXT, priority INTEGER,
                gpu_required INTEGER, status TEXT,
                input_tokens INTEGER, output_tokens INTEGER,
                latency_ms REAL,
                created_at TEXT, started_at TEXT, completed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS lb_policies (
                policy_id TEXT PRIMARY KEY,
                name TEXT, max_queue_depth INTEGER,
                health_check_interval_s INTEGER,
                failover_enabled INTEGER,
                created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS health_snapshots (
                snap_id TEXT PRIMARY KEY,
                snapshot_json TEXT,
                created_at TEXT
            );
        """)
        self._conn.commit()

    def register_node(self, node: ClusterNode) -> ClusterNode:
        """Add a node to the cluster."""
        node.status = "online"
        node.last_heartbeat = datetime.utcnow().isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (node.node_id, node.name, node.host, node.port,
             node.gpu_count, node.gpu_memory_gb, node.cpu_cores,
             node.ram_gb, node.status, node.current_load,
             node.max_concurrent_jobs, node.registered_at, node.last_heartbeat)
        )
        self._conn.commit()
        print(f"{G}✓{NC} Node {BOLD}{node.name}{NC} [{C}{node.node_id}{NC}] "
              f"online — {node.gpu_count}×GPU {node.gpu_memory_gb}GB")
        return node

    def submit_job(self, job: ClusterJob) -> ClusterJob:
        """Submit a job to the cluster queue."""
        self._conn.execute(
            "INSERT OR REPLACE INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (job.job_id, job.model_id, job.node_id, job.job_type,
             job.priority, job.gpu_required, job.status,
             job.input_tokens, job.output_tokens, job.latency_ms,
             job.created_at, job.started_at, job.completed_at)
        )
        self._conn.commit()
        print(f"{C}→{NC} Job {BOLD}{job.job_id}{NC} queued "
              f"[{job.job_type}, model={job.model_id}, priority={job.priority}]")
        return job

    def schedule_jobs(self) -> int:
        """Assign queued jobs to available nodes. Returns count of scheduled jobs."""
        queued = self._conn.execute(
            "SELECT job_id, gpu_required, priority FROM jobs "
            "WHERE status='queued' ORDER BY priority DESC, created_at ASC LIMIT 20"
        ).fetchall()
        nodes = self._conn.execute(
            "SELECT node_id, max_concurrent_jobs, current_load FROM nodes "
            "WHERE status='online' ORDER BY current_load ASC"
        ).fetchall()
        if not queued or not nodes:
            print(f"{Y}⚠{NC}  Nothing to schedule (queued={len(queued)}, "
                  f"nodes={len(nodes)})")
            return 0
        scheduled = 0
        for job_id, gpu_req, priority in queued:
            # Round-robin node assignment
            node_id, max_jobs, load = nodes[self._rr_index % len(nodes)]
            self._rr_index += 1
            running = self._conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE node_id=? AND status='running'",
                (node_id,)
            ).fetchone()[0]
            if running >= max_jobs:
                continue
            now = datetime.utcnow().isoformat()
            self._conn.execute(
                "UPDATE jobs SET status='running', node_id=?, started_at=? WHERE job_id=?",
                (node_id, now, job_id)
            )
            new_load = min(1.0, load + 0.1)
            self._conn.execute(
                "UPDATE nodes SET current_load=? WHERE node_id=?",
                (new_load, node_id)
            )
            scheduled += 1
        self._conn.commit()
        print(f"{G}✓{NC} Scheduled {scheduled}/{len(queued)} jobs across "
              f"{len(nodes)} nodes")
        return scheduled

    def get_cluster_health(self) -> ClusterHealth:
        """Return a snapshot of cluster health."""
        total, online, degraded = 0, 0, 0
        total_gpus, avg_load_sum = 0, 0.0
        rows = self._conn.execute(
            "SELECT status, gpu_count, current_load FROM nodes"
        ).fetchall()
        for status, gpus, load in rows:
            total += 1
            total_gpus += gpus
            avg_load_sum += load
            if status == "online":
                online += 1
            elif status == "degraded":
                degraded += 1
        job_counts = self._conn.execute(
            "SELECT status, COUNT(*) FROM jobs GROUP BY status"
        ).fetchall()
        counts = {r[0]: r[1] for r in job_counts}
        health = ClusterHealth(
            total_nodes=total,
            online_nodes=online,
            degraded_nodes=degraded,
            total_gpus=total_gpus,
            available_gpus=max(0, total_gpus - online),
            queued_jobs=counts.get("queued", 0),
            running_jobs=counts.get("running", 0),
            completed_jobs=counts.get("done", 0),
            avg_load=round(avg_load_sum / max(total, 1), 3),
            checked_at=datetime.utcnow().isoformat(),
        )
        snap_id = str(uuid.uuid4())[:8]
        self._conn.execute(
            "INSERT INTO health_snapshots VALUES (?, ?, ?)",
            (snap_id, json.dumps(asdict(health)), health.checked_at)
        )
        self._conn.commit()
        return health

    def balance_load(self) -> None:
        """Rebalance jobs from overloaded nodes to idle nodes."""
        nodes = self._conn.execute(
            "SELECT node_id, current_load, max_concurrent_jobs FROM nodes WHERE status='online'"
        ).fetchall()
        if len(nodes) < 2:
            print(f"{Y}⚠{NC}  Need ≥2 online nodes to balance.")
            return
        overloaded = [(nid, load, cap) for nid, load, cap in nodes if load > 0.7]
        idle = [(nid, load, cap) for nid, load, cap in nodes if load < 0.3]
        migrations = 0
        for o_id, o_load, _ in overloaded:
            if not idle:
                break
            i_id, i_load, i_cap = idle.pop(0)
            # Move one running job from overloaded → idle
            job = self._conn.execute(
                "SELECT job_id FROM jobs WHERE node_id=? AND status='running' LIMIT 1",
                (o_id,)
            ).fetchone()
            if job:
                self._conn.execute(
                    "UPDATE jobs SET node_id=? WHERE job_id=?", (i_id, job[0])
                )
                self._conn.execute(
                    "UPDATE nodes SET current_load=current_load-0.15 WHERE node_id=?", (o_id,)
                )
                self._conn.execute(
                    "UPDATE nodes SET current_load=current_load+0.15 WHERE node_id=?", (i_id,)
                )
                migrations += 1
        self._conn.commit()
        print(f"{G}✓{NC} Load balanced: {migrations} job(s) migrated, "
              f"{len(overloaded)} overloaded, {len(idle)} idle nodes")

    def list_nodes(self) -> List[ClusterNode]:
        rows = self._conn.execute("SELECT * FROM nodes ORDER BY registered_at DESC").fetchall()
        return [ClusterNode(node_id=r[0], name=r[1], host=r[2], port=r[3],
                            gpu_count=r[4], gpu_memory_gb=r[5], cpu_cores=r[6],
                            ram_gb=r[7], status=r[8], current_load=r[9],
                            max_concurrent_jobs=r[10], registered_at=r[11],
                            last_heartbeat=r[12])
                for r in rows]

    def complete_job(self, job_id: str, output_tokens: int = 128) -> None:
        """Mark a job as completed."""
        now = datetime.utcnow().isoformat()
        self._conn.execute(
            "UPDATE jobs SET status='done', completed_at=?, output_tokens=?, latency_ms=? "
            "WHERE job_id=?",
            (now, output_tokens, round(random.uniform(50, 500), 1), job_id)
        )
        job = self._conn.execute("SELECT node_id FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if job and job[0]:
            self._conn.execute(
                "UPDATE nodes SET current_load=MAX(0,current_load-0.1) WHERE node_id=?",
                (job[0],)
            )
        self._conn.commit()
        print(f"{G}✓{NC} Job {C}{job_id}{NC} completed ({output_tokens} tokens)")

    def close(self) -> None:
        self._conn.close()


# ── helpers ───────────────────────────────────────────────────────────────────
def _load_bar(val: float, width: int = 20) -> str:
    filled = int(val * width)
    color = G if val < 0.5 else (Y if val < 0.8 else R)
    return color + "█" * filled + NC + "░" * (width - filled)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ai-cluster", description="BlackRoad AI Cluster Orchestrator"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    nd = sub.add_parser("node", help="Register a node")
    nd.add_argument("--name", required=True)
    nd.add_argument("--host", default="localhost")
    nd.add_argument("--port", type=int, default=8080)
    nd.add_argument("--gpus", type=int, default=1)
    nd.add_argument("--gpu-mem", type=float, default=24.0)

    jb = sub.add_parser("job", help="Submit a job")
    jb.add_argument("--model-id", required=True)
    jb.add_argument("--type", choices=["inference", "training", "benchmark"],
                    default="inference")
    jb.add_argument("--priority", type=int, default=5)
    jb.add_argument("--gpus", type=int, default=1)

    sub.add_parser("schedule", help="Schedule queued jobs")
    sub.add_parser("health", help="Show cluster health")
    sub.add_parser("balance", help="Rebalance load")
    sub.add_parser("nodes", help="List all nodes")

    done = sub.add_parser("complete", help="Mark job done")
    done.add_argument("job_id")
    done.add_argument("--tokens", type=int, default=128)

    args = parser.parse_args()
    orch = AIClusterOrchestrator()

    try:
        if args.cmd == "node":
            n = ClusterNode(name=args.name, host=args.host, port=args.port,
                            gpu_count=args.gpus, gpu_memory_gb=args.gpu_mem)
            orch.register_node(n)

        elif args.cmd == "job":
            j = ClusterJob(model_id=args.model_id, job_type=args.type,
                           priority=args.priority, gpu_required=args.gpus)
            orch.submit_job(j)

        elif args.cmd == "schedule":
            orch.schedule_jobs()

        elif args.cmd == "health":
            h = orch.get_cluster_health()
            print(f"\n{BOLD}{B}── Cluster Health ─────────────────────{NC}")
            print(f"  {C}Nodes{NC}    {h.online_nodes}/{h.total_nodes} online "
                  f"({h.degraded_nodes} degraded)")
            print(f"  {C}GPUs{NC}     {h.total_gpus} total | "
                  f"avg load {_load_bar(h.avg_load)} {h.avg_load:.0%}")
            print(f"  {C}Jobs{NC}     queued={Y}{h.queued_jobs}{NC} "
                  f"running={C}{h.running_jobs}{NC} "
                  f"done={G}{h.completed_jobs}{NC}")

        elif args.cmd == "balance":
            orch.balance_load()

        elif args.cmd == "nodes":
            nodes = orch.list_nodes()
            if not nodes:
                print(f"{Y}No nodes registered.{NC}")
                return
            for n in nodes:
                icon = G + "●" if n.status == "online" else R + "●"
                print(f"  {icon}{NC} {C}{n.node_id}{NC} {BOLD}{n.name:<20}{NC} "
                      f"{n.host}:{n.port} "
                      f"{n.gpu_count}GPU load={_load_bar(n.current_load, 10)}")

        elif args.cmd == "complete":
            orch.complete_job(args.job_id, args.tokens)

    finally:
        orch.close()


if __name__ == "__main__":
    main()
