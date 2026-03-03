"""
BlackRoad AI Cluster — Orchestration system for distributed AI inference.
Node health monitoring, load balancing, job scheduling, and topology export.

Usage:
    python cluster.py node add --host 192.168.4.64 --port 11434 --gpus 2 --gpu-type A100
    python cluster.py node list
    python cluster.py job assign JOB-001 --gpus 1 --model qwen2.5
    python cluster.py cluster stats
    python cluster.py cluster rebalance
    python cluster.py cluster topology
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── ANSI colours ─────────────────────────────────────────────────────────────
R = "\033[0;31m"; G = "\033[0;32m"; Y = "\033[1;33m"
C = "\033[0;36m"; B = "\033[0;34m"; M = "\033[0;35m"; NC = "\033[0m"
BOLD = "\033[1m"

DB_PATH = Path.home() / ".blackroad" / "ai_cluster.db"

# ── Data models ───────────────────────────────────────────────────────────────
@dataclass
class Node:
    """Represents a compute node in the BlackRoad AI cluster."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    host: str = "localhost"
    port: int = 11434
    gpu_count: int = 1
    gpu_type: str = "unknown"           # e.g. A100, H100, RTX4090, RPI
    status: str = "offline"             # online | offline | degraded | draining
    load: float = 0.0                   # 0.0–1.0 utilisation
    gpu_memory_gb: float = 16.0
    cpu_cores: int = 8
    ram_gb: float = 32.0
    max_jobs: int = 4
    tags: List[str] = field(default_factory=list)
    registered_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_heartbeat: Optional[str] = None

    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"

    def available_capacity(self) -> float:
        """Free fraction (1.0 = fully idle)."""
        return round(max(0.0, 1.0 - self.load), 4)

    def supports(self, gpu_type: str) -> bool:
        if gpu_type in ("any", ""):
            return True
        return self.gpu_type.lower() == gpu_type.lower()


@dataclass
class Job:
    """A unit of work to be executed on a cluster node."""
    job_id: str = field(default_factory=lambda: f"job-{str(uuid.uuid4())[:8]}")
    model_id: str = ""
    node_id: Optional[str] = None
    job_type: str = "inference"         # inference | training | embedding | benchmark
    priority: int = 5                   # 1 (low) – 10 (critical)
    gpu_required: int = 1
    gpu_type_required: str = "any"
    memory_gb_required: float = 0.0
    status: str = "queued"              # queued | running | done | failed | cancelled
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    error_msg: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def is_terminal(self) -> bool:
        return self.status in ("done", "failed", "cancelled")


@dataclass
class ClusterStats:
    """Point-in-time snapshot of cluster health."""
    total_nodes: int = 0
    online_nodes: int = 0
    degraded_nodes: int = 0
    offline_nodes: int = 0
    total_gpus: int = 0
    busy_gpus: int = 0
    avg_load: float = 0.0
    peak_load: float = 0.0
    queued_jobs: int = 0
    running_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    gpu_types: Dict[str, int] = field(default_factory=dict)
    snapshot_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ── Cluster ───────────────────────────────────────────────────────────────────
class Cluster:
    """
    BlackRoad AI Cluster orchestrator.

    Manages compute nodes, schedules jobs with least-loaded routing,
    rebalances under high skew, and exports full topology.
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
                id TEXT PRIMARY KEY,
                host TEXT NOT NULL,
                port INTEGER NOT NULL,
                gpu_count INTEGER DEFAULT 1,
                gpu_type TEXT DEFAULT 'unknown',
                status TEXT DEFAULT 'offline',
                load REAL DEFAULT 0.0,
                gpu_memory_gb REAL DEFAULT 16.0,
                cpu_cores INTEGER DEFAULT 8,
                ram_gb REAL DEFAULT 32.0,
                max_jobs INTEGER DEFAULT 4,
                tags_json TEXT DEFAULT '[]',
                registered_at TEXT,
                last_heartbeat TEXT
            );
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                model_id TEXT,
                node_id TEXT,
                job_type TEXT,
                priority INTEGER DEFAULT 5,
                gpu_required INTEGER DEFAULT 1,
                gpu_type_required TEXT DEFAULT 'any',
                memory_gb_required REAL DEFAULT 0.0,
                status TEXT DEFAULT 'queued',
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                latency_ms REAL DEFAULT 0.0,
                error_msg TEXT,
                created_at TEXT,
                started_at TEXT,
                completed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS cluster_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT,
                node_id TEXT,
                job_id TEXT,
                detail TEXT,
                ts TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
            CREATE INDEX IF NOT EXISTS idx_nodes_status ON nodes(status, load);
        """)
        self._conn.commit()

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _node_from_row(self, row) -> Node:
        return Node(
            id=row["id"], host=row["host"], port=row["port"],
            gpu_count=row["gpu_count"], gpu_type=row["gpu_type"],
            status=row["status"], load=row["load"],
            gpu_memory_gb=row["gpu_memory_gb"], cpu_cores=row["cpu_cores"],
            ram_gb=row["ram_gb"], max_jobs=row["max_jobs"],
            tags=json.loads(row["tags_json"] or "[]"),
            registered_at=row["registered_at"],
            last_heartbeat=row["last_heartbeat"],
        )

    def _job_from_row(self, row) -> Job:
        return Job(
            job_id=row["job_id"], model_id=row["model_id"],
            node_id=row["node_id"], job_type=row["job_type"],
            priority=row["priority"], gpu_required=row["gpu_required"],
            gpu_type_required=row["gpu_type_required"],
            memory_gb_required=row["memory_gb_required"],
            status=row["status"], input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"], latency_ms=row["latency_ms"],
            error_msg=row["error_msg"], created_at=row["created_at"],
            started_at=row["started_at"], completed_at=row["completed_at"],
        )

    def _log_event(self, event_type: str, node_id: str = "",
                   job_id: str = "", detail: str = "") -> None:
        self._conn.execute(
            "INSERT INTO cluster_events VALUES (?,?,?,?,?,?)",
            (str(uuid.uuid4())[:8], event_type, node_id,
             job_id, detail, datetime.utcnow().isoformat())
        )

    def _running_job_count(self, node_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE node_id=? AND status='running'",
            (node_id,)
        ).fetchone()
        return row[0] if row else 0

    # ── Public API ────────────────────────────────────────────────────────────
    def add_node(self, host: str, port: int, gpus: int, gpu_type: str,
                 gpu_memory_gb: float = 16.0, cpu_cores: int = 8,
                 ram_gb: float = 32.0, tags: Optional[List[str]] = None) -> Node:
        """
        Register a new compute node and bring it online.

        Args:
            host:         Hostname or IP of the node.
            port:         Ollama / inference server port.
            gpus:         Number of GPU devices available.
            gpu_type:     GPU model string (e.g. 'A100', 'H100', 'RTX4090').
            gpu_memory_gb: Per-GPU VRAM in GB.
            cpu_cores:    Total CPU cores.
            ram_gb:       System RAM in GB.
            tags:         Arbitrary labels (e.g. ['pi', 'edge']).

        Returns:
            The registered Node object.
        """
        node = Node(
            host=host, port=port, gpu_count=gpus, gpu_type=gpu_type,
            gpu_memory_gb=gpu_memory_gb, cpu_cores=cpu_cores, ram_gb=ram_gb,
            tags=tags or [], status="online",
            last_heartbeat=datetime.utcnow().isoformat(),
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (node.id, node.host, node.port, node.gpu_count, node.gpu_type,
             node.status, node.load, node.gpu_memory_gb, node.cpu_cores,
             node.ram_gb, node.max_jobs, json.dumps(node.tags),
             node.registered_at, node.last_heartbeat)
        )
        self._log_event("node_registered", node_id=node.id,
                        detail=f"{gpus}×{gpu_type} @ {host}:{port}")
        self._conn.commit()
        print(f"{G}✓{NC} Node {BOLD}{node.id}{NC} online — "
              f"{C}{gpus}×{gpu_type}{NC} @ {host}:{port}")
        return node

    def get_available_node(self, requirements: Optional[Dict] = None) -> Optional[Node]:
        """
        Return the least-loaded online node that satisfies requirements.

        Requirements dict keys (all optional):
            gpu_type (str), gpu_required (int), memory_gb (float)

        Returns None if no node can satisfy the request.
        """
        reqs = requirements or {}
        gpu_type = reqs.get("gpu_type", "any")
        gpu_req = int(reqs.get("gpu_required", 1))
        mem_req = float(reqs.get("memory_gb", 0.0))

        rows = self._conn.execute(
            "SELECT * FROM nodes WHERE status='online' ORDER BY load ASC"
        ).fetchall()
        for row in rows:
            node = self._node_from_row(row)
            if node.gpu_count < gpu_req:
                continue
            if gpu_type not in ("any", "") and not node.supports(gpu_type):
                continue
            if mem_req > 0 and node.gpu_memory_gb < mem_req:
                continue
            if self._running_job_count(node.id) >= node.max_jobs:
                continue
            return node
        return None

    def assign_job(self, job_id: str, requirements: Dict) -> Optional[Node]:
        """
        Assign a queued job to the best available node.

        Requirements dict:
            model_id (str), job_type (str), priority (int),
            gpu_required (int), gpu_type (str), memory_gb (float)

        Returns the assigned Node, or None if no capacity available.
        """
        # Upsert the job record if not already queued
        existing = self._conn.execute(
            "SELECT job_id FROM jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        if not existing:
            job = Job(
                job_id=job_id,
                model_id=requirements.get("model_id", ""),
                job_type=requirements.get("job_type", "inference"),
                priority=int(requirements.get("priority", 5)),
                gpu_required=int(requirements.get("gpu_required", 1)),
                gpu_type_required=requirements.get("gpu_type", "any"),
                memory_gb_required=float(requirements.get("memory_gb", 0.0)),
            )
            self._conn.execute(
                "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (job.job_id, job.model_id, None, job.job_type, job.priority,
                 job.gpu_required, job.gpu_type_required, job.memory_gb_required,
                 job.status, job.input_tokens, job.output_tokens,
                 job.latency_ms, job.error_msg, job.created_at,
                 job.started_at, job.completed_at)
            )

        node = self.get_available_node(requirements)
        if node is None:
            print(f"{Y}⚠{NC}  No node available for job {C}{job_id}{NC} "
                  f"(requirements: {requirements})")
            return None

        now = datetime.utcnow().isoformat()
        gpu_req = int(requirements.get("gpu_required", 1))
        load_increment = round(gpu_req / max(node.gpu_count, 1) * 0.25, 4)
        new_load = min(1.0, node.load + load_increment)

        self._conn.execute(
            "UPDATE jobs SET status='running', node_id=?, started_at=? WHERE job_id=?",
            (node.id, now, job_id)
        )
        self._conn.execute(
            "UPDATE nodes SET load=?, last_heartbeat=? WHERE id=?",
            (new_load, now, node.id)
        )
        self._log_event("job_assigned", node_id=node.id, job_id=job_id,
                        detail=f"load {node.load:.2f}→{new_load:.2f}")
        self._conn.commit()
        print(f"{G}→{NC} Job {C}{job_id}{NC} assigned to node "
              f"{BOLD}{node.id}{NC} ({node.gpu_type}) "
              f"load={_bar(new_load, 10)}")
        return node

    def rebalance_jobs(self, skew_threshold: float = 0.35) -> int:
        """
        Migrate running jobs from overloaded nodes to underutilised ones.

        A rebalance triggers when load difference between any pair of online
        nodes exceeds `skew_threshold` (default 0.35).

        Returns the number of jobs migrated.
        """
        rows = self._conn.execute(
            "SELECT * FROM nodes WHERE status='online' ORDER BY load ASC"
        ).fetchall()
        nodes = [self._node_from_row(r) for r in rows]
        if len(nodes) < 2:
            print(f"{Y}⚠{NC}  Need ≥2 online nodes to rebalance.")
            return 0

        migrated = 0
        idle_nodes = [n for n in nodes if n.load < 0.3]
        hot_nodes  = [n for n in nodes if n.load > 0.7]

        for hot in hot_nodes:
            if not idle_nodes:
                break
            target = min(idle_nodes, key=lambda n: n.load)
            if hot.load - target.load < skew_threshold:
                continue
            # Migrate the least-critical running job
            job_row = self._conn.execute(
                "SELECT job_id FROM jobs WHERE node_id=? AND status='running' "
                "ORDER BY priority ASC LIMIT 1",
                (hot.id,)
            ).fetchone()
            if not job_row:
                continue
            jid = job_row[0]
            delta = 0.15
            self._conn.execute(
                "UPDATE jobs SET node_id=? WHERE job_id=?", (target.id, jid)
            )
            self._conn.execute(
                "UPDATE nodes SET load=MAX(0,load-?) WHERE id=?", (delta, hot.id)
            )
            self._conn.execute(
                "UPDATE nodes SET load=MIN(1,load+?) WHERE id=?", (delta, target.id)
            )
            self._log_event("job_migrated", node_id=hot.id, job_id=jid,
                            detail=f"→ {target.id}")
            migrated += 1
            # Refresh target load estimate
            target.load = min(1.0, target.load + delta)
            if target.load >= 0.6:
                idle_nodes.remove(target)

        self._conn.commit()
        status = f"{G}✓{NC}" if migrated else f"{C}–{NC}"
        print(f"{status} Rebalance: {migrated} job(s) migrated, "
              f"{len(hot_nodes)} hot, {len(idle_nodes)} idle nodes")
        return migrated

    def cluster_stats(self) -> ClusterStats:
        """Return a comprehensive snapshot of cluster health and utilisation."""
        node_rows = self._conn.execute("SELECT * FROM nodes").fetchall()
        stats = ClusterStats()
        loads = []
        gpu_type_map: Dict[str, int] = {}

        for row in node_rows:
            n = self._node_from_row(row)
            stats.total_nodes += 1
            stats.total_gpus += n.gpu_count
            loads.append(n.load)
            gpu_type_map[n.gpu_type] = gpu_type_map.get(n.gpu_type, 0) + n.gpu_count
            if n.status == "online":
                stats.online_nodes += 1
                stats.busy_gpus += int(n.gpu_count * n.load)
            elif n.status == "degraded":
                stats.degraded_nodes += 1
            else:
                stats.offline_nodes += 1

        stats.avg_load = round(sum(loads) / max(len(loads), 1), 4)
        stats.peak_load = round(max(loads, default=0.0), 4)
        stats.gpu_types = gpu_type_map

        job_counts = self._conn.execute(
            "SELECT status, COUNT(*) FROM jobs GROUP BY status"
        ).fetchall()
        for status, cnt in job_counts:
            if status == "queued":
                stats.queued_jobs = cnt
            elif status == "running":
                stats.running_jobs = cnt
            elif status == "done":
                stats.completed_jobs = cnt
            elif status == "failed":
                stats.failed_jobs = cnt

        return stats

    def export_topology(self, output_path: Optional[Path] = None) -> Dict:
        """
        Export the full cluster topology as a JSON-serialisable dict.

        Includes all nodes with their specs, current load, active job counts,
        and edge connections (node → jobs). Writes to file if output_path given.

        Returns the topology dict.
        """
        nodes = [self._node_from_row(r)
                 for r in self._conn.execute("SELECT * FROM nodes").fetchall()]
        jobs = [self._job_from_row(r)
                for r in self._conn.execute("SELECT * FROM jobs").fetchall()]

        job_map: Dict[str, List[Dict]] = {}
        for j in jobs:
            nid = j.node_id or "__unassigned__"
            job_map.setdefault(nid, []).append({
                "job_id": j.job_id, "model_id": j.model_id,
                "status": j.status, "priority": j.priority,
                "latency_ms": j.latency_ms,
            })

        topology = {
            "schema_version": "1.0",
            "exported_at": datetime.utcnow().isoformat(),
            "cluster": {
                "total_nodes": len(nodes),
                "total_gpus": sum(n.gpu_count for n in nodes),
                "gpu_types": list({n.gpu_type for n in nodes}),
            },
            "nodes": [
                {
                    **{k: v for k, v in asdict(n).items()},
                    "endpoint": n.endpoint(),
                    "available_capacity": n.available_capacity(),
                    "active_jobs": job_map.get(n.id, []),
                }
                for n in nodes
            ],
            "unassigned_jobs": job_map.get("__unassigned__", []),
            "events": [
                dict(r) for r in self._conn.execute(
                    "SELECT * FROM cluster_events ORDER BY ts DESC LIMIT 50"
                ).fetchall()
            ],
        }

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(topology, indent=2))
            print(f"{G}✓{NC} Topology exported → {C}{output_path}{NC}")

        return topology

    # ── Additional utilities ──────────────────────────────────────────────────
    def heartbeat(self, node_id: str, load: float) -> None:
        """Update a node's load and heartbeat timestamp."""
        self._conn.execute(
            "UPDATE nodes SET load=?, last_heartbeat=? WHERE id=?",
            (round(max(0.0, min(1.0, load)), 4),
             datetime.utcnow().isoformat(), node_id)
        )
        self._conn.commit()

    def complete_job(self, job_id: str, output_tokens: int = 0,
                     latency_ms: float = 0.0) -> None:
        """Mark a job as done and release load from its node."""
        row = self._conn.execute(
            "SELECT node_id, gpu_required FROM jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        if row:
            nid, gpu_req = row
            self._conn.execute(
                "UPDATE jobs SET status='done', completed_at=?, "
                "output_tokens=?, latency_ms=? WHERE job_id=?",
                (datetime.utcnow().isoformat(), output_tokens, latency_ms, job_id)
            )
            if nid:
                node_row = self._conn.execute(
                    "SELECT gpu_count FROM nodes WHERE id=?", (nid,)
                ).fetchone()
                if node_row:
                    delta = round(gpu_req / max(node_row[0], 1) * 0.25, 4)
                    self._conn.execute(
                        "UPDATE nodes SET load=MAX(0,load-?) WHERE id=?", (delta, nid)
                    )
            self._log_event("job_completed", node_id=nid or "", job_id=job_id,
                            detail=f"{output_tokens} tokens {latency_ms:.1f}ms")
            self._conn.commit()
            print(f"{G}✓{NC} Job {C}{job_id}{NC} done "
                  f"({output_tokens} tokens, {latency_ms:.1f}ms)")

    def fail_job(self, job_id: str, error: str = "") -> None:
        """Mark a job as failed and release its node load."""
        row = self._conn.execute(
            "SELECT node_id, gpu_required FROM jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        if row:
            nid, gpu_req = row
            self._conn.execute(
                "UPDATE jobs SET status='failed', error_msg=?, completed_at=? "
                "WHERE job_id=?",
                (error, datetime.utcnow().isoformat(), job_id)
            )
            if nid:
                self._conn.execute(
                    "UPDATE nodes SET load=MAX(0,load-0.25) WHERE id=?", (nid,)
                )
            self._log_event("job_failed", job_id=job_id, detail=error[:200])
            self._conn.commit()
            print(f"{R}✗{NC} Job {C}{job_id}{NC} failed: {error[:80]}")

    def drain_node(self, node_id: str) -> int:
        """Mark node as draining; returns count of running jobs to migrate."""
        self._conn.execute(
            "UPDATE nodes SET status='draining' WHERE id=?", (node_id,)
        )
        self._log_event("node_draining", node_id=node_id)
        self._conn.commit()
        count = self._running_job_count(node_id)
        print(f"{Y}⚠{NC}  Node {C}{node_id}{NC} draining "
              f"({count} jobs still running)")
        return count

    def remove_node(self, node_id: str) -> bool:
        """Remove a node (must be offline or draining with no running jobs)."""
        if self._running_job_count(node_id) > 0:
            print(f"{R}✗{NC} Cannot remove node {node_id}: jobs still running.")
            return False
        self._conn.execute("DELETE FROM nodes WHERE id=?", (node_id,))
        self._log_event("node_removed", node_id=node_id)
        self._conn.commit()
        print(f"{G}✓{NC} Node {C}{node_id}{NC} removed.")
        return True

    def list_nodes(self) -> List[Node]:
        rows = self._conn.execute(
            "SELECT * FROM nodes ORDER BY load ASC"
        ).fetchall()
        return [self._node_from_row(r) for r in rows]

    def list_jobs(self, status: Optional[str] = None,
                  limit: int = 50) -> List[Job]:
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
        self._conn.close()


# ── Ollama-first orchestration ────────────────────────────────────────────────
# All mentions of the handles below are routed to the local Ollama service.
# No external AI provider (Copilot, Claude, ChatGPT, etc.) is involved.
OLLAMA_HANDLES: frozenset = frozenset(
    {"@copilot", "@lucidia", "@blackboxprogramming", "@ollama"}
)


@dataclass
class ClusterNode:
    """Compute node registered in the BlackRoad AI cluster (Ollama-first)."""
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    host: str = "localhost"
    port: int = 11434
    gpu_count: int = 1
    gpu_memory_gb: float = 16.0
    max_concurrent_jobs: int = 4
    status: str = "offline"
    current_load: float = 0.0
    last_heartbeat: Optional[str] = None
    registered_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class ClusterJob:
    """Unit of work dispatched to a ClusterNode running Ollama."""
    job_id: str = field(
        default_factory=lambda: f"job-{str(uuid.uuid4())[:8]}"
    )
    model_id: str = ""
    node_id: Optional[str] = None
    job_type: str = "inference"
    priority: int = 5
    gpu_required: int = 1
    status: str = "queued"
    output_tokens: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class ClusterHealth:
    """Point-in-time health snapshot for the Ollama cluster."""
    total_nodes: int = 0
    online_nodes: int = 0
    total_gpus: int = 0
    avg_load: float = 0.0
    queued_jobs: int = 0
    running_jobs: int = 0


class AIClusterOrchestrator:
    """
    Manages BlackRoad cluster nodes and jobs, all served by local Ollama.

    Every inference request — whether triggered by @copilot, @lucidia,
    @blackboxprogramming, or @ollama — is routed here and dispatched to
    a local Ollama endpoint.  No external AI provider is used.
    """

    _DEFAULT_DB = Path.home() / ".blackroad" / "ai_cluster_orch.db"

    def __init__(self, db_path: Path = _DEFAULT_DB) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    # ── Schema ────────────────────────────────────────────────────────────────
    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                name TEXT DEFAULT '',
                host TEXT DEFAULT 'localhost',
                port INTEGER DEFAULT 11434,
                gpu_count INTEGER DEFAULT 1,
                gpu_memory_gb REAL DEFAULT 16.0,
                max_concurrent_jobs INTEGER DEFAULT 4,
                status TEXT DEFAULT 'offline',
                current_load REAL DEFAULT 0.0,
                last_heartbeat TEXT,
                registered_at TEXT
            );
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                model_id TEXT DEFAULT '',
                node_id TEXT,
                job_type TEXT DEFAULT 'inference',
                priority INTEGER DEFAULT 5,
                gpu_required INTEGER DEFAULT 1,
                status TEXT DEFAULT 'queued',
                output_tokens INTEGER DEFAULT 0,
                created_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_orch_jobs_status ON jobs(status);
            CREATE INDEX IF NOT EXISTS idx_orch_nodes_load
                ON nodes(status, current_load);
        """)
        self._conn.commit()

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _node_from_row(self, row) -> ClusterNode:
        return ClusterNode(
            node_id=row["node_id"], name=row["name"],
            host=row["host"], port=row["port"],
            gpu_count=row["gpu_count"], gpu_memory_gb=row["gpu_memory_gb"],
            max_concurrent_jobs=row["max_concurrent_jobs"],
            status=row["status"], current_load=row["current_load"],
            last_heartbeat=row["last_heartbeat"],
            registered_at=row["registered_at"],
        )

    def _job_from_row(self, row) -> ClusterJob:
        return ClusterJob(
            job_id=row["job_id"], model_id=row["model_id"],
            node_id=row["node_id"], job_type=row["job_type"],
            priority=row["priority"], gpu_required=row["gpu_required"],
            status=row["status"], output_tokens=row["output_tokens"],
            created_at=row["created_at"],
        )

    def _running_job_count(self, node_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE node_id=? AND status='running'",
            (node_id,),
        ).fetchone()
        return row[0] if row else 0

    # ── Public API ────────────────────────────────────────────────────────────
    def register_node(self, node: ClusterNode) -> ClusterNode:
        """Register a node and bring it online."""
        node.status = "online"
        node.last_heartbeat = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (node.node_id, node.name, node.host, node.port,
             node.gpu_count, node.gpu_memory_gb, node.max_concurrent_jobs,
             node.status, node.current_load, node.last_heartbeat,
             node.registered_at),
        )
        self._conn.commit()
        return node

    def list_nodes(self) -> List[ClusterNode]:
        """Return all nodes ordered by load (least-loaded first)."""
        rows = self._conn.execute(
            "SELECT * FROM nodes ORDER BY current_load ASC"
        ).fetchall()
        return [self._node_from_row(r) for r in rows]

    def submit_job(self, job: ClusterJob) -> ClusterJob:
        """Enqueue a job for scheduling."""
        job.status = "queued"
        self._conn.execute(
            "INSERT OR REPLACE INTO jobs VALUES (?,?,?,?,?,?,?,?,?)",
            (job.job_id, job.model_id, job.node_id, job.job_type,
             job.priority, job.gpu_required, job.status,
             job.output_tokens, job.created_at),
        )
        self._conn.commit()
        return job

    def schedule_jobs(self) -> int:
        """
        Assign queued jobs to the least-loaded available nodes.

        Returns the number of jobs successfully scheduled.
        """
        queued_rows = self._conn.execute(
            "SELECT * FROM jobs WHERE status='queued' "
            "ORDER BY priority DESC, created_at ASC"
        ).fetchall()

        scheduled = 0
        for job_row in queued_rows:
            job = self._job_from_row(job_row)
            node_rows = self._conn.execute(
                "SELECT * FROM nodes WHERE status='online' "
                "ORDER BY current_load ASC"
            ).fetchall()
            for node_row in node_rows:
                node = self._node_from_row(node_row)
                if self._running_job_count(node.node_id) < node.max_concurrent_jobs:
                    self._conn.execute(
                        "UPDATE jobs SET status='running', node_id=? "
                        "WHERE job_id=?",
                        (node.node_id, job.job_id),
                    )
                    scheduled += 1
                    break

        self._conn.commit()
        return scheduled

    def complete_job(self, job_id: str, output_tokens: int = 0) -> None:
        """Mark a job as done and record its output token count."""
        self._conn.execute(
            "UPDATE jobs SET status='done', output_tokens=? WHERE job_id=?",
            (output_tokens, job_id),
        )
        self._conn.commit()

    def get_cluster_health(self) -> ClusterHealth:
        """Return a health snapshot of the entire cluster."""
        nodes = self.list_nodes()
        health = ClusterHealth(
            total_nodes=len(nodes),
            online_nodes=sum(1 for n in nodes if n.status == "online"),
            total_gpus=sum(n.gpu_count for n in nodes),
            avg_load=round(
                sum(n.current_load for n in nodes) / max(len(nodes), 1), 4
            ) if nodes else 0.0,
        )
        for status, cnt in self._conn.execute(
            "SELECT status, COUNT(*) FROM jobs GROUP BY status"
        ).fetchall():
            if status == "queued":
                health.queued_jobs = cnt
            elif status == "running":
                health.running_jobs = cnt
        return health

    def balance_load(self) -> int:
        """
        Migrate a running job from each overloaded node to an idle one.

        Returns the number of jobs migrated.
        """
        rows = self._conn.execute(
            "SELECT * FROM nodes WHERE status='online' ORDER BY current_load ASC"
        ).fetchall()
        nodes = [self._node_from_row(r) for r in rows]
        if len(nodes) < 2:
            logging.warning("Need ≥2 online nodes to balance.")
            return 0

        migrated = 0
        idle_nodes = [n for n in nodes if n.current_load < 0.3]
        hot_nodes = [n for n in nodes if n.current_load > 0.7]

        for hot in hot_nodes:
            if not idle_nodes:
                break
            target = min(idle_nodes, key=lambda n: n.current_load)
            if hot.current_load - target.current_load < 0.35:
                continue
            job_row = self._conn.execute(
                "SELECT job_id FROM jobs WHERE node_id=? AND status='running' "
                "ORDER BY priority ASC LIMIT 1",
                (hot.node_id,),
            ).fetchone()
            if not job_row:
                continue
            delta = 0.15
            self._conn.execute(
                "UPDATE jobs SET node_id=? WHERE job_id=?",
                (target.node_id, job_row[0]),
            )
            self._conn.execute(
                "UPDATE nodes SET current_load=MAX(0,current_load-?) "
                "WHERE node_id=?",
                (delta, hot.node_id),
            )
            self._conn.execute(
                "UPDATE nodes SET current_load=MIN(1,current_load+?) "
                "WHERE node_id=?",
                (delta, target.node_id),
            )
            migrated += 1
            target.current_load = min(1.0, target.current_load + delta)
            if target.current_load >= 0.6:
                idle_nodes.remove(target)

        self._conn.commit()
        return migrated

    def close(self) -> None:
        self._conn.close()


class OllamaRouter:
    """
    Routes every request that contains an Ollama handle to the local Ollama
    service running on the BlackRoad cluster.

    Supported handles (case-insensitive):
        @copilot, @lucidia, @blackboxprogramming, @ollama

    No external AI provider is contacted — all inference stays on your
    own hardware.
    """

    DEFAULT_MODEL = "qwen2.5:7b"

    def __init__(
        self,
        orchestrator: AIClusterOrchestrator,
        default_ollama_url: str = "http://localhost:11434",
        default_model: str = DEFAULT_MODEL,
    ) -> None:
        self.orchestrator = orchestrator
        self.default_ollama_url = default_ollama_url
        self.default_model = default_model

    # ── Public API ────────────────────────────────────────────────────────────
    def should_route_to_ollama(self, message: str) -> bool:
        """Return True when *message* contains at least one Ollama handle."""
        lower = message.lower()
        return any(handle in lower for handle in OLLAMA_HANDLES)

    def resolve_endpoint(self) -> str:
        """
        Return the base URL of the least-loaded online Ollama node.

        Falls back to *default_ollama_url* when no nodes are online.
        """
        online = [
            n for n in self.orchestrator.list_nodes()
            if n.status == "online"
        ]
        if online:
            best = min(online, key=lambda n: n.current_load)
            return f"http://{best.host}:{best.port}"
        return self.default_ollama_url

    def build_request(
        self, message: str, model: Optional[str] = None
    ) -> Dict:
        """
        Build an Ollama-compatible request payload.

        Args:
            message: User message; must contain an Ollama routing handle.
            model:   Override the default model name.

        Returns:
            dict with keys: provider, endpoint, url, model, prompt.

        Raises:
            ValueError: when *message* contains no recognised Ollama handle.
        """
        if not self.should_route_to_ollama(message):
            raise ValueError(
                f"Message contains no Ollama routing handle. "
                f"Supported handles: {sorted(OLLAMA_HANDLES)}"
            )
        endpoint = self.resolve_endpoint()
        chosen_model = model or self.default_model
        return {
            "provider": "ollama",
            "endpoint": endpoint,
            "url": f"{endpoint}/api/generate",
            "model": chosen_model,
            "prompt": message,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────
def _bar(val: float, width: int = 20) -> str:
    """Coloured utilisation bar."""
    filled = int(val * width)
    color = G if val < 0.5 else (Y if val < 0.8 else R)
    return color + "█" * filled + NC + "░" * (width - filled) + f" {val:.0%}"


def _print_stats(stats: ClusterStats) -> None:
    gpu_type_str = ", ".join(f"{v}×{k}" for k, v in stats.gpu_types.items())
    print(f"\n{BOLD}{B}╔══ BlackRoad AI Cluster Stats ════════════════╗{NC}")
    print(f"  {C}Nodes{NC}      {stats.online_nodes}/{stats.total_nodes} online"
          f"  {Y}{stats.degraded_nodes}{NC} degraded  "
          f"{R}{stats.offline_nodes}{NC} offline")
    print(f"  {C}GPUs{NC}       {stats.total_gpus} total ({gpu_type_str})"
          f"  {stats.busy_gpus} busy")
    print(f"  {C}Load{NC}       avg {_bar(stats.avg_load)}  "
          f"peak {_bar(stats.peak_load)}")
    print(f"  {C}Jobs{NC}       "
          f"{Y}{stats.queued_jobs}{NC} queued  "
          f"{C}{stats.running_jobs}{NC} running  "
          f"{G}{stats.completed_jobs}{NC} done  "
          f"{R}{stats.failed_jobs}{NC} failed")
    print(f"{BOLD}{B}╚═══════════════════════════════════════════════╝{NC}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cluster",
        description="BlackRoad AI Cluster — distributed inference orchestrator",
    )
    sub = p.add_subparsers(dest="group", required=True)

    # ── node subcommands ──
    ng = sub.add_parser("node", help="Node management")
    nsub = ng.add_subparsers(dest="action", required=True)

    na = nsub.add_parser("add", help="Register a new node")
    na.add_argument("--host", required=True)
    na.add_argument("--port", type=int, default=11434)
    na.add_argument("--gpus", type=int, default=1)
    na.add_argument("--gpu-type", default="unknown")
    na.add_argument("--gpu-mem", type=float, default=16.0)
    na.add_argument("--cpu-cores", type=int, default=8)
    na.add_argument("--ram", type=float, default=32.0)
    na.add_argument("--tags", nargs="*", default=[])

    nl = nsub.add_parser("list", help="List all nodes")
    nd = nsub.add_parser("drain", help="Drain a node")
    nd.add_argument("node_id")
    nr = nsub.add_parser("remove", help="Remove a node")
    nr.add_argument("node_id")
    nh = nsub.add_parser("heartbeat", help="Update node load")
    nh.add_argument("node_id")
    nh.add_argument("--load", type=float, required=True)

    # ── job subcommands ──
    jg = sub.add_parser("job", help="Job management")
    jsub = jg.add_subparsers(dest="action", required=True)

    ja = jsub.add_parser("assign", help="Assign a job to a node")
    ja.add_argument("job_id")
    ja.add_argument("--model-id", default="")
    ja.add_argument("--type", default="inference")
    ja.add_argument("--priority", type=int, default=5)
    ja.add_argument("--gpus", type=int, default=1)
    ja.add_argument("--gpu-type", default="any")
    ja.add_argument("--memory-gb", type=float, default=0.0)

    jc = jsub.add_parser("complete", help="Mark job done")
    jc.add_argument("job_id")
    jc.add_argument("--tokens", type=int, default=0)
    jc.add_argument("--latency-ms", type=float, default=0.0)

    jf = jsub.add_parser("fail", help="Mark job failed")
    jf.add_argument("job_id")
    jf.add_argument("--error", default="")

    jl = jsub.add_parser("list", help="List jobs")
    jl.add_argument("--status", default=None)

    # ── cluster subcommands ──
    cg = sub.add_parser("cluster", help="Cluster-level operations")
    csub = cg.add_subparsers(dest="action", required=True)
    csub.add_parser("stats", help="Print cluster stats")
    rb = csub.add_parser("rebalance", help="Rebalance job load")
    rb.add_argument("--skew", type=float, default=0.35)
    tp = csub.add_parser("topology", help="Export topology JSON")
    tp.add_argument("--out", default=None)

    return p


def main() -> None:
    args = _build_parser().parse_args()
    cluster = Cluster()

    try:
        if args.group == "node":
            if args.action == "add":
                cluster.add_node(
                    host=args.host, port=args.port,
                    gpus=args.gpus, gpu_type=args.gpu_type,
                    gpu_memory_gb=args.gpu_mem,
                    cpu_cores=args.cpu_cores,
                    ram_gb=args.ram,
                    tags=args.tags,
                )
            elif args.action == "list":
                nodes = cluster.list_nodes()
                if not nodes:
                    print(f"{Y}No nodes registered.{NC}")
                    return
                print(f"\n{BOLD}{B}── Cluster Nodes ──────────────────────{NC}")
                for n in nodes:
                    dot = G + "●" if n.status == "online" else \
                          Y + "●" if n.status == "degraded" else R + "●"
                    running = cluster._running_job_count(n.id)
                    print(f"  {dot}{NC} {C}{n.id}{NC}  "
                          f"{BOLD}{n.host}:{n.port}{NC}  "
                          f"{n.gpu_count}×{n.gpu_type}  "
                          f"load={_bar(n.load, 12)}  "
                          f"jobs={running}/{n.max_jobs}")
            elif args.action == "drain":
                cluster.drain_node(args.node_id)
            elif args.action == "remove":
                cluster.remove_node(args.node_id)
            elif args.action == "heartbeat":
                cluster.heartbeat(args.node_id, args.load)

        elif args.group == "job":
            if args.action == "assign":
                cluster.assign_job(args.job_id, {
                    "model_id": args.model_id, "job_type": args.type,
                    "priority": args.priority, "gpu_required": args.gpus,
                    "gpu_type": args.gpu_type, "memory_gb": args.memory_gb,
                })
            elif args.action == "complete":
                cluster.complete_job(args.job_id, args.tokens, args.latency_ms)
            elif args.action == "fail":
                cluster.fail_job(args.job_id, args.error)
            elif args.action == "list":
                jobs = cluster.list_jobs(status=args.status)
                if not jobs:
                    print(f"{Y}No jobs found.{NC}")
                    return
                print(f"\n{BOLD}{B}── Jobs ────────────────────────────────{NC}")
                for j in jobs:
                    sc = {
                        "queued": Y, "running": C, "done": G, "failed": R
                    }.get(j.status, NC)
                    print(f"  {C}{j.job_id}{NC}  [{sc}{j.status}{NC}]  "
                          f"model={j.model_id or '—'}  "
                          f"p={j.priority}  node={j.node_id or '—'}")

        elif args.group == "cluster":
            if args.action == "stats":
                _print_stats(cluster.cluster_stats())
            elif args.action == "rebalance":
                cluster.rebalance_jobs(skew_threshold=args.skew)
            elif args.action == "topology":
                out = Path(args.out) if args.out else None
                topo = cluster.export_topology(out)
                if not out:
                    print(json.dumps(topo, indent=2))

    finally:
        cluster.close()


if __name__ == "__main__":
    main()
