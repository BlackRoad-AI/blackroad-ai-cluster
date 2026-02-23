"""Tests for src/ai_cluster.py — BlackRoad AI Cluster Orchestrator."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ai_cluster import (
    ClusterNode, ClusterJob, ClusterHealth,
    AIClusterOrchestrator,
)


@pytest.fixture
def orch(tmp_path):
    o = AIClusterOrchestrator(db_path=tmp_path / "test_cluster.db")
    yield o
    o.close()


@pytest.fixture
def orch_2nodes(orch):
    orch.register_node(ClusterNode(
        node_id="n1", name="GPU-A100-1", host="10.0.0.1",
        gpu_count=2, gpu_memory_gb=80.0, max_concurrent_jobs=4
    ))
    orch.register_node(ClusterNode(
        node_id="n2", name="GPU-A100-2", host="10.0.0.2",
        gpu_count=4, gpu_memory_gb=80.0, max_concurrent_jobs=8
    ))
    return orch


# ── dataclass defaults ────────────────────────────────────────────────────────
def test_cluster_node_defaults():
    node = ClusterNode()
    assert node.status == "offline"
    assert node.current_load == 0.0
    assert node.max_concurrent_jobs == 4
    assert node.gpu_count == 1


def test_cluster_job_defaults():
    job = ClusterJob()
    assert job.status == "queued"
    assert job.job_type == "inference"
    assert job.priority == 5
    assert job.gpu_required == 1


# ── node management ───────────────────────────────────────────────────────────
def test_register_node_sets_online(orch):
    node = ClusterNode(node_id="new-node", name="NewNode", status="offline")
    result = orch.register_node(node)
    assert result.status == "online"
    assert result.last_heartbeat is not None


def test_list_nodes_empty(orch):
    assert orch.list_nodes() == []


def test_list_nodes_after_register(orch_2nodes):
    nodes = orch_2nodes.list_nodes()
    assert len(nodes) == 2
    assert {n.node_id for n in nodes} == {"n1", "n2"}


def test_node_gpu_count_persisted(orch):
    orch.register_node(ClusterNode(node_id="gpu4", name="Big GPU", gpu_count=4))
    nodes = orch.list_nodes()
    assert nodes[0].gpu_count == 4


# ── job management ────────────────────────────────────────────────────────────
def test_submit_job_queued(orch):
    job = ClusterJob(job_id="j1", model_id="llm-3b", job_type="inference", priority=5)
    result = orch.submit_job(job)
    assert result.status == "queued"


def test_schedule_returns_count(orch_2nodes):
    for i in range(3):
        orch_2nodes.submit_job(ClusterJob(job_id=f"j{i}", model_id="m", priority=5))
    scheduled = orch_2nodes.schedule_jobs()
    assert scheduled > 0
    assert scheduled <= 3


def test_schedule_no_nodes_returns_zero(orch):
    orch.submit_job(ClusterJob(job_id="j1", model_id="m1"))
    assert orch.schedule_jobs() == 0


def test_schedule_no_jobs_returns_zero(orch_2nodes):
    assert orch_2nodes.schedule_jobs() == 0


def test_complete_job_updates_status(orch_2nodes):
    orch_2nodes.submit_job(ClusterJob(job_id="done-job", model_id="m1"))
    orch_2nodes.schedule_jobs()
    orch_2nodes.complete_job("done-job", output_tokens=200)
    row = orch_2nodes._conn.execute(
        "SELECT status FROM jobs WHERE job_id='done-job'"
    ).fetchone()
    assert row[0] == "done"


def test_complete_job_sets_output_tokens(orch_2nodes):
    orch_2nodes.submit_job(ClusterJob(job_id="tok-job", model_id="m1"))
    orch_2nodes.schedule_jobs()
    orch_2nodes.complete_job("tok-job", output_tokens=512)
    row = orch_2nodes._conn.execute(
        "SELECT output_tokens FROM jobs WHERE job_id='tok-job'"
    ).fetchone()
    assert row[0] == 512


# ── health ────────────────────────────────────────────────────────────────────
def test_cluster_health_empty(orch):
    health = orch.get_cluster_health()
    assert isinstance(health, ClusterHealth)
    assert health.total_nodes == 0
    assert health.queued_jobs == 0


def test_cluster_health_with_nodes(orch_2nodes):
    health = orch_2nodes.get_cluster_health()
    assert health.total_nodes == 2
    assert health.online_nodes == 2
    assert health.total_gpus == 6   # 2 + 4
    assert health.avg_load == 0.0


def test_cluster_health_counts_queued(orch_2nodes):
    orch_2nodes.submit_job(ClusterJob(job_id="q1", model_id="m"))
    orch_2nodes.submit_job(ClusterJob(job_id="q2", model_id="m"))
    health = orch_2nodes.get_cluster_health()
    assert health.queued_jobs == 2


# ── load balancing ────────────────────────────────────────────────────────────
def test_balance_load_single_node_no_crash(orch):
    orch.register_node(ClusterNode(node_id="solo", name="Solo"))
    orch.balance_load()   # Should warn & not raise


def test_balance_load_migrates_jobs(orch_2nodes):
    # Set one node overloaded, one idle
    orch_2nodes._conn.execute("UPDATE nodes SET current_load=0.8 WHERE node_id='n1'")
    orch_2nodes._conn.execute("UPDATE nodes SET current_load=0.1 WHERE node_id='n2'")
    orch_2nodes._conn.commit()
    orch_2nodes.submit_job(ClusterJob(job_id="ol-job", model_id="m1"))
    orch_2nodes._conn.execute(
        "UPDATE jobs SET status='running', node_id='n1' WHERE job_id='ol-job'"
    )
    orch_2nodes._conn.commit()
    orch_2nodes.balance_load()
    # n1 load should have decreased
    row = orch_2nodes._conn.execute(
        "SELECT current_load FROM nodes WHERE node_id='n1'"
    ).fetchone()
    assert row[0] < 0.8
