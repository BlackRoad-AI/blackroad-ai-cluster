# blackroad-ai-cluster

> **BlackRoad AI Cluster** — Distributed AI inference orchestration with node health monitoring, priority job scheduling, and automatic load balancing.

## ✅ Verified Working

| Component | Status | Notes |
|-----------|--------|-------|
| **CI (lint + tests)** | ✅ Passing | Python 3.11, pytest, flake8 |
| **Deploy to Cloudflare Pages** | ✅ Passing | Skips gracefully when no `package.json` |
| **Auto Deploy** | ✅ Passing | Detects project type; deploys Workers/Pages/Railway |
| **Security Scan** | ✅ Passing | CodeQL v4 (Python), dependency review on PRs |
| **Self-Healing Master** | ✅ Fixed | Fixed newline-in-`GITHUB_OUTPUT` bug; npm audit only on JS projects |
| **Self-Healing** | ✅ Passing | Health monitor + dependency auto-updates |
| **Test Auto-Heal** | ✅ Passing | Skips Node.js setup when no `package.json` |
| **Auto-Merge** | ✅ Enabled | Merges Dependabot PRs and PRs labelled `automerge` |
| **Cloudflare Workers** | ✅ Ready | Long-running AI tasks via Durable Objects |
| **All actions SHA-pinned** | ✅ Done | Every `uses:` pinned to a SHA-256 commit hash |

> Last verified: 2026-03-04 · Branch: `main`

---

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![BlackRoad AI](https://img.shields.io/badge/BlackRoad-AI-FF1D6C)](https://blackroad.ai)
[![License](https://img.shields.io/badge/license-Proprietary-black)](LICENSE)
[![CI](https://github.com/BlackRoad-AI/blackroad-ai-cluster/actions/workflows/ci.yml/badge.svg)](https://github.com/BlackRoad-AI/blackroad-ai-cluster/actions/workflows/ci.yml)
[![Security Scan](https://github.com/BlackRoad-AI/blackroad-ai-cluster/actions/workflows/security-scan.yml/badge.svg)](https://github.com/BlackRoad-AI/blackroad-ai-cluster/actions/workflows/security-scan.yml)

---

## Overview

`blackroad-ai-cluster` is the orchestration backbone for BlackRoad's distributed AI inference
infrastructure. It manages a fleet of GPU nodes, schedules inference and training jobs with
priority queuing, monitors cluster health, and automatically rebalances load from overloaded
nodes to idle ones — all persisted in SQLite for offline development and CI.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      blackroad-ai-cluster                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                  AIClusterOrchestrator                           │  │
│  │                                                                  │  │
│  │  ┌──────────────┐   ┌──────────────┐   ┌───────────────────┐   │  │
│  │  │  Node        │   │  Job         │   │  Load Balancer    │   │  │
│  │  │  Registry    │   │  Scheduler   │   │                   │   │  │
│  │  │              │   │              │   │  round_robin      │   │  │
│  │  │  GPU nodes   │   │  priority    │   │  least_loaded     │   │  │
│  │  │  heartbeat   │   │  queued      │   │  gpu_affinity     │   │  │
│  │  │  status      │   │  running     │   │                   │   │  │
│  │  │  online      │   │  done/failed │   │  threshold=0.7    │   │  │
│  │  └──────────────┘   └──────────────┘   └───────────────────┘   │  │
│  │                                                                  │  │
│  │  ┌──────────────────────────────────────────────────────────┐   │  │
│  │  │  Health Monitor  ─  Snapshots stored to health_snapshots │   │  │
│  │  └──────────────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  SQLite: ~/.blackroad/ai_cluster.db                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

### Node Lifecycle

```
REGISTER → ONLINE → BUSY (jobs running) → DEGRADED → OFFLINE
              ↑_____________REBALANCE_______________↑
```

### Job Lifecycle

```
submit() → QUEUED → schedule() → RUNNING → complete() → DONE
                                         ↘ (timeout)  → FAILED
```

---

## Features

- 🖥️ **Node Registry** — register GPU nodes with hardware specs (GPU count, VRAM, CPU, RAM)
- 📋 **Job Queue** — submit inference/training/benchmark jobs with priority 1–10
- ⚡ **Priority Scheduler** — high-priority jobs run first; round-robin across nodes
- 📊 **Health Snapshots** — timestamped cluster health records
- ⚖️ **Load Balancer** — auto-migrate jobs from overloaded (>0.7) to idle (<0.3) nodes
- 🔧 **Capacity Control** — per-node `max_concurrent_jobs` enforcement
- 🗄️ **SQLite Persistence** — zero-config local database

---

## Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| Python | ≥ 3.9 | Runtime |
| pytest | ≥ 7.0 | Testing |

---

## Installation

```bash
git clone https://github.com/BlackRoad-AI/blackroad-ai-cluster.git
cd blackroad-ai-cluster
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `BLACKROAD_CLUSTER_DB` | `~/.blackroad/ai_cluster.db` | DB path |

### Node Sizing Guide

| Node Type | GPU Count | GPU VRAM | Max Jobs | Use Case |
|-----------|-----------|----------|----------|----------|
| Development | 0 | 0 | 2 | CPU-only testing |
| Small GPU | 1 | 24 GB | 4 | 7B–13B models |
| Large GPU | 2 | 80 GB | 6 | 70B models |
| Multi-GPU | 8 | 80 GB | 16 | 405B models / training |

---

## Usage

### CLI

#### Register a GPU node

```bash
python src/ai_cluster.py node \
    --name "A100-Node-01" \
    --host 10.0.0.101 \
    --port 8080 \
    --gpus 8 \
    --gpu-mem 80.0
```

**Output:**
```
✓ Node A100-Node-01 [a3f2c1e8] online — 8×GPU 80.0GB
```

#### Submit a job

```bash
python src/ai_cluster.py job \
    --model-id llama3-70b \
    --type inference \
    --priority 8 \
    --gpus 2
```

**Output:**
```
→ Job b4d2f1c0 queued [inference, model=llama3-70b, priority=8]
```

#### Schedule queued jobs

```bash
python src/ai_cluster.py schedule
```

**Output:**
```
✓ Scheduled 5/7 jobs across 3 nodes
```

#### Check cluster health

```bash
python src/ai_cluster.py health
```

**Output:**
```
── Cluster Health ─────────────────────
  Nodes    3/3 online (0 degraded)
  GPUs     24 total | avg load ████████░░░░░░░░░░░░ 40%
  Jobs     queued=2 running=5 done=148
```

#### Rebalance load

```bash
python src/ai_cluster.py balance
```

**Output:**
```
✓ Load balanced: 2 job(s) migrated, 1 overloaded, 1 idle nodes
```

#### List all nodes

```bash
python src/ai_cluster.py nodes
```

**Output:**
```
  ● a3f2c1e8 A100-Node-01         10.0.0.101:8080 8GPU load=████░░░░░░
  ● b1c2d3e4 A100-Node-02         10.0.0.102:8080 8GPU load=██░░░░░░░░
  ● c4d5e6f7 RTX-Node-01          10.0.0.103:8080 4GPU load=░░░░░░░░░░
```

#### Mark a job complete

```bash
python src/ai_cluster.py complete b4d2f1c0 --tokens 1024
```

**Output:**
```
✓ Job b4d2f1c0 completed (1024 tokens)
```

---

### Python API

```python
from src.ai_cluster import (
    AIClusterOrchestrator, ClusterNode, ClusterJob
)

orch = AIClusterOrchestrator()

# Register nodes
orch.register_node(ClusterNode(
    node_id="gpu-01",
    name="A100-Primary",
    host="10.0.0.1",
    gpu_count=8,
    gpu_memory_gb=80.0,
    max_concurrent_jobs=16,
))

# Submit jobs
for i in range(10):
    orch.submit_job(ClusterJob(
        model_id="llama3-70b",
        job_type="inference",
        priority=5 + (i % 5),
        gpu_required=2,
    ))

# Schedule
scheduled = orch.schedule_jobs()
print(f"Scheduled {scheduled} jobs")

# Health check
health = orch.get_cluster_health()
print(f"Load: {health.avg_load:.1%}")

# Balance
orch.balance_load()

# Complete a job
orch.complete_job("job-id-here", output_tokens=512)

orch.close()
```

---

## API Reference

### `AIClusterOrchestrator`

| Method | Returns | Description |
|--------|---------|-------------|
| `register_node(node)` | `ClusterNode` | Add/update a GPU node |
| `list_nodes()` | `List[ClusterNode]` | All registered nodes |
| `submit_job(job)` | `ClusterJob` | Queue a new job |
| `schedule_jobs()` | `int` | Assign queued jobs → nodes |
| `complete_job(job_id, tokens)` | `None` | Mark job done |
| `get_cluster_health()` | `ClusterHealth` | Health snapshot |
| `balance_load()` | `None` | Migrate overloaded jobs |
| `close()` | `None` | Close DB |

### `ClusterNode` Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `node_id` | `str` | auto-UUID8 | Unique node ID |
| `name` | `str` | `""` | Node name |
| `host` | `str` | `"localhost"` | IP/hostname |
| `port` | `int` | `8080` | Port |
| `gpu_count` | `int` | `1` | Number of GPUs |
| `gpu_memory_gb` | `float` | `24.0` | VRAM per GPU |
| `max_concurrent_jobs` | `int` | `4` | Capacity limit |

### `ClusterJob` Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `job_id` | `str` | auto-UUID8 | Unique job ID |
| `model_id` | `str` | `""` | Target model |
| `job_type` | `str` | `"inference"` | inference/training/benchmark |
| `priority` | `int` | `5` | 1 (low) – 10 (high) |
| `gpu_required` | `int` | `1` | GPUs needed |

### `ClusterHealth` Fields

| Field | Description |
|-------|-------------|
| `total_nodes` | All registered nodes |
| `online_nodes` | Online nodes |
| `total_gpus` | Sum of all node GPUs |
| `queued_jobs` | Jobs waiting to run |
| `running_jobs` | Currently executing |
| `completed_jobs` | Finished jobs |
| `avg_load` | Average load 0.0–1.0 |

---

## Running Tests

```bash
pytest tests/test_ai_cluster.py -v
# Expected: 19 passed
```

---

## Database Schema

```sql
-- ~/.blackroad/ai_cluster.db

CREATE TABLE nodes (
    node_id              TEXT PRIMARY KEY,
    name                 TEXT,
    host                 TEXT,
    port                 INTEGER,
    gpu_count            INTEGER,
    gpu_memory_gb        REAL,
    cpu_cores            INTEGER,
    ram_gb               REAL,
    status               TEXT,        -- online | offline | degraded
    current_load         REAL,
    max_concurrent_jobs  INTEGER,
    registered_at        TEXT,
    last_heartbeat       TEXT
);

CREATE TABLE jobs (
    job_id       TEXT PRIMARY KEY,
    model_id     TEXT,
    node_id      TEXT,
    job_type     TEXT,               -- inference | training | benchmark
    priority     INTEGER,            -- 1 (low) to 10 (high)
    gpu_required INTEGER,
    status       TEXT,               -- queued | running | done | failed
    input_tokens  INTEGER,
    output_tokens INTEGER,
    latency_ms   REAL,
    created_at   TEXT,
    started_at   TEXT,
    completed_at TEXT
);

CREATE TABLE health_snapshots (
    snap_id       TEXT PRIMARY KEY,
    snapshot_json TEXT,
    created_at    TEXT
);
```

---

## BlackRoad AI Cluster Fleet

| Node | IP | GPUs | Role |
|------|----|------|------|
| octavia-pi | 192.168.4.38 | — | Primary agent host (22,500 agents) |
| lucidia-pi | 192.168.4.64 | — | Secondary agent host (7,500 agents) |
| blackroad os-infinity | 159.65.43.12 | — | Cloud failover |
| Railway A100 | managed | A100 80GB | Primary GPU inference |
| Railway H100 | managed | H100 80GB | Specialist reasoning |

---

*© BlackRoad OS, Inc. All rights reserved. Proprietary — not open source.*
