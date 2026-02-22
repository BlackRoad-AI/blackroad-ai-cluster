#!/bin/bash
# BlackRoad AI Cluster Bootstrap
# Run on each node to join the cluster

set -euo pipefail

NODE_ROLE=${1:-"secondary"}
CLUSTER_PRIMARY="192.168.4.38"
GATEWAY_URL="http://127.0.0.1:8787"

echo "🚀 Bootstrapping BlackRoad AI Cluster Node ($NODE_ROLE)"

# Install Ollama
if ! command -v ollama &>/dev/null; then
  curl -fsSL https://ollama.ai/install.sh | sh
fi

# Pull base models
echo "📦 Pulling models..."
ollama pull qwen2.5:7b
ollama pull llama3.2:3b

if [ "$NODE_ROLE" == "primary" ]; then
  ollama pull qwen2.5:72b
  ollama pull deepseek-r1:32b
fi

# Start memory bridge
pip install -q fastapi uvicorn httpx
echo "🧠 Starting memory bridge on :8001..."
nohup python3 -m uvicorn memory_bridge.server:app --host 0.0.0.0 --port 8001 &

# Register with cluster
echo "📡 Registering node with cluster..."
curl -s -X POST "$CLUSTER_PRIMARY:8080/cluster/register" \
  -H "Content-Type: application/json" \
  -d "{\"role\": \"$NODE_ROLE\", \"ip\": \"$(hostname -I | awk '{print $1}')\"}" \
  && echo "✓ Registered" || echo "⚠ Could not reach primary (will retry)"

echo "✅ Node bootstrap complete"
