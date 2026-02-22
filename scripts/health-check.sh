#!/bin/bash
# Cluster health check
set -euo pipefail

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'

check() {
  local name=$1 url=$2
  if curl -sf "$url" &>/dev/null; then
    echo -e "${GREEN}✓${NC} $name"
  else
    echo -e "${RED}✗${NC} $name ($url)"
  fi
}

echo "=== BlackRoad AI Cluster Health ==="
check "Ollama (primary)"    "http://192.168.4.38:11434/api/tags"
check "Ollama (secondary)"  "http://192.168.4.64:11434/api/tags"
check "Memory Bridge"       "http://192.168.4.38:8001/health"
check "Gateway"             "http://127.0.0.1:8787/health"
echo "==================================="
