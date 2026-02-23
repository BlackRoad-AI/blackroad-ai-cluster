# BlackRoad vLLM Deployment

Production LLM serving with vLLM on Railway GPU instances.

## Supported Models

| Model | VRAM | Throughput | Best For |
|-------|------|-----------|---------|
| Qwen2.5-72B-Instruct | 80GB A100 | 1200 tok/s | General agents |
| DeepSeek-R1-7B | 16GB | 4800 tok/s | Reasoning |
| Llama-3.2-3B | 6GB | 8000 tok/s | Edge/Pi |
| Mistral-7B | 14GB | 5200 tok/s | Balanced |

## Quick Deploy

```bash
# Railway A100 deployment
railway up --service blackroad-vllm

# Local with CUDA
docker run --gpus all -p 8000:8000 \
  -e MODEL=Qwen/Qwen2.5-7B-Instruct \
  blackroadai/vllm:latest
```

## API (OpenAI-compatible)

```python
from openai import OpenAI

client = OpenAI(
    api_key="bk_live_xxx",
    base_url="https://vllm.blackroad.ai/v1"
)

response = client.chat.completions.create(
    model="blackroad-qwen-72b",
    temperature=0.7,
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

## vLLM Configuration

```yaml
model: Qwen/Qwen2.5-72B-Instruct
tensor-parallel-size: 1
max-model-len: 32768
gpu-memory-utilization: 0.90
max-num-seqs: 256
enable-chunked-prefill: true
quantization: awq
served-model-name: blackroad-qwen-72b
```

## Performance Tuning

Enable continuous batching, prefix caching, and speculative decoding:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-72B-Instruct \
  --max-num-seqs 512 \
  --enable-prefix-caching \
  --speculative-model Qwen/Qwen2.5-0.5B-Instruct \
  --num-speculative-tokens 5
```

## Metrics

Prometheus metrics endpoint exposes:
- GPU cache utilization
- Active request count  
- Time to first token (TTFT)
- Time per output token (TPOT)
