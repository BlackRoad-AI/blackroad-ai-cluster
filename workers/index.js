/**
 * BlackRoad AI Cluster — Cloudflare Worker
 *
 * Handles long-running AI inference tasks via Durable Objects,
 * proxies requests to the on-premises GPU cluster, and provides
 * a health endpoint for monitoring.
 *
 * Routes:
 *   GET  /health          — Worker health check (always fast)
 *   POST /tasks           — Enqueue a new AI task (Durable Object)
 *   GET  /tasks/:id       — Poll task status
 *   POST /v1/chat         — Proxy to primary cluster node
 */

export default {
  /**
   * @param {Request} request
   * @param {Env} env
   * @param {ExecutionContext} ctx
   */
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    // ------------------------------------------------------------------ //
    // Health check — verified working, returns cluster metadata
    // ------------------------------------------------------------------ //
    if (url.pathname === '/health' && request.method === 'GET') {
      return Response.json({
        status: 'ok',
        service: 'blackroad-ai-cluster-worker',
        environment: env.ENVIRONMENT ?? 'unknown',
        timestamp: new Date().toISOString(),
        version: '1.0.0',
      });
    }

    // ------------------------------------------------------------------ //
    // Task queue — backed by Durable Objects for persistence
    // ------------------------------------------------------------------ //
    if (url.pathname === '/tasks') {
      if (request.method === 'POST') {
        return handleEnqueueTask(request, env);
      }
      if (request.method === 'GET') {
        return handleListTasks(request, env);
      }
    }

    const taskMatch = url.pathname.match(/^\/tasks\/([^/]+)$/);
    if (taskMatch) {
      const taskId = taskMatch[1];
      return handleGetTask(taskId, env);
    }

    // ------------------------------------------------------------------ //
    // OpenAI-compatible chat proxy → primary cluster node
    // ------------------------------------------------------------------ //
    if (url.pathname === '/v1/chat' || url.pathname === '/v1/chat/completions') {
      return handleChatProxy(request, env, ctx);
    }

    return new Response('Not Found', { status: 404 });
  },
};

// -------------------------------------------------------------------------- //
// Task enqueueing via Durable Object
// -------------------------------------------------------------------------- //

async function handleEnqueueTask(request, env) {
  let body;
  try {
    body = await request.json();
  } catch {
    return Response.json({ error: 'Invalid JSON body' }, { status: 400 });
  }

  if (!body.model || !body.prompt) {
    return Response.json({ error: 'Missing required fields: model, prompt' }, { status: 422 });
  }

  const taskId = crypto.randomUUID();
  const stub = env.TASK_QUEUE.get(env.TASK_QUEUE.idFromName(taskId));

  await stub.fetch('https://worker/init', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ taskId, ...body, status: 'queued', createdAt: Date.now() }),
  });

  return Response.json({ taskId, status: 'queued' }, { status: 202 });
}

async function handleGetTask(taskId, env) {
  try {
    const stub = env.TASK_QUEUE.get(env.TASK_QUEUE.idFromName(taskId));
    const resp = await stub.fetch('https://worker/status');
    const data = await resp.json();
    return Response.json(data);
  } catch {
    return Response.json({ error: 'Task not found' }, { status: 404 });
  }
}

async function handleListTasks(_request, _env) {
  return Response.json({ message: 'List tasks via Durable Objects — implement with storage listing' });
}

// -------------------------------------------------------------------------- //
// Chat completion proxy to the on-prem cluster
// -------------------------------------------------------------------------- //

async function handleChatProxy(request, env, ctx) {
  const clusterUrl = env.CLUSTER_URL ?? 'http://192.168.4.38:11434';
  const upstreamUrl = `${clusterUrl}/api/chat`;

  let body;
  try {
    body = await request.json();
  } catch {
    return Response.json({ error: 'Invalid JSON' }, { status: 400 });
  }

  let upstream;
  try {
    upstream = await fetch(upstreamUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch (err) {
    return Response.json({ error: `Upstream unreachable: ${err.message}` }, { status: 502 });
  }

  return upstream;
}

// -------------------------------------------------------------------------- //
// Durable Object — TaskQueue
// Persists task state; processes jobs asynchronously via alarm()
// -------------------------------------------------------------------------- //

export class TaskQueue {
  constructor(state, env) {
    this.state = state;
    this.env = env;
  }

  async fetch(request) {
    const url = new URL(request.url);

    if (url.pathname === '/init' && request.method === 'POST') {
      const task = await request.json();
      await this.state.storage.put('task', task);

      // Schedule processing after 1 second
      await this.state.storage.setAlarm(Date.now() + 1_000);
      return Response.json({ ok: true });
    }

    if (url.pathname === '/status') {
      const task = (await this.state.storage.get('task')) ?? { status: 'not_found' };
      return Response.json(task);
    }

    return new Response('Not Found', { status: 404 });
  }

  async alarm() {
    const task = await this.state.storage.get('task');
    if (!task || task.status !== 'queued') return;

    // Mark as running
    task.status = 'running';
    task.startedAt = Date.now();
    await this.state.storage.put('task', task);

    const clusterUrl = this.env.CLUSTER_URL ?? 'http://192.168.4.38:11434';

    try {
      const resp = await fetch(`${clusterUrl}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: task.model, prompt: task.prompt, stream: false }),
      });

      if (!resp.ok) throw new Error(`Upstream HTTP ${resp.status}`);

      const result = await resp.json();
      task.status = 'done';
      task.result = result;
      task.completedAt = Date.now();
    } catch (err) {
      task.status = 'failed';
      task.error = err.message;
      task.failedAt = Date.now();
    }

    await this.state.storage.put('task', task);
  }
}
