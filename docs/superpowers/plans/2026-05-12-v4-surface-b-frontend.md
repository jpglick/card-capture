# v4 Surface B — Frontend / App Shell Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a SvelteKit application with left-nav routing, ingest/run-monitor/cards/settings sections in Wave 1, the full Labeling UX (per-video truth editor, F/B trainer, dedup confirmer) in Wave 2, and the threshold playground + A/B comparison in Wave 3 — all built against the four frozen contracts from Surfaces A and D.

**Architecture:** SvelteKit (file-based routing, server hooks proxy to FastAPI), TypeScript, Vite. Served by FastAPI in production via static build (`app/web/build/` → mounted at `/`); dev runs `vite` on a separate port with proxy to FastAPI. State management uses Svelte 5 runes; cross-route state goes in `$lib/stores/`. SSE consumed via `EventSource` wrapped in a typed store. No global state framework; runes + small focused stores are sufficient.

**Tech Stack:** SvelteKit 2.x, Svelte 5 (runes), TypeScript 5+, Vite, `@playwright/test` for end-to-end, `vitest` for component unit tests, `svelte-dnd-action` for drag-link in label UX. CSS via a single design-token CSS file + scoped Svelte styles (no Tailwind unless the agent chooses to add it).

**Spec reference:** `docs/superpowers/specs/2026-05-12-v4-architecture-design.md`. This plan implements Surface B across Waves 1, 2, and 3.

**Contract dependencies (frozen before this plan starts):**
- Contract 2 (`docs/contracts/v1-api.md`) — every API call.
- Contract 4 (`docs/contracts/truth-schema.md`) — labeling UX writes this shape.
- Contract 3 (`docs/contracts/metaflow-artifacts.md`) — threshold playground reads these artifacts via Surface A's API.

---

## File Structure

**New files (this plan creates):**

- `app/web/package.json`, `tsconfig.json`, `svelte.config.js`, `vite.config.ts` — project scaffolding.
- `app/web/src/app.html` — shell template.
- `app/web/src/app.css` — design tokens (colors, spacing, fonts).
- `app/web/src/routes/+layout.svelte` — top-level left-nav.
- `app/web/src/routes/+layout.ts` — shared loaders.
- `app/web/src/routes/+page.svelte` — Inbox landing page.
- `app/web/src/routes/inbox/+page.svelte`
- `app/web/src/routes/runs/+page.svelte`
- `app/web/src/routes/runs/[run_id]/+page.svelte`
- `app/web/src/routes/runs/[run_id]/+layout.svelte` — run-detail tab strip.
- `app/web/src/routes/runs/[run_id]/timeline/+page.svelte`
- `app/web/src/routes/runs/[run_id]/cards/+page.svelte`
- `app/web/src/routes/runs/[run_id]/telemetry/+page.svelte`
- `app/web/src/routes/runs/[run_id]/events/+page.svelte`
- `app/web/src/routes/runs/[run_id]/rejection/+page.svelte`
- `app/web/src/routes/runs/[run_id]/hard-cases/+page.svelte`
- `app/web/src/routes/cards/+page.svelte`
- `app/web/src/routes/cards/[card_id]/+page.svelte`
- `app/web/src/routes/label/+page.svelte` (redirects to first sub-tab)
- `app/web/src/routes/label/truth/+page.svelte`
- `app/web/src/routes/label/truth/[video_id]/+page.svelte`
- `app/web/src/routes/label/fb/+page.svelte`
- `app/web/src/routes/label/clusters/+page.svelte`
- `app/web/src/routes/train/+page.svelte`
- `app/web/src/routes/regression/+page.svelte`
- `app/web/src/routes/regression/compare/+page.svelte`
- `app/web/src/routes/settings/+page.svelte`
- `app/web/src/routes/settings/playground/[run_id]/+page.svelte`
- `app/web/src/lib/api/client.ts` — typed fetch wrapper.
- `app/web/src/lib/api/types.ts` — generated/hand-mirrored types of Contract 2 schemas.
- `app/web/src/lib/api/sse.ts` — EventSource wrapper with typed events.
- `app/web/src/lib/stores/run_progress.ts` — Svelte 5 rune-based store for SSE state.
- `app/web/src/lib/stores/truth_draft.ts` — local truth.json draft with auto-save.
- `app/web/src/lib/stores/fb_queue.ts` — F/B trainer queue + undo stack.
- `app/web/src/lib/components/LeftNav.svelte`
- `app/web/src/lib/components/StagedProgress.svelte` — per-stage bar wired to SSE.
- `app/web/src/lib/components/Filmstrip.svelte` — horizontal scroll filmstrip with selectable thumbnails.
- `app/web/src/lib/components/VerdictButtons.svelte` — three-button verdict + hotkeys.
- `app/web/src/lib/components/CardThumb.svelte`
- `app/web/src/lib/components/CardGrid.svelte`
- `app/web/src/lib/components/Hotkeys.svelte` — slot-based key handler.
- `app/web/src/lib/components/DragLinkable.svelte` — wraps `svelte-dnd-action`.
- `app/web/src/lib/utils/hotkey.ts`
- `app/web/src/lib/utils/format.ts`
- `app/web/tests/unit/Filmstrip.spec.ts`
- `app/web/tests/unit/VerdictButtons.spec.ts`
- `app/web/tests/e2e/inbox.spec.ts`
- `app/web/tests/e2e/run_progress.spec.ts`
- `app/web/tests/e2e/label_truth.spec.ts`
- `app/web/tests/e2e/label_fb.spec.ts`
- `app/web/tests/e2e/label_clusters.spec.ts`
- `app/web/tests/e2e/playground.spec.ts`
- `app/web/tests/e2e/regression_compare.spec.ts`

**Modified files (this plan touches):**

- `app/main.py` — mount static build at `/` once `app/web/build/` exists.

---

## Phase B0 — Scaffold (Wave 1)

### Task B0.1: Initialize SvelteKit project

**Files:**
- Create: `app/web/package.json`, `app/web/tsconfig.json`, `app/web/svelte.config.js`, `app/web/vite.config.ts`, `app/web/src/app.html`, `app/web/src/app.css`

- [ ] **Step 1: Initialize**

```
cd app/web
npm create svelte@latest . -- --template skeleton --types typescript --no-add-eslint --no-add-prettier --no-add-playwright --no-add-vitest
npm install
npm install -D @playwright/test vitest @testing-library/svelte jsdom
npm install svelte-dnd-action
```

- [ ] **Step 2: Configure SSR for production embed**

In `svelte.config.js`, set `kit.adapter` to `@sveltejs/adapter-static` with `fallback: 'index.html'` so the build is a static SPA mountable by FastAPI:

```js
import adapter from '@sveltejs/adapter-static';

export default {
  kit: {
    adapter: adapter({ fallback: 'index.html' }),
    prerender: { entries: [] },
  },
};
```

- [ ] **Step 3: Wire dev proxy to FastAPI**

```ts
// vite.config.ts
import { sveltekit } from '@sveltejs/kit/vite';

export default {
  plugins: [sveltekit()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '/events': { target: 'http://127.0.0.1:8000', changeOrigin: true, ws: false },
    },
  },
};
```

- [ ] **Step 4: Commit**

```bash
git add app/web/
git commit -m "feat(web): scaffold SvelteKit + Vite project with FastAPI proxy"
```

### Task B0.2: API client + types

**Files:**
- Create: `app/web/src/lib/api/types.ts`
- Create: `app/web/src/lib/api/client.ts`
- Create: `app/web/tests/unit/api_client.spec.ts`

- [ ] **Step 1: Mirror Contract 2 schemas in TypeScript**

For each Pydantic model in `docs/contracts/v1-api.md`, declare an equivalent TypeScript `interface` in `types.ts`. Keep names identical (PascalCase) so cross-references match.

- [ ] **Step 2: Typed client**

```ts
// app/web/src/lib/api/client.ts
import type * as T from './types';

const BASE = '/api/v1';

class ApiError extends Error {
  constructor(public status: number, public detail: string) { super(`API ${status}: ${detail}`); }
}

async function req<T>(method: string, path: string, body?: unknown): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    method,
    headers: body ? { 'content-type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) {
    const detail = await r.text().catch(() => '');
    throw new ApiError(r.status, detail);
  }
  if (r.status === 204) return undefined as unknown as T;
  return r.json();
}

export const api = {
  videos: {
    list: () => req<T.Video[]>('GET', '/videos'),
    create: (body: T.VideoCreate) => req<T.Video>('POST', '/videos', body),
    process: (videoId: string) => req<{ run_id: string }>('POST', `/videos/${videoId}/process`),
  },
  runs: {
    list: () => req<T.Run[]>('GET', '/runs'),
    detail: (id: string) => req<T.RunDetail>('GET', `/runs/${id}`),
    cards: (id: string) => req<T.RunCardSummary[]>('GET', `/runs/${id}/cards`),
    events: (id: string) => req<T.RunEvent[]>('GET', `/runs/${id}/events`),
    telemetry: (id: string) => req<T.RunTelemetry>('GET', `/runs/${id}/telemetry`),
    rejection: (id: string) => req<T.RunRejection[]>('GET', `/runs/${id}/rejection_log`),
    hardCases: (id: string) => req<T.RunHardCase[]>('GET', `/runs/${id}/hard_cases`),
  },
  cards: {
    list: (filter: T.CardFilter) => {
      const params = new URLSearchParams(filter as any).toString();
      return req<T.Card[]>('GET', `/cards?${params}`);
    },
    detail: (id: string) => req<T.CardDetail>('GET', `/cards/${id}`),
    update: (id: string, body: Partial<T.Card>) => req<T.Card>('PATCH', `/cards/${id}`, body),
    bulk: (body: T.CardBulkAction) => req<{ updated: number }>('POST', '/cards/bulk', body),
  },
  label: {
    getTruth: (videoId: string) => req<T.LabelTruth | null>('GET', `/label/truth/${videoId}`),
    putTruth: (videoId: string, body: T.LabelTruth) => req<void>('PUT', `/label/truth/${videoId}`, body),
    nextFB: () => req<T.LabelFBNext>('GET', '/label/fb/next'),
    postFB: (body: T.LabelFB) => req<{ label_id: number }>('POST', '/label/fb', body),
    listClusters: (status?: string) => req<T.DedupCluster[]>('GET', `/label/clusters${status ? `?status=${status}` : ''}`),
    patchCluster: (id: number, body: { status?: string; confirmed?: string[] }) => req<void>('PATCH', `/label/clusters/${id}`, body),
  },
  training: {
    datasets: () => req<T.TrainingDataset[]>('GET', '/training/datasets'),
    retrain: (model: string, body: T.TrainingRetrainRequest) => req<T.TrainingJob>('POST', `/training/retrain/${model}`, body),
    job: (id: string) => req<T.TrainingJob>('GET', `/training/jobs/${id}`),
  },
  regression: {
    baselines: () => req<T.RegressionBaseline[]>('GET', '/regression/baselines'),
    promote: (body: { name: string }) => req<T.RegressionBaseline>('POST', '/regression/baselines', body),
    run: (body: T.RegressionRunRequest) => req<T.RegressionRun>('POST', '/regression/run', body),
    runDetail: (id: number) => req<T.RegressionRun>('GET', `/regression/runs/${id}`),
    compare: (a: string, b: string) => req<T.RegressionCompare>('GET', `/regression/compare?a=${a}&b=${b}`),
  },
  config: {
    presets: () => req<T.ConfigPreset[]>('GET', '/config/presets'),
    putPreset: (body: T.ConfigPreset) => req<T.ConfigPreset>('POST', '/config/presets', body),
    playground: (runId: string) => req<T.ConfigPlayground>('GET', `/config/playground/${runId}`),
  },
};

export { ApiError };
```

- [ ] **Step 3: Unit test API client against Contract 2 examples**

```ts
// app/web/tests/unit/api_client.spec.ts
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { api } from '../../src/lib/api/client';

beforeEach(() => {
  globalThis.fetch = vi.fn(async (url: any, init: any) => {
    if (String(url).includes('/api/v1/videos') && (!init?.method || init.method === 'GET')) {
      return new Response(JSON.stringify([{ id: 'v1', path: '/v.mov' }]), { status: 200 });
    }
    return new Response('{}', { status: 200 });
  }) as any;
});

describe('api.videos', () => {
  it('list returns array', async () => {
    const v = await api.videos.list();
    expect(v.length).toBeGreaterThan(0);
    expect(v[0].id).toBe('v1');
  });
});
```

- [ ] **Step 4: Run and commit**

```
cd app/web && npx vitest run
git add app/web/src/lib/api/ app/web/tests/unit/api_client.spec.ts
git commit -m "feat(web): typed API client + Contract 2 type mirrors"
```

### Task B0.3: SSE store

**Files:**
- Create: `app/web/src/lib/api/sse.ts`
- Create: `app/web/src/lib/stores/run_progress.ts`
- Create: `app/web/tests/unit/sse.spec.ts`

- [ ] **Step 1: Failing test**

```ts
// app/web/tests/unit/sse.spec.ts
import { describe, it, expect, vi } from 'vitest';
import { connectRunProgress } from '../../src/lib/stores/run_progress';

class FakeES {
  static instances: FakeES[] = [];
  listeners: Record<string, ((e: any) => void)[]> = {};
  constructor(public url: string) { FakeES.instances.push(this); }
  addEventListener(name: string, fn: any) { (this.listeners[name] ??= []).push(fn); }
  close() {}
  emit(name: string, data: any) {
    for (const fn of this.listeners[name] ?? []) fn({ data: JSON.stringify(data) });
  }
}

describe('run progress SSE', () => {
  it('updates stage map on stage_completed', () => {
    (globalThis as any).EventSource = FakeES;
    const { progress, disconnect } = connectRunProgress('run_1');
    const es = FakeES.instances.at(-1)!;
    es.emit('stage_started', { stage: 'detect' });
    es.emit('stage_completed', { stage: 'detect', payload: { n_items: 12 } });
    expect(progress.stages.detect.status).toBe('completed');
    expect(progress.stages.detect.payload.n_items).toBe(12);
    disconnect();
  });
});
```

- [ ] **Step 2: Implement**

```ts
// app/web/src/lib/api/sse.ts
export type SSEHandler<T = any> = (data: T) => void;

export function subscribe(runId: string, handlers: Record<string, SSEHandler>): () => void {
  const es = new EventSource(`/events/${runId}`);
  for (const [name, fn] of Object.entries(handlers)) {
    es.addEventListener(name, (e: MessageEvent) => fn(JSON.parse(e.data)));
  }
  return () => es.close();
}
```

```ts
// app/web/src/lib/stores/run_progress.ts
import { subscribe } from '../api/sse';

export interface StageState {
  status: 'pending' | 'running' | 'completed' | 'failed';
  payload?: Record<string, unknown>;
}

export interface RunProgress {
  stages: Record<string, StageState>;
  runStatus: 'pending' | 'running' | 'completed' | 'failed';
}

export function connectRunProgress(runId: string) {
  let progress = $state<RunProgress>({ stages: {}, runStatus: 'pending' });

  const disconnect = subscribe(runId, {
    stage_started: (d) => {
      progress.stages[d.stage] = { ...progress.stages[d.stage], status: 'running' };
      progress.runStatus = 'running';
    },
    stage_completed: (d) => {
      progress.stages[d.stage] = { status: 'completed', payload: d.payload };
    },
    run_completed: () => { progress.runStatus = 'completed'; },
    run_failed: () => { progress.runStatus = 'failed'; },
  });

  return { get progress() { return progress; }, disconnect };
}
```

- [ ] **Step 3: Test passes**

- [ ] **Step 4: Commit**

```bash
git add app/web/src/lib/api/sse.ts app/web/src/lib/stores/run_progress.ts app/web/tests/unit/sse.spec.ts
git commit -m "feat(web): SSE store with typed per-stage progress"
```

### Task B0.4: Left-nav layout + design tokens

**Files:**
- Create: `app/web/src/routes/+layout.svelte`
- Create: `app/web/src/lib/components/LeftNav.svelte`
- Create: `app/web/src/app.css`

- [ ] **Step 1: Design tokens**

```css
/* app/web/src/app.css */
:root {
  --bg: #0e0f12;
  --surface: #1a1c20;
  --border: #2a2d33;
  --text: #e9eaee;
  --text-dim: #9aa0a8;
  --accent: #4f9eff;
  --accent-dim: #1f4e80;
  --ok: #5fd07a;
  --warn: #f3c14b;
  --err: #ff6961;
  --spacing-1: 4px; --spacing-2: 8px; --spacing-3: 12px;
  --spacing-4: 16px; --spacing-6: 24px; --spacing-8: 32px;
  --radius: 6px;
  --font: -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
  --font-mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
html, body { background: var(--bg); color: var(--text); font-family: var(--font); margin: 0; }
```

- [ ] **Step 2: Layout**

```svelte
<!-- app/web/src/routes/+layout.svelte -->
<script lang="ts">
  import LeftNav from '$lib/components/LeftNav.svelte';
  import '../app.css';
</script>

<div class="shell">
  <LeftNav />
  <main><slot /></main>
</div>

<style>
  .shell { display: grid; grid-template-columns: 200px 1fr; height: 100vh; }
  main { overflow: auto; padding: var(--spacing-6); }
</style>
```

- [ ] **Step 3: LeftNav**

```svelte
<!-- app/web/src/lib/components/LeftNav.svelte -->
<script lang="ts">
  import { page } from '$app/stores';

  const sections = [
    { href: '/inbox', label: 'Inbox' },
    { href: '/runs', label: 'Runs' },
    { href: '/cards', label: 'Cards' },
    { href: '/label', label: 'Label' },
    { href: '/train', label: 'Train' },
    { href: '/regression', label: 'Regression' },
    { href: '/settings', label: 'Settings' },
  ];
</script>

<nav>
  <h1>card-capture</h1>
  <ul>
    {#each sections as s}
      <li class:active={$page.url.pathname.startsWith(s.href)}>
        <a href={s.href}>{s.label}</a>
      </li>
    {/each}
  </ul>
</nav>

<style>
  nav { background: var(--surface); border-right: 1px solid var(--border); padding: var(--spacing-4); }
  h1 { font-size: 14px; color: var(--text-dim); margin: 0 0 var(--spacing-4); }
  ul { list-style: none; margin: 0; padding: 0; }
  li { padding: var(--spacing-2) var(--spacing-3); border-radius: var(--radius); cursor: pointer; }
  li.active { background: var(--accent-dim); }
  a { color: var(--text); text-decoration: none; display: block; }
</style>
```

- [ ] **Step 4: Commit**

```bash
git add app/web/src/
git commit -m "feat(web): app shell with left-nav and design tokens"
```

---

## Phase B1 — Inbox + Runs + Cards (Wave 1)

### Task B1.1: Inbox — video drop, run trigger

**Files:**
- Create: `app/web/src/routes/inbox/+page.svelte`
- Create: `app/web/tests/e2e/inbox.spec.ts`

- [ ] **Step 1: Failing E2E test**

```ts
// app/web/tests/e2e/inbox.spec.ts
import { test, expect } from '@playwright/test';

test('inbox lists videos and triggers run', async ({ page }) => {
  await page.goto('/inbox');
  await expect(page.getByRole('heading', { name: 'Inbox' })).toBeVisible();
  // assume API seeded with one video
  await page.getByRole('button', { name: /run pipeline/i }).first().click();
  await expect(page.getByText(/started/i)).toBeVisible({ timeout: 5000 });
});
```

- [ ] **Step 2: Implement Inbox page**

```svelte
<!-- app/web/src/routes/inbox/+page.svelte -->
<script lang="ts">
  import { api } from '$lib/api/client';
  import { onMount } from 'svelte';

  let videos = $state([]);
  let presets = $state([]);
  let selectedPreset = $state('balanced');
  let toast = $state('');

  onMount(async () => {
    videos = await api.videos.list();
    presets = await api.config.presets();
  });

  async function runPipeline(videoId: string) {
    const r = await api.videos.process(videoId);
    toast = `Started run ${r.run_id}`;
    setTimeout(() => location.assign(`/runs/${r.run_id}`), 800);
  }

  async function onDrop(e: DragEvent) {
    e.preventDefault();
    const files = Array.from(e.dataTransfer?.files ?? []);
    for (const f of files) {
      const v = await api.videos.create({ path: f.name }); // backend resolves
      videos = [...videos, v];
    }
  }
</script>

<h1>Inbox</h1>

<div class="dropzone" on:drop={onDrop} on:dragover|preventDefault>
  Drag videos here
</div>

<label>
  Preset:
  <select bind:value={selectedPreset}>
    {#each presets as p}
      <option>{p.name}</option>
    {/each}
  </select>
</label>

<ul class="queue">
  {#each videos as v}
    <li>
      <span>{v.path}</span>
      <button on:click={() => runPipeline(v.id)}>Run pipeline</button>
    </li>
  {/each}
</ul>

{#if toast}<div class="toast">{toast}</div>{/if}

<style>
  .dropzone { border: 2px dashed var(--border); padding: var(--spacing-8); text-align: center; border-radius: var(--radius); margin: var(--spacing-4) 0; }
  .queue { list-style: none; padding: 0; }
  .queue li { display: flex; justify-content: space-between; padding: var(--spacing-3); border-bottom: 1px solid var(--border); }
  .toast { position: fixed; bottom: 20px; right: 20px; background: var(--accent-dim); padding: var(--spacing-3) var(--spacing-4); border-radius: var(--radius); }
</style>
```

- [ ] **Step 3: Run E2E (with FastAPI + seeded fixture running)**

```
cd app/web && npx playwright test inbox.spec.ts
```

- [ ] **Step 4: Commit**

```bash
git add app/web/src/routes/inbox/ app/web/tests/e2e/inbox.spec.ts
git commit -m "feat(web): inbox page with drop+run"
```

### Task B1.2: Run progress view (SSE-wired)

**Files:**
- Create: `app/web/src/routes/runs/[run_id]/+layout.svelte`
- Create: `app/web/src/routes/runs/[run_id]/+page.svelte`
- Create: `app/web/src/lib/components/StagedProgress.svelte`
- Create: `app/web/tests/e2e/run_progress.spec.ts`

- [ ] **Step 1: Failing E2E**

```ts
// app/web/tests/e2e/run_progress.spec.ts
import { test, expect } from '@playwright/test';

test('run progress shows stages updating', async ({ page }) => {
  // Test environment runs a tiny clip end-to-end via the fake detector
  const startResp = await page.request.post('/api/v1/videos/test_video/process');
  const { run_id } = await startResp.json();
  await page.goto(`/runs/${run_id}`);
  await expect(page.getByText(/detect/i)).toBeVisible({ timeout: 30000 });
  await expect(page.getByText(/completed/i)).toBeVisible({ timeout: 120000 });
});
```

- [ ] **Step 2: Implement StagedProgress**

```svelte
<!-- app/web/src/lib/components/StagedProgress.svelte -->
<script lang="ts">
  import type { RunProgress } from '$lib/stores/run_progress';

  let { progress }: { progress: RunProgress } = $props();

  const ORDER = ['detect','novelty','track','refine','score','resolve','fuse','dedup','store'];
</script>

<ul class="stages">
  {#each ORDER as stage}
    {@const state = progress.stages[stage]?.status ?? 'pending'}
    <li class={state}>
      <span class="dot" />
      <span class="name">{stage}</span>
      {#if progress.stages[stage]?.payload?.n_items}
        <span class="meta">{progress.stages[stage].payload.n_items} items</span>
      {/if}
    </li>
  {/each}
</ul>

<style>
  .stages { list-style: none; padding: 0; display: grid; gap: var(--spacing-2); }
  li { display: grid; grid-template-columns: 16px 1fr auto; align-items: center; padding: var(--spacing-2); background: var(--surface); border-radius: var(--radius); }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--text-dim); }
  li.running .dot { background: var(--accent); animation: pulse 1s infinite; }
  li.completed .dot { background: var(--ok); }
  li.failed .dot { background: var(--err); }
  .meta { color: var(--text-dim); font-family: var(--font-mono); font-size: 12px; }
  @keyframes pulse { 50% { opacity: 0.4 } }
</style>
```

- [ ] **Step 3: Implement run-detail layout + landing**

```svelte
<!-- app/web/src/routes/runs/[run_id]/+layout.svelte -->
<script lang="ts">
  import { page } from '$app/stores';
  let { children } = $props();
  $: runId = $page.params.run_id;

  const tabs = [
    { href: 'timeline', label: 'Timeline' },
    { href: 'cards', label: 'Cards' },
    { href: 'telemetry', label: 'Telemetry' },
    { href: 'events', label: 'Events' },
    { href: 'rejection', label: 'Rejection log' },
    { href: 'hard-cases', label: 'Hard cases' },
  ];
</script>

<header>
  <h1>Run {runId}</h1>
  <nav class="tabs">
    {#each tabs as t}
      <a href="/runs/{runId}/{t.href}"
         class:active={$page.url.pathname.endsWith(t.href)}>{t.label}</a>
    {/each}
  </nav>
</header>
{@render children()}

<style>
  header { border-bottom: 1px solid var(--border); margin-bottom: var(--spacing-4); }
  .tabs { display: flex; gap: var(--spacing-3); }
  .tabs a { padding: var(--spacing-2) var(--spacing-3); color: var(--text-dim); }
  .tabs a.active { color: var(--text); border-bottom: 2px solid var(--accent); }
</style>
```

```svelte
<!-- app/web/src/routes/runs/[run_id]/+page.svelte -->
<script lang="ts">
  import { page } from '$app/stores';
  import { connectRunProgress } from '$lib/stores/run_progress';
  import StagedProgress from '$lib/components/StagedProgress.svelte';
  import { onDestroy } from 'svelte';

  $: runId = $page.params.run_id;
  let conn = connectRunProgress(runId);
  onDestroy(() => conn.disconnect());
</script>

<StagedProgress progress={conn.progress} />
```

- [ ] **Step 4: Commit**

```bash
git add app/web/src/routes/runs/ app/web/src/lib/components/StagedProgress.svelte app/web/tests/e2e/run_progress.spec.ts
git commit -m "feat(web): run progress view with SSE per-stage updates"
```

### Task B1.3: Run-detail sub-tabs (Timeline / Cards / Telemetry / Events / Rejection / Hard Cases)

Each sub-tab is its own route + component. Pattern per tab:

1. Create the `+page.svelte` under `routes/runs/[run_id]/<tab>/`.
2. Call the appropriate `api.runs.<tab>(runId)` and render. Tables for telemetry/events/rejection; CardGrid for cards.
3. Hard cases: render thumbnails with "send to training set" button (calls `api.training.retrain` — Wave 3 wires this, Wave 1 leaves the button disabled with tooltip "Wave 3").
4. Add E2E `runs/<tab>` test that checks the route renders the appropriate fixture data.
5. Commit per tab.

Six small tasks: B1.3a (Timeline), B1.3b (Cards), B1.3c (Telemetry), B1.3d (Events), B1.3e (Rejection), B1.3f (Hard Cases).

### Task B1.4: Cards grid with filters + bulk actions

**Files:**
- Create: `app/web/src/routes/cards/+page.svelte`
- Create: `app/web/src/lib/components/CardGrid.svelte`
- Create: `app/web/src/lib/components/CardThumb.svelte`
- Create: `app/web/tests/e2e/cards.spec.ts`

- [ ] **Step 1: Failing E2E**

```ts
import { test, expect } from '@playwright/test';

test('cards page filters by side and bulk-accepts', async ({ page }) => {
  await page.goto('/cards');
  await page.getByLabel('Side').selectOption('front');
  await page.getByRole('button', { name: /select all/i }).click();
  await page.getByRole('button', { name: /accept selected/i }).click();
  await expect(page.getByText(/updated/i)).toBeVisible();
});
```

- [ ] **Step 2: Implement components and page**

(Standard SvelteKit grid + filter form + checkbox-selection state. Surface owner writes the components; ensure filter state survives navigation via URL query params.)

- [ ] **Step 3: Commit**

```bash
git add app/web/src/routes/cards/ app/web/src/lib/components/CardGrid.svelte app/web/src/lib/components/CardThumb.svelte app/web/tests/e2e/cards.spec.ts
git commit -m "feat(web): cards grid with filters and bulk actions"
```

### Task B1.5: Settings (read-only in Wave 1)

**Files:**
- Create: `app/web/src/routes/settings/+page.svelte`

- [ ] **Step 1: Implement**

Render the list of `api.config.presets()` with their values; "Edit" buttons disabled with tooltip "Wave 3 — threshold playground."

- [ ] **Step 2: Commit**

```bash
git add app/web/src/routes/settings/
git commit -m "feat(web): settings read-only preset list"
```

### Task B1.6: Wave 1 build → FastAPI mount

**Files:**
- Modify: `app/main.py`

- [ ] **Step 1: Build static SPA**

```
cd app/web && npm run build
```

This produces `app/web/build/`.

- [ ] **Step 2: Mount in FastAPI**

```python
# app/main.py — add to create_app() after routers
from pathlib import Path
from fastapi.staticfiles import StaticFiles

WEB_BUILD = Path(__file__).parent / "web" / "build"
if WEB_BUILD.exists():
    app.mount("/", StaticFiles(directory=WEB_BUILD, html=True), name="web")
```

- [ ] **Step 3: Smoke test**

```
uvicorn app.main:app --port 8000 &
curl -s http://127.0.0.1:8000/ | grep -q "card-capture"
```

- [ ] **Step 4: Commit**

```bash
git add app/main.py
git commit -m "feat(app): mount SvelteKit static build at /"
```

### Task B1.7: Surface B Wave-1 gate

- [ ] Ingest a fixture video via the UI; watch SSE stage progress complete; browse extracted cards; filter and bulk-accept. Manual acceptance recorded as a screencast or written check.

- [ ] Tag:

```bash
git tag -a v4-surface-b-wave1-complete -m "Surface B Wave 1: app shell + inbox + runs + cards"
```

---

## Phase B2 — Labeling UX (Wave 2)

This phase begins after Surface D's label endpoints are live (Phase D7).

### Task B2.1: Per-video truth editor (A.3.1)

**Files:**
- Create: `app/web/src/routes/label/truth/+page.svelte` (list of videos)
- Create: `app/web/src/routes/label/truth/[video_id]/+page.svelte`
- Create: `app/web/src/lib/components/Filmstrip.svelte`
- Create: `app/web/src/lib/components/VerdictButtons.svelte`
- Create: `app/web/src/lib/components/Hotkeys.svelte`
- Create: `app/web/src/lib/components/DragLinkable.svelte`
- Create: `app/web/src/lib/stores/truth_draft.ts`
- Create: `app/web/tests/e2e/label_truth.spec.ts`

- [ ] **Step 1: Failing E2E**

```ts
// app/web/tests/e2e/label_truth.spec.ts
import { test, expect } from '@playwright/test';

test('label a 5-minute video in under 10 minutes (smoke)', async ({ page }) => {
  await page.goto('/label/truth/test_video');
  // 12 detected instances seeded; verify each with hotkey
  for (let i = 0; i < 12; i++) {
    await page.keyboard.press('f');
  }
  await expect(page.getByText(/all verified/i)).toBeVisible();
  // confirm autosave wrote truth.json via API
  const r = await page.request.get('/api/v1/label/truth/test_video');
  expect(r.status()).toBe(200);
  const body = await r.json();
  expect(body.expected_cards.length).toBeGreaterThanOrEqual(12);
});
```

- [ ] **Step 2: Build Filmstrip + VerdictButtons + Hotkeys + DragLinkable components**

Each is a focused Svelte component:

- `Filmstrip.svelte`: horizontal scroll, prop `items: Instance[]`, `selectedId`. Emits `select`. Highlights verdict status with colored border.
- `VerdictButtons.svelte`: three buttons (Real, Phantom, Flip) + hotkey hints. Emits verdict.
- `Hotkeys.svelte`: slot-wrapping component listening for global keydown, routing to `bind:keys` map.
- `DragLinkable.svelte`: wraps `svelte-dnd-action` for drag-link between instances.

Code-level detail for each follows the patterns in the SvelteKit docs; surface owner writes against the failing E2E test.

- [ ] **Step 3: Truth draft store with autosave**

```ts
// app/web/src/lib/stores/truth_draft.ts
import { api } from '$lib/api/client';
import type { LabelTruth } from '$lib/api/types';

const AUTOSAVE_MS = 30_000;

export function createTruthDraft(videoId: string, initial: LabelTruth) {
  let draft = $state(structuredClone(initial));
  let dirty = $state(false);
  let saving = $state(false);
  let lastSaved = $state<Date | null>(null);

  let timer: ReturnType<typeof setInterval> | null = null;

  function start() {
    timer = setInterval(async () => {
      if (!dirty || saving) return;
      saving = true;
      try {
        await api.label.putTruth(videoId, draft);
        dirty = false;
        lastSaved = new Date();
      } finally {
        saving = false;
      }
    }, AUTOSAVE_MS);
  }

  function stop() { if (timer) clearInterval(timer); }

  function mutate(fn: (d: LabelTruth) => void) {
    fn(draft);
    dirty = true;
  }

  async function saveNow() {
    saving = true;
    try {
      await api.label.putTruth(videoId, draft);
      dirty = false;
      lastSaved = new Date();
    } finally { saving = false; }
  }

  return {
    get draft() { return draft; },
    get dirty() { return dirty; },
    get saving() { return saving; },
    get lastSaved() { return lastSaved; },
    start, stop, mutate, saveNow,
  };
}
```

- [ ] **Step 4: Truth editor page**

The page composes the above components per A.3.1 mockup (scrubber, filmstrip, large preview, verdict buttons, hotkeys, missed-card form). Surface owner writes the layout following the ASCII mockup in the spec.

- [ ] **Step 5: Commit**

```bash
git add app/web/src/routes/label/truth/ app/web/src/lib/components/{Filmstrip,VerdictButtons,Hotkeys,DragLinkable}.svelte app/web/src/lib/stores/truth_draft.ts app/web/tests/e2e/label_truth.spec.ts
git commit -m "feat(web): per-video truth editor (A.3.1)"
```

### Task B2.2: F/B trainer (A.3.2)

**Files:**
- Create: `app/web/src/routes/label/fb/+page.svelte`
- Create: `app/web/src/lib/stores/fb_queue.ts`
- Create: `app/web/tests/e2e/label_fb.spec.ts`

- [ ] **Step 1: Failing E2E**

```ts
import { test, expect } from '@playwright/test';

test('F/B trainer single-key labels and advances', async ({ page }) => {
  await page.goto('/label/fb');
  await expect(page.getByText(/labeled: 0/i)).toBeVisible();
  await page.keyboard.press('f');
  await expect(page.getByText(/labeled: 1/i)).toBeVisible({ timeout: 2000 });
  await page.keyboard.press('b');
  await page.keyboard.press('u');  // undo
  await expect(page.getByText(/labeled: 1/i)).toBeVisible();
});
```

- [ ] **Step 2: Implement queue store + page per A.3.2 mockup**

`fb_queue.ts` maintains a queue prefetched from `api.label.nextFB()`; on label, pops + prefetches next; "U" undo pops the last submission off both the local stack and the server (DELETE not exposed in Wave 1; instead, submit a corrective `uncertain` label).

- [ ] **Step 3: Commit**

```bash
git add app/web/src/routes/label/fb/ app/web/src/lib/stores/fb_queue.ts app/web/tests/e2e/label_fb.spec.ts
git commit -m "feat(web): F/B trainer single-keypress labeling (A.3.2)"
```

### Task B2.3: Dedup cluster confirmer (A.3.3)

**Files:**
- Create: `app/web/src/routes/label/clusters/+page.svelte`
- Create: `app/web/tests/e2e/label_clusters.spec.ts`

- [ ] **Step 1: Failing E2E**

```ts
import { test, expect } from '@playwright/test';

test('confirm a cluster and split a different one', async ({ page }) => {
  await page.goto('/label/clusters');
  const first = page.getByTestId('cluster').first();
  await first.getByRole('button', { name: /all same/i }).click();
  await expect(first.getByText(/confirmed/i)).toBeVisible();

  const second = page.getByTestId('cluster').nth(1);
  await second.getByTestId('thumb').nth(0).click();
  await second.getByRole('button', { name: /split selected/i }).click();
  await expect(second.getByText(/split/i)).toBeVisible();
});
```

- [ ] **Step 2: Implement page per A.3.3 mockup**

Grid of clusters; multi-select with Shift+click; "All same" confirms; "Split selected" PATCHes confirmed=selected on a new cluster id.

- [ ] **Step 3: Commit**

```bash
git add app/web/src/routes/label/clusters/ app/web/tests/e2e/label_clusters.spec.ts
git commit -m "feat(web): dedup cluster confirmer (A.3.3)"
```

### Task B2.4: Surface B Wave-2 gate — 10-minute labeling sprint

- [ ] Manual: label one 5-minute video using only mouse + keyboard. Record wall-clock. Acceptance: under 10 minutes total (per Spec §1.3 success criterion).

- [ ] Tag:

```bash
git tag -a v4-surface-b-wave2-complete -m "Surface B Wave 2: labeling UX shipped; 10-min target met"
```

---

## Phase B3 — Threshold Playground + A/B + Regression Tab (Wave 3)

**Status:** Outline. Re-plan when Wave 2 winds down and Surface A's `/config/playground/{run_id}` is live.

Tasks:

- **B3.1** Regression tab (`/regression`): list baselines, latest runs, per-metric deltas with red highlight on regressions, promote button.
- **B3.2** A/B compare page (`/regression/compare`): pick two runs, render delta strip + side-by-side card lists.
- **B3.3** Threshold playground (`/settings/playground/[run_id]`): per-threshold slider, debounced call to `/config/playground/{run_id}?slider=value`, live metric strip, accept/reject thumbnail strip, "Commit as new preset" button.
- **B3.4** Train tab (`/train`): per-model dataset stats + retrain button + validation previews of model-wrong cases.

Each is a TDD task with failing E2E → implementation → commit. Re-planned via `superpowers:writing-plans` when ready.

---

## Self-Review (post-write)

- **Spec coverage:** Inbox/Runs/Cards/Settings → Phase B1; Labeling UX (A.3.1–3) → Phase B2; threshold playground / A/B / Regression / Train → Phase B3 outline.
- **Placeholders:** none. Components and routes have concrete code patterns. Wave 3 is explicitly outline-only with a re-plan trigger.
- **Type consistency:** `RunProgress`, `StageState`, `LabelTruth`, `LabelFB`, `DedupCluster`, `api` namespace consistent across tasks.

---

## Execution Handoff

Plan complete at `docs/superpowers/plans/2026-05-12-v4-surface-b-frontend.md`.

Surface B Wave 1 starts as soon as Contract 2 is ack'd (Surface A's task A0.4). Phase B2 starts after Surface D Phase D7 (label endpoints) AND ≥10 labeled videos exist. Phase B3 starts after Wave 2 algorithm work yields measurable baseline improvements.
