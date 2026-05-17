<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { api } from '$lib/api/client';
  import { createTrainingStore } from '$lib/stores/training.svelte';

  const store = createTrainingStore();

  let retraining = $state(false);
  let retrainError = $state<string | null>(null);
  let retrainLogs = $state<string[]>([]);

  interface BenchmarkRow { video: string; before: number; after: number; delta: number; }
  let benchmarkRows = $state<BenchmarkRow[]>([]);
  let benchmarking = $state(false);

  onMount(() => store.refresh());

  async function retrainAll() {
    retraining = true;
    retrainError = null;
    retrainLogs = [];
    try {
      for (const model of ['presence', 'fb_classifier']) {
        const job = await api.training.retrain(model, { epochs: 30, learning_rate: 0.001 });
        await pollJob(job.job_id);
      }
      await store.refresh();
    } catch (e: any) {
      retrainError = e.message ?? 'Retrain failed';
    } finally {
      retraining = false;
    }
  }

  async function pollJob(id: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const interval = setInterval(async () => {
        const job = await api.training.getJob(id);
        if (job.logs) retrainLogs = job.logs;
        if (job.status === 'completed') { clearInterval(interval); resolve(); }
        else if (job.status === 'failed') {
          clearInterval(interval);
          reject(new Error(job.error ?? 'job failed'));
        }
      }, 2000);
    });
  }

  async function runBenchmark() {
    benchmarking = true;
    benchmarkRows = [];
    const { job_id } = await api.training.startBenchmark(3);
    await pollBenchmarkJob(job_id);
    benchmarking = false;
  }

  async function pollBenchmarkJob(id: string) {
    return new Promise<void>((resolve) => {
      const interval = setInterval(async () => {
        const j = await api.training.getBenchmark(id);
        if (j.status === 'completed') {
          benchmarkRows = j.rows ?? [];
          clearInterval(interval); resolve();
        } else if (j.status === 'failed') { clearInterval(interval); resolve(); }
      }, 3000);
    });
  }

  function scrollBottom(node: HTMLElement, _dep: any) {
    $effect(() => { void _dep; node.scrollTop = node.scrollHeight; });
  }

  function pct(v: number | null | undefined) {
    return v != null ? `${Math.round(v * 100)}%` : '—';
  }

  function modelHistory(model: string) {
    return store.stats?.history?.filter(h => h.model === model) ?? [];
  }

  function accLine(model: string) {
    const hist = modelHistory(model);
    const cur = hist.at(-1);
    const prev = hist.at(-2);
    return { cur, prev };
  }

  function deltaClass(cur: any, prev: any) {
    if (!prev) return 'delta-neutral';
    return cur.accuracy > prev.accuracy ? 'delta-pos' : cur.accuracy < prev.accuracy ? 'delta-neg' : 'delta-neutral';
  }

  function deltaArrow(cur: any, prev: any) {
    if (!prev) return '';
    return cur.accuracy > prev.accuracy ? '↑' : cur.accuracy < prev.accuracy ? '↓' : '→';
  }
</script>

<div class="hub">
  <h1>Training</h1>

  <!-- Workflow guide -->
  <div class="workflow">
    <div class="step">
      <div class="step-num">1</div>
      <div class="step-body">
        <strong>Process videos</strong>
        <span>Run pipeline from the Videos page. Each run queues presence frames automatically.</span>
      </div>
    </div>
    <div class="step-arrow">→</div>
    <div class="step">
      <div class="step-num">2</div>
      <div class="step-body">
        <strong>Label everything</strong>
        <span>Clear all three labeling queues below. More labels = better models.</span>
      </div>
    </div>
    <div class="step-arrow">→</div>
    <div class="step">
      <div class="step-num">3</div>
      <div class="step-body">
        <strong>Retrain</strong>
        <span>Hit "Retrain all" once queues are cleared. Needs ≥10 labeled samples per model.</span>
      </div>
    </div>
    <div class="step-arrow">→</div>
    <div class="step">
      <div class="step-num">4</div>
      <div class="step-body">
        <strong>Benchmark</strong>
        <span>Run the last 3 videos through the new model and compare card counts.</span>
      </div>
    </div>
  </div>

  <!-- Labeling queues -->
  <h2 class="section-title">Labeling Queues</h2>
  <div class="panels">
    <a class="panel" href="/training/presence">
      <div class="panel-title">Presence</div>
      <div class="pending">{store.stats?.pending.presence ?? '…'}</div>
      <div class="pending-label">frames to label</div>
      <div class="acc">
        {#if accLine('presence').cur}
          {pct(accLine('presence').cur.accuracy)}
          {#if accLine('presence').prev}
            <span class={deltaClass(accLine('presence').cur, accLine('presence').prev)}>
              ({deltaArrow(accLine('presence').cur, accLine('presence').prev)} from {pct(accLine('presence').prev.accuracy)})
            </span>
          {/if}
        {:else}
          not yet trained
        {/if}
      </div>
      <div class="panel-cta">Label Y / N →</div>
    </a>

    <a class="panel" href="/training/fb">
      <div class="panel-title">Front / Back</div>
      <div class="pending">{store.stats?.pending.fb ?? '…'}</div>
      <div class="pending-label">cards to label</div>
      <div class="acc">
        {#if accLine('fb_classifier').cur}
          {pct(accLine('fb_classifier').cur.accuracy)}
          {#if accLine('fb_classifier').prev}
            <span class={deltaClass(accLine('fb_classifier').cur, accLine('fb_classifier').prev)}>
              ({deltaArrow(accLine('fb_classifier').cur, accLine('fb_classifier').prev)} from {pct(accLine('fb_classifier').prev.accuracy)})
            </span>
          {/if}
        {:else}
          not yet trained
        {/if}
      </div>
      <div class="panel-cta">Label F / B →</div>
    </a>

    <a class="panel" href="/training/corners">
      <div class="panel-title">YOLO Corners</div>
      <div class="pending">{store.stats?.pending.corners ?? '…'}</div>
      <div class="pending-label">detections to review</div>
      <div class="acc">Used for YOLO fine-tuning</div>
      <div class="panel-cta">Review Y / E / N →</div>
    </a>
  </div>

  <!-- Ground truth & verification tools -->
  <h2 class="section-title">Ground Truth & Verification</h2>
  <div class="panels panels-sm">
    <a class="panel panel-sm" href="/label/truth">
      <div class="panel-title">Per-Video Truth</div>
      <div class="panel-desc">Verify detections and create ground truth files for regression testing.</div>
    </a>
    <a class="panel panel-sm" href="/label/clusters">
      <div class="panel-title">Dedup Clusters</div>
      <div class="panel-desc">Confirm or reject deduplication groups across sessions.</div>
    </a>
    <a class="panel panel-sm" href="/label/hard_cases">
      <div class="panel-title">Hard Cases</div>
      <div class="panel-desc">Graduate failure cases into the permanent training set.</div>
    </a>
  </div>

  <!-- Retrain -->
  <h2 class="section-title">Retrain</h2>
  <div class="retrain-row">
    <button class="retrain-btn" disabled={retraining} onclick={retrainAll}>
      {retraining ? 'Training… (this takes a few minutes)' : 'Retrain all models'}
    </button>
    <span class="retrain-note">Trains presence + front/back classifiers sequentially. Requires ≥10 labeled samples each.</span>
  </div>
  {#if retraining || retrainLogs.length > 0}
    <div class="log-box" class:log-error={!!retrainError}
      use:scrollBottom={retrainLogs}>
      {#each retrainLogs as line}
        <div class="log-line">{line}</div>
      {/each}
      {#if retraining}
        <div class="log-line log-cursor">▋</div>
      {/if}
    </div>
  {/if}
  {#if retrainError}
    <div class="error-box">{retrainError}</div>
  {/if}

  <!-- Benchmark -->
  <h2 class="section-title">Benchmark</h2>
  <div class="retrain-row">
    <button class="bm-btn" disabled={benchmarking} onclick={runBenchmark}>
      {benchmarking ? 'Running pipeline…' : 'Run last 3 videos'}
    </button>
    <span class="retrain-note">Re-runs the pipeline on your last 3 videos and shows before/after card counts.</span>
  </div>
  {#if benchmarkRows.length > 0}
    <table class="bm-table">
      <thead><tr><th>Video</th><th>Before</th><th>After</th><th>Δ</th></tr></thead>
      <tbody>
        {#each benchmarkRows as row}
          <tr>
            <td>{row.video}</td>
            <td>{row.before} cards</td>
            <td>{row.after} cards</td>
            <td class:positive={row.delta > 0} class:neutral={row.delta === 0}>
              {row.delta > 0 ? '+' : ''}{row.delta}
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
  {/if}

  <!-- Accuracy history -->
  {#if store.stats?.history?.length}
    <h2 class="section-title">Accuracy History</h2>
    {#each ['presence', 'fb_classifier'] as model}
      {@const points = [...(store.stats?.history ?? [])].filter(h => h.model === model && h.accuracy != null).reverse()}
      {#if points.length > 0}
        <div class="history-model">
          <div class="history-model-name">{model === 'presence' ? 'Presence classifier' : 'Front/Back classifier'}</div>
          <table class="history-table">
            <thead><tr><th>#</th><th>Date</th><th>Accuracy</th><th>Change</th></tr></thead>
            <tbody>
              {#each points as row, i}
                {@const prev = points[i + 1]}
                {@const delta = prev != null ? (row.accuracy ?? 0) - (prev.accuracy ?? 0) : null}
                <tr class:latest={i === 0}>
                  <td class="run-num">{points.length - i}</td>
                  <td class="run-date">{new Date(row.created_at).toLocaleString()}</td>
                  <td class="run-acc">{pct(row.accuracy)}</td>
                  <td class="run-delta">
                    {#if delta === null}
                      <span class="delta-neutral">first run</span>
                    {:else if delta > 0.001}
                      <span class="delta-pos">↑ +{pct(delta)}</span>
                    {:else if delta < -0.001}
                      <span class="delta-neg">↓ {pct(delta)}</span>
                    {:else}
                      <span class="delta-neutral">→ no change</span>
                    {/if}
                  </td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {/if}
    {/each}
  {/if}
</div>

<style>
  .hub { max-width: 900px; margin: 2rem auto; padding: 0 1rem 4rem; }
  h1 { margin-bottom: 1.5rem; }
  .section-title { margin: 2.5rem 0 1rem; font-size: 1rem; text-transform: uppercase;
    letter-spacing: 0.05em; color: #888; font-weight: 600; }

  /* Workflow */
  .workflow {
    display: flex; align-items: flex-start; gap: 0.5rem;
    background: #1a1a2e; border-radius: 12px; padding: 1.25rem 1.5rem;
    margin-bottom: 2rem; flex-wrap: wrap; color: #e0e0e0;
  }
  .step { display: flex; gap: 0.75rem; align-items: flex-start; flex: 1; min-width: 160px; }
  .step-num {
    width: 28px; height: 28px; border-radius: 50%;
    background: #6366f1; color: white; font-weight: 700; font-size: 0.85rem;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
  }
  .step-body { display: flex; flex-direction: column; gap: 0.2rem; }
  .step-body strong { font-size: 0.9rem; }
  .step-body span { font-size: 0.78rem; color: #999; line-height: 1.4; }
  .step-arrow { color: #444; font-size: 1.25rem; padding-top: 4px; flex-shrink: 0; }

  /* Panels */
  .panels { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }
  .panels-sm { grid-template-columns: repeat(3, 1fr); }
  .panel {
    background: #1e1e2e; border-radius: 12px; padding: 1.25rem 1.5rem;
    text-decoration: none; color: #e0e0e0;
    display: flex; flex-direction: column; gap: 0.35rem;
    border: 1px solid transparent; transition: border-color 0.15s, background 0.15s;
  }
  .panel:hover { background: #252535; border-color: #6366f1; }
  .panel-title { font-weight: 700; font-size: 1rem; color: #e0e0e0; }
  .pending { font-size: 2rem; font-weight: 800; color: #fff; line-height: 1; }
  .pending-label { font-size: 0.75rem; color: #888; }
  .acc { font-size: 0.8rem; color: #aaa; margin-top: 0.25rem; }
  .panel-cta {
    margin-top: auto; padding-top: 0.75rem;
    font-size: 0.85rem; font-weight: 600; color: #6366f1;
  }
  .panel-sm { gap: 0.5rem; }
  .panel-desc { font-size: 0.82rem; color: #999; line-height: 1.5; }

  /* Retrain */
  .retrain-row { display: flex; align-items: center; gap: 1.25rem; flex-wrap: wrap; }
  .retrain-btn {
    background: #6366f1; color: white; border: none; border-radius: 8px;
    padding: 0.7rem 1.5rem; cursor: pointer; font-size: 0.95rem; font-weight: 600;
    white-space: nowrap;
  }
  .retrain-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .bm-btn {
    background: #0acf97; color: white; border: none; border-radius: 8px;
    padding: 0.7rem 1.5rem; cursor: pointer; font-size: 0.95rem; font-weight: 600;
    white-space: nowrap;
  }
  .bm-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .retrain-note { font-size: 0.82rem; color: #888; }
  .log-box {
    margin-top: 0.75rem; background: #0d0d1a; border: 1px solid #2a2a4a;
    border-radius: 8px; padding: 0.75rem 1rem; font-family: monospace;
    font-size: 0.82rem; color: #a0cfff; max-height: 220px; overflow-y: auto;
    display: flex; flex-direction: column; gap: 0.15rem;
  }
  .log-line { white-space: pre; line-height: 1.5; }
  .log-cursor { animation: blink 1s step-end infinite; color: #6366f1; }
  @keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0; } }

  .error-box {
    margin-top: 0.75rem; background: #3a1a1a; border: 1px solid #fa5c7c;
    border-radius: 8px; padding: 0.75rem 1rem; font-size: 0.85rem; color: #fa5c7c;
    white-space: pre-wrap;
  }

  /* Benchmark table */
  .bm-table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
  .bm-table th, .bm-table td {
    padding: 0.5rem 1rem; text-align: left; border-bottom: 1px solid #2a2a3e;
    font-size: 0.9rem;
  }
  .positive { color: #0acf97; font-weight: 600; }
  .neutral { color: #aaa; }

  /* Accuracy history */
  .history-model { margin-bottom: 1.5rem; }
  .history-model-name { font-size: 0.9rem; font-weight: 600; color: #ccc; margin-bottom: 0.5rem; }
  .history-table { width: 100%; border-collapse: collapse; }
  .history-table th {
    text-align: left; font-size: 0.75rem; text-transform: uppercase;
    letter-spacing: 0.05em; color: #666; padding: 0.4rem 0.75rem;
    border-bottom: 1px solid #2a2a3e;
  }
  .history-table td { padding: 0.5rem 0.75rem; font-size: 0.88rem; border-bottom: 1px solid #1a1a2e; }
  .history-table tr.latest td { background: #1a1a2e; }
  .run-num { color: #555; width: 2rem; }
  .run-date { color: #999; font-size: 0.82rem; }
  .run-acc { font-weight: 700; color: #e0e0e0; }
  .delta-pos { color: #0acf97; font-weight: 600; }
  .delta-neg { color: #fa5c7c; font-weight: 600; }
  .delta-neutral { color: #666; }
</style>
