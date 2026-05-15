<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { api } from '$lib/api/client';
  import { createTrainingStore } from '$lib/stores/training.svelte';

  const store = createTrainingStore();

  let retraining = $state(false);

  interface BenchmarkRow { video: string; before: number; after: number; delta: number; }
  let benchmarkRows = $state<BenchmarkRow[]>([]);
  let benchmarking = $state(false);

  onMount(() => store.refresh());

  async function retrain(model: string) {
    retraining = true;
    const r = await api.post(`/training/retrain/${model}`, { epochs: 30, learning_rate: 0.001 });
    const job = await r.json();
    await pollJob(job.job_id);
  }

  async function retrainAll() {
    retraining = true;
    for (const model of ['presence', 'fb_classifier']) {
      const r = await api.post(`/training/retrain/${model}`, { epochs: 30, learning_rate: 0.001 });
      const job = await r.json();
      await pollJob(job.job_id);
    }
    await store.refresh();
    retraining = false;
  }

  async function pollJob(id: string): Promise<void> {
    return new Promise((resolve) => {
      const interval = setInterval(async () => {
        const r = await api.get(`/training/jobs/${id}`);
        const job = await r.json();
        if (job.status === 'completed' || job.status === 'failed') {
          clearInterval(interval);
          resolve();
        }
      }, 2000);
    });
  }

  async function runBenchmark() {
    benchmarking = true;
    benchmarkRows = [];
    const r = await api.post('/training/benchmark', { n: 3 });
    const job = await r.json();
    await pollBenchmarkJob(job.job_id);
    benchmarking = false;
  }

  async function pollBenchmarkJob(id: string) {
    return new Promise<void>((resolve) => {
      const interval = setInterval(async () => {
        const r = await api.get(`/training/benchmark/${id}`);
        const j = await r.json();
        if (j.status === 'completed') {
          benchmarkRows = j.rows ?? [];
          clearInterval(interval);
          resolve();
        } else if (j.status === 'failed') {
          clearInterval(interval);
          resolve();
        }
      }, 3000);
    });
  }

  function pct(v: number | null | undefined) {
    return v != null ? `${Math.round(v * 100)}%` : '—';
  }
</script>

<div class="hub">
  <h1>Training</h1>

  <div class="panels">
    <div class="panel" onclick={() => goto('/training/presence')}>
      <div class="panel-title">Presence</div>
      <div class="pending">{store.stats?.pending.presence ?? '…'} pending</div>
      <div class="acc">acc: {pct(store.stats?.accuracy['presence'])}</div>
      <button class="label-btn" onclick|stopPropagation={() => goto('/training/presence')}>
        Label now →
      </button>
    </div>

    <div class="panel" onclick={() => goto('/training/fb')}>
      <div class="panel-title">Front / Back</div>
      <div class="pending">{store.stats?.pending.fb ?? '…'} pending</div>
      <div class="acc">acc: {pct(store.stats?.accuracy['fb_classifier'])}</div>
      <button class="label-btn" onclick|stopPropagation={() => goto('/training/fb')}>
        Label now →
      </button>
    </div>

    <div class="panel" onclick={() => goto('/training/corners')}>
      <div class="panel-title">YOLO Corners</div>
      <div class="pending">{store.stats?.pending.corners ?? '…'} pending</div>
      <div class="acc">acc: —</div>
      <button class="label-btn" onclick|stopPropagation={() => goto('/training/corners')}>
        Label now →
      </button>
    </div>

    <div class="panel benchmark">
      <div class="panel-title">Retrain</div>
      <button class="retrain-btn" disabled={retraining} onclick={retrainAll}>
        {retraining ? 'Training…' : 'Retrain all'}
      </button>
    </div>
  </div>

  <!-- Benchmark section -->
  <div class="benchmark-section">
    <div class="bm-header">
      <h2>Benchmark</h2>
      <button class="bm-btn" disabled={benchmarking} onclick={runBenchmark}>
        {benchmarking ? 'Running…' : 'Run pipeline on last 3 videos'}
      </button>
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
                {row.delta > 0 ? '+' : ''}{row.delta} {row.delta > 0 ? '✓' : '→'}
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}
  </div>

  <!-- Accuracy history chart -->
  {#if store.stats?.history?.length}
    <div class="chart-section">
      <h2>Accuracy over time</h2>
      <div class="chart">
        {#each ['presence', 'fb_classifier'] as model}
          {@const points = store.stats.history.filter(h => h.model === model && h.accuracy != null)}
          {#if points.length > 1}
            <div class="series">
              <span class="series-label">{model === 'presence' ? 'Presence' : 'Front/Back'}</span>
              <svg viewBox="0 0 200 60" class="sparkline">
                <polyline
                  points={points.map((p, i) =>
                    `${(i / (points.length - 1)) * 190 + 5},${55 - (p.accuracy ?? 0) * 50}`
                  ).join(' ')}
                  fill="none"
                  stroke={model === 'presence' ? '#6366f1' : '#0acf97'}
                  stroke-width="2"
                />
              </svg>
              <span class="series-pct">{pct(points[points.length - 1]?.accuracy)}</span>
            </div>
          {/if}
        {/each}
      </div>
    </div>
  {/if}
</div>

<style>
  .hub { max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
  h1 { margin-bottom: 1.5rem; }
  .panels { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; }
  .panel {
    background: #1e1e2e; border-radius: 12px; padding: 1.25rem;
    cursor: pointer; transition: background 0.15s;
    display: flex; flex-direction: column; gap: 0.5rem;
  }
  .panel:hover { background: #2a2a3e; }
  .panel-title { font-weight: 700; font-size: 1rem; }
  .pending { font-size: 1.4rem; font-weight: 700; }
  .acc { color: #aaa; font-size: 0.85rem; }
  .label-btn {
    margin-top: auto; background: #6366f1; color: white;
    border: none; border-radius: 8px; padding: 0.5rem 1rem;
    cursor: pointer; font-size: 0.85rem;
  }
  .retrain-btn {
    background: #0acf97; color: white; border: none;
    border-radius: 8px; padding: 0.6rem 1.2rem; cursor: pointer;
    font-weight: 600; margin-top: auto;
  }
  .retrain-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .benchmark { cursor: default; }
  .benchmark:hover { background: #1e1e2e; }

  .benchmark-section { margin-top: 2rem; }
  .bm-header { display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem; }
  .bm-header h2 { margin: 0; }
  .bm-btn {
    background: #6366f1; color: white; border: none;
    border-radius: 8px; padding: 0.5rem 1.2rem; cursor: pointer;
  }
  .bm-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .bm-table { width: 100%; border-collapse: collapse; }
  .bm-table th, .bm-table td { padding: 0.5rem 1rem; text-align: left; border-bottom: 1px solid #333; }
  .positive { color: #0acf97; font-weight: 600; }
  .neutral { color: #aaa; }
  .chart-section { margin-top: 2rem; }
  .chart { display: flex; flex-direction: column; gap: 0.75rem; }
  .series { display: flex; align-items: center; gap: 1rem; }
  .series-label { width: 90px; font-size: 0.85rem; color: #aaa; }
  .sparkline { width: 200px; height: 60px; }
  .series-pct { font-weight: 700; width: 40px; }
</style>
