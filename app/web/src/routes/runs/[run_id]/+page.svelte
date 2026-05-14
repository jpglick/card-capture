<script lang="ts">
    import { onMount, onDestroy } from 'svelte';
    import { page } from '$app/state';
    import { api } from '$lib/api/client';
    import type { RunDetail } from '$lib/api/types';

    const runId = page.params.run_id;
    let run: RunDetail | null = $state(null);
    let loading = $state(true);
    let error = $state<string | null>(null);
    let liveLines = $state<string[]>([]);
    let evtSource: EventSource | null = null;

    async function loadRun() {
        try {
            loading = true;
            run = await api.runs.detail(runId);
        } catch (e: any) {
            error = e.message;
        } finally {
            loading = false;
        }
    }

    function connectSSE() {
        evtSource = new EventSource(`/events/${runId}`);
        evtSource.addEventListener('log', (e: any) => {
            try {
                const d = JSON.parse(e.data);
                if (d.payload?.line) liveLines = [...liveLines, d.payload.line];
            } catch {}
        });
        evtSource.addEventListener('run_completed', () => {
            evtSource?.close();
            loadRun(); // refresh summary stats
        });
        evtSource.addEventListener('run_failed', (e: any) => {
            try {
                const d = JSON.parse(e.data);
                if (d.payload?.error) liveLines = [...liveLines, `ERROR: ${d.payload.error}`];
            } catch {}
            evtSource?.close();
            loadRun();
        });
        evtSource.onerror = () => evtSource?.close();
    }

    onMount(() => {
        loadRun().then(() => {
            if (run?.status === 'running') connectSSE();
        });
    });

    onDestroy(() => evtSource?.close());

    function fmt(ms: number) {
        if (ms < 1000) return `${ms}ms`;
        if (ms < 60000) return `${(ms/1000).toFixed(1)}s`;
        return `${Math.floor(ms/60000)}m ${Math.floor((ms%60000)/1000)}s`;
    }
</script>

<div class="page-header">
    <h1>Run <code>{runId}</code></h1>
    {#if run}
        <a href="/runs" class="back-link">← All runs</a>
    {/if}
</div>

{#if loading}
    <p class="muted">Loading…</p>
{:else if error}
    <p class="error">{error}</p>
{:else if run}
    <div class="summary-cards">
        <div class="stat-card">
            <div class="stat-label">Status</div>
            <div class="stat-value status-{run.status}">{run.status}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Cards extracted</div>
            <div class="stat-value">{run.cards_extracted ?? 0}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Elapsed</div>
            <div class="stat-value">{run.elapsed_ms ? fmt(run.elapsed_ms) : '—'}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Video</div>
            <div class="stat-value small">{run.video_id}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Started</div>
            <div class="stat-value small">{run.created_at ? new Date(run.created_at).toLocaleString() : '—'}</div>
        </div>
    </div>

    {#if run.cards_extracted > 0}
        <a href="/cards?run_id={runId}" class="view-cards-btn">View {run.cards_extracted} cards →</a>
    {/if}

    <h2>{run.status === 'running' ? 'Live log' : 'Run log'}</h2>
    <div class="log-box">
        {#if run.status === 'running' && liveLines.length === 0}
            <span class="muted">Waiting for pipeline output…</span>
        {:else if run.status !== 'running' && liveLines.length === 0 && (!run.events || run.events.length === 0)}
            <span class="muted">No log lines recorded for this run.</span>
        {/if}

        {#each liveLines as line}
            <div class="log-line live">{line}</div>
        {/each}

        {#if run.events?.length > 0}
            <div class="log-divider">— pipeline events —</div>
            {#each run.events as ev}
                <div class="log-line">
                    <span class="log-ts">{ev.created_at}</span>
                    <span class="log-type">{ev.event_type}</span>
                    {#if ev.data_json}
                        <span class="log-data">{ev.data_json}</span>
                    {/if}
                </div>
            {/each}
        {/if}
    </div>
{/if}

<style>
    .page-header {
        display: flex;
        align-items: baseline;
        gap: 1.5rem;
        margin-bottom: 1.5rem;
    }
    h1 { margin: 0; }
    h1 code { font-size: 1rem; background: #f0f1f9; padding: 0.2rem 0.5rem; border-radius: 4px; }
    .back-link { font-size: 0.9rem; color: #727cf5; text-decoration: none; }
    .back-link:hover { text-decoration: underline; }

    .summary-cards {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin-bottom: 1.5rem;
    }

    .stat-card {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.07);
        min-width: 130px;
    }

    .stat-label { font-size: 0.75rem; color: #6c757d; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.4rem; }
    .stat-value { font-size: 1.4rem; font-weight: 700; color: #313a46; }
    .stat-value.small { font-size: 0.9rem; font-weight: 500; word-break: break-all; }
    .stat-value.status-completed { color: #0acf97; }
    .stat-value.status-running  { color: #727cf5; }
    .stat-value.status-failed   { color: #fa5c7c; }

    .view-cards-btn {
        display: inline-block;
        background: #727cf5;
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 8px;
        text-decoration: none;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 2rem;
    }
    .view-cards-btn:hover { background: #5a65e8; }

    .log-box {
        background: #1e1e2d;
        color: #c9d1d9;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        font-family: monospace;
        font-size: 0.82rem;
        min-height: 120px;
        max-height: 480px;
        overflow-y: auto;
    }

    .log-line { padding: 1px 0; line-height: 1.5; white-space: pre-wrap; word-break: break-all; }
    .log-line.live { color: #e6edf3; }
    .log-ts { color: #727cf5; margin-right: 0.75rem; }
    .log-type { color: #ffbc00; margin-right: 0.75rem; font-weight: 600; }
    .log-data { color: #8b949e; }
    .log-divider { color: #444; margin: 0.5rem 0; text-align: center; font-size: 0.75rem; }

    .muted { color: #6c757d; font-style: italic; }
    .error { color: #fa5c7c; }
</style>
