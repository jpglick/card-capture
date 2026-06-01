<script lang="ts">
    import { onMount, onDestroy, tick } from 'svelte';
    import { page } from '$app/state';
    import { api } from '$lib/api/client';
    import type { RunDetail, RunResources, ResourceSample, StageMarker } from '$lib/api/types';
    import ResourceChart from '$lib/components/ResourceChart.svelte';
    import PipelineSparkline from '$lib/components/PipelineSparkline.svelte';

    const runId = page.params.run_id!;
    let run = $state<RunDetail | null>(null);
    let resources = $state<RunResources | null>(null);
    let loading = $state(true);
    let error = $state<string | null>(null);
    let liveLines = $state<string[]>([]);
    let progressMap = $state<Record<string, { stage_id: string; pct: number; detail: string }>>({});
    let evtSource: EventSource | null = null;
    let pollTimer: ReturnType<typeof setInterval> | null = null;
    let logBoxElement: HTMLElement | undefined = $state();

    $effect(() => {
        if (logBoxElement && (liveLines.length > 0 || run?.logs?.length)) {
            logBoxElement.scrollTop = logBoxElement.scrollHeight;
        }
    });

    // Stage colors — shared between the legend and chart markers
    const STAGE_COLORS: Record<string, string> = {
        detect:  '#6366f1',
        novelty: '#10b981',
        track:   '#f59e0b',
        refine:  '#ef4444',
        score:   '#3b82f6',
        resolve: '#8b5cf6',
        dedup:   '#f97316',
        store:   '#06b6d4',
    };

    function stageColor(name: string): string {
        return STAGE_COLORS[name] ?? '#6c757d';
    }

    async function loadRun() {
        try {
            loading = true;
            run = await api.runs.detail(runId);
            
            // Reconstruct progressMap from historical events if the run is finished
            if (run && run.status !== 'running' && run.events) {
                const newProgress: Record<string, { stage_id: string; pct: number; detail: string }> = {};
                for (const ev of run.events) {
                    if (ev.event_type === 'stage_progress' && ev.data_json) {
                        try {
                            const d = JSON.parse(ev.data_json);
                            if (d.stage_id) {
                                newProgress[d.stage_id] = d;
                            }
                        } catch {}
                    } else if (ev.event_type === 'stage_finished' && ev.data_json) {
                        try {
                            const d = JSON.parse(ev.data_json);
                            if (d.stage) {
                                newProgress[d.stage] = { stage_id: d.stage, pct: 100, detail: `elapsed_ms=${d.elapsed_ms}` };
                            }
                        } catch {}
                    }
                }
                
                // Fallback: If we don't have stage_progress events (older runs), use stage_timings
                if (Object.keys(newProgress).length === 0 && run.stage_timings && run.stage_timings.length > 0) {
                     for (const t of run.stage_timings) {
                         newProgress[t.stage] = { stage_id: t.stage, pct: 100, detail: `elapsed_ms=${t.elapsed_ms}` };
                     }
                }

                // Fallback 2: If we don't have stage_timings either, parse the logs
                if (Object.keys(newProgress).length === 0 && run.logs) {
                    for (const line of run.logs) {
                        const m = line.match(/\[stage\] (\w+) finished in (\d+)ms/);
                        if (m) {
                            newProgress[m[1]] = { stage_id: m[1], pct: 100, detail: `elapsed_ms=${m[2]}` };
                        }
                    }
                }
                
                progressMap = newProgress;
            }
            
        } catch (e: any) {
            error = e.message;
        } finally {
            loading = false;
        }
    }

    async function loadResources() {
        try {
            resources = await api.runs.resources(runId);
        } catch { /* no-op — endpoint may not have data yet */ }
    }

    function connectSSE() {
        evtSource = new EventSource(`/events/${runId}`);
        evtSource.addEventListener('log', (e: any) => {
            try {
                const d = JSON.parse(e.data);
                if (d.payload?.line) liveLines = [...liveLines, d.payload.line];
            } catch {}
        });
        evtSource.addEventListener('stage_progress', (e: any) => {
            try {
                const d = JSON.parse(e.data);
                if (d.payload?.stage_id) {
                    progressMap[d.payload.stage_id] = d.payload;
                }
            } catch {}
        });
        evtSource.addEventListener('run_completed', () => {
            evtSource?.close();
            if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
            loadRun();
            loadResources();
        });
        evtSource.addEventListener('run_failed', (e: any) => {
            try {
                const d = JSON.parse(e.data);
                if (d.payload?.error) liveLines = [...liveLines, `ERROR: ${d.payload.error}`];
            } catch {}
            evtSource?.close();
            if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
            loadRun();
            loadResources();
        });
        evtSource.onerror = () => evtSource?.close();
    }

    onMount(() => {
        loadRun().then(() => {
            loadResources();
            if (run?.status === 'running') {
                connectSSE();
                pollTimer = setInterval(loadResources, 3000);
            }
        });
    });

    onDestroy(() => {
        evtSource?.close();
        if (pollTimer) clearInterval(pollTimer);
    });

    function fmt(ms: number) {
        if (ms < 1000) return `${ms}ms`;
        if (ms < 60000) return `${(ms/1000).toFixed(1)}s`;
        return `${Math.floor(ms/60000)}m ${Math.floor((ms%60000)/1000)}s`;
    }

    // Derived chart data from resources
    const stagesWithColor = $derived.by(() => {
        const markers: StageMarker[] = resources?.stage_markers ?? [];
        return markers.map(m => ({ ...m, color: stageColor(m.name) }));
    });

    const cpuSamples = $derived.by(() => {
        const s: ResourceSample[] = resources?.samples ?? [];
        return s.map(r => ({ elapsed_s: r.elapsed_s, value: r.cpu_pct }));
    });
    const memSamples = $derived.by(() => {
        const s: ResourceSample[] = resources?.samples ?? [];
        return s.map(r => ({ elapsed_s: r.elapsed_s, value: r.mem_pct }));
    });
    const gpuSamples = $derived.by(() => {
        const s: ResourceSample[] = resources?.samples ?? [];
        return s.map(r => ({ elapsed_s: r.elapsed_s, value: r.gpu_pct }));
    });
    const vramSamples = $derived.by(() => {
        const hi = resources?.host_info;
        const totalMb = hi?.vram_total_gb ? hi.vram_total_gb * 1000 : null;
        const s: ResourceSample[] = resources?.samples ?? [];
        return s.map(r => ({
            elapsed_s: r.elapsed_s,
            value: r.vram_used_mb != null && totalMb ? (r.vram_used_mb / totalMb) * 100 : null,
        }));
    });
    
    const neuralSamples = $derived.by(() => {
        const s: ResourceSample[] = resources?.samples ?? [];
        return s.map(r => ({ elapsed_s: r.elapsed_s, value: r.neural_pct ?? null }));
    });

    const hasGpu = $derived.by(() => (resources?.samples ?? []).some((s: ResourceSample) => s.gpu_pct != null));
    const hasVram = $derived.by(() => (resources?.samples ?? []).some((s: ResourceSample) => s.vram_used_mb != null));
    const hasNeural = $derived.by(() => (resources?.samples ?? []).some((s: ResourceSample) => s.neural_pct != null));
    const isUnified = $derived(resources?.host_info?.vram_is_unified ?? false);
    const knownStageNames = $derived([...new Set(stagesWithColor.map(s => s.name))]);
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

    {#if (run.stage_timings?.length ?? 0) > 0}
        <h2>Stage timings</h2>
        <table class="timings-table">
            <thead>
                <tr><th>Stage</th><th>Elapsed</th><th>Diagnostics</th></tr>
            </thead>
            <tbody>
                {#each run.stage_timings as t}
                    <tr>
                        <td class="stage-name">{t.stage}</td>
                        <td class="stage-time">{fmt(t.elapsed_ms)}</td>
                        <td class="stage-diag">
                            {#if resources?.stage_metrics?.[t.stage]}
                                {#each Object.entries(resources.stage_metrics[t.stage]) as [k, v]}
                                    <span class="diag-chip">{k}={v}</span>
                                {/each}
                            {:else}—{/if}
                        </td>
                    </tr>
                {/each}
            </tbody>
        </table>
    {/if}

    {#if run.detect_telemetry}
        {@const dt = run.detect_telemetry}
        <h2>Detect diagnostics</h2>
        <div class="diag-grid">
            <div class="diag-section">
                <div class="diag-title">Frame pipeline</div>
                <div class="diag-row">
                    <span class="diag-label">Sampled frames</span>
                    <span class="diag-value">{dt.frame_count?.toLocaleString() ?? '—'}</span>
                </div>
                <div class="diag-row">
                    <span class="diag-label">Passed triage → YOLO</span>
                    <span class="diag-value">
                        {dt.accepted_frame_count?.toLocaleString() ?? '—'}
                        {#if dt.triage_pass_rate != null}
                            <span class="diag-pct {dt.triage_pass_rate < 0.02 ? 'pct-warn' : ''}">
                                ({(dt.triage_pass_rate * 100).toFixed(1)}%)
                            </span>
                        {/if}
                    </span>
                </div>
                <div class="diag-row">
                    <span class="diag-label">Presence windows</span>
                    <span class="diag-value">{dt.presence_windows ?? '—'}</span>
                </div>
                <div class="diag-row">
                    <span class="diag-label">Sampler</span>
                    <span class="diag-value small">{dt.sampler_type ?? '—'}</span>
                </div>
            </div>

            <div class="diag-section">
                <div class="diag-title">YOLO inference</div>
                <div class="diag-row">
                    <span class="diag-label">Device</span>
                    <span class="diag-value device-badge device-{dt.yolo_device ?? 'unknown'}">
                        {dt.yolo_device ?? '—'}
                    </span>
                </div>
                <div class="diag-row">
                    <span class="diag-label">Frames processed</span>
                    <span class="diag-value">{dt.yolo_frames?.toLocaleString() ?? '—'}</span>
                </div>
                <div class="diag-row">
                    <span class="diag-label">Batches (×{dt.yolo_frames && dt.yolo_batches ? Math.round(dt.yolo_frames / dt.yolo_batches) : '?'} frames)</span>
                    <span class="diag-value">{dt.yolo_batches ?? '—'}</span>
                </div>
                <div class="diag-row">
                    <span class="diag-label">Total YOLO time</span>
                    <span class="diag-value">{dt.yolo_elapsed_s != null ? fmt(dt.yolo_elapsed_s * 1000) : '—'}</span>
                </div>
                {#if dt.yolo_elapsed_s && dt.yolo_frames}
                    <div class="diag-row">
                        <span class="diag-label">ms / frame</span>
                        <span class="diag-value">{((dt.yolo_elapsed_s * 1000) / dt.yolo_frames).toFixed(1)} ms</span>
                    </div>
                {/if}
            </div>
        </div>
    {/if}

    {#if resources?.host_info || (resources?.samples?.length ?? 0) > 0}
        <h2>Resource usage</h2>

        {#if resources?.host_info}
            {@const hi = resources.host_info}
            <div class="host-bar">
                {#if hi.hostname}<span class="host-chip"><span class="host-key">host</span>{hi.hostname}</span>{/if}
                {#if hi.platform}<span class="host-chip"><span class="host-key">platform</span>{hi.platform}</span>{/if}
                {#if hi.cpu_count_physical}<span class="host-chip"><span class="host-key">cpu</span>{hi.cpu_count_physical}c / {hi.cpu_count_logical}t</span>{/if}
                {#if hi.mem_total_gb}<span class="host-chip"><span class="host-key">ram</span>{hi.mem_total_gb} GB{hi.vram_is_unified ? ' (unified)' : ''}</span>{/if}
                {#if hi.gpu_name}<span class="host-chip"><span class="host-key">gpu</span>{hi.gpu_name}</span>{/if}
                {#if hi.vram_total_gb && !hi.vram_is_unified}<span class="host-chip"><span class="host-key">vram</span>{hi.vram_total_gb} GB</span>{/if}
            </div>
        {/if}

        {#if knownStageNames.length > 0}
            <div class="stage-legend">
                {#each knownStageNames as name}
                    <span class="legend-item">
                        <span class="legend-swatch" style="background:{stageColor(name)}"></span>
                        {name}
                    </span>
                {/each}
            </div>
        {/if}

        <div class="charts-grid">
            <ResourceChart
                samples={cpuSamples}
                stages={stagesWithColor}
                label="CPU %"
                color="#727cf5"
                noData={cpuSamples.length === 0}
            />
            <ResourceChart
                samples={memSamples}
                stages={stagesWithColor}
                label="Memory %"
                color="#0acf97"
                noData={memSamples.length === 0}
            />
            {#if hasGpu}
                <ResourceChart
                    samples={gpuSamples}
                    stages={stagesWithColor}
                    label="GPU %"
                    color="#fa5c7c"
                    noData={gpuSamples.length === 0}
                />
            {/if}
            {#if hasVram}
                <ResourceChart
                    samples={vramSamples}
                    stages={stagesWithColor}
                    label={isUnified ? 'VRAM % (unified)' : 'VRAM %'}
                    color="#ffbc00"
                    noData={vramSamples.length === 0}
                />
            {/if}
            {#if hasNeural}
                <ResourceChart
                    samples={neuralSamples}
                    stages={stagesWithColor}
                    label="Neural Engine %"
                    color="#8b5cf6"
                    noData={neuralSamples.length === 0}
                />
            {/if}
        </div>
    {/if}

    {#if Object.keys(progressMap).length > 0}
        <PipelineSparkline {progressMap} />
    {/if}

    <h2>{run.status === 'running' ? 'Live log' : 'Run log'}</h2>
    <div class="log-box" bind:this={logBoxElement}>
        {#if run.status === 'running' && liveLines.length === 0}
            <span class="muted">Waiting for pipeline output…</span>
        {:else if run.status !== 'running' && liveLines.length === 0 && (!run.logs || run.logs.length === 0) && (!run.events || run.events.length === 0)}
            <span class="muted">No log lines recorded for this run.</span>
        {/if}

        {#if (run.logs?.length ?? 0) > 0 && liveLines.length === 0}
            {#each run.logs as line}
                <div class="log-line">{line}</div>
            {/each}
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

    .cloud-badge {
        display: inline-block;
        background: #e8f4fd;
        color: #0078d4;
        border: 1px solid #b3d9f5;
        border-radius: 4px;
        padding: 0.1rem 0.5rem;
        font-size: 0.75rem;
        font-weight: 600;
        vertical-align: middle;
        margin-left: 0.5rem;
    }

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
        height: 480px;
        overflow-y: auto;
        scroll-behavior: smooth;
    }

    .log-line { padding: 1px 0; line-height: 1.5; white-space: pre-wrap; word-break: break-all; }
    .log-line.live { color: #e6edf3; }
    .log-ts { color: #727cf5; margin-right: 0.75rem; }
    .log-type { color: #ffbc00; margin-right: 0.75rem; font-weight: 600; }
    .log-data { color: #8b949e; }
    .log-divider { color: #444; margin: 0.5rem 0; text-align: center; font-size: 0.75rem; }

    .muted { color: #6c757d; font-style: italic; }
    .error { color: #fa5c7c; }

    .timings-table {
        width: 100%;
        max-width: 480px;
        border-collapse: collapse;
        margin-bottom: 1.5rem;
        background: white;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 6px rgba(0,0,0,0.07);
        font-size: 0.9rem;
    }
    .timings-table th {
        background: #f0f1f9;
        color: #6c757d;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
        padding: 0.5rem 1rem;
        text-align: left;
    }
    .timings-table td { padding: 0.45rem 1rem; border-top: 1px solid #f0f1f9; }
    .stage-name { font-weight: 600; color: #313a46; text-transform: capitalize; }
    .stage-time { color: #727cf5; font-family: monospace; }
    .stage-diag { display: flex; flex-wrap: wrap; gap: 0.3rem; min-width: 220px; }
    .diag-chip {
        display: inline-block;
        background: #f0f1f9;
        border: 1px solid #dfe3f2;
        border-radius: 999px;
        padding: 0.1rem 0.5rem;
        font-size: 0.72rem;
        color: #495057;
        font-family: monospace;
    }

    .diag-grid {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin-bottom: 1.5rem;
    }
    .diag-section {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.07);
        min-width: 260px;
        flex: 1;
    }
    .diag-title {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #6c757d;
        font-weight: 700;
        margin-bottom: 0.75rem;
    }
    .diag-row {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 1rem;
        padding: 0.3rem 0;
        border-bottom: 1px solid #f8f9fa;
        font-size: 0.875rem;
    }
    .diag-row:last-child { border-bottom: none; }
    .diag-label { color: #6c757d; }
    .diag-value { font-weight: 600; color: #313a46; font-family: monospace; }
    .diag-value.small { font-size: 0.78rem; font-family: sans-serif; font-weight: 500; }
    .diag-pct { color: #6c757d; font-size: 0.8rem; font-weight: 400; margin-left: 0.3rem; }
    .diag-pct.pct-warn { color: #fa5c7c; }

    .device-badge { padding: 0.1rem 0.5rem; border-radius: 4px; font-size: 0.8rem; }
    .device-mps   { background: #e8f5e9; color: #2e7d32; }
    .device-cpu   { background: #fff3e0; color: #e65100; }
    .device-unknown { background: #f5f5f5; color: #757575; }

    /* Resource usage */
    .host-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 0.75rem;
    }
    .host-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: white;
        border-radius: 6px;
        padding: 0.25rem 0.6rem;
        font-size: 0.82rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
        color: #313a46;
        font-weight: 500;
    }
    .host-key {
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6c757d;
        font-weight: 700;
    }

    .stage-legend {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem 1rem;
        margin-bottom: 0.75rem;
    }
    .legend-item {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.8rem;
        color: #495057;
    }
    .legend-swatch {
        width: 10px;
        height: 10px;
        border-radius: 2px;
        flex-shrink: 0;
    }

    .charts-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.75rem;
        margin-bottom: 1.5rem;
    }
    @media (max-width: 700px) {
        .charts-grid { grid-template-columns: 1fr; }
    }

    .progress-container {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 0.75rem;
        border-left: 4px solid #727cf5;
    }
    .multi-progress {
        margin-bottom: 2rem;
    }

    .progress-info {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 0.75rem;
        font-size: 0.95rem;
    }
    .progress-stage { color: #313a46; font-weight: 500; }
    .progress-stage strong { text-transform: capitalize; color: #727cf5; }
    .progress-detail { color: #495057; font-size: 0.85rem; font-weight: 500; }
    .progress-pct { font-weight: 700; color: #313a46; font-family: monospace; }
    
    .progress-bar-bg {
        height: 10px;
        background: #f1f3fa;
        border-radius: 5px;
        overflow: hidden;
    }
    .progress-bar-fill {
        height: 100%;
        transition: width 0.3s ease;
    }
</style>
