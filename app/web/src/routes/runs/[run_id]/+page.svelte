<script lang="ts">
    import { onMount, onDestroy } from 'svelte';
    import { page } from '$app/state';
    import { api } from '$lib/api/client';
    import { createSSEStore } from '$lib/sse';
    import type { RunDetail, RunEvent } from '$lib/api/types';

    const runId = page.params.run_id;
    let run: RunDetail | null = null;
    let loading = true;
    let error: string | null = null;

    const eventsStore = createSSEStore(runId);

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

    onMount(() => {
        loadRun();
        eventsStore.connect();
    });

    onDestroy(() => {
        eventsStore.disconnect();
    });
</script>

<h1>Run: {runId}</h1>

{#if loading}
    <p>Loading run details...</p>
{:else if error}
    <p class="error">{error}</p>
{:else if run}
    <div class="run-info">
        <p><strong>Status:</strong> <span class="status-badge {run.status}">{run.status}</span></p>
        <p><strong>Created:</strong> {new Date(run.created_at).toLocaleString()}</p>
    </div>

    <h2>Live Events</h2>
    <div class="events-log">
        {#each $eventsStore as event}
            <div class="event-item">
                <span class="timestamp">{new Date(event.timestamp).toLocaleTimeString()}</span>
                <span class="event-type">{event.type}</span>
                {#if event.stage_id}
                    <span class="stage">[{event.stage_id}]</span>
                {/if}
            </div>
        {/each}
        {#each run.events as event}
            <div class="event-item legacy">
                <span class="timestamp">{new Date(event.timestamp).toLocaleTimeString()}</span>
                <span class="event-type">{event.event_name}</span>
                {#if event.stage_id}
                    <span class="stage">[{event.stage_id}]</span>
                {/if}
            </div>
        {/each}
    </div>
{/if}

<style>
    .run-info {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .events-log {
        background: #1e1e2d;
        color: #aab8c2;
        padding: 1rem;
        border-radius: 8px;
        font-family: monospace;
        height: 400px;
        overflow-y: auto;
    }

    .event-item {
        margin-bottom: 0.25rem;
        border-bottom: 1px solid #2b2b3d;
        padding-bottom: 0.25rem;
    }

    .event-item.legacy {
        opacity: 0.7;
    }

    .timestamp { color: #727cf5; margin-right: 1rem; }
    .event-type { font-weight: bold; color: white; margin-right: 1rem; }
    .stage { color: #ffbc00; }

    .status-badge {
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.875rem;
    }
    .status-badge.completed { background: #0acf97; color: white; }
    .status-badge.running { background: #727cf5; color: white; }
</style>
