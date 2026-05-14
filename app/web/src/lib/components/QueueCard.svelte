<!-- app/web/src/lib/components/QueueCard.svelte -->
<script lang="ts">
    import { onMount, onDestroy } from 'svelte';
    import { api } from '$lib/api/client';

    export let video: { video_id: string; filename: string; status: string };

    let runId: string | null = null;
    let progress: string = '';
    let stage: string = '';
    let evtSource: EventSource | null = null;

    async function startProcessing() {
        try {
            const run = await api.videos.process(video.video_id);
            runId = run.run_id;
            progress = 'Starting…';
            stage = 'pending';
            subscribeToEvents(run.run_id);
        } catch (e: any) {
            progress = `Error: ${e.message}`;
            stage = 'failed';
        }
    }

    function subscribeToEvents(id: string) {
        evtSource = new EventSource(`/api/v1/events/subscribe?run_id=${id}`);
        evtSource.onmessage = (e) => {
            try {
                const data = JSON.parse(e.data);
                stage = data.stage ?? stage;
                progress = data.message ?? progress;
                if (data.stage === 'completed' || data.stage === 'failed') {
                    evtSource?.close();
                }
            } catch {}
        };
        evtSource.onerror = () => {
            evtSource?.close();
        };
    }

    onDestroy(() => {
        evtSource?.close();
    });

    const statusColor: Record<string, string> = {
        pending: '#6c757d',
        processing: '#727cf5',
        completed: '#0acf97',
        failed: '#fa5c7c',
    };
</script>

<div class="queue-card" style="--status-color: {statusColor[video.status] ?? statusColor['pending']}">
    <div class="card-header">
        <span class="filename">{video.filename}</span>
        <span class="status-badge">{stage || video.status}</span>
    </div>
    {#if progress}
        <div class="progress-text">{progress}</div>
    {/if}
    {#if !runId && video.status !== 'processing'}
        <button class="process-btn" on:click={startProcessing}>▶ Process</button>
    {/if}
    {#if runId && (stage === 'completed' || video.status === 'completed')}
        <a href="/runs/{runId}" class="view-link">View results →</a>
    {/if}
</div>

<style>
    .queue-card {
        background: white;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        border-left: 4px solid var(--status-color);
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .filename {
        font-weight: 600;
        font-size: 0.95rem;
        color: #313a46;
        word-break: break-all;
    }

    .status-badge {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        color: var(--status-color);
        padding: 2px 8px;
        border-radius: 20px;
        border: 1px solid var(--status-color);
        white-space: nowrap;
    }

    .progress-text {
        font-size: 0.82rem;
        color: #6c757d;
        font-style: italic;
    }

    .process-btn {
        align-self: flex-start;
        background: #727cf5;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.4rem 1rem;
        font-size: 0.85rem;
        cursor: pointer;
        font-weight: 600;
    }

    .process-btn:hover { background: #5a65e8; }

    .view-link {
        font-size: 0.85rem;
        color: #727cf5;
        text-decoration: none;
        font-weight: 600;
    }
    .view-link:hover { text-decoration: underline; }
</style>
