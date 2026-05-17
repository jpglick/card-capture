<script lang="ts">
    import { onMount } from 'svelte';
    import { api } from '$lib/api/client';
    import type { Video, BatchStatus } from '$lib/api/types';

    let videos = $state<Video[]>([]);
    let selected = $state<Set<string>>(new Set());
    let loading = $state(true);
    let submitting = $state(false);
    let batchStatus = $state<BatchStatus | null>(null);
    let pollTimer: ReturnType<typeof setInterval> | null = null;

    onMount(async () => {
        try {
            videos = await api.videos.list();
        } catch (e) {
            console.error(e);
        } finally {
            loading = false;
        }
    });

    function toggle(id: string) {
        const s = new Set(selected);
        s.has(id) ? s.delete(id) : s.add(id);
        selected = s;
    }

    function toggleAll() {
        selected = selected.size === videos.length
            ? new Set<string>()
            : new Set(videos.map(v => v.video_id));
    }

    async function submitBatch() {
        if (selected.size === 0) return;
        submitting = true;
        try {
            const { batch_id } = await api.batch.create([...selected]);
            batchStatus = await api.batch.status(batch_id);
            pollTimer = setInterval(async () => {
                batchStatus = await api.batch.status(batch_id);
                if (batchStatus && ['complete', 'failed', 'partial'].includes(batchStatus.status)) {
                    if (pollTimer) clearInterval(pollTimer);
                    pollTimer = null;
                }
            }, 3000);
        } catch (e) {
            console.error(e);
        } finally {
            submitting = false;
        }
    }

    function statusColor(s: string) {
        return s === 'complete' ? '#28a745' : s === 'failed' ? '#dc3545' : s === 'running' ? '#007bff' : '#6c757d';
    }
</script>

<h1>Batch Process</h1>

{#if batchStatus}
    <section class="batch-status">
        <h2>Batch {batchStatus.batch_id} — <span style="color:{statusColor(batchStatus.status)}">{batchStatus.status}</span></h2>
        <p>{batchStatus.completed}/{batchStatus.total} complete · {batchStatus.failed} failed</p>
        <ul class="job-list">
            {#each batchStatus.jobs as job}
                <li>
                    <span class="job-id">Video {job.video_id}</span>
                    <span class="job-status" style="color:{statusColor(job.status)}">{job.status}</span>
                    {#if job.run_id}<a href="/runs/{job.run_id}">view run</a>{/if}
                    {#if job.error}<span class="err">{job.error}</span>{/if}
                </li>
            {/each}
        </ul>
        <button onclick={() => batchStatus = null}>Start new batch</button>
    </section>
{:else}
    <p class="section-desc">Select videos to process on the cloud GPU, then click Process Batch.</p>

    <div class="toolbar">
        <button onclick={toggleAll} class="btn-secondary">
            {selected.size === videos.length ? 'Deselect all' : 'Select all'}
        </button>
        <button onclick={submitBatch} class="btn-primary"
                disabled={selected.size === 0 || submitting}>
            {submitting ? 'Submitting…' : `Process Batch (${selected.size})`}
        </button>
    </div>

    {#if loading}
        <p>Loading videos…</p>
    {:else if videos.length === 0}
        <p class="empty">No videos uploaded yet.</p>
    {:else}
        <table class="video-table">
            <thead><tr><th></th><th>Filename</th><th>Status</th></tr></thead>
            <tbody>
                {#each videos as video}
                    <tr class:selected={selected.has(video.video_id)}
                        onclick={() => toggle(video.video_id)}>
                        <td><input type="checkbox" checked={selected.has(video.video_id)}
                                   onchange={() => toggle(video.video_id)} /></td>
                        <td>{video.filename ?? video.source_path?.split('/').pop()}</td>
                        <td>{video.status ?? '—'}</td>
                    </tr>
                {/each}
            </tbody>
        </table>
    {/if}
{/if}

<style>
    .toolbar { display: flex; gap: 1rem; margin-bottom: 1rem; align-items: center; }
    .video-table { width: 100%; border-collapse: collapse; }
    .video-table th, .video-table td { padding: 0.5rem 0.75rem; text-align: left;
        border-bottom: 1px solid #dee2e6; }
    .video-table tbody tr { cursor: pointer; }
    .video-table tbody tr:hover, .video-table tbody tr.selected { background: #f0f1ff; }
    .job-list { list-style: none; padding: 0; }
    .job-list li { display: flex; gap: 1rem; padding: 0.4rem 0; border-bottom: 1px solid #dee2e6; }
    .job-id { font-family: monospace; }
    .err { color: #dc3545; font-size: 0.8rem; }
    .empty { color: #6c757d; }
    .section-desc { color: #6c757d; margin-bottom: 1rem; }
    
    .btn-primary {
        background: #727cf5;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        font-size: 0.9rem;
        font-weight: 600;
        cursor: pointer;
    }
    .btn-primary:hover { background: #5a65e8; }
    .btn-primary:disabled { opacity: 0.6; cursor: not-allowed; }

    .btn-secondary {
        background: none;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 0.4rem 0.9rem;
        font-size: 0.82rem;
        cursor: pointer;
        color: #495057;
    }
    .btn-secondary:hover { background: #f8f9fa; }
</style>
