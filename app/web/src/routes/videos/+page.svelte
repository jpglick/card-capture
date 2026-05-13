<script lang="ts">
    import { onMount } from 'svelte';
    import { api } from '$lib/api/client';
    import type { Video } from '$lib/api/types';

    let videos: Video[] = [];
    let loading = true;
    let error: string | null = null;

    async function loadVideos() {
        try {
            loading = true;
            videos = await api.videos.list();
        } catch (e: any) {
            error = e.message;
        } finally {
            loading = false;
        }
    }

    async function startProcessing(videoId: string) {
        try {
            const run = await api.videos.process(videoId);
            window.location.href = `/runs/${run.run_id}`;
        } catch (e: any) {
            alert(`Failed to start processing: ${e.message}`);
        }
    }

    onMount(loadVideos);
</script>

<h1>Videos</h1>

{#if loading}
    <p>Loading videos...</p>
{:else if error}
    <p class="error">{error}</p>
{:else}
    <table class="video-table">
        <thead>
            <tr>
                <th>ID</th>
                <th>Filename</th>
                <th>Status</th>
                <th>Created At</th>
                <th>Actions</th>
            </tr>
        </thead>
        <tbody>
            {#each videos as video}
                <tr>
                    <td>{video.video_id}</td>
                    <td>{video.filename}</td>
                    <td><span class="status-badge {video.status}">{video.status}</span></td>
                    <td>{new Date(video.created_at).toLocaleString()}</td>
                    <td>
                        <button onclick={() => startProcessing(video.video_id)}>Process</button>
                    </td>
                </tr>
            {/each}
        </tbody>
    </table>
{/if}

<style>
    .video-table {
        width: 100%;
        border-collapse: collapse;
        background: white;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    th, td {
        padding: 1rem;
        text-align: left;
        border-bottom: 1px solid #eee;
    }

    th {
        background: #f8f9fa;
        font-weight: 600;
    }

    .error {
        color: #fa5c7c;
    }

    .status-badge {
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.875rem;
        text-transform: capitalize;
    }

    .status-badge.completed { background: #0acf97; color: white; }
    .status-badge.processing { background: #727cf5; color: white; }
    .status-badge.failed { background: #fa5c7c; color: white; }
    .status-badge.pending { background: #ffbc00; color: white; }
</style>
