<script lang="ts">
    import { onMount } from 'svelte';
    import { api } from '$lib/api/client';
    import type { Video } from '$lib/api/types';

    let videos: Video[] = [];
    let loading = true;

    async function load() {
        try {
            loading = true;
            videos = await api.videos.list();
        } finally {
            loading = false;
        }
    }

    onMount(load);
</script>

<h1>Truth Editor</h1>
<p>Select a video to verify detections and create ground truth.</p>

{#if loading}
    <p>Loading videos...</p>
{:else}
    <div class="video-list">
        {#each videos as video}
            <a href="/label/truth/{video.video_id}" class="video-card">
                <h3>{video.filename}</h3>
                <span class="status {video.status}">{video.status}</span>
            </a>
        {/each}
    </div>
{/if}

<style>
    .video-list {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
        gap: 1rem;
    }

    .video-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        text-decoration: none;
        color: inherit;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }

    .video-card:hover { transform: translateY(-4px); }

    .status {
        font-size: 0.75rem;
        text-transform: uppercase;
        padding: 2px 6px;
        border-radius: 4px;
        background: #eee;
    }
</style>
