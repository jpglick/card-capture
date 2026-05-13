<script lang="ts">
    import { onMount } from 'svelte';
    import { api } from '$lib/api/client';
    import type { Video, RunSummary } from '$lib/api/types';

    let videos: Video[] = [];
    let runs: RunSummary[] = [];
    let loading = true;

    async function load() {
        try {
            loading = true;
            [videos, runs] = await Promise.all([
                api.videos.list(),
                api.runs.list()
            ]);
        } catch (e) {
            console.error(e);
        } finally {
            loading = false;
        }
    }

    onMount(load);

    $: activeRuns = runs.filter(r => r.status === 'running');
    $: pendingVideos = videos.filter(v => v.status === 'pending');
</script>

<h1>Dashboard</h1>

<div class="summary-grid">
    <div class="card">
        <h3>Active Runs</h3>
        <p class="count">{activeRuns.length}</p>
    </div>
    <div class="card">
        <h3>Pending Videos</h3>
        <p class="count">{pendingVideos.length}</p>
    </div>
    <div class="card">
        <h3>Total Videos</h3>
        <p class="count">{videos.length}</p>
    </div>
</div>

<div class="recent-sections">
    <div class="section">
        <h2>Recent Runs</h2>
        {#each runs.slice(0, 5) as run}
            <div class="item">
                <a href="/runs/{run.run_id}">{run.run_id}</a>
                <span class="status {run.status}">{run.status}</span>
            </div>
        {/each}
    </div>
    <div class="section">
        <h2>Recently Added Videos</h2>
        {#each videos.slice(0, 5) as video}
            <div class="item">
                <span>{video.filename}</span>
                <span class="status {video.status}">{video.status}</span>
            </div>
        {/each}
    </div>
</div>

<style>
    .summary-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin-bottom: 3rem;
    }

    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
    }

    .count {
        font-size: 2.5rem;
        font-weight: bold;
        color: #727cf5;
        margin: 0;
    }

    .recent-sections {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 2rem;
    }

    .section {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

    .item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 0;
        border-bottom: 1px solid #eee;
    }

    .status {
        font-size: 0.75rem;
        padding: 0.2rem 0.5rem;
        border-radius: 10px;
        text-transform: uppercase;
    }

    .status.completed { background: #e1fcef; color: #0acf97; }
    .status.running { background: #eef2ff; color: #727cf5; }
    .status.failed { background: #fff5f5; color: #fa5c7c; }
</style>
