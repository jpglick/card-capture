<script lang="ts">
    import { onMount } from 'svelte';
    import { api } from '$lib/api/client';
    import type { RunSummary } from '$lib/api/types';

    let runs: RunSummary[] = [];
    let loading = true;
    let error: string | null = null;

    async function loadRuns() {
        try {
            loading = true;
            runs = await api.runs.list();
        } catch (e: any) {
            error = e.message;
        } finally {
            loading = false;
        }
    }

    onMount(loadRuns);
</script>

<h1>Pipeline Runs</h1>

{#if loading}
    <p>Loading runs...</p>
{:else if error}
    <p class="error">{error}</p>
{:else}
    <table class="run-table">
        <thead>
            <tr>
                <th>Run ID</th>
                <th>Status</th>
                <th>Cards</th>
                <th>Elapsed</th>
                <th>Created At</th>
            </tr>
        </thead>
        <tbody>
            {#each runs as run}
                <tr>
                    <td><a href="/runs/{run.run_id}">{run.run_id}</a></td>
                    <td><span class="status-badge {run.status}">{run.status}</span></td>
                    <td>{run.cards_extracted}</td>
                    <td>{(run.elapsed_ms / 1000).toFixed(1)}s</td>
                    <td>{new Date(run.created_at).toLocaleString()}</td>
                </tr>
            {/each}
        </tbody>
    </table>
{/if}

<style>
    .run-table {
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
    .status-badge.running { background: #727cf5; color: white; }
    .status-badge.failed { background: #fa5c7c; color: white; }
    .status-badge.pending { background: #ffbc00; color: white; }
</style>
