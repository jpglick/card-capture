<script lang="ts">
    import { onMount } from 'svelte';
    import { api } from '$lib/api/client';
    import type { Baseline } from '$lib/api/types';

    let baselines: Baseline[] = [];
    let loading = true;
    let error: string | null = null;

    async function load() {
        try {
            loading = true;
            baselines = await api.regression.listBaselines();
        } catch (e: any) {
            error = e.message;
        } finally {
            loading = false;
        }
    }

    onMount(load);

    function handleCompare() {
        // Simple logic for now: compare top two if possible
        if (baselines.length >= 2) {
            window.location.href = `/regression/compare?a=${baselines[1].name}&b=${baselines[0].name}`;
        }
    }
</script>

<h1>Regression Testing</h1>
<p>Monitor pipeline performance stability against saved baselines.</p>

{#if loading}
    <p>Loading baselines...</p>
{:else if error}
    <p class="error">{error}</p>
{:else}
    <div class="actions">
        <button onclick={handleCompare} disabled={baselines.length < 2}>Compare Top 2</button>
    </div>

    <table class="baseline-table">
        <thead>
            <tr>
                <th>Name</th>
                <th>Commit SHA</th>
                <th>Created At</th>
                <th>Actions</th>
            </tr>
        </thead>
        <tbody>
            {#each baselines as baseline}
                <tr>
                    <td><strong>{baseline.name}</strong></td>
                    <td><code>{baseline.code_sha.slice(0, 7)}</code></td>
                    <td>{new Date(baseline.created_at).toLocaleString()}</td>
                    <td>
                        <button class="secondary">View Runs</button>
                    </td>
                </tr>
            {/each}
        </tbody>
    </table>
{/if}

<style>
    .baseline-table {
        width: 100%;
        border-collapse: collapse;
        background: white;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

    th, td {
        padding: 1rem;
        text-align: left;
        border-bottom: 1px solid #eee;
    }

    th { background: #f8f9fa; font-weight: 600; }

    .actions { margin-bottom: 1rem; }

    .error { color: #fa5c7c; }

    button {
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        background: #727cf5;
        color: white;
    }

    button:disabled { background: #ccc; cursor: not-allowed; }

    button.secondary { background: #eee; color: #333; }
</style>
