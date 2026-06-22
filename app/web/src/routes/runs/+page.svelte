<script lang="ts">
    import { onMount } from 'svelte';
    import { api } from '$lib/api/client';
    import type { RunSummary, CdpSubmission } from '$lib/api/types';

    let runs: RunSummary[] = [];
    let loading = true;
    let error: string | null = null;
    // Map of run_id -> {submitted, identified}
    let cdpCounts: Record<string, { submitted: number; identified: number }> = {};

    async function loadRuns() {
        try {
            loading = true;
            runs = await api.runs.list();
            // Load CDP counts for each run in parallel (best-effort)
            const pairs = await Promise.allSettled(
                runs.map(r => api.cdp.getRunSubmissions(r.run_id).then(m => ({ run_id: r.run_id, m })).catch(() => null))
            );
            const counts: Record<string, { submitted: number; identified: number }> = {};
            for (const p of pairs) {
                if (p.status === 'fulfilled' && p.value) {
                    const { run_id, m } = p.value;
                    const subs = Object.values(m);
                    if (subs.length > 0) {
                        counts[run_id] = {
                            submitted: subs.length,
                            identified: subs.filter((s: CdpSubmission) => s.status === 'identified').length,
                        };
                    }
                }
            }
            cdpCounts = counts;
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
                <th>CDP</th>
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
                    <td>
                        {#if cdpCounts[run.run_id]}
                            {@const c = cdpCounts[run.run_id]}
                            <span class="cdp-pill cdp-{c.identified === c.submitted ? 'done' : 'partial'}">
                                ✓ {c.identified}/{c.submitted}
                            </span>
                        {:else if run.cards_extracted > 0}
                            <span class="cdp-pill cdp-none">—</span>
                        {:else}
                            <span class="cdp-pill cdp-na">n/a</span>
                        {/if}
                    </td>
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

    .cdp-pill {
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.2rem 0.5rem;
        border-radius: 999px;
        white-space: nowrap;
    }
    .cdp-done    { background: #e8f5e9; color: #2e7d32; }
    .cdp-partial { background: #fff3e0; color: #e65100; }
    .cdp-none    { background: #f0f0f0; color: #9e9e9e; }
    .cdp-na      { color: #bbb; font-weight: 400; }
</style>
