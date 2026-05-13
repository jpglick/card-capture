<script lang="ts">
    import { onMount } from 'svelte';
    import { page } from '$app/state';
    import { api } from '$lib/api/client';
    import type { RegressionCompare } from '$lib/api/types';

    const a = page.url.searchParams.get('a') || '';
    const b = page.url.searchParams.get('b') || '';

    let comparison: RegressionCompare | null = null;
    let loading = true;

    async function load() {
        try {
            loading = true;
            comparison = await api.regression.compare(a, b);
        } finally {
            loading = false;
        }
    }

    onMount(load);

    function formatDelta(val: number | null) {
        if (val === null) return '-';
        const sign = val > 0 ? '+' : '';
        const color = val > 0 ? '#0acf97' : val < 0 ? '#fa5c7c' : '#666';
        return `<span style="color: ${color}">${sign}${val.toFixed(4)}</span>`;
    }
</script>

<h1>A/B Comparison</h1>
<p>Comparing <strong>{a}</strong> vs <strong>{b}</strong></p>

{#if loading}
    <p>Loading comparison data...</p>
{:else if comparison}
    <div class="metrics-grid">
        {#each Object.entries(comparison.metric_deltas) as [key, delta]}
            <div class="metric-card">
                <h3>{key.replace(/_/g, ' ')}</h3>
                <div class="delta">{@html formatDelta(delta)}</div>
            </div>
        {/each}
    </div>

    <h2>Detailed Changes</h2>
    <div class="regressions">
        {#if comparison.regressions.length === 0}
            <p class="success">No significant regressions detected!</p>
        {:else}
            <ul>
                {#each comparison.regressions as reg}
                    <li class="fail">{reg}</li>
                {/each}
            </ul>
        {/if}
    </div>
{/if}

<style>
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }

    .delta { font-size: 1.5rem; font-weight: bold; }

    h3 { font-size: 0.9rem; color: #666; text-transform: uppercase; margin-top: 0; }

    .success { color: #0acf97; font-weight: bold; }
    .fail { color: #fa5c7c; }
</style>
