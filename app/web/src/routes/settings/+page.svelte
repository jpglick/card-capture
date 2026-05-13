<script lang="ts">
    import { onMount } from 'svelte';
    import { api } from '$lib/api/client';
    import type { RunSummary, ConfigPreset } from '$lib/api/types';

    let runs: RunSummary[] = [];
    let presets: ConfigPreset[] = [];

    async function load() {
        [runs, presets] = await Promise.all([
            api.runs.list(),
            api.config.listPresets()
        ]);
    }

    onMount(load);
</script>

<h1>Settings & Tuning</h1>

<section>
    <h2>Config Presets</h2>
    <div class="presets-grid">
        {#each presets as preset}
            <div class="preset-card">
                <h3>{preset.preset_name}</h3>
                <p>{preset.description}</p>
                <pre>{JSON.stringify(preset.config, null, 2)}</pre>
            </div>
        {/each}
    </div>
</section>

<section>
    <h2>Threshold Playground</h2>
    <p>Select a completed run to tune thresholds in real-time.</p>
    <div class="run-selector">
        {#each runs.filter(r => r.status === 'completed') as run}
            <a href="/settings/playground/{run.run_id}" class="run-link">
                {run.run_id} ({new Date(run.created_at).toLocaleDateString()})
            </a>
        {/each}
    </div>
</section>

<style>
    section { margin-bottom: 3rem; }
    
    .presets-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
    }

    .preset-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

    pre {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 6px;
        font-size: 0.8rem;
        overflow-x: auto;
    }

    .run-selector {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    .run-link {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-decoration: none;
        color: #727cf5;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    .run-link:hover { background: #eef2ff; }
</style>
