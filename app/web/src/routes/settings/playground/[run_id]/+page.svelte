<script lang="ts">
    import { onMount } from 'svelte';
    import { page } from '$app/state';
    import { api } from '$lib/api/client';
    import type { ConfigPlayground } from '$lib/api/types';

    const runId = page.params.run_id;
    let initial: ConfigPlayground | null = null;
    let config = $state<any>({});
    let results = $state<any>(null);
    let loading = true;
    let updating = $state(false);

    async function load() {
        try {
            loading = true;
            initial = await api.config.getPlayground(runId);
            config = { ...initial.config };
            await recompute();
        } finally {
            loading = false;
        }
    }

    async function recompute() {
        updating = true;
        try {
            // Using fetch directly as we didn't add recompute to the client yet
            const r = await fetch(`/api/v1/config/playground/${runId}/recompute`, {
                method: 'POST',
                headers: { 'content-type': 'application/json' },
                body: JSON.stringify(config)
            });
            results = await r.json();
        } finally {
            updating = false;
        }
    }

    function handleSliderChange() {
        recompute();
    }

    onMount(load);
</script>

<div class="playground">
    <header>
        <h1>Threshold Playground</h1>
        <p>Run: <code>{runId}</code></p>
    </header>

    {#if loading}
        <p>Loading run artifacts...</p>
    {:else}
        <div class="layout">
            <aside class="controls">
                <h2>Parameters</h2>
                
                <div class="control-group">
                    <label>Corner Confidence ({config.corner_confidence_threshold})</label>
                    <input type="range" min="0.1" max="0.9" step="0.05" 
                           bind:value={config.corner_confidence_threshold}
                           onchange={handleSliderChange} />
                    <span class="hint">Higher = fewer phantoms, more misses.</span>
                </div>

                <div class="control-group">
                    <label>Background Novelty ({config.background_novelty_threshold})</label>
                    <input type="range" min="0.01" max="0.3" step="0.01" 
                           bind:value={config.background_novelty_threshold}
                           onchange={handleSliderChange} />
                    <span class="hint">Gating vs empty stand.</span>
                </div>

                <div class="control-group">
                    <label>Min Track Length ({config.min_track_length})</label>
                    <input type="range" min="1" max="30" step="1" 
                           bind:value={config.min_track_length}
                           onchange={handleSliderChange} />
                </div>

                <button class="save-preset">Save as New Preset</button>
            </aside>

            <main class="results">
                {#if updating}
                    <div class="overlay">Recomputing...</div>
                {/if}

                <div class="metrics-summary">
                    <div class="metric">
                        <h3>Extracted Cards</h3>
                        <div class="value">{results?.metrics.card_count || 0}</div>
                    </div>
                </div>

                <h2>Resulting Tracks</h2>
                <div class="tracks-grid">
                    {#if results?.tracks}
                        {#each results.tracks as track}
                            <div class="track-card">
                                <div class="badge">{track.angle}</div>
                                <div class="id">{track.instance_id.slice(0,8)}</div>
                                <div class="frames">{track.candidate_detection_ids.length} frames</div>
                            </div>
                        {/each}
                    {/if}
                </div>
            </main>
        </div>
    {/if}
</div>

<style>
    .playground { height: 100%; display: flex; flex-direction: column; }
    
    .layout { display: grid; grid-template-columns: 300px 1fr; gap: 2rem; flex: 1; }

    .controls {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

    .control-group { margin-bottom: 2rem; }
    label { display: block; font-weight: bold; margin-bottom: 0.5rem; }
    input[type="range"] { width: 100%; }
    .hint { font-size: 0.75rem; color: #666; }

    .results { position: relative; }
    .overlay {
        position: absolute; top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(255,255,255,0.7);
        display: flex; align-items: center; justify-content: center;
        z-index: 10; font-weight: bold;
    }

    .metrics-summary {
        display: flex; gap: 1rem; margin-bottom: 2rem;
    }
    .metric {
        background: #727cf5; color: white;
        padding: 1.5rem; border-radius: 10px; flex: 1;
    }
    .metric h3 { margin: 0; font-size: 0.9rem; opacity: 0.8; }
    .metric .value { font-size: 2.5rem; font-weight: bold; }

    .tracks-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
        gap: 1rem;
    }
    .track-card {
        background: white; padding: 0.5rem; border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center;
    }
    .badge { font-size: 0.7rem; background: #eee; border-radius: 4px; margin-bottom: 0.5rem; }
    .id { font-family: monospace; font-size: 0.8rem; }
    .frames { font-size: 0.7rem; color: #666; }

    .save-preset {
        width: 100%; padding: 0.75rem; background: #0acf97; color: white;
        border: none; border-radius: 6px; font-weight: bold; cursor: pointer;
    }
</style>
