<script lang="ts">
    import { onMount } from 'svelte';
    import { api } from '$lib/api/client';
    import type { RunSummary, ConfigPreset } from '$lib/api/types';

    let runs: RunSummary[] = [];
    let presets: ConfigPreset[] = [];
    let error: string | null = null;
    let saveSuccess: string | null = null;

    // Pipeline performance config
    let pipelineConfig = $state<Record<string, number>>({});
    let pipelineSaving = $state(false);

    // Edit state
    let editingPreset: ConfigPreset | null = null;
    let newPresetName = '';
    let newPresetDesc = '';
    let editConfig: Record<string, number> = {};

    const THRESHOLD_META: Record<string, { label: string; min: number; max: number; step: number; tip: string }> = {
        corner_confidence: {
            label: 'Corner Confidence',
            min: 0.2, max: 0.9, step: 0.05,
            tip: 'Minimum YOLO confidence to accept a card corner detection. Higher = fewer false positives.',
        },
        background_novelty_threshold: {
            label: 'Background Novelty',
            min: 0.02, max: 0.20, step: 0.01,
            tip: 'How different a frame must be from the empty workspace. Higher = stricter empty-frame rejection.',
        },
        centroid_jump_ratio: {
            label: 'Centroid Jump Ratio',
            min: 0.10, max: 0.60, step: 0.05,
            tip: 'Fraction of frame width that triggers a new track on rapid card movement.',
        },
        valley_drop_ratio: {
            label: 'Valley Drop Ratio',
            min: 0.20, max: 0.70, step: 0.05,
            tip: 'Sharpness valley sensitivity for detecting card swaps. Higher = more session splits.',
        },
        foil_threshold: {
            label: 'Foil Threshold',
            min: 0.0, max: 200.0, step: 5.0,
            tip: 'Laplacian variance threshold for foil card detection. 0 = always use median fusion.',
        },
    };

    async function load() {
        try {
            [runs, presets] = await Promise.all([
                api.runs.list(),
                api.config.listPresets(),
            ]);
            pipelineConfig = await api.config.getPipeline();
        } catch (e: any) {
            error = e.message;
        }
    }

    async function savePipelineConfig() {
        pipelineSaving = true;
        try {
            pipelineConfig = await api.config.patchPipeline(pipelineConfig);
            saveSuccess = 'Pipeline config saved — takes effect on the next run.';
        } catch (e: any) {
            error = e.message;
        } finally {
            pipelineSaving = false;
        }
    }

    function startEdit(preset: ConfigPreset) {
        editingPreset = preset;
        newPresetName = preset.preset_name + '_copy';
        newPresetDesc = preset.description;
        editConfig = { ...preset.config as Record<string, number> };
    }

    function startNew() {
        editingPreset = null;
        newPresetName = '';
        newPresetDesc = '';
        editConfig = {
            corner_confidence: 0.50,
            background_novelty_threshold: 0.08,
            centroid_jump_ratio: 0.30,
            valley_drop_ratio: 0.40,
            foil_threshold: 50.0,
        };
    }

    function cancelEdit() {
        editingPreset = null;
        newPresetName = '';
        editConfig = {};
    }

    async function savePreset() {
        if (!newPresetName.trim()) {
            error = 'Preset name is required.';
            return;
        }
        try {
            await api.config.createPreset({
                preset_name: newPresetName.trim(),
                description: newPresetDesc,
                config: { ...editConfig },
            });
            saveSuccess = `Preset "${newPresetName.trim()}" saved.`;
            cancelEdit();
            await load();
        } catch (e: any) {
            error = e.message;
        }
    }

    const builtinNames = new Set(['fast', 'balanced', 'quality']);

    onMount(load);
</script>

<h1>Settings & Tuning</h1>

{#if error}
    <div class="banner error">{error} <button onclick={() => error = null}>✕</button></div>
{/if}
{#if saveSuccess}
    <div class="banner success">{saveSuccess} <button onclick={() => saveSuccess = null}>✕</button></div>
{/if}

<section>
    <h2>Pipeline Performance</h2>
    <p class="section-desc">Applies to all runs. Changes take effect on the next run.</p>
    <div class="perf-grid">
        <div class="perf-field">
            <label for="batch-size">YOLO Batch Size</label>
            <span class="perf-tip">Frames per GPU call. Larger = better GPU utilization. 256 is a good max for MPS.</span>
            <div class="perf-input-row">
                <input id="batch-size" type="number" min="8" max="512" step="8"
                    bind:value={pipelineConfig.inference_batch_size} />
                <span class="perf-unit">frames / batch</span>
            </div>
        </div>
        <div class="perf-field">
            <label for="triage-pct">Triage Keep %</label>
            <span class="perf-tip">Fraction of frames sent to YOLO after sharpness ranking. 0.05 = top 5%.</span>
            <div class="perf-input-row">
                <input id="triage-pct" type="number" min="0.01" max="1.0" step="0.01"
                    bind:value={pipelineConfig.triage_keep_percentile} />
                <span class="perf-unit">ratio</span>
            </div>
        </div>
        <div class="perf-field">
            <label for="corner-conf">Corner Confidence</label>
            <span class="perf-tip">Minimum YOLO detection confidence. Lower = more detections, more false positives.</span>
            <div class="perf-input-row">
                <input id="corner-conf" type="number" min="0.1" max="0.95" step="0.05"
                    bind:value={pipelineConfig.corner_confidence} />
                <span class="perf-unit">threshold</span>
            </div>
        </div>
    </div>
    <button class="btn-primary" onclick={savePipelineConfig} disabled={pipelineSaving}>
        {pipelineSaving ? 'Saving…' : 'Save'}
    </button>
</section>

<section>
    <div class="section-header">
        <h2>Config Presets</h2>
        <button class="btn-primary" onclick={startNew}>+ New Preset</button>
    </div>

    <div class="presets-grid">
        {#each presets as preset}
            <div class="preset-card" class:builtin={builtinNames.has(preset.preset_name)}>
                <div class="preset-top">
                    <h3>{preset.preset_name}</h3>
                    {#if builtinNames.has(preset.preset_name)}
                        <span class="badge">built-in</span>
                    {:else}
                        <span class="badge user">user</span>
                    {/if}
                </div>
                <p class="desc">{preset.description}</p>
                <dl class="config-dl">
                    {#each Object.entries(preset.config) as [k, v]}
                        <div class="dl-row">
                            <dt>{THRESHOLD_META[k]?.label ?? k}</dt>
                            <dd>{v}</dd>
                        </div>
                    {/each}
                </dl>
                <button class="btn-secondary" onclick={() => startEdit(preset)}>
                    Edit / Copy
                </button>
            </div>
        {/each}
    </div>
</section>

<!-- Inline editor (shown when New or Edit/Copy is clicked) -->
{#if newPresetName !== '' || Object.keys(editConfig).length > 0}
    <section class="editor-section">
        <h2>{editingPreset ? `Copy of "${editingPreset.preset_name}"` : 'New Preset'}</h2>

        <div class="editor-form">
            <div class="form-row">
                <label for="preset-name">Preset name</label>
                <input id="preset-name" type="text" bind:value={newPresetName} placeholder="my_preset" />
            </div>
            <div class="form-row">
                <label for="preset-desc">Description</label>
                <input id="preset-desc" type="text" bind:value={newPresetDesc} placeholder="Optional description" />
            </div>

            {#each Object.entries(THRESHOLD_META) as [key, meta]}
                <div class="form-row slider-row">
                    <label for="slider-{key}" title={meta.tip}>
                        {meta.label}
                        <span class="tip-icon" title={meta.tip}>?</span>
                    </label>
                    <div class="slider-group">
                        <input
                            id="slider-{key}"
                            type="range"
                            min={meta.min}
                            max={meta.max}
                            step={meta.step}
                            bind:value={editConfig[key]}
                        />
                        <input
                            type="number"
                            min={meta.min}
                            max={meta.max}
                            step={meta.step}
                            bind:value={editConfig[key]}
                            class="number-input"
                        />
                    </div>
                </div>
            {/each}

            <div class="editor-actions">
                <button class="btn-primary" onclick={savePreset}>Save Preset</button>
                <button class="btn-ghost" onclick={cancelEdit}>Cancel</button>
            </div>
        </div>
    </section>
{/if}

<section>
    <h2>Threshold Playground</h2>
    <p class="section-desc">Select a completed run to tune thresholds live against persisted stage outputs.</p>
    <div class="run-selector">
        {#each runs.filter(r => r.status === 'completed') as run}
            <a href="/settings/playground/{run.run_id}" class="run-link">
                {run.run_id}
                <span class="run-date">{new Date(run.created_at).toLocaleDateString()}</span>
            </a>
        {/each}
        {#if runs.filter(r => r.status === 'completed').length === 0}
            <p class="empty">No completed runs yet.</p>
        {/if}
    </div>
</section>

<style>
    section { margin-bottom: 3rem; }

    .section-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }

    .banner {
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .banner.error { background: #fff3f3; border: 1px solid #fa5c7c; color: #fa5c7c; }
    .banner.success { background: #f0fff8; border: 1px solid #0acf97; color: #0acf97; }
    .banner button { background: none; border: none; cursor: pointer; color: inherit; font-size: 1rem; }

    .presets-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 1.25rem;
    }

    .preset-card {
        background: white;
        padding: 1.25rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
    }

    .preset-card.builtin { border-top: 3px solid #727cf5; }

    .preset-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .preset-top h3 { margin: 0; font-size: 1rem; color: #313a46; }

    .badge {
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 20px;
        background: #eef0fe;
        color: #727cf5;
        font-weight: 600;
        text-transform: uppercase;
    }
    .badge.user { background: #f0fff8; color: #0acf97; }

    .desc { font-size: 0.85rem; color: #6c757d; margin: 0; }

    .config-dl { margin: 0; }
    .dl-row { display: flex; justify-content: space-between; font-size: 0.8rem; padding: 2px 0; }
    dt { color: #6c757d; }
    dd { font-weight: 600; color: #313a46; margin: 0; }

    .editor-section {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
    }

    .editor-form { display: flex; flex-direction: column; gap: 1rem; max-width: 600px; }

    .form-row { display: flex; flex-direction: column; gap: 0.25rem; }
    .form-row label { font-size: 0.85rem; font-weight: 600; color: #495057; }

    .form-row input[type="text"] {
        padding: 0.5rem 0.75rem;
        border: 1px solid #ced4da;
        border-radius: 6px;
        font-size: 0.9rem;
    }

    .compute-select {
        padding: 0.5rem 0.75rem;
        border: 1px solid #ced4da;
        border-radius: 6px;
        font-size: 0.9rem;
        background: white;
    }

    .slider-row label { display: flex; align-items: center; gap: 0.4rem; }

    .tip-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: #dee2e6;
        font-size: 0.7rem;
        color: #6c757d;
        cursor: help;
        flex-shrink: 0;
    }

    .slider-group { display: flex; gap: 0.75rem; align-items: center; }

    .slider-group input[type="range"] { flex: 1; accent-color: #727cf5; }

    .number-input {
        width: 80px;
        padding: 0.35rem 0.5rem;
        border: 1px solid #ced4da;
        border-radius: 6px;
        font-size: 0.85rem;
        text-align: right;
    }

    .editor-actions { display: flex; gap: 0.75rem; margin-top: 0.5rem; }

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

    .btn-ghost {
        background: none;
        border: none;
        color: #6c757d;
        cursor: pointer;
        font-size: 0.9rem;
    }

    .section-desc { color: #6c757d; font-size: 0.9rem; margin-bottom: 1rem; }

    .run-selector { display: flex; flex-direction: column; gap: 0.5rem; }

    .run-link {
        background: white;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        text-decoration: none;
        color: #727cf5;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        display: flex;
        justify-content: space-between;
    }
    .run-link:hover { background: #eef2ff; }
    .run-date { color: #6c757d; font-weight: normal; font-size: 0.85rem; }

    .empty { color: #6c757d; font-style: italic; font-size: 0.9rem; }

    .perf-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .perf-field {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        display: flex;
        flex-direction: column;
        gap: 0.35rem;
    }
    .perf-field label { font-size: 0.85rem; font-weight: 700; color: #313a46; }
    .perf-tip { font-size: 0.75rem; color: #6c757d; line-height: 1.4; }
    .perf-input-row { display: flex; align-items: center; gap: 0.5rem; margin-top: 0.25rem; }
    .perf-input-row input[type="number"] {
        width: 90px;
        padding: 0.4rem 0.6rem;
        border: 1px solid #ced4da;
        border-radius: 6px;
        font-size: 1rem;
        font-weight: 600;
        color: #313a46;
        text-align: right;
    }
    .perf-unit { font-size: 0.8rem; color: #6c757d; }
</style>
