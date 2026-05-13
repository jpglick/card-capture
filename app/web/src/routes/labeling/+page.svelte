<script lang="ts">
    import { onMount } from 'svelte';
    import { api } from '$lib/api/client';
    import type { DatasetSummary, LabelFBNext } from '$lib/api/types';

    let datasets: DatasetSummary[] = [];
    let nextFB: LabelFBNext | null = null;
    let loading = true;

    async function load() {
        loading = true;
        try {
            datasets = await api.training.listDatasets();
            const fb = await api.label.nextFB();
            if ('instance_id' in fb) {
                nextFB = fb as LabelFBNext;
            }
        } catch (e) {
            console.error(e);
        } finally {
            loading = false;
        }
    }

    async function submitLabel(side: 'front' | 'back' | 'uncertain') {
        if (!nextFB) return;
        try {
            await api.label.postFB({
                instance_id: nextFB.instance_id,
                frame_index: nextFB.frame_index,
                side
            });
            await load(); // Load next
        } catch (e) {
            alert(e);
        }
    }

    onMount(load);
</script>

<h1>Labeling</h1>

<div class="stats">
    {#each datasets as ds}
        <div class="stat-card">
            <h3>{ds.model_name}</h3>
            <p>Total Labels: {ds.total_labels}</p>
        </div>
    {/each}
</div>

{#if nextFB}
    <div class="label-tool">
        <h2>Front/Back Classification</h2>
        <div class="image-container">
            <img src={nextFB.canonical_url} alt="Candidate card" />
        </div>
        <div class="actions">
            <button onclick={() => submitLabel('front')} class="front">Front (F)</button>
            <button onclick={() => submitLabel('back')} class="back">Back (B)</button>
            <button onclick={() => submitLabel('uncertain')} class="uncertain">Uncertain (U)</button>
        </div>
    </div>
{:else if !loading}
    <p>No more candidates for labeling at the moment.</p>
{/if}

<style>
    .stats {
        display: flex;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .stat-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        flex: 1;
    }
    .label-tool {
        background: white;
        padding: 2rem;
        border-radius: 8px;
        text-align: center;
    }
    .image-container img {
        max-width: 100%;
        height: 400px;
        object-fit: contain;
        border: 1px solid #eee;
        margin-bottom: 1rem;
    }
    .actions {
        display: flex;
        justify-content: center;
        gap: 1rem;
    }
    button {
        padding: 1rem 2rem;
        font-size: 1.25rem;
        cursor: pointer;
        border: none;
        border-radius: 4px;
        transition: opacity 0.2s;
    }
    button:hover { opacity: 0.8; }
    .front { background: #0acf97; color: white; }
    .back { background: #727cf5; color: white; }
    .uncertain { background: #6c757d; color: white; }
</style>
