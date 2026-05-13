<script lang="ts">
    import { onMount } from 'svelte';
    import { api } from '$lib/api/client';

    let cases = $state<any[]>([]);
    let loading = true;

    async function load() {
        loading = true;
        try {
            // Using fetch directly as we didn't add hard_cases to the client yet
            const r = await fetch('/api/v1/training/hard_cases');
            cases = await r.json();
        } finally {
            loading = false;
        }
    }

    async function promote(case_id: number, model: string, label: string) {
        try {
            await fetch(`/api/v1/training/hard_cases/${case_id}/promote?model_name=${model}&label=${label}`, {
                method: 'POST'
            });
            await load();
        } catch (e) {
            alert(e);
        }
    }

    onMount(load);
</script>

<h1>Hard Case Review</h1>
<p>Graduate high-value failure cases into the permanent training set.</p>

{#if loading}
    <p>Loading cases...</p>
{:else if cases.length === 0}
    <p>No new hard cases for review.</p>
{:else}
    <div class="case-grid">
        {#each cases as c}
            <div class="case-card">
                <div class="badge">{c.stage_id}</div>
                <img src={c.thumbnail_path} alt="Hard case" />
                <div class="reason">{c.reason}</div>
                
                <div class="actions">
                    <button onclick={() => promote(c.case_id, 'fb_classifier', 'front')}>F/B: Front</button>
                    <button onclick={() => promote(c.case_id, 'fb_classifier', 'back')}>F/B: Back</button>
                    <button class="secondary" onclick={() => promote(c.case_id, 'phantom', 'true')}>Phantom</button>
                </div>
            </div>
        {/each}
    </div>
{/if}

<style>
    .case-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 2rem;
    }

    .case-card {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

    .badge {
        font-size: 0.7rem;
        background: #eee;
        display: inline-block;
        padding: 2px 6px;
        border-radius: 4px;
        margin-bottom: 0.5rem;
    }

    img {
        width: 100%;
        height: 250px;
        object-fit: contain;
        background: #f8f9fa;
        border-radius: 8px;
    }

    .reason {
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #666;
    }

    .actions {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }

    button {
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        background: #727cf5;
        color: white;
        font-size: 0.8rem;
    }

    button.secondary { background: #eee; color: #333; }
</style>
