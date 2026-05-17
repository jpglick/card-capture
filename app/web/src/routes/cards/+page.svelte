<script lang="ts">
    import { onMount } from 'svelte';
    import { page } from '$app/stores';
    import { api } from '$lib/api/client';
    import type { Card } from '$lib/api/types';

    let cards: Card[] = [];
    let loading = true;
    let runId: string | null = null;

    async function load() {
        try {
            loading = true;
            runId = $page.url.searchParams.get('run_id');
            const filter: Record<string, string> = {};
            if (runId) filter.run_id = runId;
            cards = await api.cards.listAll(filter as any);
        } catch (e) {
            console.error(e);
        } finally {
            loading = false;
        }
    }

    onMount(load);
</script>

<h1>Cards{runId ? ` — ${runId}` : ''}</h1>

{#if loading}
    <p>Loading cards...</p>
{:else if cards.length === 0}
    <p class="empty">No cards found{runId ? ` for run ${runId}` : ''}.</p>
{:else}
    <div class="card-grid">
        {#each cards as card}
            <div class="card-item">
                <img src={card.fused_url ?? card.canonical_url ?? ''} alt="Card" />
                <div class="info">
                    <span class="angle">{card.side}</span>
                    <span class="score">{(card.confidence * 100).toFixed(0)}%</span>
                </div>
            </div>
        {/each}
    </div>
{/if}

<style>
    .card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
        gap: 1.5rem;
    }

    .card-item {
        background: white;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }

    .card-item:hover { transform: translateY(-4px); }

    img { width: 100%; height: 210px; object-fit: cover; background: #111; }

    .info {
        padding: 0.5rem;
        display: flex;
        justify-content: space-between;
        font-size: 0.75rem;
    }

    .angle { font-weight: bold; color: #727cf5; }
    .score { color: #0acf97; }
    .empty { color: #aaa; margin-top: 2rem; }
</style>
