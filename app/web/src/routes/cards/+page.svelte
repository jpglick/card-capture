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
                <div class="img-wrap">
                    <img src={card.fused_url ?? card.canonical_url ?? ''} alt="Card" />
                    <div class="hover-ids">
                        <button
                            class="id-chip"
                            title="Click to copy instance_id"
                            onclick={() => navigator.clipboard.writeText(card.instance_id)}
                        >{card.instance_id.slice(0, 8)}</button>
                        <span class="run-chip">{card.run_id}</span>
                    </div>
                </div>
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

    .img-wrap { position: relative; }

    img { width: 100%; height: 210px; object-fit: cover; background: #111; display: block; }

    .hover-ids {
        position: absolute;
        inset: 0;
        background: rgba(0,0,0,0.55);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 0.4rem;
        opacity: 0;
        transition: opacity 0.15s;
        pointer-events: none;
    }

    .img-wrap:hover .hover-ids {
        opacity: 1;
        pointer-events: auto;
    }

    .id-chip {
        font-family: monospace;
        font-size: 0.8rem;
        font-weight: 700;
        color: #fff;
        background: rgba(114,124,245,0.85);
        border: none;
        border-radius: 4px;
        padding: 0.2rem 0.5rem;
        cursor: pointer;
        letter-spacing: 0.04em;
    }

    .id-chip:active { background: #4e5bd4; }

    .run-chip {
        font-family: monospace;
        font-size: 0.65rem;
        color: rgba(255,255,255,0.7);
        background: rgba(0,0,0,0.4);
        border-radius: 3px;
        padding: 0.1rem 0.4rem;
        max-width: 90%;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

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
