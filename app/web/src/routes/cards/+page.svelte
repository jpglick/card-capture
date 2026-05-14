<script lang="ts">
    import { onMount } from 'svelte';
    import { api } from '$lib/api/client';
    import type { Card } from '$lib/api/types';

    let cards: Card[] = [];
    let loading = true;

    async function load() {
        try {
            loading = true;
            cards = await api.cards.listAll({});
        } catch (e) {
            console.error(e);
        } finally {
            loading = false;
        }
    }

    onMount(load);
</script>

<h1>Cards</h1>

{#if loading}
    <p>Loading cards...</p>
{:else}
    <div class="card-grid">
        {#each cards as card}
            <div class="card-item">
                <img src={card.fused_image_url} alt="Fused card" />
                <div class="info">
                    <span class="angle">{card.angle}</span>
                    <span class="score">{(card.quality_score * 100).toFixed(0)}%</span>
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

    img { width: 100%; height: 210px; object-fit: cover; }

    .info {
        padding: 0.5rem;
        display: flex;
        justify-content: space-between;
        font-size: 0.75rem;
    }

    .angle { font-weight: bold; color: #727cf5; }
    .score { color: #0acf97; }
</style>
