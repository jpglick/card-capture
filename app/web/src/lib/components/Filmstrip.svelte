<script lang="ts">
    import type { Card } from '$lib/api/types';

    let { items = [], selectedId = null, onSelect } = $props();

    function getVerdictClass(card: any) {
        if (card.verdict === 'real') return 'verdict-real';
        if (card.verdict === 'phantom') return 'verdict-phantom';
        return '';
    }
</script>

<div class="filmstrip">
    {#each items as item}
        <button 
            class="item {selectedId === item.instance_id ? 'selected' : ''} {getVerdictClass(item)}"
            onclick={() => onSelect(item.instance_id)}
        >
            <img src={item.fused_url ?? item.canonical_url ?? ''} alt="Thumbnail" />
            <div class="badge">{item.side}</div>
        </button>
    {/each}
</div>

<style>
    .filmstrip {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        padding: 0.5rem 0;
    }

    .item {
        width: 100%;
        aspect-ratio: 5 / 7;
        position: relative;
        background: #2b2b3d;
        border: 2px solid transparent;
        border-radius: 4px;
        overflow: hidden;
        cursor: pointer;
        padding: 0;
    }

    .item.selected { border-color: #727cf5; }
    .item.verdict-real { border-color: #0acf97; }
    .item.verdict-phantom { border-color: #fa5c7c; }

    img { width: 100%; height: 100%; object-fit: cover; }

    .badge {
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(0,0,0,0.6);
        color: white;
        font-size: 0.6rem;
        padding: 2px;
        text-align: center;
    }
</style>
