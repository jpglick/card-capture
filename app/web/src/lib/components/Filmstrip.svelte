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
            <img src={item.fused_image_url} alt="Thumbnail" />
            <div class="badge">{item.angle}</div>
        </button>
    {/each}
</div>

<style>
    .filmstrip {
        display: flex;
        gap: 0.5rem;
        overflow-x: auto;
        padding: 1rem 0;
        background: #1e1e2d;
    }

    .item {
        flex: 0 0 100px;
        height: 140px;
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
