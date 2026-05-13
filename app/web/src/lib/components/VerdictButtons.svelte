<script lang="ts">
    let { onVerdict } = $props();
</script>

<div class="verdict-buttons">
    <button class="real" onclick={() => onVerdict('real')}>
        <span class="label">Real</span>
        <span class="key">R</span>
    </button>
    <button class="phantom" onclick={() => onVerdict('phantom')}>
        <span class="label">Phantom</span>
        <span class="key">P</span>
    </button>
    <button class="flip" onclick={() => onVerdict('flip')}>
        <span class="label">Flip Side</span>
        <span class="key">F</span>
    </button>
</div>

<style>
    .verdict-buttons {
        display: flex;
        gap: 1rem;
        justify-content: center;
        padding: 1rem;
    }

    button {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 1rem 2rem;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        transition: opacity 0.2s;
        min-width: 120px;
    }

    button:hover { opacity: 0.8; }

    .real { background: #0acf97; color: white; }
    .phantom { background: #fa5c7c; color: white; }
    .flip { background: #727cf5; color: white; }

    .label { font-size: 1.1rem; font-weight: bold; }
    .key { font-size: 0.8rem; opacity: 0.7; }
</style>
