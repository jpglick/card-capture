<script lang="ts">
    import { onMount } from 'svelte';
    import { createFBQueue } from '$lib/stores/fb_queue.svelte';
    import Hotkeys from '$lib/components/Hotkeys.svelte';

    const q = createFBQueue();

    onMount(() => {
        q.fetchNext();
        q.fetchNext();
        q.fetchNext();
    });

    const keys = {
        'f': () => q.submitLabel('front'),
        'b': () => q.submitLabel('back'),
        'u': () => q.submitLabel('uncertain')
    };
</script>

<Hotkeys {keys}>
    <div class="trainer-container">
        <header>
            <h1>Front/Back Trainer</h1>
            <div class="stats">
                Labeled this session: <strong>{q.labeledCount}</strong>
            </div>
        </header>

        {#if q.current}
            <div class="label-area">
                <div class="card-frame">
                    <img src={q.current.canonical_url} alt="Candidate" />
                </div>

                <div class="instruction">
                    Is this the <strong>Front</strong> or <strong>Back</strong> of a card?
                </div>

                <div class="actions">
                    <button class="front" onclick={() => q.submitLabel('front')}>
                        <span class="label">Front</span>
                        <span class="key">F</span>
                    </button>
                    <button class="back" onclick={() => q.submitLabel('back')}>
                        <span class="label">Back</span>
                        <span class="key">B</span>
                    </button>
                    <button class="uncertain" onclick={() => q.submitLabel('uncertain')}>
                        <span class="label">Uncertain</span>
                        <span class="key">U</span>
                    </button>
                </div>
            </div>
        {:else}
            <p>Queue is empty. Fetching more candidates...</p>
        {/if}
    </div>
</Hotkeys>

<style>
    .trainer-container {
        max-width: 800px;
        margin: 0 auto;
        text-align: center;
    }

    header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
    }

    .label-area {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }

    .card-frame {
        width: 350px;
        height: 490px;
        margin: 0 auto 2rem;
        border: 1px solid #eee;
        border-radius: 8px;
        overflow: hidden;
        background: #f8f9fa;
    }

    .card-frame img { width: 100%; height: 100%; object-fit: contain; }

    .instruction { font-size: 1.25rem; margin-bottom: 2rem; }

    .actions { display: flex; gap: 1rem; justify-content: center; }

    button {
        display: flex;
        flex-direction: column;
        padding: 1rem 2rem;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        min-width: 120px;
        color: white;
    }

    .front { background: #0acf97; }
    .back { background: #727cf5; }
    .uncertain { background: #6c757d; }

    .label { font-weight: bold; font-size: 1.1rem; }
    .key { font-size: 0.8rem; opacity: 0.7; }
</style>
