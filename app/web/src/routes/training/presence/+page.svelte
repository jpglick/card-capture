<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import Hotkeys from '$lib/components/Hotkeys.svelte';
  import { api } from '$lib/api/client';

  let sample = $state<any>(null);
  let done = $state(false);
  let labeled = $state(0);

  async function fetchNext() {
    const r = await api.get('/training/presence/next');
    if (r.status === 204) { done = true; return; }
    sample = await r.json();
  }

  async function submit(label: 'present' | 'absent') {
    if (!sample) return;
    await api.post('/training/presence/label', { sample_id: sample.sample_id, label });
    labeled++;
    await fetchNext();
  }

  async function skip() { await fetchNext(); }

  onMount(fetchNext);

  const keys = {
    y: () => submit('present'),
    n: () => submit('absent'),
    s: skip,
  };
</script>

<Hotkeys {keys}>
  <div class="container">
    <header>
      <button onclick={() => goto('/training')} class="back">← Training</button>
      <span class="counter">
        {#if sample}
          {sample.pending_count} remaining · {labeled} labeled this session
        {/if}
      </span>
    </header>

    {#if done}
      <div class="empty">
        <p>Queue is empty.</p>
        <button onclick={() => goto('/training')}>Back to hub</button>
      </div>
    {:else if sample}
      <div class="label-area">
        <div class="frame-box">
          <img src={sample.image_url} alt="scan frame" class="scan-frame" />
        </div>

        <div class="question">Is there a card in this frame?</div>

        <div class="actions">
          <button class="yes" onclick={() => submit('present')}>
            <span class="label">Card present</span>
            <span class="key">Y</span>
          </button>
          <button class="no" onclick={() => submit('absent')}>
            <span class="label">No card</span>
            <span class="key">N</span>
          </button>
          <button class="skip" onclick={skip}>
            <span class="label">Skip</span>
            <span class="key">S</span>
          </button>
        </div>

        <div class="progress">
          <div class="bar" style="width: {Math.max(0, 100 - (sample.pending_count / (sample.pending_count + labeled)) * 100)}%"></div>
        </div>
      </div>
    {:else}
      <p>Loading…</p>
    {/if}
  </div>
</Hotkeys>

<style>
  .container { max-width: 600px; margin: 2rem auto; padding: 0 1rem; }
  header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem; }
  .back { background: none; border: none; color: #aaa; cursor: pointer; font-size: 0.9rem; }
  .counter { color: #aaa; font-size: 0.85rem; }
  .label-area { display: flex; flex-direction: column; gap: 1.5rem; align-items: center; }
  .frame-box { background: #111; border-radius: 8px; padding: 1rem; }
  .scan-frame { image-rendering: pixelated; width: 384px; height: auto; display: block; }
  .question { font-size: 1.1rem; font-weight: 600; }
  .actions { display: flex; gap: 1rem; }
  button { display: flex; flex-direction: column; align-items: center; padding: 0.75rem 1.5rem;
    border: none; border-radius: 8px; cursor: pointer; min-width: 100px; }
  .yes { background: #0acf97; color: white; }
  .no { background: #fa5c7c; color: white; }
  .skip { background: #444; color: white; }
  .label { font-weight: 600; }
  .key { font-size: 0.75rem; opacity: 0.7; }
  .progress { width: 100%; height: 4px; background: #333; border-radius: 2px; }
  .bar { height: 100%; background: #6366f1; border-radius: 2px; transition: width 0.3s; }
  .empty { text-align: center; margin-top: 4rem; }
</style>
