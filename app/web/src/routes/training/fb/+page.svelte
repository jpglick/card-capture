<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import Hotkeys from '$lib/components/Hotkeys.svelte';
  import { api } from '$lib/api/client';

  let sample = $state<any>(null);
  let done = $state(false);
  let labeled = $state(0);

  async function fetchNext() {
    const r = await api.get('/label/fb/next');
    if (r.status === 204) { done = true; return; }
    sample = await r.json();
  }

  async function submit(side: string) {
    if (!sample) return;
    await api.post('/label/fb', { instance_id: sample.instance_id, frame_index: sample.frame_index, side });
    labeled++;
    await fetchNext();
  }

  async function skip() { await fetchNext(); }

  onMount(fetchNext);

  const keys = {
    f: () => submit('front'),
    b: () => submit('back'),
    u: () => submit('uncertain'),
    x: () => submit('no_card'),
    s: skip,
  };
</script>

<Hotkeys {keys}>
  <div class="container">
    <header>
      <button onclick={() => goto('/training')} class="back">← Training</button>
      <span class="counter">{labeled} labeled this session</span>
    </header>

    {#if done}
      <div class="empty">
        <p>Queue is empty.</p>
        <button onclick={() => goto('/training')}>Back to hub</button>
      </div>
    {:else if sample}
      <div class="label-area">
        <div class="card-box">
          <img src={sample.canonical_url} alt="card" class="card-img" />
        </div>

        <div class="question">Which side is this?</div>

        <div class="actions">
          <button class="front" onclick={() => submit('front')}>
            <span class="label">Front</span><span class="key">F</span>
          </button>
          <button class="back-btn" onclick={() => submit('back')}>
            <span class="label">Back</span><span class="key">B</span>
          </button>
          <button class="uncertain" onclick={() => submit('uncertain')}>
            <span class="label">Unsure</span><span class="key">U</span>
          </button>
          <button class="nocard" onclick={() => submit('no_card')}>
            <span class="label">Not a card</span><span class="key">X</span>
          </button>
          <button class="skip" onclick={skip}>
            <span class="label">Skip</span><span class="key">S</span>
          </button>
        </div>
      </div>
    {:else}
      <p>Loading…</p>
    {/if}
  </div>
</Hotkeys>

<style>
  .container { max-width: 500px; margin: 2rem auto; padding: 0 1rem; }
  header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem; }
  .back { background: none; border: none; color: #aaa; cursor: pointer; }
  .counter { color: #aaa; font-size: 0.85rem; }
  .label-area { display: flex; flex-direction: column; gap: 1.5rem; align-items: center; }
  .card-box { background: #111; border-radius: 8px; padding: 0.75rem; }
  .card-img { width: 280px; height: auto; display: block; border-radius: 4px; }
  .question { font-size: 1.1rem; font-weight: 600; }
  .actions { display: flex; gap: 0.75rem; flex-wrap: wrap; justify-content: center; }
  button { display: flex; flex-direction: column; align-items: center; padding: 0.65rem 1.2rem;
    border: none; border-radius: 8px; cursor: pointer; }
  .front { background: #6366f1; color: white; }
  .back-btn { background: #8b5cf6; color: white; }
  .uncertain { background: #f59e0b; color: white; }
  .nocard { background: #fa5c7c; color: white; }
  .skip { background: #444; color: white; }
  .label { font-weight: 600; font-size: 0.9rem; }
  .key { font-size: 0.7rem; opacity: 0.7; }
  .empty { text-align: center; margin-top: 4rem; }
</style>
