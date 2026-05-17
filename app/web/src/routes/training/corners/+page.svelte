<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import Hotkeys from '$lib/components/Hotkeys.svelte';
  import { api } from '$lib/api/client';

  let sample = $state<any>(null);
  let done = $state(false);
  let labeled = $state(0);
  let adjusting = $state(false);
  let corners = $state<[number, number][]>([]);
  let dragging = $state<number | null>(null);
  let imgEl = $state<HTMLImageElement | null>(null);
  let imgNaturalW = $state(1);
  let imgNaturalH = $state(1);
  let imgDisplayW = $state(1);
  let imgDisplayH = $state(1);

  async function fetchNext() {
    adjusting = false;
    const r = await api.training.nextCorner();
    if (r.status === 204) { done = true; return; }
    sample = await r.json();
    corners = JSON.parse(sample.predicted_corners);
  }

  async function submit(label: string, corrected: [number,number][] | null = null) {
    if (!sample) return;
    const body: any = { sample_id: sample.sample_id, label };
    if (corrected) body.corrected_corners = JSON.stringify(corrected);
    await api.training.labelCorner(body);
    labeled++;
    await fetchNext();
  }

  function startAdjust() { adjusting = true; }

  function scaleToNatural(x: number, y: number): [number, number] {
    const sx = imgNaturalW / imgDisplayW;
    const sy = imgNaturalH / imgDisplayH;
    return [x * sx, y * sy];
  }

  function scaleToDisplay(x: number, y: number): [number, number] {
    return [x * (imgDisplayW / imgNaturalW), y * (imgDisplayH / imgNaturalH)];
  }

  function onMouseDown(i: number) { dragging = i; }

  function onMouseMove(e: MouseEvent) {
    if (dragging === null || !imgEl) return;
    const rect = imgEl.getBoundingClientRect();
    const dx = e.clientX - rect.left;
    const dy = e.clientY - rect.top;
    const [nx, ny] = scaleToNatural(dx, dy);
    corners = corners.map((c, i) => i === dragging ? [nx, ny] : c) as [number,number][];
  }

  function onMouseUp() { dragging = null; }

  function confirmAdjust() { submit('adjusted', corners); }

  function onImgLoad() {
    if (!imgEl) return;
    imgNaturalW = imgEl.naturalWidth;
    imgNaturalH = imgEl.naturalHeight;
    imgDisplayW = imgEl.clientWidth;
    imgDisplayH = imgEl.clientHeight;
  }

  onMount(fetchNext);

  const keys = {
    y: () => !adjusting && submit('correct'),
    n: () => !adjusting && submit('negative'),
    e: () => !adjusting && startAdjust(),
    ' ': () => adjusting && confirmAdjust(),
    Escape: () => { adjusting = false; if (sample) corners = JSON.parse(sample.predicted_corners); },
    s: () => !adjusting && fetchNext(),
  };
</script>

<Hotkeys {keys}>
  <div class="container">
    <header>
      <button onclick={() => goto('/training')} class="back">← Training</button>
      <span class="counter">
        {#if sample}
          conf {(sample.confidence * 100).toFixed(0)}% · {sample.pending_count} remaining · {labeled} labeled
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
        <div
          class="img-wrap"
          onmousemove={onMouseMove}
          onmouseup={onMouseUp}
          role="img"
          aria-label="frame with corner overlay"
        >
          <img
            bind:this={imgEl}
            src={sample.image_url}
            alt="detection frame"
            class="frame-img"
            onload={onImgLoad}
          />
          <svg class="overlay" width={imgDisplayW} height={imgDisplayH}>
            <polygon
              points={corners.map(([x, y]) => scaleToDisplay(x, y).join(',')).join(' ')}
              fill="rgba(0,255,120,0.15)"
              stroke="#00ff78"
              stroke-width="2"
            />
            {#if adjusting}
              {#each corners as [cx, cy], i}
                {@const [dx, dy] = scaleToDisplay(cx, cy)}
                <circle
                  cx={dx} cy={dy} r="8"
                  fill="#00ff78" stroke="white" stroke-width="2"
                  style="cursor: grab"
                  onmousedown={() => onMouseDown(i)}
                  role="button"
                  aria-label={`corner ${i}`}
                  tabindex={i}
                />
              {/each}
            {/if}
          </svg>
        </div>

        {#if adjusting}
          <div class="hint">Drag corners to adjust · <kbd>Space</kbd> confirm · <kbd>Esc</kbd> cancel</div>
          <div class="actions">
            <button class="confirm" onclick={confirmAdjust}>Confirm (Space)</button>
            <button class="skip" onclick={() => { adjusting = false; }}>Cancel (Esc)</button>
          </div>
        {:else}
          <div class="question">Do the highlighted corners look right?</div>
          <div class="actions">
            <button class="yes" onclick={() => submit('correct')}>
              <span class="label">Correct</span><span class="key">Y</span>
            </button>
            <button class="adjust" onclick={startAdjust}>
              <span class="label">Adjust</span><span class="key">E</span>
            </button>
            <button class="no" onclick={() => submit('negative')}>
              <span class="label">No card</span><span class="key">N</span>
            </button>
            <button class="skip" onclick={fetchNext}>
              <span class="label">Skip</span><span class="key">S</span>
            </button>
          </div>
        {/if}
      </div>
    {:else}
      <p>Loading…</p>
    {/if}
  </div>
</Hotkeys>

<style>
  .container { max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
  header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }
  .back { background: none; border: none; color: #aaa; cursor: pointer; }
  .counter { color: #aaa; font-size: 0.85rem; }
  .label-area { display: flex; flex-direction: column; gap: 1.25rem; align-items: center; }
  .img-wrap { position: relative; display: inline-block; }
  .frame-img { max-width: 100%; max-height: 60vh; display: block; }
  .overlay { position: absolute; top: 0; left: 0; pointer-events: none; }
  .overlay circle { pointer-events: all; }
  .question { font-size: 1.05rem; font-weight: 600; }
  .hint { color: #aaa; font-size: 0.85rem; }
  .actions { display: flex; gap: 0.75rem; flex-wrap: wrap; justify-content: center; }
  button { display: flex; flex-direction: column; align-items: center; padding: 0.65rem 1.2rem;
    border: none; border-radius: 8px; cursor: pointer; }
  .yes { background: #0acf97; color: white; }
  .adjust { background: #f59e0b; color: white; }
  .no { background: #fa5c7c; color: white; }
  .skip { background: #444; color: white; }
  .confirm { background: #6366f1; color: white; }
  .label { font-weight: 600; font-size: 0.9rem; }
  .key { font-size: 0.7rem; opacity: 0.7; }
  .empty { text-align: center; margin-top: 4rem; }
  kbd { background: #333; padding: 0.1rem 0.4rem; border-radius: 4px; font-size: 0.8rem; }
</style>
