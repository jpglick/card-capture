<script lang="ts">
    import { onMount } from 'svelte';
    import { page } from '$app/stores';
    import { api } from '$lib/api/client';
    import type { Card, CdpSubmission } from '$lib/api/types';

    let cards: Card[] = [];
    let loading = true;
    let runId: string | null = null;
    let toast: { card: Card; index: number } | null = null;
    let toastTimer: ReturnType<typeof setTimeout> | null = null;

    // CDP state: instance_id -> submission
    let cdpMap: Record<string, CdpSubmission> = {};
    let cdpSubmitting: Record<string, boolean> = {};

    async function load() {
        try {
            loading = true;
            runId = $page.url.searchParams.get('run_id');
            const filter: Record<string, string> = {};
            if (runId) filter.run_id = runId;
            cards = await api.cards.listAll(filter as any);

            // Load existing CDP submissions if viewing a run
            if (runId) {
                try {
                    cdpMap = await api.cdp.getRunSubmissions(runId);
                } catch { /* no submissions yet */ }
            } else {
                // Load individual submissions for all cards
                await loadAllCdpStatuses();
            }
        } catch (e) {
            console.error(e);
        } finally {
            loading = false;
        }
    }

    async function loadAllCdpStatuses() {
        const results = await Promise.allSettled(
            cards.map(c => api.cdp.getSubmission(c.instance_id).catch(() => null))
        );
        const newMap: Record<string, CdpSubmission> = {};
        for (let i = 0; i < cards.length; i++) {
            const r = results[i];
            if (r.status === 'fulfilled' && r.value) {
                newMap[cards[i].instance_id] = r.value;
            }
        }
        cdpMap = newMap;
    }

    async function submitToCdp(card: Card) {
        cdpSubmitting[card.instance_id] = true;
        try {
            const sub = await api.cdp.submitCard(card.instance_id);
            cdpMap = { ...cdpMap, [card.instance_id]: sub };
        } catch (e) {
            console.error('CDP submit failed:', e);
            alert(`Failed to submit to CardDealerPro: ${(e as Error).message}`);
        } finally {
            cdpSubmitting[card.instance_id] = false;
        }
    }

    async function pollCdp(card: Card) {
        cdpSubmitting[card.instance_id] = true;
        try {
            const sub = await api.cdp.pollCard(card.instance_id);
            cdpMap = { ...cdpMap, [card.instance_id]: sub };
        } catch (e) {
            console.error('CDP poll failed:', e);
        } finally {
            cdpSubmitting[card.instance_id] = false;
        }
    }

    async function notACard(card: Card) {
        const index = cards.findIndex((c) => c.instance_id === card.instance_id);
        cards = cards.filter((c) => c.instance_id !== card.instance_id);
        try {
            await api.cards.hide(card.instance_id);
            if (toastTimer) clearTimeout(toastTimer);
            toast = { card, index };
            toastTimer = setTimeout(() => (toast = null), 5000);
        } catch (e) {
            console.error(e);
            cards = [...cards.slice(0, index), card, ...cards.slice(index)];
        }
    }

    async function undoHide() {
        if (!toast) return;
        const { card, index } = toast;
        toast = null;
        if (toastTimer) clearTimeout(toastTimer);
        try {
            await api.cards.unhide(card.instance_id);
            cards = [...cards.slice(0, index), card, ...cards.slice(index)];
        } catch (e) {
            console.error(e);
        }
    }

    function cdpStatusLabel(sub: CdpSubmission): string {
        switch (sub.status) {
            case 'submitted':   return '⏳ Submitted';
            case 'processing':  return '🔄 Processing';
            case 'identified':  return sub.identified_name
                ? `✓ ${sub.identified_name}` + (sub.suggested_price ? ` · $${sub.suggested_price.toFixed(2)}` : '')
                : '✓ Identified';
            case 'failed':      return '✗ Failed';
            default:            return sub.status;
        }
    }

    onMount(load);
</script>

<div class="header">
    <h1>Cards{runId ? ` — ${runId}` : ''}</h1>
    <div class="header-actions">
        {#if runId && cards.length > 0}
            <a
                class="download-btn"
                href={`/api/v1/cards/download?run_id=${encodeURIComponent(runId)}`}
                download
            >⬇ Download all</a>
        {/if}
    </div>
</div>

{#if loading}
    <p>Loading cards...</p>
{:else if cards.length === 0}
    <p class="empty">No cards found{runId ? ` for run ${runId}` : ''}.</p>
{:else}
    <div class="card-grid">
        {#each cards as card (card.instance_id)}
            {@const sub = cdpMap[card.instance_id]}
            <div class="card-item" class:cdp-identified={sub?.status === 'identified'}>
                <div class="img-wrap">
                    <img src={card.fused_url ?? card.canonical_url ?? ''} alt="Card" />
                    <div class="hover-ids">
                        <button
                            class="id-chip"
                            title="Click to copy instance_id"
                            onclick={() => navigator.clipboard.writeText(card.instance_id)}
                        >{card.instance_id.slice(0, 8)}</button>
                        <span class="run-chip">{card.run_id}</span>
                        <button
                            class="not-card-btn"
                            title="Mark as not a card (hide it)"
                            onclick={() => notACard(card)}
                        >✕ Not a card</button>
                    </div>
                </div>
                <div class="info">
                    <span class="angle">{card.side}</span>
                    <span class="score">{(card.confidence * 100).toFixed(0)}%</span>
                </div>

                <!-- CDP submission status / button -->
                <div class="cdp-row">
                    {#if sub}
                        <span class="cdp-badge cdp-{sub.status}" title={sub.identified_name ?? sub.status}>
                            {cdpStatusLabel(sub)}
                        </span>
                        {#if sub.status === 'submitted' || sub.status === 'processing'}
                            <button
                                class="cdp-poll-btn"
                                disabled={cdpSubmitting[card.instance_id]}
                                onclick={() => pollCdp(card)}
                                title="Refresh status from CardDealerPro"
                            >↻</button>
                        {/if}
                    {:else}
                        <button
                            class="cdp-submit-btn"
                            disabled={cdpSubmitting[card.instance_id]}
                            onclick={() => submitToCdp(card)}
                            title="Submit to CardDealerPro for identification & pricing"
                        >
                            {cdpSubmitting[card.instance_id] ? 'Submitting…' : '📤 Submit to CDP'}
                        </button>
                    {/if}
                </div>
            </div>
        {/each}
    </div>
{/if}

{#if toast}
    <div class="toast" role="status">
        <span>Marked &ldquo;not a card&rdquo;.</span>
        <button class="undo-btn" onclick={undoHide}>Undo</button>
    </div>
{/if}

<style>
    .header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }

    .header-actions {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .download-btn {
        background: #727cf5;
        color: #fff;
        text-decoration: none;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 0.5rem 0.9rem;
        border-radius: 6px;
        white-space: nowrap;
    }

    .download-btn:hover { background: #4e5bd4; }

    .card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
        gap: 1.5rem;
    }

    .card-item {
        background: white;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .card-item:hover { transform: translateY(-4px); box-shadow: 0 6px 16px rgba(0,0,0,0.12); }
    .card-item.cdp-identified { box-shadow: 0 2px 4px rgba(10,207,151,0.2), 0 0 0 1px rgba(10,207,151,0.3); }

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

    .not-card-btn {
        font-size: 0.7rem;
        font-weight: 600;
        color: #fff;
        background: rgba(250,92,124,0.9);
        border: none;
        border-radius: 4px;
        padding: 0.2rem 0.5rem;
        cursor: pointer;
    }

    .not-card-btn:hover { background: #e23b5f; }

    .info {
        padding: 0.4rem 0.5rem 0;
        display: flex;
        justify-content: space-between;
        font-size: 0.75rem;
    }

    .angle { font-weight: bold; color: #727cf5; }
    .score { color: #0acf97; }
    .empty { color: #aaa; margin-top: 2rem; }

    /* CDP row below card info */
    .cdp-row {
        padding: 0.35rem 0.5rem 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.3rem;
    }

    .cdp-submit-btn {
        width: 100%;
        font-size: 0.7rem;
        font-weight: 600;
        color: #fff;
        background: linear-gradient(135deg, #667eea, #764ba2);
        border: none;
        border-radius: 5px;
        padding: 0.3rem 0.5rem;
        cursor: pointer;
        transition: opacity 0.15s;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .cdp-submit-btn:disabled { opacity: 0.6; cursor: default; }
    .cdp-submit-btn:hover:not(:disabled) { opacity: 0.88; }

    .cdp-badge {
        flex: 1;
        font-size: 0.65rem;
        font-weight: 600;
        border-radius: 4px;
        padding: 0.2rem 0.4rem;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .cdp-submitted  { background: #e8eaf6; color: #3949ab; }
    .cdp-processing { background: #fff3e0; color: #e65100; }
    .cdp-identified { background: #e8f5e9; color: #2e7d32; }
    .cdp-failed     { background: #fce4ec; color: #c62828; }

    .cdp-poll-btn {
        background: none;
        border: 1px solid #ccc;
        border-radius: 4px;
        padding: 0.15rem 0.3rem;
        cursor: pointer;
        font-size: 0.8rem;
        color: #666;
        line-height: 1;
        flex-shrink: 0;
    }
    .cdp-poll-btn:hover { background: #f0f0f0; }

    .toast {
        position: fixed;
        bottom: 1.5rem;
        left: 50%;
        transform: translateX(-50%);
        background: #2a2a3c;
        color: #fff;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        display: flex;
        align-items: center;
        gap: 0.8rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        font-size: 0.85rem;
        z-index: 100;
    }

    .undo-btn {
        background: #727cf5;
        color: #fff;
        border: none;
        border-radius: 4px;
        padding: 0.25rem 0.7rem;
        cursor: pointer;
        font-weight: 600;
    }
</style>
