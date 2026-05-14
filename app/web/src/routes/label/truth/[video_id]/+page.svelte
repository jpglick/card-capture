<script lang="ts">
    import { onMount, onDestroy } from 'svelte';
    import { page } from '$app/state';
    import { api } from '$lib/api/client';
    import { createTruthDraft } from '$lib/stores/truth_draft.svelte';
    import Filmstrip from '$lib/components/Filmstrip.svelte';
    import VerdictButtons from '$lib/components/VerdictButtons.svelte';
    import Hotkeys from '$lib/components/Hotkeys.svelte';
    import type { Card, LabelTruth } from '$lib/api/types';

    const videoId = page.params.video_id;
    let instances: (Card & { verdict?: 'real' | 'phantom' })[] = $state([]);
    let selectedId = $state<string | null>(null);
    let loading = $state(true);
    let loadError = $state<string | null>(null);
    let draftStore: any = $state(null);

    async function load() {
        try {
            loading = true;
            loadError = null;
            const [video, truth, cards] = await Promise.all([
                api.videos.get(videoId),
                api.label.getTruth(videoId),
                api.cards.listAll({ video_id: videoId })
            ]);

            const initialTruth: LabelTruth = truth || {
                video_id: videoId,
                schema_version: 1,
                expected_cards: []
            };

            draftStore = createTruthDraft(videoId, initialTruth);
            draftStore.start();

            // Map existing truth to instances
            instances = cards.map(c => {
                const gt = initialTruth.expected_cards.find(gc => gc.card_id === c.instance_id);
                return {
                    ...c,
                    verdict: gt ? 'real' : undefined
                };
            });

            if (instances.length > 0) {
                selectedId = instances[0].instance_id;
            }
        } finally {
        } catch (e: any) {
            loadError = e?.message ?? String(e);
        } finally {
            loading = false;
        }
    }

    const selectedInstance = $derived(instances.find(i => i.instance_id === selectedId));

    function handleVerdict(verdict: 'real' | 'phantom' | 'flip') {
        if (!selectedId) return;

        instances = instances.map(i => {
            if (i.instance_id === selectedId) {
                if (verdict === 'flip') {
                    return { ...i, angle: i.angle === 'Front' ? 'Back' : 'Front' };
                }
                return { ...i, verdict };
            }
            return i;
        });

        // Update draft
        draftStore.mutate((d: LabelTruth) => {
            if (verdict === 'real') {
                if (!d.expected_cards.find(c => c.card_id === selectedId)) {
                    const inst = instances.find(i => i.instance_id === selectedId)!;
                    d.expected_cards.push({
                        card_id: inst.instance_id,
                        front_present: inst.angle === 'Front',
                        back_present: inst.angle === 'Back',
                        physical_card_key: inst.instance_id,
                        is_foil: false
                    });
                }
            } else if (verdict === 'phantom') {
                d.expected_cards = d.expected_cards.filter(c => c.card_id !== selectedId);
            }
        });

        // Auto-advance
        const idx = instances.findIndex(i => i.instance_id === selectedId);
        if (idx < instances.length - 1) {
            selectedId = instances[idx + 1].instance_id;
        }
    }

    onMount(load);
    onDestroy(() => draftStore?.stop());

    const keys = {
        'r': () => handleVerdict('real'),
        'p': () => handleVerdict('phantom'),
        'f': () => handleVerdict('flip'),
        'arrowright': () => {
             const idx = instances.findIndex(i => i.instance_id === selectedId);
             if (idx < instances.length - 1) selectedId = instances[idx + 1].instance_id;
        },
        'arrowleft': () => {
             const idx = instances.findIndex(i => i.instance_id === selectedId);
             if (idx > 0) selectedId = instances[idx - 1].instance_id;
        }
    };
</script>

<Hotkeys {keys}>
    <div class="editor-container">
        <header>
            <h1>Truth Editor: {videoId}</h1>
            {#if draftStore}
                <div class="save-status">
                    {#if draftStore.saving}Saving...{:else if draftStore.dirty}Unsaved changes{:else}Saved{/if}
                </div>
            {/if}
        </header>

        {#if loading}
            <p>Loading...</p>
        {:else if loadError}
            <p class="error">Error loading instances: {loadError}</p>
        {:else}
            <div class="main-layout">
                <div class="preview-area">
                    {#if selectedInstance}
                        <div class="large-preview">
                            <img src={selectedInstance.fused_image_url} alt="Preview" />
                            <div class="meta">
                                <span>{selectedInstance.instance_id}</span>
                                <span>{selectedInstance.angle}</span>
                            </div>
                        </div>
                    {:else}
                        <p>No instances detected.</p>
                    {/if}

                    <VerdictButtons onVerdict={handleVerdict} />
                </div>

                <div class="sidebar">
                    <Filmstrip 
                        items={instances} 
                        {selectedId} 
                        onSelect={(id) => selectedId = id} 
                    />
                </div>
            </div>
        {/if}
    </div>
</Hotkeys>

<style>
    .editor-container {
        display: flex;
        flex-direction: column;
        height: 100%;
    }

    header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 1rem;
        border-bottom: 1px solid #eee;
    }

    .save-status { font-size: 0.8rem; color: #666; }

    .main-layout {
        display: grid;
        grid-template-columns: 1fr 300px;
        gap: 2rem;
        flex: 1;
        padding-top: 1rem;
    }

    .preview-area {
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    .large-preview {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        max-width: 500px;
        width: 100%;
    }

    .large-preview img {
        width: 100%;
        height: 700px;
        object-fit: contain;
        background: #f8f9fa;
    }

    .meta {
        display: flex;
        justify-content: space-between;
        padding-top: 1rem;
        font-family: monospace;
        font-size: 0.9rem;
    }

    .sidebar {
        background: #1e1e2d;
        border-radius: 8px;
        padding: 1rem;
        overflow-y: auto;
    }
</style>
