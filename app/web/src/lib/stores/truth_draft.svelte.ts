import { api } from '../api/client';
import type { LabelTruth } from '../api/types';

const AUTOSAVE_MS = 30_000;

export function createTruthDraft(videoId: string, initial: LabelTruth) {
    let draft = $state(structuredClone(initial));
    let dirty = $state(false);
    let saving = $state(false);
    let lastSaved = $state<Date | null>(null);

    let timer: any = null;

    function start() {
        timer = setInterval(async () => {
            if (!dirty || saving) return;
            await saveNow();
        }, AUTOSAVE_MS);
    }

    function stop() {
        if (timer) clearInterval(timer);
    }

    function mutate(fn: (d: LabelTruth) => void) {
        fn(draft);
        dirty = true;
    }

    async function saveNow() {
        saving = true;
        try {
            await api.label.putTruth(videoId, draft);
            dirty = false;
            lastSaved = new Date();
        } catch (e) {
            console.error('Failed to save truth:', e);
        } finally {
            saving = false;
        }
    }

    return {
        get draft() { return draft; },
        get dirty() { return dirty; },
        get saving() { return saving; },
        get lastSaved() { return lastSaved; },
        start,
        stop,
        mutate,
        saveNow,
    };
}
