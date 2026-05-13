import { api } from '../api/client';
import type { LabelFBNext } from '../api/types';

export function createFBQueue() {
    let queue = $state<LabelFBNext[]>([]);
    let labeledCount = $state(0);
    let history = $state<{ instance_id: string, side: string }[]>([]);
    let loading = $state(false);

    async function fetchNext() {
        if (loading || queue.length >= 3) return;
        loading = true;
        try {
            const next = await api.label.nextFB();
            if ('instance_id' in next) {
                queue.push(next as LabelFBNext);
            }
        } finally {
            loading = false;
        }
    }

    async function submitLabel(side: 'front' | 'back' | 'uncertain') {
        const item = queue.shift();
        if (!item) return;

        try {
            await api.label.postFB({
                instance_id: item.instance_id,
                frame_index: item.frame_index,
                side
            });
            labeledCount++;
            history.push({ instance_id: item.instance_id, side });
            if (history.length > 50) history.shift();
            
            // Prefetch more
            fetchNext();
            fetchNext();
        } catch (e) {
            console.error('Failed to submit label:', e);
            queue.unshift(item); // Put back
        }
    }

    return {
        get queue() { return queue; },
        get labeledCount() { return labeledCount; },
        get current() { return queue[0]; },
        fetchNext,
        submitLabel
    };
}
