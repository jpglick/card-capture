import { writable } from 'svelte/store';
import type { SSEEvent } from './types';

export function createSSEStore(runId: string) {
    const { subscribe, set, update } = writable<SSEEvent[]>([]);

    let eventSource: EventSource | null = null;

    function connect() {
        if (eventSource) return;

        eventSource = new EventSource(`/api/v1/events/subscribe?run_id=${runId}`);

        eventSource.onmessage = (event) => {
            const data: SSEEvent = JSON.parse(event.data);
            update(events => [...events, data]);
        };

        eventSource.onerror = (err) => {
            console.error('SSE error:', err);
            eventSource?.close();
            eventSource = null;
        };
    }

    function disconnect() {
        eventSource?.close();
        eventSource = null;
    }

    return {
        subscribe,
        connect,
        disconnect,
        clear: () => set([])
    };
}
