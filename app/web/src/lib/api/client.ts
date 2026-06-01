import type * as T from './types';

const BASE = '/api/v1';

class ApiError extends Error {
    constructor(public status: number, public detail: string) {
        super(`API ${status}: ${detail}`);
    }
}

async function req<T>(method: string, path: string, body?: unknown): Promise<T> {
    const r = await fetch(`${BASE}${path}`, {
        method,
        headers: body ? { 'content-type': 'application/json' } : {},
        body: body ? JSON.stringify(body) : undefined,
    });
    if (!r.ok) {
        const detail = await r.text().catch(() => '');
        throw new ApiError(r.status, detail);
    }
    if (r.status === 204) return undefined as unknown as T;
    return r.json();
}

export const api = {
    videos: {
        list: () => req<T.Video[]>('GET', '/videos'),
        get: (id: string) => req<T.Video>('GET', `/videos/${id}`),
        create: (body: T.VideoCreate) => req<T.Video>('POST', '/videos', body),
        upload: (file: File, onProgress?: (pct: number) => void): Promise<T.Video> => {
            return new Promise((resolve, reject) => {
                const xhr = new XMLHttpRequest();
                const form = new FormData();
                form.append('file', file);
                xhr.upload.addEventListener('progress', (e) => {
                    if (e.lengthComputable && onProgress) {
                        onProgress(Math.round((e.loaded / e.total) * 100));
                    }
                });
                xhr.addEventListener('load', () => {
                    if (xhr.status >= 200 && xhr.status < 300) {
                        resolve(JSON.parse(xhr.responseText));
                    } else {
                        reject(new ApiError(xhr.status, xhr.responseText));
                    }
                });
                xhr.addEventListener('error', () => reject(new ApiError(0, 'Network error')));
                xhr.open('POST', `${BASE}/videos/upload`);
                xhr.send(form);
            });
        },
        process: (videoId: string) => req<T.RunSummary>('POST', `/videos/${videoId}/process`),
        reset: (videoId: string) => req<{ video_id: string; status: string }>('POST', `/videos/${videoId}/reset`),
        delete: (id: string) => req<void>('DELETE', `/videos/${id}`),
    },
    runs: {
        list: (videoId?: string) => {
            const path = videoId ? `/runs?video_id=${videoId}` : '/runs';
            return req<T.RunSummary[]>('GET', path);
        },
        detail: (id: string) => req<T.RunDetail>('GET', `/runs/${id}`),
        cards: (id: string) => req<T.RunCardSummary[]>('GET', `/runs/${id}/cards`),
        events: (id: string) => req<T.RunEvent[]>('GET', `/runs/${id}/events`),
        telemetry: (id: string) => req<T.RunTelemetry>('GET', `/runs/${id}/telemetry`),
        rejection: (id: string) => req<T.RunRejection[]>('GET', `/runs/${id}/rejection_log`),
        hardCases: (id: string) => req<T.RunHardCase[]>('GET', `/runs/${id}/hard_cases`),
        resources: (id: string) => req<T.RunResources>('GET', `/runs/${id}/resources`),
    },
    cards: {
        list: (filter: T.CardFilter) => {
            const params = new URLSearchParams();
            Object.entries(filter).forEach(([k, v]) => {
                if (v !== undefined) params.append(k, v.toString());
            });
            return req<T.PaginatedCards>('GET', `/cards?${params.toString()}`);
        },
        listAll: async (filter: T.CardFilter): Promise<T.Card[]> => {
            const params = new URLSearchParams();
            Object.entries(filter).forEach(([k, v]) => {
                if (v !== undefined) params.append(k, v.toString());
            });
            const r = await req<T.PaginatedCards>('GET', `/cards?${params.toString()}`);
            return r.items;
        },
        detail: (id: string) => req<T.CardDetail>('GET', `/cards/${id}`),
        update: (id: string, body: Partial<T.Card>) => req<T.Card>('PATCH', `/cards/${id}`, body),
        bulk: (body: T.CardBulkAction) => req<{ updated: number }>('POST', '/cards/bulk', body),
    },
    label: {
        getTruth: (videoId: string) => req<T.LabelTruth | null>('GET', `/label/truth/${videoId}`),
        putTruth: (videoId: string, body: T.LabelTruth) => req<void>('PUT', `/label/truth/${videoId}`, body),
        nextFB: () => req<T.LabelFBNext | {}>('GET', '/label/fb/next'),
        postFB: (body: T.LabelFB) => req<{ label_id: number }>('POST', '/label/fb', body),
        listClusters: (status?: string) => {
            const path = status ? `/label/clusters?status=${status}` : '/label/clusters';
            return req<T.DedupCluster[]>('GET', path);
        },
        patchCluster: (id: number, body: { status?: string, confirmed?: string[] }) => 
            req<void>('PATCH', `/label/clusters/${id}`, body),
    },
    training: {
        listDatasets: () => req<T.DatasetSummary[]>('GET', '/training/datasets'),
        retrain: (modelName: string, body: { epochs?: number, learning_rate?: number }) =>
            req<T.TrainingJobSummary>('POST', `/training/retrain/${modelName}`, body),
        getJob: (id: string) => req<T.TrainingJobDetail>('GET', `/training/jobs/${id}`),
        getStats: () => req<any>('GET', '/training/stats'),
        nextPresence: () => fetch(`${BASE}/training/presence/next`),
        labelPresence: (body: { sample_id: number; label: string }) =>
            fetch(`${BASE}/training/presence/label`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) }),
        nextCorner: () => fetch(`${BASE}/training/corners/next`),
        labelCorner: (body: { sample_id: number; label: string; corrected_corners?: string }) =>
            fetch(`${BASE}/training/corners/label`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) }),
        startBenchmark: (n: number) => req<{ job_id: string }>('POST', '/training/benchmark', { n }),
        getBenchmark: (id: string) => req<any>('GET', `/training/benchmark/${id}`),
    },
    regression: {
        listBaselines: () => req<T.Baseline[]>('GET', '/regression/baselines'),
        compare: (a: string, b: string) => req<T.RegressionCompare>('GET', `/regression/compare?a=${a}&b=${b}`),
    },
    config: {
        listPresets: () => req<T.ConfigPreset[]>('GET', '/config/presets'),
        createPreset: (payload: T.ConfigPreset) => req<T.ConfigPreset>('POST', '/config/presets', payload),
        getPlayground: (runId: string) => req<T.ConfigPlayground>('GET', `/config/playground/${runId}`),
        getPipeline: () => req<Record<string, number>>('GET', '/config/pipeline'),
        patchPipeline: (body: Record<string, number>) => req<Record<string, number>>('PATCH', '/config/pipeline', body),
    },
    batch: {
        create: (video_ids: string[], config_preset?: string) =>
            req<{ batch_id: string }>('POST', '/runs/batch', { video_ids, config_preset }),
        status: (batch_id: string) => req<T.BatchStatus>('GET', `/runs/batch/${batch_id}`),
    },
};
