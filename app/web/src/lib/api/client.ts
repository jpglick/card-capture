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
        process: (videoId: string) => req<T.RunSummary>('POST', `/videos/${videoId}/process`),
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
    },
    cards: {
        list: (filter: T.CardFilter) => {
            const params = new URLSearchParams();
            Object.entries(filter).forEach(([k, v]) => {
                if (v !== undefined) params.append(k, v.toString());
            });
            return req<T.Card[]>('GET', `/cards?${params.toString()}`);
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
    },
    regression: {
        listBaselines: () => req<T.Baseline[]>('GET', '/regression/baselines'),
        compare: (a: string, b: string) => req<T.RegressionCompare>('GET', `/regression/compare?a=${a}&b=${b}`),
    },
    config: {
        listPresets: () => req<T.ConfigPreset[]>('GET', '/config/presets'),
        createPreset: (payload: T.ConfigPreset) => req<T.ConfigPreset>('POST', '/config/presets', payload),
        getPlayground: (runId: string) => req<T.ConfigPlayground>('GET', `/config/playground/${runId}`),
    }
};
