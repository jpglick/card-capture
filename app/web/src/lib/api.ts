import type { 
    Video, VideoCreate, RunSummary, RunDetail, RunTelemetry, 
    Card, CardDetail, CardFilter, CardBulkAction,
    ConfigPreset, ConfigPlayground,
    LabelTruth, LabelFBNext, LabelFB, DedupCluster,
    DatasetSummary, TrainingJobSummary, TrainingJobDetail,
    Baseline, RegressionRun, RegressionCompare
} from './api/types';

const BASE_URL = '/api/v1';

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
    const res = await fetch(`${BASE_URL}${path}`, {
        ...options,
        headers: {
            'Content-Type': 'application/json',
            ...options.headers,
        },
    });
    if (res.status === 204) return null as any;
    if (!res.ok) {
        const error = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(error.detail || res.statusText);
    }
    return res.json();
}

export const api = {
    videos: {
        list: () => request<Video[]>('/videos'),
        get: (id: string) => request<Video>(`/videos/${id}`),
        create: (payload: VideoCreate) => request<Video>('/videos', { method: 'POST', body: JSON.stringify(payload) }),
        delete: (id: string) => request<void>(`/videos/${id}`, { method: 'DELETE' }),
        process: (id: string, config: any = {}) => request<RunSummary>(`/videos/${id}/process`, { method: 'POST', body: JSON.stringify(config) }),
    },
    runs: {
        list: (videoId?: string) => request<RunSummary[]>(videoId ? `/runs?video_id=${videoId}` : '/runs'),
        get: (id: string) => request<RunDetail>(`/runs/${id}`),
        telemetry: (id: string) => request<RunTelemetry>(`/runs/${id}/telemetry`),
        cards: (id: string) => request<Card[]>(`/runs/${id}/cards`),
    },
    cards: {
        list: (filter: CardFilter = {}) => {
            const params = new URLSearchParams();
            Object.entries(filter).forEach(([k, v]) => {
                if (v !== undefined) params.append(k, v.toString());
            });
            return request<{ total: number; page: number; page_size: number; items: Card[] }>(`/cards?${params}`);
        },
        get: (id: string) => request<CardDetail>(`/cards/${id}`),
        update: (id: string, payload: any) => request<CardDetail>(`/cards/${id}`, { method: 'PATCH', body: JSON.stringify(payload) }),
        bulk: (payload: CardBulkAction) => request<{ updated: number; failed: string[] }>('/cards/bulk', { method: 'POST', body: JSON.stringify(payload) }),
    },
    label: {
        getTruth: (videoId: string) => request<LabelTruth>(`/label/truth/${videoId}`),
        putTruth: (videoId: string, payload: LabelTruth) => request<void>(`/label/truth/${videoId}`, { method: 'PUT', body: JSON.stringify(payload) }),
        nextFB: () => request<LabelFBNext | null>('/label/fb/next'),
        submitFB: (payload: LabelFB) => request<any>('/label/fb', { method: 'POST', body: JSON.stringify(payload) }),
        clusters: (status?: string) => request<DedupCluster[]>(status ? `/label/clusters?status=${status}` : '/label/clusters'),
        patchCluster: (id: number, payload: any) => request<void>(`/label/clusters/${id}`, { method: 'PATCH', body: JSON.stringify(payload) }),
    },
    training: {
        datasets: () => request<DatasetSummary[]>('/training/datasets'),
        retrain: (modelName: string, body: { epochs: number; learning_rate: number }) => 
            request<TrainingJobSummary>(`/training/retrain/${modelName}`, { method: 'POST', body: JSON.stringify(body) }),
        getJob: (jobId: string) => request<TrainingJobDetail>(`/training/jobs/${jobId}`),
    },
    regression: {
        baselines: () => request<Baseline[]>('/regression/baselines'),
        createBaseline: (payload: { name: string; code_sha: string }) => request<Baseline>('/regression/baselines', { method: 'POST', body: JSON.stringify(payload) }),
        run: (payload: { baseline_id: number; video_subset?: string[] }) => request<RegressionRun>('/regression/run', { method: 'POST', body: JSON.stringify(payload) }),
        getRun: (id: number) => request<any>(`/regression/runs/${id}`),
        compare: (a: string, b: string) => request<RegressionCompare>(`/regression/compare?a=${a}&b=${b}`),
    },
    config: {
        presets: () => request<ConfigPreset[]>('/config/presets'),
        playground: (runId: string) => request<ConfigPlayground>(`/config/playground/${runId}`),
    }
};
