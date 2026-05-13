import type { Video, RunSummary, Card } from './types';

const BASE_URL = '/api/v1';

export async function listVideos(): Promise<Video[]> {
    const res = await fetch(`${BASE_URL}/videos`);
    if (!res.ok) throw new Error('Failed to list videos');
    const data = await res.json();
    return data.items;
}

export async function listRuns(videoId?: string): Promise<RunSummary[]> {
    const url = videoId ? `${BASE_URL}/runs?video_id=${videoId}` : `${BASE_URL}/runs`;
    const res = await fetch(url);
    if (!res.ok) throw new Error('Failed to list runs');
    const data = await res.json();
    return data.items;
}

export async function getCards(runId: string): Promise<Card[]> {
    const res = await fetch(`${BASE_URL}/cards?run_id=${runId}`);
    if (!res.ok) throw new Error('Failed to get cards');
    const data = await res.json();
    return data.items;
}
