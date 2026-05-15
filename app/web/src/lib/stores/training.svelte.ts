import { api } from '$lib/api/client';

export interface TrainingStats {
  pending: { presence: number; fb: number; corners: number };
  accuracy: Record<string, number | null>;
  history: Array<{ model: string; accuracy: number; created_at: string }>;
}

export function createTrainingStore() {
  let stats = $state<TrainingStats | null>(null);
  let loading = $state(true);

  async function refresh() {
    loading = true;
    try {
      const r = await api.get('/training/stats');
      stats = await r.json();
    } finally {
      loading = false;
    }
  }

  return { get stats() { return stats; }, get loading() { return loading; }, refresh };
}
