<script lang="ts">
  interface StageData {
      stage_id: string;
      pct: number;
      detail: string;
  }

  let {
      progressMap = {}
  }: {
      progressMap?: Record<string, StageData>
  } = $props();

  const ALL_STAGES = [
    { id: 'sample', name: 'Video Sampler', phase: 1 },
    { id: 'detect', name: 'Object Det (YOLO)', phase: 1 },
    { id: 'novelty', name: 'Novelty Gate', phase: 1 },
    { id: 'track', name: 'ByteTrack', phase: 2 },
    { id: 'refine', name: 'Entity Warping', phase: 2 },
    { id: 'score', name: 'Quality Scorer', phase: 2 },
    { id: 'resolve', name: 'F/B Resolver', phase: 2 },
    { id: 'fuse', name: 'Image Fusion', phase: 3 },
    { id: 'dedup', name: 'Deduplication', phase: 3 },
    { id: 'store', name: 'Data Export', phase: 3 },
  ];

  const stages = $derived(ALL_STAGES.map((s) => {
      const p = progressMap[s.id];
      let status = 'pending';
      let pct = 0;
      let detail = '';
      
      if (p) {
          pct = p.pct;
          detail = p.detail;
          status = pct >= 100 ? 'complete' : 'running';
      }
      return { ...s, status, pct, detail };
  }));

</script>

<div class="sparkline-grid">
  {#each stages as stage}
    <div class="card status-{stage.status} phase-{stage.phase}">
      <div class="phase-label">PHASE {stage.phase}</div>
      <div class="title" title={stage.detail}>{stage.name}</div>
      
      {#if stage.status !== 'pending'}
        <div class="progress-track">
          <div class="progress-fill" style="width: {stage.pct}%"></div>
        </div>
      {/if}

      <div class="status-label">
        {#if stage.status === 'complete'}
          DONE
        {:else if stage.status === 'running'}
          RUNNING...
        {:else}
          PENDING
        {/if}
      </div>
    </div>
  {/each}
</div>

<style>
  .sparkline-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    margin-bottom: 2rem;
    padding: 1rem 0;
  }

  .card {
    background-color: #1a1b1e;
    border-radius: 8px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 110px;
    border: 1px solid transparent;
    transition: all 0.2s ease;
  }

  /* Status Colors */
  /* Phase 1: Blue */
  .card.phase-1 .progress-fill { background-color: #7497d5; }
  .card.phase-1.status-running { border-color: #7497d5; box-shadow: 0 0 8px rgba(116, 151, 213, 0.4); }
  
  /* Phase 2: Green */
  .card.phase-2 .progress-fill { background-color: #72b866; }
  .card.phase-2.status-running { border-color: #72b866; box-shadow: 0 0 8px rgba(114, 184, 102, 0.4); }

  /* Phase 3: Orange */
  .card.phase-3 .progress-fill { background-color: #de8953; }
  .card.phase-3.status-running { border-color: #de8953; box-shadow: 0 0 8px rgba(222, 137, 83, 0.4); }

  /* Pending styling */
  .card.status-pending {
    opacity: 0.7;
  }

  .phase-label {
    font-size: 0.7rem;
    font-weight: 700;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
  }

  .title {
    font-size: 1rem;
    font-weight: 500;
    color: #e0e0e0;
    margin-bottom: 12px;
  }

  .progress-track {
    height: 6px;
    background-color: #333;
    border-radius: 3px;
    margin-top: auto;
    margin-bottom: 12px;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s ease;
  }

  .status-label {
    font-size: 0.75rem;
    color: #999;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
</style>
