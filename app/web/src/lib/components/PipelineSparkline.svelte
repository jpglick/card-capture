<script lang="ts">
  // PipelineSparkline.svelte
  
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
    { id: 'sample', name: 'AdaptivePresenceSampler' },
    { id: 'detect', name: 'Feature Extraction (YOLO)' },
    { id: 'novelty', name: 'Background Novelty' },
    { id: 'track', name: 'ByteTrack Integration' },
    { id: 'refine', name: 'Entity Cropping & Warping' },
    { id: 'score', name: 'Quality Scoring' },
    { id: 'resolve', name: 'Session Aggregation' },
    { id: 'fuse', name: 'Lighting-Diverse Fusion' },
    { id: 'dedup', name: 'Global Deduplication' },
    { id: 'store', name: 'Data Export & Store' },
  ];

  const stages = $derived(ALL_STAGES.map((s, index) => {
      const p = progressMap[s.id];
      let status = 'pending';
      let detail = 'Waiting for upstream...';
      let pct = 0;
      
      if (p) {
          pct = p.pct;
          detail = p.detail;
          status = pct >= 100 ? 'complete' : 'running';
      }
      return { ...s, status, detail, pct, phase: Math.floor(index / 5) + 1 };
  }));

  let hoveredStage = $state<any>(null);

</script>

<div class="dashboard">
  <header>
    <h2>Pipeline Telemetry</h2>
  </header>

  <div class="sparkline-grid">
    {#each stages as stage}
      <!-- svelte-ignore a11y_no_static_element_interactions -->
      <div 
        class="cell {stage.status}" 
        onmouseenter={() => hoveredStage = stage}
        onmouseleave={() => hoveredStage = null}
      >
        </div>
    {/each}
  </div>

  <div class="tooltip-area">
    {#if hoveredStage}
      <div class="tooltip">
        <h3>{hoveredStage.name}</h3>
        <p><strong>Status:</strong> <span class="status-text {hoveredStage.status}">{hoveredStage.status} ({hoveredStage.pct}%)</span></p>
        <p><strong>Detail:</strong> {hoveredStage.detail}</p>
        <p><strong>Phase Group:</strong> {hoveredStage.phase}</p>
      </div>
    {:else}
      <p class="placeholder">Hover over a grid cell to inspect pipeline span data.</p>
    {/if}
  </div>
</div>

<style>
  .dashboard {
    font-family: system-ui, -apple-system, sans-serif;
    max-width: 480px;
    margin: 0;
    padding: 1.5rem;
    border-radius: 8px;
    background: #1e1e2d;
    color: #c9d1d9;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin-bottom: 1.5rem;
  }

  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
  }

  header h2 {
    margin: 0;
    font-size: 1.2rem;
    color: #e6edf3;
  }

  /* Sparkline Grid Layout */
  .sparkline-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin-bottom: 1.5rem;
  }

  /* Base Cell Styling */
  .cell {
    aspect-ratio: 1;
    border-radius: 6px;
    background: #374151;
    border: 1px solid #4b5563;
    transition: all 0.2s ease;
    cursor: crosshair;
  }

  .cell:hover {
    transform: scale(1.1);
    border-color: #9ca3af;
  }

  /* Status Animations */
  .cell.pending {
    background: transparent;
  }

  .cell.running {
    background: #727cf5;
    border-color: #9098f8;
    animation: pulse 1.5s infinite;
  }

  .cell.complete {
    background: #0acf97; /* Green fill */
    border-color: #34d399;
  }

  @keyframes pulse {
    0% { opacity: 0.6; }
    50% { opacity: 1; box-shadow: 0 0 8px #727cf5; }
    100% { opacity: 0.6; }
  }

  /* Tooltip Styling */
  .tooltip-area {
    min-height: 140px;
    padding: 1rem;
    background: #28283a;
    border-radius: 6px;
    border: 1px solid #37374a;
  }

  .tooltip h3 {
    margin: 0 0 0.5rem 0;
    font-size: 1.1rem;
    color: #e6edf3;
  }

  .tooltip p {
    margin: 0.35rem 0;
    font-size: 0.85rem;
    color: #9ca3af;
  }

  .status-text {
    text-transform: capitalize;
    font-weight: bold;
  }

  .status-text.running { color: #9098f8; }
  .status-text.complete { color: #0acf97; }
  .status-text.pending { color: #6b7280; }
  
  .placeholder {
    color: #6b7280;
    font-style: italic;
    text-align: center;
    margin-top: 1.5rem;
    font-size: 0.85rem;
  }
</style>
