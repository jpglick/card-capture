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

    // The ordered stages of our pipeline
    const STAGE_ORDER = [
        "sample", "detect", "novelty", "track", "refine",
        "score", "resolve", "fuse", "dedup", "store"
    ];

    const STAGE_COLORS: Record<string, string> = {
        sample:  '#adb5bd',
        detect:  '#6366f1',
        novelty: '#10b981',
        track:   '#f59e0b',
        refine:  '#ef4444',
        score:   '#3b82f6',
        resolve: '#8b5cf6',
        fuse:    '#ec4899',
        dedup:   '#f97316',
        store:   '#06b6d4',
    };

    function stageColor(name: string): string {
        return STAGE_COLORS[name] ?? '#6c757d';
    }

    // We only display stages that have reported some progress
    const activeStages = $derived(
        STAGE_ORDER.map(s => progressMap[s]).filter(Boolean) as StageData[]
    );
</script>

<div class="waterfall-wrap">
    {#if activeStages.length === 0}
        <div class="empty-state">Waiting for pipeline stages to start...</div>
    {:else}
        {#each activeStages as stage}
            <div class="stage-row">
                <div class="stage-info">
                    <span class="stage-name" style="color: {stageColor(stage.stage_id)}">{stage.stage_id}</span>
                    <span class="stage-detail">{stage.detail}</span>
                    <span class="stage-pct">{stage.pct}%</span>
                </div>
                <div class="bar-bg">
                    <div class="bar-fill" style="width: {stage.pct}%; background-color: {stageColor(stage.stage_id)}"></div>
                </div>
            </div>
        {/each}
    {/if}
</div>

<style>
    .waterfall-wrap {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }
    .empty-state {
        color: #adb5bd;
        font-style: italic;
        text-align: center;
        padding: 1rem 0;
    }
    .stage-row {
        margin-bottom: 0.75rem;
    }
    .stage-row:last-child {
        margin-bottom: 0;
    }
    .stage-info {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 0.3rem;
        font-size: 0.85rem;
    }
    .stage-name {
        font-weight: 700;
        text-transform: capitalize;
        width: 80px;
    }
    .stage-detail {
        color: #6c757d;
        flex: 1;
        text-align: left;
        padding-left: 1rem;
        font-size: 0.8rem;
    }
    .stage-pct {
        font-weight: 700;
        color: #313a46;
        font-family: monospace;
    }
    .bar-bg {
        height: 6px;
        background: #f1f3fa;
        border-radius: 3px;
        overflow: hidden;
    }
    .bar-fill {
        height: 100%;
        transition: width 0.2s ease-out;
    }
</style>
