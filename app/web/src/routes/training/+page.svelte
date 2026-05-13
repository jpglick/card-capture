<script lang="ts">
    import { onMount, onDestroy } from 'svelte';
    import { api } from '$lib/api/client';
    import type { DatasetSummary, TrainingJobSummary, TrainingJobDetail } from '$lib/api/types';

    let datasets: DatasetSummary[] = [];
    let activeJob: TrainingJobDetail | null = null;
    let loading = true;
    let error: string | null = null;
    let pollInterval: any;

    async function load() {
        try {
            loading = true;
            datasets = await api.training.listDatasets();
        } catch (e: any) {
            error = e.message;
        } finally {
            loading = false;
        }
    }

    async function startRetrain(modelName: string) {
        try {
            const job = await api.training.retrain(modelName, { epochs: 20, learning_rate: 0.001 });
            activeJob = { ...job, progress: undefined };
            startPolling(job.job_id);
        } catch (e: any) {
            alert(e.message);
        }
    }

    function startPolling(jobId: string) {
        if (pollInterval) clearInterval(pollInterval);
        pollInterval = setInterval(async () => {
            try {
                const job = await api.training.getJob(jobId);
                activeJob = job;
                if (job.status === 'completed' || job.status === 'failed') {
                    clearInterval(pollInterval);
                    load(); // Refresh dataset last_updated
                }
            } catch (e) {
                clearInterval(pollInterval);
            }
        }, 1000);
    }

    onMount(load);
    onDestroy(() => clearInterval(pollInterval));
</script>

<h1>Model Training</h1>

<div class="datasets-grid">
    {#each datasets as ds}
        <div class="dataset-card">
            <h2>{ds.model_name}</h2>
            <div class="stat">
                <span class="label">Total Labels:</span>
                <span class="value">{ds.total_labels}</span>
            </div>
            <div class="distribution">
                {#each Object.entries(ds.class_distribution) as [cls, count]}
                    <div class="class-item">
                        <span class="cls">{cls}:</span>
                        <span class="cnt">{count}</span>
                    </div>
                {/each}
            </div>
            <div class="meta">Last updated: {new Date(ds.last_updated).toLocaleString()}</div>
            
            <button 
                class="retrain-btn" 
                disabled={activeJob?.status === 'running'}
                onclick={() => startRetrain(ds.model_name)}
            >
                Start Retraining
            </button>
        </div>
    {/each}
</div>

{#if activeJob}
    <div class="active-job">
        <h3>Training Job: {activeJob.job_id}</h3>
        <div class="job-status">
            Status: <span class="status {activeJob.status}">{activeJob.status}</span>
        </div>
        
        {#if activeJob.progress}
            <div class="progress-bar">
                <div class="fill" style="width: {(activeJob.progress.epoch / activeJob.progress.total_epochs) * 100}%"></div>
            </div>
            <div class="progress-text">
                Epoch {activeJob.progress.epoch}/{activeJob.progress.total_epochs} 
                - Val Accuracy: {(activeJob.progress.val_accuracy * 100).toFixed(1)}%
            </div>
        {/if}

        {#if activeJob.status === 'completed'}
            <p class="success">Training completed successfully!</p>
        {:else if activeJob.status === 'failed'}
            <p class="error">Training failed: {activeJob.progress?.error || 'Unknown error'}</p>
        {/if}
    </div>
{/if}

<style>
    .datasets-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2rem;
        margin-bottom: 3rem;
    }

    .dataset-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }

    h2 { margin-top: 0; font-size: 1.25rem; color: #333; }

    .stat { margin-bottom: 1rem; }
    .label { color: #666; margin-right: 0.5rem; }
    .value { font-weight: bold; font-size: 1.1rem; }

    .distribution {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 1.5rem;
    }

    .class-item {
        background: #f1f3f9;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
    }

    .meta { font-size: 0.75rem; color: #999; margin-bottom: 1.5rem; }

    .retrain-btn {
        width: 100%;
        padding: 0.75rem;
        background: #727cf5;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        cursor: pointer;
    }

    .retrain-btn:disabled { background: #cfd4ed; cursor: not-allowed; }

    .active-job {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #727cf5;
    }

    .status { font-weight: bold; text-transform: uppercase; }
    .status.running { color: #727cf5; }
    .status.completed { color: #0acf97; }
    .status.failed { color: #fa5c7c; }

    .progress-bar {
        height: 12px;
        background: #eef2ff;
        border-radius: 6px;
        overflow: hidden;
        margin: 1.5rem 0 0.5rem;
    }

    .fill {
        height: 100%;
        background: #727cf5;
        transition: width 0.3s ease;
    }

    .progress-text { font-size: 0.875rem; color: #666; }
    .success { color: #0acf97; font-weight: bold; margin-top: 1rem; }
    .error { color: #fa5c7c; font-weight: bold; margin-top: 1rem; }
</style>
