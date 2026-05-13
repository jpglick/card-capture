<script lang="ts">
    import { onMount } from 'svelte';
    import { api } from '$lib/api/client';
    import type { DedupCluster } from '$lib/api/types';

    let clusters: DedupCluster[] = $state([]);
    let loading = true;

    async function load() {
        try {
            loading = true;
            clusters = await api.label.listClusters('pending');
        } finally {
            loading = false;
        }
    }

    async function confirmCluster(clusterId: number, memberIds: string[]) {
        try {
            await api.label.patchCluster(clusterId, {
                status: 'confirmed',
                confirmed: memberIds
            });
            clusters = clusters.filter(c => c.cluster_id !== clusterId);
        } catch (e) {
            alert(e);
        }
    }

    async function rejectCluster(clusterId: number) {
        try {
            await api.label.patchCluster(clusterId, {
                status: 'rejected'
            });
            clusters = clusters.filter(c => c.cluster_id !== clusterId);
        } catch (e) {
            alert(e);
        }
    }

    onMount(load);
</script>

<h1>Dedup Clusters</h1>
<p>Confirm whether these detections are all the same physical card.</p>

{#if loading}
    <p>Loading clusters...</p>
{:else if clusters.length === 0}
    <p>All clusters verified!</p>
{:else}
    <div class="cluster-list">
        {#each clusters as cluster}
            <div class="cluster-card" data-testid="cluster">
                <div class="header">
                    <h3>Cluster #{cluster.cluster_id}</h3>
                    <div class="actions">
                        <button class="confirm" onclick={() => confirmCluster(cluster.cluster_id, cluster.predicted)}>
                            All Same
                        </button>
                        <button class="reject" onclick={() => rejectCluster(cluster.cluster_id)}>
                            Not Same
                        </button>
                    </div>
                </div>

                <div class="thumbs">
                    {#each cluster.member_thumbnails as thumb}
                        <img src={thumb} alt="Member" data-testid="thumb" />
                    {/each}
                </div>
            </div>
        {/each}
    </div>
{/if}

<style>
    .cluster-list {
        display: flex;
        flex-direction: column;
        gap: 2rem;
    }

    .cluster-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

    .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }

    .actions { display: flex; gap: 0.5rem; }

    .thumbs {
        display: flex;
        gap: 0.5rem;
        overflow-x: auto;
        padding-bottom: 0.5rem;
    }

    .thumbs img {
        height: 150px;
        border-radius: 4px;
        border: 1px solid #eee;
    }

    button {
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-weight: bold;
    }

    .confirm { background: #0acf97; color: white; }
    .reject { background: #fa5c7c; color: white; }
</style>
