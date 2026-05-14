<script lang="ts">
    import { onMount } from 'svelte';
    import { api } from '$lib/api/client';
    import type { Video } from '$lib/api/types';
    import QueueCard from '$lib/components/QueueCard.svelte';

    let videos: Video[] = [];
    let loading = true;
    let error: string | null = null;
    let dragging = false;

    async function loadVideos() {
        try {
            loading = true;
            videos = await api.videos.list();
        } catch (e: any) {
            error = e.message;
        } finally {
            loading = false;
        }
    }

    async function registerVideo(filename: string) {
        try {
            await api.videos.create({ filename });
            await loadVideos();
        } catch (e: any) {
            error = `Failed to add video: ${e.message}`;
        }
    }

    function onDragOver(e: DragEvent) {
        e.preventDefault();
        dragging = true;
    }

    function onDragLeave() {
        dragging = false;
    }

    async function onDrop(e: DragEvent) {
        e.preventDefault();
        dragging = false;
        const files = e.dataTransfer?.files;
        if (!files?.length) return;
        for (const file of Array.from(files)) {
            await registerVideo(file.name);
        }
    }

    async function onFileInput(e: Event) {
        const input = e.target as HTMLInputElement;
        if (!input.files?.length) return;
        for (const file of Array.from(input.files)) {
            await registerVideo(file.name);
        }
        input.value = '';
    }

    onMount(loadVideos);
</script>

<div class="page-header">
    <h1>Videos</h1>
    <label class="add-btn">
        + Add Videos
        <input type="file" accept="video/*" multiple on:change={onFileInput} style="display:none" />
    </label>
</div>

{#if error}
    <div class="error-banner">{error} <button on:click={() => error = null}>✕</button></div>
{/if}

<!-- Drag-drop dropzone -->
<div
    class="dropzone"
    class:active={dragging}
    on:dragover={onDragOver}
    on:dragleave={onDragLeave}
    on:drop={onDrop}
    role="region"
    aria-label="Drop video files here"
>
    {#if dragging}
        <span class="drop-hint">Release to add videos</span>
    {:else}
        <span class="drop-hint">Drag & drop video files here, or use the button above</span>
    {/if}
</div>

<!-- Queue -->
{#if loading}
    <p class="loading">Loading…</p>
{:else if videos.length === 0}
    <p class="empty">No videos yet. Add one above.</p>
{:else}
    <div class="queue-list">
        {#each videos as video (video.video_id)}
            <QueueCard {video} />
        {/each}
    </div>
{/if}

<style>
    .page-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
    }

    h1 { margin: 0; }

    .add-btn {
        background: #727cf5;
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 8px;
        cursor: pointer;
        font-weight: 600;
        font-size: 0.9rem;
        user-select: none;
    }
    .add-btn:hover { background: #5a65e8; }

    .error-banner {
        background: #fff3f3;
        border: 1px solid #fa5c7c;
        color: #fa5c7c;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .error-banner button {
        background: none;
        border: none;
        cursor: pointer;
        color: #fa5c7c;
        font-size: 1rem;
    }

    .dropzone {
        border: 2px dashed #dee2e6;
        border-radius: 12px;
        padding: 2.5rem;
        text-align: center;
        margin-bottom: 1.5rem;
        transition: all 0.2s;
        background: #f8f9fa;
    }

    .dropzone.active {
        border-color: #727cf5;
        background: #eef0fe;
    }

    .drop-hint {
        color: #6c757d;
        font-size: 0.95rem;
    }

    .dropzone.active .drop-hint {
        color: #727cf5;
        font-weight: 600;
    }

    .loading, .empty {
        color: #6c757d;
        text-align: center;
        padding: 2rem;
    }

    .queue-list {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
    }
</style>
