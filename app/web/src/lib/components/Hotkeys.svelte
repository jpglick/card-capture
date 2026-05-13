<script lang="ts">
    import { onMount } from 'svelte';

    let { keys = {}, children } = $props();

    function handleKeydown(event: KeyboardEvent) {
        if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
            return;
        }

        const handler = keys[event.key.toLowerCase()];
        if (handler) {
            event.preventDefault();
            handler();
        }
    }

    onMount(() => {
        window.addEventListener('keydown', handleKeydown);
        return () => window.removeEventListener('keydown', handleKeydown);
    });
</script>

{@render children()}
