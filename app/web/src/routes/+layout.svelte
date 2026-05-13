<script lang="ts">
    import '../app.css';
    let { children } = $props();
</script>

<div class="app-container">
    <nav class="sidebar">
        <div class="logo">Card Capture v4</div>
        <ul class="nav-links">
            <li><a href="/">Dashboard</a></li>
            <li><a href="/videos">Videos</a></li>
            <li><a href="/runs">Runs</a></li>
            <li><a href="/labeling">Labeling</a></li>
            <li><a href="/regression">Regression</a></li>
        </ul>
    </nav>

    <main class="content">
        {@render children()}
    </main>
</div>

<style>
    :global(body) {
        margin: 0;
        padding: 0;
        font-family: system-ui, sans-serif;
        background: #f4f4f9;
    }

    .app-container {
        display: flex;
        height: 100vh;
    }

    .sidebar {
        width: 240px;
        background: #1e1e2d;
        color: white;
        padding: 1rem;
        display: flex;
        flex-direction: column;
    }

    .logo {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
        color: #727cf5;
    }

    .nav-links {
        list-style: none;
        padding: 0;
        margin: 0;
    }

    .nav-links li {
        margin-bottom: 0.5rem;
    }

    .nav-links a {
        color: #aab8c2;
        text-decoration: none;
        display: block;
        padding: 0.5rem;
        border-radius: 4px;
        transition: background 0.2s;
    }

    .nav-links a:hover {
        background: #2b2b3d;
        color: white;
    }

    .content {
        flex: 1;
        padding: 2rem;
        overflow-y: auto;
    }
</style>
