<h1>Labeling Tools</h1>
<p>Select a labeling task to continue building the training set and ground truth.</p>

<div class="tool-grid">
    <a href="/label/truth" class="tool-card">
        <h2>Per-Video Truth</h2>
        <p>Verify detections and create ground truth files for regression testing.</p>
    </a>
    <a href="/label/fb" class="tool-card">
        <h2>F/B Trainer</h2>
        <p>Quickly classify individual crops as Front or Back side.</p>
    </a>
    <a href="/label/clusters" class="tool-card">
        <h2>Cluster Confirmer</h2>
        <p>Verify deduplication groups across sessions.</p>
    </a>
</div>

<style>
    .tool-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2rem;
        margin-top: 2rem;
    }

    .tool-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        text-decoration: none;
        color: inherit;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .tool-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px rgba(0,0,0,0.1);
    }

    h2 { color: #727cf5; margin-top: 0; }
</style>
