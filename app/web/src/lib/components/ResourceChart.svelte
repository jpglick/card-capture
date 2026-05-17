<script lang="ts">
    interface Sample {
        elapsed_s: number;
        value: number | null;
    }

    interface StageLine {
        name: string;
        elapsed_s: number;
        color: string;
    }

    let {
        samples = [],
        stages = [],
        yMax = 100,
        yUnit = '%',
        color = '#727cf5',
        label = '',
        noData = false,
    }: {
        samples?: Sample[];
        stages?: StageLine[];
        yMax?: number;
        yUnit?: string;
        color?: string;
        label?: string;
        noData?: boolean;
    } = $props();

    // SVG geometry constants
    const W = 600, H = 130;
    const ML = 44, MR = 14, MT = 10, MB = 32;
    const CW = W - ML - MR;
    const CH = H - MT - MB;

    const xMax = $derived(
        samples.length > 0 ? Math.max(...samples.map(s => s.elapsed_s), 10) : 60
    );

    const chartData = $derived.by(() => {
        const xm = xMax;
        const x = (t: number) => ML + (t / xm) * CW;
        const y = (v: number) => MT + CH - (Math.min(Math.max(v, 0), yMax) / yMax) * CH;

        // Build polyline segments, splitting on null
        const segs: string[][] = [];
        let cur: string[] = [];
        for (const s of samples) {
            if (s.value == null) {
                if (cur.length) { segs.push(cur); cur = []; }
            } else {
                cur.push(`${x(s.elapsed_s).toFixed(1)},${y(s.value).toFixed(1)}`);
            }
        }
        if (cur.length) segs.push(cur);

        // Area fill paths
        const fills = segs.map(seg => {
            if (!seg.length) return '';
            const bY = (MT + CH).toFixed(1);
            const fX = seg[0].split(',')[0];
            const lX = seg[seg.length - 1].split(',')[0];
            return `M${fX},${bY} L${seg.join(' L')} L${lX},${bY} Z`;
        }).filter(Boolean);

        // X-axis ticks
        const interval = xm > 600 ? 120 : xm > 300 ? 60 : xm > 120 ? 30 : xm > 60 ? 15 : 10;
        const xTicks: { x: number; lbl: string }[] = [];
        for (let t = 0; t <= xm + 0.01; t += interval) {
            const m = Math.floor(t / 60), s = Math.round(t % 60);
            xTicks.push({ x: x(t), lbl: m > 0 ? `${m}m${s > 0 ? s + 's' : ''}` : `${t}s` });
        }

        // Y-axis ticks
        const yStep = yMax > 1 ? 25 : yMax > 0.1 ? 0.25 : 0.025;
        const yTicks: { y: number; lbl: string }[] = [];
        for (let v = 0; v <= yMax + 0.001; v += yStep) {
            const rounded = Math.round(v * 1000) / 1000;
            if (rounded > yMax) break;
            yTicks.push({ y: y(rounded), lbl: `${rounded}${yUnit}` });
        }

        // Stage lines clamped to chart width
        const stageLines = stages
            .filter(s => s.elapsed_s >= 0 && s.elapsed_s <= xm)
            .map(s => ({ ...s, x: x(s.elapsed_s) }));

        return { segs, fills, xTicks, yTicks, stageLines };
    });
</script>

<div class="chart-wrap">
    <div class="chart-label">{label}</div>
    <svg viewBox="0 0 {W} {H}" width="100%" preserveAspectRatio="none">
        <!-- Grid lines -->
        {#each chartData.yTicks as tick}
            <line
                x1={ML} y1={tick.y} x2={W - MR} y2={tick.y}
                stroke="#f0f1f9" stroke-width="1"
            />
            <text x={ML - 4} y={tick.y + 3.5} text-anchor="end" font-size="9" fill="#adb5bd">
                {tick.lbl}
            </text>
        {/each}

        <!-- Stage marker lines (behind the data) -->
        {#each chartData.stageLines as st}
            <line
                x1={st.x} y1={MT} x2={st.x} y2={MT + CH}
                stroke={st.color} stroke-width="1.5" stroke-dasharray="3 3" opacity="0.8"
            />
            <text
                x={st.x + 2} y={MT + 9}
                font-size="8" fill={st.color} font-weight="600"
            >{st.name}</text>
        {/each}

        <!-- Area fill -->
        {#if !noData}
            {#each chartData.fills as d}
                <path {d} fill={color} opacity="0.12" />
            {/each}
            <!-- Line -->
            {#each chartData.segs as seg}
                <polyline
                    points={seg.join(' ')}
                    fill="none"
                    stroke={color}
                    stroke-width="1.5"
                    stroke-linejoin="round"
                    stroke-linecap="round"
                />
            {/each}
        {:else}
            <text x={ML + CW/2} y={MT + CH/2 + 4} text-anchor="middle" font-size="11" fill="#adb5bd">
                No data
            </text>
        {/if}

        <!-- X-axis ticks -->
        {#each chartData.xTicks as tick}
            <line x1={tick.x} y1={MT + CH} x2={tick.x} y2={MT + CH + 4} stroke="#dee2e6" stroke-width="1" />
            <text x={tick.x} y={MT + CH + 14} text-anchor="middle" font-size="9" fill="#adb5bd">
                {tick.lbl}
            </text>
        {/each}

        <!-- Axes -->
        <line x1={ML} y1={MT} x2={ML} y2={MT + CH} stroke="#dee2e6" stroke-width="1" />
        <line x1={ML} y1={MT + CH} x2={W - MR} y2={MT + CH} stroke="#dee2e6" stroke-width="1" />
    </svg>
</div>

<style>
    .chart-wrap {
        background: white;
        border-radius: 10px;
        padding: 0.75rem 0.75rem 0.25rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.07);
        min-width: 0;
    }
    .chart-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #6c757d;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    svg { display: block; overflow: visible; }
</style>
