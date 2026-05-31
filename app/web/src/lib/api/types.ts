/**
 * Card Capture API types (Contract 2).
 */

export interface Video {
    video_id: string;
    filename: string;
    duration_ms: number;
    status: 'pending' | 'processing' | 'completed' | 'failed';
    created_at: string;
}

export interface VideoCreate {
    filename: string;
    file_path?: string;
}

export interface RunSummary {
    run_id: string;
    video_id: string;
    status: string;
    cards_extracted: number;
    elapsed_ms: number;
    created_at: string;
}

export interface Run {
    run_id: string;
    video_id: string;
    status: string;
    cards_extracted: number;
    elapsed_ms: number;
    created_at: string;
}

export interface ResourceSample {
    elapsed_s: number;
    cpu_pct: number | null;
    mem_used_mb: number | null;
    mem_pct: number | null;
    gpu_pct: number | null;
    vram_used_mb: number | null;
    neural_pct?: number | null;
}

export interface StageMarker {
    name: string;
    elapsed_s: number;
}

export interface HostInfo {
    hostname?: string;
    platform?: string;
    cpu_count_physical?: number;
    cpu_count_logical?: number;
    mem_total_gb?: number;
    gpu_device?: string;
    gpu_name?: string;
    vram_total_gb?: number;
    vram_is_unified?: boolean;
}

export interface RunResources {
    run_id: string;
    host_info: HostInfo | null;
    samples: ResourceSample[];
    stage_markers: StageMarker[];
    stage_metrics?: Record<string, Record<string, number>>;
}

export interface DetectTelemetry {
    frame_count?: number;
    accepted_frame_count?: number;
    triage_pass_rate?: number;
    yolo_frames?: number;
    yolo_batches?: number;
    yolo_elapsed_s?: number;
    yolo_device?: string;
    presence_windows?: number;
    sampler_type?: string;
}

export interface RunDetail extends RunSummary {
    events: RunEvent[];
    logs?: string[];
    stage_timings?: Array<{ stage: string; elapsed_ms: number }>;
    detect_telemetry?: DetectTelemetry;
}

export interface RunEvent {
    created_at: string;
    event_type: string;
    data_json?: string;
}

export interface RunTelemetry {
    run_id: string;
    throughput_fps: number;
    stage_durations_ms: Record<string, number>;
}

export interface RunRejection {
    frame_index: number;
    reason: string;
    score: number;
}

export interface RunHardCase {
    case_id: number;
    stage_id: string;
    reason: string;
    thumbnail_url: string;
}

export interface RunCardSummary {
    instance_id: string;
    angle: string;
    thumbnail_url: string;
}

export interface Card {
    card_id: string;
    instance_id: string;
    video_id: string;
    run_id: string;
    side: 'Front' | 'Back';
    is_foil: boolean;
    confidence: number;
    review_state?: string;
    canonical_url: string | null;
    fused_url: string | null;
    created_at: string;
}

export interface CardDetail extends Card {
    quality_score_detail: any;
    source_frame_indices: number[];
    dedup_group_id?: number;
}

export interface CardFilter {
    run_id?: string;
    video_id?: string;
    dedup_group_id?: number;
    review_state?: string;
    side?: string;
    is_foil?: boolean;
    confidence_min?: number;
    confidence_max?: number;
    page?: number;
    page_size?: number;
}

export interface PaginatedCards {
    total: number;
    page: number;
    page_size: number;
    items: Card[];
}

export interface CardBulkAction {
    card_ids: string[];
    review_state?: string;
    dedup_group_id?: number;
}

export interface ConfigPreset {
    preset_name: string;
    description: string;
    config: Record<string, any>;
}

export interface ConfigPlayground {
    run_id: string;
    config: Record<string, any>;
}

export interface SSEEvent {
    type: 'stage_started' | 'stage_completed' | 'run_completed' | 'run_failed';
    run_id: string;
    stage_id?: string;
    artifact_ref?: string;
    timestamp: string;
}

export interface LabelTruthExpectedCard {
    card_id: string;
    front_present: boolean;
    back_present: boolean;
    physical_card_key: string;
    is_foil: boolean;
    approx_front_window_ms?: [number, number];
    approx_back_window_ms?: [number, number];
    notes?: string;
}

export interface LabelTruth {
    video_id: string;
    schema_version: number;
    expected_cards: LabelTruthExpectedCard[];
}

export interface LabelFBNext {
    instance_id: string;
    frame_index: number;
    canonical_url?: string;
    video_id: string;
    run_id: string;
    labels_collected: number;
    labels_target: number;
}

export interface LabelFB {
    instance_id: string;
    frame_index: number;
    side: 'front' | 'back' | 'uncertain';
}

export interface DedupCluster {
    cluster_id: number;
    status: string;
    predicted_member_ids: string[];
    confirmed_member_ids?: string[];
    member_thumbnails: string[];
    updated_at: string;
}

export interface DatasetSummary {
    model_name: string;
    total_labels: number;
    class_distribution: Record<string, number>;
    last_updated: string;
}

export interface TrainingJobSummary {
    job_id: string;
    model_name: string;
    status: string;
    created_at: string;
}

export interface TrainingJobDetail {
    job_id: string;
    model_name: string;
    status: string;
    progress?: Record<string, any>;
    created_at: string;
    completed_at?: string;
}

export interface Baseline {
    baseline_id: number;
    name: string;
    code_sha: string;
    created_at: string;
}

export interface RegressionRun {
    run_id: number;
    baseline_id: number;
    status: string;
    created_at: string;
}

export interface RegressionCompare {
    run_a: number;
    run_b: number;
    metric_deltas: Record<string, any>;
    regressions: any[];
    per_video_deltas: any[];
}

export interface VastConfig {
    pipeline_backend: 'mps' | 'cuda';
    cuda_gpu_type: 'RTX 4090' | 'Flagship' | 'RTX 5060 Ti';
    vast_template_id: string;
    cuda_idle_timeout_s: number;
}

export interface BatchJob {
    video_id: string;
    status: 'pending' | 'running' | 'complete' | 'failed';
    run_id: string | null;
    error?: string;
}

export interface BatchStatus {
    batch_id: string;
    status: 'queued' | 'running' | 'complete' | 'partial' | 'failed';
    total: number;
    completed: number;
    failed: number;
    jobs: BatchJob[];
    error?: string;
}
