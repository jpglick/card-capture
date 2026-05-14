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

export interface RunDetail extends RunSummary {
    events: RunEvent[];
}

export interface RunEvent {
    timestamp: string;
    event_name: string;
    stage_id?: string;
    artifact_ref?: string;
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
    instance_id: string;
    run_id: string;
    angle: 'Front' | 'Back';
    fused_image_url: string;
    quality_score: number;
    primary_hash: string;
    review_state?: string;
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
