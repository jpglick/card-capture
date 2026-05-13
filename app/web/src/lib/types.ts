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

export interface RunSummary {
    run_id: string;
    video_id: string;
    status: string;
    cards_extracted: number;
    elapsed_ms: number;
    created_at: string;
}

export interface Card {
    instance_id: string;
    run_id: string;
    angle: 'Front' | 'Back';
    fused_image_url: string;
    quality_score: number;
    primary_hash: string;
}

export interface SSEEvent {
    type: 'stage_started' | 'stage_completed' | 'run_completed' | 'run_failed';
    run_id: string;
    stage_id?: string;
    artifact_ref?: string;
    timestamp: string;
}
