"""Hard-case capture for active learning.

Identifies ambiguous sessions for manual review and ground-truth labeling:
1. Multiple fronts (>2 Fronts in session) → indicates unresolved multi-card ambiguity
2. Borderline Hamming distance ([18, 26] around threshold 22) → pHash collision risk
3. Borderline confidence ([0.45, 0.55]) → classifier uncertainty near decision boundary
4. Low border purity (<0.3) → cropped cards have significant occlusion
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


def is_hard_case(session: Dict[str, Any]) -> Optional[str]:
    """Check if session is a hard case for active learning.

    Args:
        session: Session dict with keys:
            - video_id (int)
            - session_id (str)
            - front_tracks (list of dicts with 'angle' key)
            - hamming_values (list of floats)
            - confidence_scores (list of floats)
            - border_purity_scores (list of floats)

    Returns:
        Reason string if hard case (e.g., "multiple_fronts_3"), None otherwise.
    """
    # Check 1: Multiple fronts (>2)
    front_tracks = session.get("front_tracks", [])
    front_count = sum(1 for track in front_tracks if track.get("angle") == "Front")
    if front_count > 2:
        return f"multiple_fronts_{front_count}"

    # Check 2: Borderline Hamming distance [18, 26] (within 4 of threshold 22)
    hamming_values = session.get("hamming_values", [])
    for hamming in hamming_values:
        if 18 <= hamming <= 26:
            return f"borderline_hamming_{hamming}"

    # Check 3: Borderline confidence [0.45, 0.55]
    confidence_scores = session.get("confidence_scores", [])
    for confidence in confidence_scores:
        if 0.45 <= confidence <= 0.55:
            return f"borderline_confidence_{confidence:.3f}"

    # Check 4: Low border purity (<0.3)
    border_purity_scores = session.get("border_purity_scores", [])
    for purity in border_purity_scores:
        if purity < 0.3:
            return f"low_border_purity_{purity:.3f}"

    return None


def capture_hard_case(
    session: Dict[str, Any],
    reason: str,
    output_file: str = "hard_cases.jsonl",
) -> None:
    """Persist hard case to JSONL file.

    Args:
        session: Session dict with relevant data
        reason: Classification reason (from is_hard_case)
        output_file: JSONL file path (default "hard_cases.jsonl")
    """
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "reason": reason,
        "video_id": session.get("video_id"),
        "session_id": session.get("session_id"),
        "front_count": sum(1 for track in session.get("front_tracks", [])
                          if track.get("angle") == "Front"),
        "total_tracks": len(session.get("front_tracks", [])),
        "hamming_values": session.get("hamming_values", []),
        "confidence_scores": session.get("confidence_scores", []),
        "border_purity_scores": session.get("border_purity_scores", []),
    }

    # Append to JSONL file
    output_path = Path(output_file)
    with open(output_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_hard_cases(input_file: str = "hard_cases.jsonl") -> List[Dict[str, Any]]:
    """Load hard cases from JSONL file.

    Args:
        input_file: JSONL file path (default "hard_cases.jsonl")

    Returns:
        List of hard case records (empty list if file doesn't exist)
    """
    input_path = Path(input_file)
    if not input_path.exists():
        return []

    records = []
    with open(input_path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records
