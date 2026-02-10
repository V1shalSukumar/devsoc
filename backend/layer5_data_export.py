"""
LAYER 5: Data Export & Analytics

Provides:
1. Batch CSV export from analysis reports (minimum 6 files required)
2. ML-ready flattened feature extraction
3. Aggregate analytics computation
4. Time-series trend analysis

CSV is ONLY generated when batch threshold (6+ files) is met.
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from collections import Counter
import uuid

# Minimum files required to generate CSV
MIN_BATCH_SIZE = 6

# Output directories
DATA_DIR = Path(__file__).parent.parent / "data"
EXPORTS_DIR = DATA_DIR / "exports"
OUTPUTS_DIR = DATA_DIR / "outputs"

# Ensure export directory exists
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ── CSV Schema Definition ───────────────────────────────────────────────────

CSV_COLUMNS = [
    # Metadata
    "report_id",
    "processed_at",
    "audio_file",
    "duration_seconds",
    "language",
    
    # Layer 1: Audio metrics
    "segment_count",
    "avg_confidence",
    "overall_confidence",
    
    # Layer 2: Text analysis
    "transcript_length",
    "pii_count",
    "profanity_count",
    "obligation_count",
    "text_risk_level",
    
    # Layer 3: Intelligence
    "compliance_score",
    "overall_compliance",
    "violation_count",
    "primary_intent",
    "customer_sentiment",
    "agent_sentiment",
    "call_outcome",
    
    # FinBERT analysis
    "financial_tone",
    "financial_term_count",
    
    # Speaker distribution (%)
    "speaker_agent_pct",
    "speaker_customer_pct",
    "speaker_unknown_pct",
    
    # Emotion distribution (%)
    "emotion_neutral_pct",
    "emotion_urgent_pct",
    "emotion_hesitant_pct",
    "emotion_excited_pct",
    
    # Stress levels (%)
    "stress_low_pct",
    "stress_medium_pct",
    "stress_high_pct",
    
    # Risk metrics
    "overall_risk_score",
    "overall_risk_level",
    "risk_factor_count",
]


# ── Feature Extraction ──────────────────────────────────────────────────────

def extract_flat_features(report: dict) -> dict:
    """
    Flatten nested report JSON into ML-ready row.
    
    Args:
        report: Full analysis report dictionary
        
    Returns:
        Flattened dictionary matching CSV_COLUMNS
    """
    # Calculate speaker distribution from segments
    segments = report.get("segments", [])
    speaker_counts = Counter(seg.get("speaker", "Unknown") for seg in segments)
    total_segs = len(segments) if segments else 1
    
    # Calculate emotion distribution
    emotion_counts = Counter(seg.get("emotion", "neutral") for seg in segments)
    
    # Calculate stress distribution
    stress_counts = Counter(seg.get("stress_level", "medium") for seg in segments)
    
    # Calculate average confidence
    confidences = [seg.get("confidence", 0) for seg in segments if seg.get("confidence")]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    # Extract compliance data
    compliance = report.get("regulatory_compliance", {})
    intent = report.get("intent_classification", {})
    risk = report.get("overall_risk", {})
    finbert = report.get("finbert_analysis", {})
    
    return {
        # Metadata
        "report_id": report.get("audio_file", "unknown"),
        "processed_at": report.get("processed_at", ""),
        "audio_file": report.get("audio_file", ""),
        "duration_seconds": report.get("duration_seconds", 0),
        "language": report.get("language", "unknown"),
        
        # Layer 1: Audio metrics
        "segment_count": len(segments),
        "avg_confidence": round(avg_confidence, 3),
        "overall_confidence": report.get("overall_confidence", 0),
        
        # Layer 2: Text analysis
        "transcript_length": len(report.get("transcript", "")),
        "pii_count": report.get("pii_count", 0),
        "profanity_count": len(report.get("profanity_findings", [])),
        "obligation_count": len(report.get("obligation_sentences", [])),
        "text_risk_level": report.get("text_risk_level", "unknown"),
        
        # Layer 3: Intelligence
        "compliance_score": compliance.get("compliance_score", 0),
        "overall_compliance": compliance.get("overall_compliance", "unknown"),
        "violation_count": len(compliance.get("violations", [])),
        "primary_intent": intent.get("primary_intent", "unknown"),
        "customer_sentiment": intent.get("customer_sentiment", "unknown"),
        "agent_sentiment": intent.get("agent_sentiment", "unknown"),
        "call_outcome": intent.get("call_outcome", "unknown"),
        
        # FinBERT analysis
        "financial_tone": finbert.get("overall_financial_tone", "neutral"),
        "financial_term_count": finbert.get("unique_terms_found", 0),
        
        # Speaker distribution
        "speaker_agent_pct": round(speaker_counts.get("Agent", 0) / total_segs * 100, 1),
        "speaker_customer_pct": round(speaker_counts.get("Customer", 0) / total_segs * 100, 1),
        "speaker_unknown_pct": round(speaker_counts.get("Unknown", 0) / total_segs * 100, 1),
        
        # Emotion distribution
        "emotion_neutral_pct": round(emotion_counts.get("neutral", 0) / total_segs * 100, 1),
        "emotion_urgent_pct": round(emotion_counts.get("urgent", 0) / total_segs * 100, 1),
        "emotion_hesitant_pct": round(emotion_counts.get("hesitant", 0) / total_segs * 100, 1),
        "emotion_excited_pct": round(emotion_counts.get("excited", 0) / total_segs * 100, 1),
        
        # Stress distribution
        "stress_low_pct": round(stress_counts.get("low", 0) / total_segs * 100, 1),
        "stress_medium_pct": round(stress_counts.get("medium", 0) / total_segs * 100, 1),
        "stress_high_pct": round(stress_counts.get("high", 0) / total_segs * 100, 1),
        
        # Risk metrics
        "overall_risk_score": risk.get("score", 0),
        "overall_risk_level": risk.get("level", "unknown"),
        "risk_factor_count": len(risk.get("risk_factors", [])),
    }


# ── CSV Generation ──────────────────────────────────────────────────────────

def generate_batch_csv(reports: list[dict], batch_id: str = None) -> tuple[str, str]:
    """
    Generate CSV from multiple reports.
    
    Args:
        reports: List of analysis reports
        batch_id: Optional batch identifier
        
    Returns:
        Tuple of (csv_path, error_message)
        
    Raises:
        ValueError: If reports count is below MIN_BATCH_SIZE
    """
    if len(reports) < MIN_BATCH_SIZE:
        return None, f"Minimum {MIN_BATCH_SIZE} reports required. Got {len(reports)}."
    
    if not batch_id:
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    
    csv_path = EXPORTS_DIR / f"{batch_id}.csv"
    
    # Extract features from all reports
    rows = [extract_flat_features(report) for report in reports]
    
    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    
    return str(csv_path), None


def export_all_reports_csv() -> tuple[str, str]:
    """
    Export all saved reports as CSV if threshold met.
    
    Returns:
        Tuple of (csv_path, error_message)
    """
    reports = load_all_reports()
    
    if len(reports) < MIN_BATCH_SIZE:
        return None, f"Need at least {MIN_BATCH_SIZE} reports. Currently have {len(reports)}."
    
    return generate_batch_csv(reports, batch_id="all_reports")


def load_all_reports() -> list[dict]:
    """Load all saved report JSON files."""
    reports = []
    
    if not OUTPUTS_DIR.exists():
        return reports
    
    for f in OUTPUTS_DIR.glob("report_*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            reports.append(data)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    
    return reports


# ── Analytics Computation ───────────────────────────────────────────────────

def compute_analytics(reports: list[dict] = None) -> dict:
    """
    Compute aggregate analytics from reports.
    
    Args:
        reports: List of reports, or None to load all saved reports
        
    Returns:
        Analytics summary dictionary
    """
    if reports is None:
        reports = load_all_reports()
    
    if not reports:
        return {"error": "No reports available", "report_count": 0}
    
    # Extract features for all reports
    features = [extract_flat_features(r) for r in reports]
    
    # Compute aggregates
    compliance_scores = [f["compliance_score"] for f in features]
    risk_scores = [f["overall_risk_score"] for f in features]
    durations = [f["duration_seconds"] for f in features]
    
    # Count distributions
    intent_counts = Counter(f["primary_intent"] for f in features)
    compliance_counts = Counter(f["overall_compliance"] for f in features)
    risk_level_counts = Counter(f["overall_risk_level"] for f in features)
    outcome_counts = Counter(f["call_outcome"] for f in features)
    
    return {
        "report_count": len(reports),
        "can_export_csv": len(reports) >= MIN_BATCH_SIZE,
        "min_batch_size": MIN_BATCH_SIZE,
        
        # Summary statistics
        "avg_compliance_score": round(sum(compliance_scores) / len(compliance_scores), 1),
        "min_compliance_score": min(compliance_scores),
        "max_compliance_score": max(compliance_scores),
        
        "avg_risk_score": round(sum(risk_scores) / len(risk_scores), 1),
        "min_risk_score": min(risk_scores),
        "max_risk_score": max(risk_scores),
        
        "total_duration_seconds": round(sum(durations), 1),
        "avg_duration_seconds": round(sum(durations) / len(durations), 1),
        
        # Distributions
        "intent_distribution": dict(intent_counts),
        "compliance_distribution": dict(compliance_counts),
        "risk_level_distribution": dict(risk_level_counts),
        "outcome_distribution": dict(outcome_counts),
        
        # Totals
        "total_violations": sum(f["violation_count"] for f in features),
        "total_pii_detected": sum(f["pii_count"] for f in features),
        "total_obligations": sum(f["obligation_count"] for f in features),
    }


def compute_trends(reports: list[dict] = None) -> dict:
    """
    Compute time-series trends from reports.
    
    Args:
        reports: List of reports, or None to load all saved reports
        
    Returns:
        Time-series data for visualization
    """
    if reports is None:
        reports = load_all_reports()
    
    if not reports:
        return {"error": "No reports available", "data_points": []}
    
    # Sort by processed_at
    sorted_reports = sorted(reports, key=lambda r: r.get("processed_at", ""))
    
    data_points = []
    for r in sorted_reports:
        compliance = r.get("regulatory_compliance", {})
        risk = r.get("overall_risk", {})
        
        data_points.append({
            "timestamp": r.get("processed_at", ""),
            "audio_file": r.get("audio_file", ""),
            "compliance_score": compliance.get("compliance_score", 0),
            "risk_score": risk.get("score", 0),
            "violation_count": len(compliance.get("violations", [])),
            "duration_seconds": r.get("duration_seconds", 0),
        })
    
    return {
        "report_count": len(data_points),
        "data_points": data_points,
        "can_export_csv": len(data_points) >= MIN_BATCH_SIZE,
    }


# ── Batch Processing Support ────────────────────────────────────────────────

class BatchProcessor:
    """Manages batch analysis state and CSV generation."""
    
    def __init__(self):
        self._batches: dict[str, dict] = {}
    
    def create_batch(self) -> str:
        """Create a new batch and return its ID."""
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self._batches[batch_id] = {
            "created_at": datetime.now().isoformat(),
            "reports": [],
            "status": "in_progress",
            "csv_path": None,
        }
        return batch_id
    
    def add_report(self, batch_id: str, report: dict) -> dict:
        """Add a report to a batch."""
        if batch_id not in self._batches:
            return {"error": f"Batch {batch_id} not found"}
        
        self._batches[batch_id]["reports"].append(report)
        count = len(self._batches[batch_id]["reports"])
        
        return {
            "batch_id": batch_id,
            "report_count": count,
            "remaining_for_csv": max(0, MIN_BATCH_SIZE - count),
            "can_generate_csv": count >= MIN_BATCH_SIZE,
        }
    
    def finalize_batch(self, batch_id: str) -> dict:
        """Finalize batch and generate CSV if threshold met."""
        if batch_id not in self._batches:
            return {"error": f"Batch {batch_id} not found"}
        
        batch = self._batches[batch_id]
        reports = batch["reports"]
        
        if len(reports) < MIN_BATCH_SIZE:
            batch["status"] = "incomplete"
            return {
                "batch_id": batch_id,
                "status": "incomplete",
                "report_count": len(reports),
                "error": f"Need at least {MIN_BATCH_SIZE} reports. Got {len(reports)}.",
                "csv_generated": False,
            }
        
        # Generate CSV
        csv_path, error = generate_batch_csv(reports, batch_id)
        
        if error:
            batch["status"] = "error"
            return {"batch_id": batch_id, "status": "error", "error": error}
        
        batch["status"] = "completed"
        batch["csv_path"] = csv_path
        
        return {
            "batch_id": batch_id,
            "status": "completed",
            "report_count": len(reports),
            "csv_path": csv_path,
            "csv_generated": True,
        }
    
    def get_batch_status(self, batch_id: str) -> dict:
        """Get status of a batch."""
        if batch_id not in self._batches:
            return {"error": f"Batch {batch_id} not found"}
        
        batch = self._batches[batch_id]
        return {
            "batch_id": batch_id,
            "created_at": batch["created_at"],
            "status": batch["status"],
            "report_count": len(batch["reports"]),
            "can_generate_csv": len(batch["reports"]) >= MIN_BATCH_SIZE,
            "remaining_for_csv": max(0, MIN_BATCH_SIZE - len(batch["reports"])),
            "csv_path": batch.get("csv_path"),
        }


# Module-level batch processor instance
_batch_processor: Optional[BatchProcessor] = None


def get_batch_processor() -> BatchProcessor:
    """Get the global batch processor instance."""
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = BatchProcessor()
    return _batch_processor
