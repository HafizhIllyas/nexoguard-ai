"""
nexoguard.api.schemas
Pydantic schemas for NexoGuard API requests and responses.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    model_loaded: bool = True
    guardrails_active: bool = True


class PatientSummary(BaseModel):
    patient_id: str
    pH: float
    BDecf: Optional[float] = None
    ground_truth_category: str
    Apgar1: Optional[float] = None
    delivery_type: Optional[str] = None


class PatientListResponse(BaseModel):
    total: int
    patients: List[PatientSummary]


class PatientDetailResponse(PatientSummary):
    has_recording: bool = True
    total_samples: int = 0
    duration_seconds: float = 0.0


class StreamResponse(BaseModel):
    patient_id: str
    sampling_rate_hz: float = 4.0
    fhr: List[float]
    uc: List[float]
    duration_seconds: float


class PredictRequest(BaseModel):
    fhr: List[float] = Field(..., description="Fetal Heart Rate signal in bpm (at 4 Hz)")
    uc: Optional[List[float]] = Field(default=None, description="Uterine Contraction signal (0-100 mm Hg)")
    sampling_rate_hz: float = Field(default=4.0, description="Sampling rate in Hz")


class PredictResponse(BaseModel):
    category: str = Field(..., description="FIGO 2015 Category: Category I, II, or III")
    risk_score: float = Field(..., description="Calibrated Acidemia Risk percentage (0-100%)")
    acidemia_probability: float = Field(..., description="Raw model probability (0.0 - 1.0)")
    guardrail_override: bool = Field(default=False, description="Whether deterministic FIGO guardrail fired")
    guardrail_reasons: List[str] = Field(default_factory=list, description="Reasons triggering guardrail")
    features: Dict[str, Any] = Field(default_factory=dict, description="Extracted clinical DSP features")
    clinical_recommendation: str = Field(..., description="Actionable FIGO guideline guidance")


class FileUploadResponse(BaseModel):
    filename: str
    total_samples: int
    duration_seconds: float
    windows_analyzed: int
    category: str
    risk_score: float
    guardrail_override: bool
    guardrail_reasons: List[str]
    clinical_recommendation: str
