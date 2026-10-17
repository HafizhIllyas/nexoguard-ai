"""
nexoguard.api.app
FastAPI / ASGI application for NexoGuard AI Clinical Decision Support.
Exposes high-performance REST and WebSocket endpoints for real-time telemetry streaming,
FIGO 2015 risk scoring, and clinical safety guardrail enforcement.
"""

import logging
from typing import Optional
from fastapi import FastAPI, Request, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
import uvicorn

from nexoguard.api.errors import AppError
from nexoguard.api.schemas import (
    HealthResponse,
    PatientListResponse,
    PatientDetailResponse,
    StreamResponse,
    PredictRequest,
    PredictResponse,
    FileUploadResponse
)
from nexoguard.api.service import NexoGuardService

logger = logging.getLogger("nexoguard.api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="NexoGuard AI Clinical Telemetry & Risk Scoring API",
    description="High-performance ASGI API for intrapartum CTG telemetry, FIGO 2015 classification, and deterministic safety guardrails.",
    version="1.0.0"
)

# Enable CORS for browser telemetry monitors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = NexoGuardService()


# ==============================================================================
# Global Exception Handlers (ECC error-handling pattern)
# ==============================================================================

@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    logger.warning(f"AppError on {request.url.path}: [{exc.code}] {exc.message}")
    return JSONResponse(status_code=exc.status_code, content=exc.to_dict())


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    logger.warning(f"Validation error on {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "REQUEST_VALIDATION_ERROR",
                "message": "Invalid request parameters or payload structure.",
                "details": exc.errors()
            }
        }
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(f"Unexpected internal server error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred during request processing."
            }
        }
    )


# ==============================================================================
# REST Endpoints
# ==============================================================================

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def get_health():
    """System health check and module operational readiness."""
    return HealthResponse()


@app.get("/api/patients", response_model=PatientListResponse, tags=["Patients"])
async def list_patients():
    """List all available patients in the clinical cohort with outcomes and FIGO ground truth."""
    patients = service.list_patients()
    return PatientListResponse(total=len(patients), patients=patients)


@app.get("/api/patients/{patient_id}", response_model=PatientDetailResponse, tags=["Patients"])
async def get_patient_detail(patient_id: str):
    """Retrieve detailed clinical outcomes and telemetry duration for a specific patient."""
    return service.get_patient(patient_id)


@app.get("/api/patients/{patient_id}/stream", response_model=StreamResponse, tags=["Telemetry"])
async def get_patient_stream(patient_id: str):
    """Retrieve raw 4 Hz FHR and UC telemetry arrays for replay and inspection."""
    return service.get_patient_stream(patient_id)


@app.post("/api/predict", response_model=PredictResponse, tags=["Inference"])
async def predict_signal(request: PredictRequest):
    """
    Evaluate arbitrary dual-channel CTG signal (FHR + UC at 4 Hz).
    Executes DSP cleaning, feature extraction, ML inference, and deterministic FIGO guardrails.
    """
    return service.predict_signal(
        fhr_raw=request.fhr,
        uc_raw=request.uc,
        sampling_rate_hz=request.sampling_rate_hz
    )


@app.post("/api/predict/upload", response_model=FileUploadResponse, tags=["Inference"])
async def predict_file_upload(file: UploadFile = File(...)):
    """Upload a hospital CTG CSV file for automated parsing and FIGO risk evaluation."""
    content = await file.read()
    content_str = content.decode("utf-8", errors="replace")
    return service.process_csv_upload(content_str=content_str, filename=file.filename or "upload.csv")


# ==============================================================================
# WebSocket Live Streaming Telemetry
# ==============================================================================

@app.websocket("/api/ws/stream/{patient_id}")
async def websocket_stream_telemetry(websocket: WebSocket, patient_id: str):
    """
    Bidirectional WebSocket stream broadcasting 4 Hz telemetry packets.
    Allows real-time telemetry streaming into web monitors with live latency < 10ms.
    """
    await websocket.accept()
    try:
        stream_data = service.get_patient_stream(patient_id)
        fhr = stream_data.fhr
        uc = stream_data.uc
        chunk_size = 4  # 1 second of data at 4 Hz
        import asyncio

        for i in range(0, len(fhr), chunk_size):
            packet = {
                "patient_id": patient_id,
                "index": i,
                "fhr_chunk": fhr[i:i + chunk_size],
                "uc_chunk": uc[i:i + chunk_size] if uc else [0.0] * chunk_size,
                "timestamp_sec": round(i / 4.0, 2)
            }
            await websocket.send_json(packet)
            await asyncio.sleep(0.25)  # Stream in real-time cadence (4 chunks/sec)

        await websocket.send_json({"status": "STREAM_FINISHED", "patient_id": patient_id})
        await websocket.close()
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected for patient {patient_id}")
    except Exception as e:
        logger.error(f"WebSocket streaming error for patient {patient_id}: {e}")
        try:
            await websocket.close(code=1011, reason=str(e))
        except Exception:
            pass


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Console script entrypoint for running the API server."""
    uvicorn.run("nexoguard.api.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run_server()
