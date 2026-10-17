"""
nexoguard.api.errors
Structured, typed exception hierarchy and error handlers following ECC error-handling skill patterns.
"""

from typing import Any, Optional


class AppError(Exception):
    """Base application exception for NexoGuard API."""
    def __init__(self, message: str, code: str, status_code: int = 500, details: Optional[Any] = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}

    def to_dict(self) -> dict:
        result = {
            "error": {
                "code": self.code,
                "message": self.message,
            }
        }
        if self.details:
            result["error"]["details"] = self.details
        return result


class NotFoundError(AppError):
    def __init__(self, resource: str, resource_id: str, code: str = "NOT_FOUND"):
        super().__init__(
            message=f"{resource} not found: '{resource_id}'",
            code=code,
            status_code=404,
            details={"resource": resource, "id": resource_id}
        )


class PatientNotFoundError(NotFoundError):
    def __init__(self, patient_id: str):
        super().__init__(
            resource="Patient",
            resource_id=patient_id,
            code="PATIENT_NOT_FOUND"
        )


class ValidationError(AppError):
    def __init__(self, message: str, code: str = "VALIDATION_ERROR", details: Optional[Any] = None):
        super().__init__(
            message=message,
            code=code,
            status_code=422,
            details=details
        )


class InsufficientDataError(ValidationError):
    def __init__(self, duration_seconds: float, min_required: float = 60.0):
        super().__init__(
            message=f"Signal duration {duration_seconds:.1f}s is insufficient. Minimum required is {min_required:.1f}s.",
            code="INSUFFICIENT_DATA",
            details={"duration_seconds": duration_seconds, "min_required": min_required}
        )
