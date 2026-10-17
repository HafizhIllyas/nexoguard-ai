"""
tests/test_api.py
TDD Test Suite for NexoGuard AI Decoupled FastAPI / ASGI Service Layer.
Tests REST endpoints, error boundaries, typed errors, validation, and clinical FIGO predictions.
"""

import unittest
from fastapi.testclient import TestClient


class TestNexoGuardAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nexoguard.api.app import app
        cls.client = TestClient(app)

    def test_health_check(self):
        """GET /api/health should return 200 with status ok and model loaded."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["version"], "1.0.0")
        self.assertTrue(data["model_loaded"])
        self.assertTrue(data["guardrails_active"])

    def test_get_patients_list(self):
        """GET /api/patients should return a list of patients with metadata."""
        response = self.client.get("/api/patients")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("patients", data)
        self.assertIn("total", data)
        self.assertGreaterEqual(data["total"], 40)
        
        # Check structure of first patient
        patient = data["patients"][0]
        self.assertIn("patient_id", patient)
        self.assertIn("pH", patient)
        self.assertIn("ground_truth_category", patient)

    def test_get_patient_detail_success(self):
        """GET /api/patients/{id} should return details for valid patient."""
        response = self.client.get("/api/patients/1024")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["patient_id"], "1024")
        self.assertEqual(data["pH"], 7.43)
        self.assertEqual(data["ground_truth_category"], "Category I")

    def test_get_patient_detail_not_found(self):
        """GET /api/patients/{id} with invalid id should return 404 with typed error structure."""
        response = self.client.get("/api/patients/99999")
        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertIn("error", data)
        self.assertEqual(data["error"]["code"], "PATIENT_NOT_FOUND")

    def test_get_patient_stream(self):
        """GET /api/patients/{id}/stream should return 4 Hz FHR and UC arrays."""
        response = self.client.get("/api/patients/1024/stream")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["patient_id"], "1024")
        self.assertIn("fhr", data)
        self.assertIn("uc", data)
        self.assertIn("sampling_rate_hz", data)
        self.assertEqual(data["sampling_rate_hz"], 4.0)
        self.assertGreater(len(data["fhr"]), 0)

    def test_predict_reassuring_signal(self):
        """POST /api/predict with authentic reassuring clinical signal should return Category I."""
        import pandas as pd
        from pathlib import Path
        csv_path = Path(__file__).resolve().parent.parent / "data" / "sample_uploads" / "sample_category_i_reassuring.csv"
        df = pd.read_csv(csv_path)
        fhr = df["fhr"].tolist()
        uc = df["uc"].tolist()
        
        response = self.client.post("/api/predict", json={"fhr": fhr, "uc": uc})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("category", data)
        self.assertEqual(data["category"], "Category I")
        self.assertLess(data["risk_score"], 25.0)
        self.assertFalse(data["guardrail_override"])

    def test_predict_pathological_bradycardia_guardrail(self):
        """POST /api/predict with severe bradycardia (85 bpm) must trigger Category III guardrail."""
        fhr = [85.0 + (i % 2) for i in range(2400)]
        uc = [20.0 for _ in range(2400)]
        
        response = self.client.post("/api/predict", json={"fhr": fhr, "uc": uc})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["category"], "Category III")
        self.assertTrue(data["guardrail_override"])
        self.assertGreaterEqual(data["risk_score"], 80.0)
        self.assertTrue(len(data["guardrail_reasons"]) > 0)

    def test_predict_validation_error_too_short(self):
        """POST /api/predict with insufficient data (< 60s) should return 422 error."""
        response = self.client.post("/api/predict", json={"fhr": [140.0] * 10, "uc": [15.0] * 10})
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertIn("error", data)
        self.assertEqual(data["error"]["code"], "INSUFFICIENT_DATA")

    def test_predict_file_upload(self):
        """POST /api/predict/upload should parse CSV and return classification."""
        import numpy as np
        np.random.seed(42)
        t = np.linspace(0, 600, 2400)
        fhr_vals = 140.0 + 8.0 * np.sin(2 * np.pi * t / 60.0) + np.random.normal(0, 2.0, 2400)
        lines = ["seconds,fhr,uc"] + [f"{t[i]:.2f},{fhr_vals[i]:.2f},20.0" for i in range(2400)]
        csv_content = "\n".join(lines)
        files = {"file": ("test_patient.csv", csv_content.encode("utf-8"), "text/csv")}
        
        response = self.client.post("/api/predict/upload", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("category", data)
        self.assertIn("risk_score", data)
        self.assertIn("windows_analyzed", data)


if __name__ == "__main__":
    unittest.main()
