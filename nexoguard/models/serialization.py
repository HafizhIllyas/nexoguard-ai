"""
nexoguard.models.serialization
Model persistence and metadata sidecar management.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import joblib

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODEL_VERSION = "1.0.0"


def save_model_bundle(
    model_bundle: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None
) -> str:
    """
    Serializes trained model package and writes metadata JSON sidecar.
    """
    if output_path is None:
        DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        target_path = DEFAULT_MODEL_DIR / "model_xgb.pkl"
    else:
        target_path = Path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

    bundle_to_save = dict(model_bundle)
    bundle_to_save["version"] = MODEL_VERSION

    joblib.dump(bundle_to_save, str(target_path))

    # Save companion JSON metadata sidecar
    sidecar_path = target_path.with_suffix(".json")
    meta = {
        "version": MODEL_VERSION,
        "feature_cols": bundle_to_save.get("feature_cols", []),
        "metrics": bundle_to_save.get("metrics", {})
    }
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return str(target_path)


def load_model_bundle(model_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Loads serialized model package with path resolution fallback.
    """
    if model_path is not None:
        target_path = Path(model_path)
    else:
        candidate_cwd = Path("models") / "model_xgb.pkl"
        if candidate_cwd.is_file():
            target_path = candidate_cwd
        elif (DEFAULT_MODEL_DIR / "model_xgb.pkl").is_file():
            target_path = DEFAULT_MODEL_DIR / "model_xgb.pkl"
        else:
            raise FileNotFoundError("Could not find serialized model_xgb.pkl in models/.")

    if not target_path.is_file():
        raise FileNotFoundError(f"Model file not found: {target_path}")

    return joblib.load(str(target_path))
