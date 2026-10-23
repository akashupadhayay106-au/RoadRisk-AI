"""
Phase 8 Unit Tests — CCTV Accident Detection & Edge Computer Vision
====================================================================
Tests dataset scanning, PyTorch transforms, MobileNetV2 architecture,
inference engine, latency benchmarking, and Grad-CAM interpretability.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import json
import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.cctv_preprocessing import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    scan_cctv_manifest,
    get_image_transforms,
    load_cctv_dataset,
)
from src.cctv_inference import (
    build_cctv_classifier,
    load_cctv_model,
    predict_cctv_frame,
    generate_cctv_gradcam,
    CCTV_MODEL_PATH,
    CCTV_METRICS_PATH,
)


class TestCCTVPreprocessing:
    """Validate CCTV manifest indexing and data preparation."""

    def test_scan_manifest_classes_and_splits(self):
        df = scan_cctv_manifest(verify_images=False)
        assert not df.empty
        assert "split" in df.columns
        assert "class_name" in df.columns
        assert "filepath" in df.columns

        splits = set(df["split"].unique())
        assert {"train", "val", "test"}.issubset(splits)

        classes = set(df["class_name"].unique())
        assert {"Accident", "Non Accident"} == classes

    def test_image_transforms(self):
        train_t = get_image_transforms(is_train=True)
        eval_t = get_image_transforms(is_train=False)
        assert train_t is not None
        assert eval_t is not None

        # Test transformation on dummy image
        dummy_img = Image.new("RGB", (300, 300), color="blue")
        t_tensor = eval_t(dummy_img)
        assert t_tensor.shape == (3, 224, 224)
        assert isinstance(t_tensor, torch.Tensor)

    def test_dataset_item_structure(self):
        ds = load_cctv_dataset("test")
        assert len(ds) > 0
        img, label, fname = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 224, 224)
        assert label in [0, 1]
        assert isinstance(fname, str)


class TestCCTVModelArchitecture:
    """Validate MobileNetV2 architecture customization."""

    def test_build_classifier_structure(self):
        model = build_cctv_classifier(pretrained=False)
        assert isinstance(model, torch.nn.Module)

        # Check output features of classifier head
        last_linear = model.classifier[-1]
        assert last_linear.out_features == len(CLASS_NAMES)  # 2

    def test_forward_pass_shape(self):
        model = build_cctv_classifier(pretrained=False)
        model.eval()
        dummy_batch = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy_batch)
        assert out.shape == (2, 2)


class TestCCTVInference:
    """Validate frame inference engine and performance metrics."""

    def test_model_artifacts_exist(self):
        assert CCTV_MODEL_PATH.exists()
        assert CCTV_METRICS_PATH.exists()

        with open(CCTV_METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        assert "test_metrics" in metrics
        assert metrics["test_metrics"]["accuracy"] > 0.70

    def test_predict_single_frame(self):
        model, dev = load_cctv_model()
        dummy_img = Image.new("RGB", (224, 224), color="red")

        res = predict_cctv_frame(dummy_img, model=model, device=dev)
        assert "prediction" in res
        assert res["prediction"] in CLASS_NAMES
        assert 0.0 <= res["confidence"] <= 1.0
        assert "probabilities" in res
        assert "Accident" in res["probabilities"]
        assert "Non Accident" in res["probabilities"]

        prob_sum = res["probabilities"]["Accident"] + res["probabilities"]["Non Accident"]
        assert pytest.approx(prob_sum, abs=1e-3) == 1.0
        assert res["latency_ms"] > 0.0


class TestCCTVGradCAM:
    """Validate Grad-CAM visual attention mapping."""

    def test_gradcam_overlay_generation(self):
        model, dev = load_cctv_model()
        dummy_img = Image.new("RGB", (224, 224), color="gray")

        overlay, cam = generate_cctv_gradcam(dummy_img, model=model, device=dev)
        assert isinstance(overlay, Image.Image)
        assert overlay.size == (224, 224)
        assert isinstance(cam, np.ndarray)
        assert cam.shape == (224, 224)
        assert cam.min() >= 0.0
        assert cam.max() <= 1.0
