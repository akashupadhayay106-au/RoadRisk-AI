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


class TestCCTVVideoSequenceGrouping:
    """Validate video sequence extraction, leakage auditing, and grouped splitting."""

    def test_extract_source_video_id(self):
        from src.cctv_preprocessing import extract_source_video_id

        assert extract_source_video_id("acc1 (12).jpg") == "acc1"
        assert extract_source_video_id("acc1 (1).jpg") == "acc1"
        assert extract_source_video_id("test10_12.jpg") == "test10"
        assert extract_source_video_id("5_14.jpg") == "5"
        assert extract_source_video_id("highway_cam_01_100.jpg") == "highway_cam_01"
        assert extract_source_video_id("singleframe.jpg") == "singleframe"

    def test_audit_split_leakage_detection(self):
        from src.cctv_preprocessing import scan_cctv_manifest, audit_split_leakage

        manifest = scan_cctv_manifest(verify_images=False)
        audit_res = audit_split_leakage(manifest)

        assert audit_res["status"] == "audited"
        assert audit_res["total_frames"] >= 980
        assert audit_res["total_groups"] > 0
        assert audit_res["leaking_groups_count"] > 0
        assert audit_res["leakage_percentage"] > 50.0  # Naive split has severe leakage
        assert "class_breakdown" in audit_res
        assert "Accident" in audit_res["class_breakdown"]
        assert "Non Accident" in audit_res["class_breakdown"]

    def test_create_leakage_free_splits(self):
        from src.cctv_preprocessing import scan_cctv_manifest, create_leakage_free_splits

        manifest = scan_cctv_manifest(verify_images=False)
        df_grouped = create_leakage_free_splits(manifest, test_ratio=0.15, val_ratio=0.15, random_state=42)

        assert "grouped_split" in df_grouped.columns
        splits = set(df_grouped["grouped_split"].unique())
        assert {"train", "val", "test"} == splits

        # Check ZERO group overlap across train, val, and test splits
        for cls in ["Accident", "Non Accident"]:
            cls_df = df_grouped[df_grouped["class_name"] == cls]
            train_groups = set(cls_df[cls_df["grouped_split"] == "train"]["video_group"].unique())
            val_groups = set(cls_df[cls_df["grouped_split"] == "val"]["video_group"].unique())
            test_groups = set(cls_df[cls_df["grouped_split"] == "test"]["video_group"].unique())

            # Intersections MUST BE EMPTY
            assert len(train_groups.intersection(val_groups)) == 0
            assert len(train_groups.intersection(test_groups)) == 0
            assert len(val_groups.intersection(test_groups)) == 0


class TestCCTVBaselineAndVideo:
    """Validate baseline model benchmarking and video stream processing."""

    def test_cctv_baseline_metrics_artifact(self):
        from src.cctv_inference import CCTV_BASELINE_METRICS_PATH

        assert CCTV_BASELINE_METRICS_PATH.exists()
        with open(CCTV_BASELINE_METRICS_PATH, "r", encoding="utf-8") as f:
            b_metrics = json.load(f)

        assert "test_metrics" in b_metrics
        assert 0.40 <= b_metrics["test_metrics"]["accuracy"] <= 1.0
        assert "macro_f1" in b_metrics["test_metrics"]
        assert "confusion_matrix" in b_metrics["test_metrics"]

    def test_sample_video_frames_invalid_file(self):
        from src.cctv_inference import sample_video_frames

        with pytest.raises(Exception):
            sample_video_frames("nonexistent_video_feed_xyz.mp4")

    def test_load_cctv_model_missing_weights(self):
        from src.cctv_inference import load_cctv_model

        # When pointing to a nonexistent weights file with require_custom_weights=True, it should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            load_cctv_model(model_path=Path("nonexistent_model_weights.pth"), force_reload=True, require_custom_weights=True)

