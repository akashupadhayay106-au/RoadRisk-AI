"""
RoadRisk AI — CCTV Accident Detection & Deep Learning Inference Engine
========================================================================
Implements transfer learning with MobileNetV2 for real-time binary CCTV crash
classification (Accident vs Non-Accident), model training/evaluation,
latency benchmarking, and Grad-CAM visual attention interpretability.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import json
import time
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from src.config import MODELS_DIR
from src.cctv_preprocessing import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    get_image_transforms,
    scan_cctv_manifest,
    get_cctv_dataloaders,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Default artifact paths
CCTV_MODEL_PATH = MODELS_DIR / "cctv_mobilenet_v2.pth"
CCTV_METRICS_PATH = MODELS_DIR / "cctv_model_metrics.json"
CCTV_BASELINE_METRICS_PATH = MODELS_DIR / "cctv_baseline_metrics.json"

# In-memory singleton cache
_LOADED_MODEL = None
_LOADED_DEVICE = None


# ── Model Architecture ────────────────────────────────────────────────────────

def build_cctv_classifier(
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_features: bool = False,
) -> nn.Module:
    """
    Construct MobileNetV2 architecture with customized binary classification head.
    Pretrained on ImageNet-1K; lightweight footprint (~9 MB) optimized for real-time edge CCTV.
    """
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, len(CLASS_NAMES)),
    )
    return model


# ── Model Training & Evaluation Pipeline ─────────────────────────────────────

def train_cctv_model(
    epochs: int = 4,
    batch_size: int = 32,
    lr: float = 1e-3,
    freeze_features: bool = True,
    device: Optional[str] = None,
    save_model: bool = True,
    model_path: Path = CCTV_MODEL_PATH,
    metrics_path: Path = CCTV_METRICS_PATH,
) -> Dict[str, Any]:
    """
    Train / fine-tune MobileNetV2 on the CCTV crash dataset.
    Evaluates on validation set after each epoch and performs complete test evaluation.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    logger.info(f"Training CCTV classifier on device: {device} for {epochs} epochs...")

    # 1. Scan manifest and construct DataLoaders
    manifest_df = scan_cctv_manifest()
    loaders = get_cctv_dataloaders(batch_size=batch_size, manifest_df=manifest_df)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    # 2. Build model
    model = build_cctv_classifier(pretrained=True, freeze_features=freeze_features)
    model.to(device_obj)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    training_history = []
    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels, _ in train_loader:
            images = images.to(device_obj)
            labels = labels.to(device_obj)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_train_loss = running_loss / total_train if total_train else 0.0
        epoch_train_acc = correct_train / total_train if total_train else 0.0

        # Validation step
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device_obj)
                labels = labels.to(device_obj)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        epoch_val_loss = val_loss / total_val if total_val else 0.0
        epoch_val_acc = correct_val / total_val if total_val else 0.0
        epoch_sec = round(time.time() - t0, 2)

        logger.info(
            f"Epoch {epoch}/{epochs} ({epoch_sec}s) — "
            f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc*100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc*100:.2f}%"
        )

        training_history.append({
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 4),
            "train_acc": round(epoch_train_acc, 4),
            "val_loss": round(epoch_val_loss, 4),
            "val_acc": round(epoch_val_acc, 4),
            "duration_sec": epoch_sec,
        })

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights = model.state_dict().copy()

    # Load best weights for test evaluation
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # 3. Comprehensive Evaluation on Test Set
    test_metrics = evaluate_cctv_model(model, test_loader, device=device)
    logger.info(f"Test Accuracy: {test_metrics['accuracy']*100:.2f}%, Macro-F1: {test_metrics['macro_f1']:.4f}")

    results = {
        "model_architecture": "MobileNetV2 (ImageNet Transfer Learning)",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "device": device,
        "test_metrics": test_metrics,
        "training_history": training_history,
        "classes": CLASS_NAMES,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 4. Save artifacts
    if save_model:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved CCTV model to {model_path} and metrics to {metrics_path}")

    return results


def evaluate_cctv_model(
    model: nn.Module,
    test_loader,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Compute Accuracy, Precision, Recall, Macro-F1, and Confusion Matrix on test DataLoader.
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        roc_auc_score,
    )

    device_obj = torch.device(device)
    model.eval()
    model.to(device_obj)

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device_obj)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            preds = outputs.argmax(dim=1).cpu().numpy()
            labels_np = labels.numpy()
            probs_np = probs.cpu().numpy()

            y_true.extend(labels_np)
            y_pred.extend(preds)
            y_prob.extend(probs_np[:, 0])  # Prob of Accident (class 0)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    try:
        # Note: class 0 is Accident, class 1 is Non Accident.
        # Compute ROC-AUC on class 1 vs class 0
        auc = float(roc_auc_score([1 - y for y in y_true], y_prob))
    except Exception:
        auc = 0.5

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(prec, 4),
        "macro_recall": round(rec, 4),
        "macro_f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm,
        "sample_count": len(y_true),
    }


# ── High-Performance Inference Engine ────────────────────────────────────────

def load_cctv_model(
    model_path: Path = CCTV_MODEL_PATH,
    device: Optional[str] = None,
    force_reload: bool = False,
    require_custom_weights: bool = False,
) -> Tuple[nn.Module, str]:
    """
    Load the trained MobileNetV2 CCTV classifier into memory (singleton pattern).
    If no saved weights exist, instantiates a pre-trained backbone unless require_custom_weights=True.
    """
    global _LOADED_MODEL, _LOADED_DEVICE

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if _LOADED_MODEL is not None and not force_reload and _LOADED_DEVICE == device and not require_custom_weights:
        return _LOADED_MODEL, _LOADED_DEVICE

    if require_custom_weights and not model_path.exists():
        raise FileNotFoundError(f"Required CCTV model weights not found at {model_path}")

    model = build_cctv_classifier(pretrained=True, freeze_features=False)

    if model_path.exists():
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            logger.info(f"Loaded fine-tuned CCTV weights from {model_path}")
        except Exception as e:
            logger.warning(f"Could not load custom weights from {model_path} ({e}). Using base pretrained model.")
    else:
        logger.info(f"No custom model weights found at {model_path}. Using base transfer model.")

    model.to(torch.device(device))
    model.eval()

    _LOADED_MODEL = model
    _LOADED_DEVICE = device
    return _LOADED_MODEL, _LOADED_DEVICE


def predict_cctv_frame(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    model: Optional[nn.Module] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run sub-50ms inference on a single CCTV frame.

    Returns:
    {
        "prediction": "Accident" or "Non Accident",
        "is_accident": bool,
        "confidence": float (0.0 to 1.0),
        "probabilities": {"Accident": p_acc, "Non Accident": p_non},
        "latency_ms": float,
        "device": str
    }
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model is None:
        model, device = load_cctv_model(device=device)

    # Convert input to PIL Image
    if isinstance(image_input, (str, Path)):
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        pil_img = Image.fromarray(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # Preprocess
    transform = get_image_transforms(is_train=False)
    tensor = transform(pil_img).unsqueeze(0).to(torch.device(device))

    # Inference with timing
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    prob_accident = float(probs[CLASS_TO_IDX["Accident"]].item())
    prob_non_accident = float(probs[CLASS_TO_IDX["Non Accident"]].item())

    pred_idx = int(probs.argmax().item())
    predicted_class = IDX_TO_CLASS[pred_idx]
    confidence = float(probs[pred_idx].item())

    return {
        "prediction": predicted_class,
        "is_accident": (predicted_class == "Accident"),
        "confidence": round(confidence, 4),
        "probabilities": {
            "Accident": round(prob_accident, 4),
            "Non Accident": round(prob_non_accident, 4),
        },
        "latency_ms": latency_ms,
        "device": device,
    }


# ── Grad-CAM Visual Attention Engine ─────────────────────────────────────────

def generate_cctv_gradcam(
    image_input: Union[str, Path, Image.Image],
    model: Optional[nn.Module] = None,
    device: Optional[str] = None,
    target_class: Optional[int] = None,
) -> Tuple[Image.Image, np.ndarray]:
    """
    Compute Gradient-weighted Class Activation Mapping (Grad-CAM) on the final
    convolutional layer of MobileNetV2 (model.features[-1]).
    Produces a heatmap overlay showing regions the CNN focused on.
    """
    import cv2

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model is None:
        model, device = load_cctv_model(device=device)

    # Convert to PIL
    if isinstance(image_input, (str, Path)):
        pil_img = Image.open(image_input).convert("RGB")
    else:
        pil_img = image_input.convert("RGB")

    orig_w, orig_h = pil_img.size
    transform = get_image_transforms(is_train=False)
    tensor = transform(pil_img).unsqueeze(0).to(torch.device(device))

    # Hook the last convolutional feature block
    target_layer = model.features[-1]
    feature_maps = []
    gradients = []

    def forward_hook(module, inp, out):
        feature_maps.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    model.zero_grad()
    logits = model(tensor)
    if target_class is None:
        target_class = int(logits.argmax(dim=1).item())

    score = logits[0, target_class]
    score.backward()

    h1.remove()
    h2.remove()

    if not gradients or not feature_maps:
        # Fallback if hook failed
        return pil_img, np.zeros((orig_h, orig_w), dtype=np.float32)

    # Grad-CAM weight calculation
    grads = gradients[0].cpu().data.numpy()[0]  # [C, H, W]
    fmaps = feature_maps[0].cpu().data.numpy()[0]  # [C, H, W]

    # Global average pooling on gradients
    weights = np.mean(grads, axis=(1, 2))  # [C]
    cam = np.zeros(fmaps.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * fmaps[i, :, :]

    # ReLU and normalization
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    # Resize CAM to original image size
    cam_resized = cv2.resize(cam, (orig_w, orig_h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Alpha overlay
    img_np = np.array(pil_img)
    overlay = np.uint8(0.6 * img_np + 0.4 * heatmap)
    overlay_img = Image.fromarray(overlay)

    return overlay_img, cam_resized


# ── Baseline Model Benchmark & Video Frame Sampling ──────────────────────────

def evaluate_cctv_baseline(
    manifest_df: Optional[pd.DataFrame] = None,
    split_col: str = "split",
    save_metrics: bool = True,
    metrics_path: Path = CCTV_BASELINE_METRICS_PATH,
) -> Dict[str, Any]:
    """
    Train and evaluate a lightweight feature-engineering baseline
    (Color Histograms + Spatial Downsampling + Logistic Regression).
    
    Provides an empirical benchmark against which deep transfer learning (MobileNetV2)
    can be rigorously compared.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

    if manifest_df is None:
        manifest_df = scan_cctv_manifest(verify_images=False)

    col = split_col if split_col in manifest_df.columns else "split"
    train_df = manifest_df[manifest_df[col] == "train"].reset_index(drop=True)
    test_df = manifest_df[manifest_df[col] == "test"].reset_index(drop=True)

    if train_df.empty or test_df.empty:
        logger.warning("Train or test split empty for baseline evaluation.")
        return {}

    def extract_baseline_features(filepaths: List[str]) -> np.ndarray:
        feats = []
        for fp in filepaths:
            try:
                with Image.open(fp) as im:
                    im_rgb = im.convert("RGB").resize((32, 32))
                    arr = np.array(im_rgb, dtype=np.float32) / 255.0
                    # Spatial thumbnail (32x32x3 = 3072) + 8-bin channel histograms (24)
                    thumb_flat = arr.flatten()
                    r_hist, _ = np.histogram(arr[:, :, 0], bins=8, range=(0, 1))
                    g_hist, _ = np.histogram(arr[:, :, 1], bins=8, range=(0, 1))
                    b_hist, _ = np.histogram(arr[:, :, 2], bins=8, range=(0, 1))
                    feat = np.concatenate([thumb_flat, r_hist, g_hist, b_hist])
                    feats.append(feat)
            except Exception as e:
                logger.warning(f"Error extracting features for {fp}: {e}")
                feats.append(np.zeros(32 * 32 * 3 + 24, dtype=np.float32))
        return np.array(feats, dtype=np.float32)

    logger.info("Extracting baseline feature vectors from CCTV images...")
    X_train = extract_baseline_features(train_df["filepath"].tolist())
    y_train = train_df["class_idx"].values
    X_test = extract_baseline_features(test_df["filepath"].tolist())
    y_test = test_df["class_idx"].values

    clf = LogisticRegression(max_iter=300, random_state=42)
    start_t = time.time()
    clf.fit(X_train, y_train)
    train_duration = round(time.time() - start_t, 3)

    y_pred = clf.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()

    try:
        # roc_auc computed on class 1 vs 0
        roc_auc = float(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
    except Exception:
        roc_auc = 0.5

    baseline_metrics = {
        "model_architecture": "Baseline (Color Histograms + 32x32 Spatial Thumbnails + LogisticRegression)",
        "split_used": col,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "train_duration_sec": train_duration,
        "test_metrics": {
            "accuracy": round(acc, 4),
            "macro_precision": round(prec, 4),
            "macro_recall": round(rec, 4),
            "macro_f1": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "confusion_matrix": cm,
            "sample_count": len(test_df),
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if save_metrics:
        try:
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(baseline_metrics, f, indent=2)
            logger.info(f"Saved baseline CCTV metrics to {metrics_path}")
        except Exception as e:
            logger.warning(f"Could not save baseline metrics: {e}")

    return baseline_metrics


def sample_video_frames(
    video_input: Union[str, Path, bytes, Any],
    sample_interval_sec: float = 1.0,
    max_frames: int = 60,
) -> List[Dict[str, Any]]:
    """
    Extract evenly spaced frames from a surveillance video file for offline crash auditing.
    
    Parameters:
    - video_input: file path, bytes, or Streamlit UploadedFile object.
    - sample_interval_sec: temporal sampling interval in seconds (default: 1.0s).
    - max_frames: safety upper-bound on extracted frames to preserve memory (default: 60).
    
    Returns:
    - List of dicts: {"timestamp_sec": float, "frame_index": int, "image": PIL.Image}
    """
    import cv2
    import tempfile

    temp_path = None
    if isinstance(video_input, (str, Path)):
        video_path = str(video_input)
    elif hasattr(video_input, "read"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_input.read())
            temp_path = tmp.name
        video_path = temp_path
    elif isinstance(video_input, bytes):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_input)
            temp_path = tmp.name
        video_path = temp_path
    else:
        raise ValueError("Unsupported video_input type. Expected Path, str, bytes, or file-like object.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise IOError(f"Could not open video stream from {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0  # standard fallback CCTV FPS

    frame_interval = max(1, int(round(fps * sample_interval_sec)))

    sampled_results = []
    current_frame = 0

    try:
        while cap.isOpened() and len(sampled_results) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame % frame_interval == 0:
                # Convert BGR OpenCV frame to RGB PIL Image
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                timestamp = round(current_frame / fps, 2)
                sampled_results.append({
                    "timestamp_sec": timestamp,
                    "frame_index": current_frame,
                    "image": pil_img,
                })

            current_frame += 1
    finally:
        cap.release()
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

    return sampled_results


if __name__ == "__main__":
    print("Testing CCTV Inference Pipeline...")
    model, dev = load_cctv_model()
    print(f"Loaded MobileNetV2 model on {dev}.")

    # Test manifest
    manifest = scan_cctv_manifest()
    sample_row = manifest.iloc[0]
    print(f"Testing inference on: {sample_row['filepath']}")

    res = predict_cctv_frame(sample_row["filepath"], model=model, device=dev)
    print(f"Prediction: {res['prediction']} (Confidence: {res['confidence']*100:.2f}%) in {res['latency_ms']}ms")
