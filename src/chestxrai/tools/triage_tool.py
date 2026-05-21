import os
import json
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

IMG_SIZE = 224
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_densenet121_nih.pth")
GRADCAM_DIR = os.path.join(os.path.dirname(__file__), "..", "gradcam_outputs")

PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax"
]

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

val_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


class DenseNet121MultiLabel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.densenet = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        features = self.densenet.features(x)
        out = torch.relu(features)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.densenet.classifier(out)
        return out


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        target_layer = self.model.densenet.features
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.clone().detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        logits = self.model(input_tensor)
        target = logits[0, class_idx]
        target.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        return cam


def predict_and_explain(model, image_path, grad_cam, top_k=3):
    orig_img = Image.open(image_path).convert("RGB")
    img_resized = orig_img.resize((IMG_SIZE, IMG_SIZE))
    image_np = np.array(img_resized) / 255.0
    input_tensor = val_transform(orig_img).unsqueeze(0).to(DEVICE)
    input_tensor.requires_grad_(True)
    with torch.no_grad():
        logits = model(input_tensor)
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    top_indices = np.argsort(probs)[::-1][:top_k]
    results = []
    for idx in top_indices:
        inp = val_transform(orig_img).unsqueeze(0).to(DEVICE)
        inp.requires_grad_(True)
        heatmap = grad_cam.generate(inp, int(idx))
        results.append({
            "pathology": PATHOLOGIES[idx],
            "confidence": float(probs[idx]),
            "heatmap": heatmap,
        })
    return results, image_np


def overlay_heatmap(image_np, heatmap, alpha=0.4):
    heatmap_colored = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0
    blended = alpha * heatmap_colored + (1 - alpha) * image_np
    return np.clip(blended, 0, 1)


_model = None
_grad_cam = None

def _get_model():
    global _model, _grad_cam
    if _model is None:
        _model = DenseNet121MultiLabel(num_classes=14, pretrained=False).to(DEVICE)
        _model.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        )
        _model.eval()
        _grad_cam = GradCAM(_model)
    return _model, _grad_cam


def analyze_xray(image_path: str, top_k: int = 5) -> dict:
    model, grad_cam = _get_model()
    orig_img = Image.open(image_path).convert("RGB")
    img_resized = orig_img.resize((IMG_SIZE, IMG_SIZE))
    image_np = np.array(img_resized) / 255.0
    input_tensor = val_transform(orig_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(input_tensor)
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    all_predictions = {
        PATHOLOGIES[i]: round(float(probs[i]), 4) for i in range(len(PATHOLOGIES))
    }
    detected = [
        {"pathology": PATHOLOGIES[i], "probability": round(float(probs[i]), 4)}
        for i in range(len(PATHOLOGIES)) if probs[i] > 0.3
    ]
    borderline = [
        {"pathology": PATHOLOGIES[i], "probability": round(float(probs[i]), 4)}
        for i in range(len(PATHOLOGIES)) if 0.15 <= probs[i] <= 0.3
    ]

    max_prob = float(probs.max())
    if max_prob > 0.7:
        severity = "CRITICAL"
    elif max_prob > 0.3:
        severity = "ABNORMAL"
    else:
        severity = "NORMAL"

    top_idx = int(np.argmax(probs))
    top_finding = PATHOLOGIES[top_idx]

    os.makedirs(GRADCAM_DIR, exist_ok=True)
    top_indices = np.argsort(probs)[::-1][:top_k]
    heatmap_paths = []

    for idx in top_indices:
        inp = val_transform(orig_img).unsqueeze(0).to(DEVICE)
        inp.requires_grad_(True)
        heatmap = grad_cam.generate(inp, int(idx))
        blended = overlay_heatmap(image_np, heatmap)
        basename = os.path.splitext(os.path.basename(image_path))[0]
        fname = os.path.join(GRADCAM_DIR, f"{basename}_{PATHOLOGIES[idx]}.png")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.imsave(fname, blended)
        heatmap_paths.append(fname)

    return {
        "source_image": image_path,
        "severity": severity,
        "top_finding": top_finding,
        "top_finding_confidence": round(float(probs[top_idx]), 4),
        "detected_pathologies": detected,
        "borderline_pathologies": borderline,
        "all_predictions": all_predictions,
        "gradcam_heatmap_paths": heatmap_paths,
    }


class XrayToolInput(BaseModel):
    image_path: str = Field(..., description="Path to the chest X-ray image file")

class XrayTool(BaseTool):
    name: str = "xray_triage"
    description: str = (
        "Analyze a chest X-ray image using a DenseNet121 deep learning model "
        "trained with the HERD optimizer on NIH ChestX-ray14 (112K images, 14 pathologies). "
        "Returns predicted pathologies with confidence scores, severity classification, "
        "and Grad-CAM heatmap paths highlighting regions of concern. "
        "Input: path to a chest X-ray image file (JPEG or PNG)."
    )
    args_schema: Type[BaseModel] = XrayToolInput

    def _run(self, image_path: str) -> str:
        try:
            result = analyze_xray(image_path)
            return json.dumps(result, indent=2)
        except FileNotFoundError:
            return json.dumps({"error": f"Image not found: {image_path}"})
        except Exception as e:
            return json.dumps({"error": f"Analysis failed: {str(e)}"})