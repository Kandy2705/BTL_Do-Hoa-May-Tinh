"""Roboflow Workflow inference helpers.

The UI keeps Roboflow optional: this module imports ``inference_sdk`` only when
the user actually runs the workflow, so the rest of the app can work offline.
"""

from __future__ import annotations

import csv
import base64
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from PIL import Image, ImageDraw


# Bảng mapping chuẩn: Tên class -> ID mới (0-6)
# Bỏ qua hoặc ID từ API, dùng mapping theo tên
BTL2_CLASS_NAME_TO_ID = {
    "person": 0,
    "car": 1,
    "bus": 2,
    "truck": 3,
    "motorcycle": 4,
    "motorbike": 4,  # alias
    "traffic light": 5,
    "traffic-light": 5,
    "trafficlight": 5,
    "stop sign": 6,
    "stop_sign": 6,
    "stopsign": 6,
    "traffic sign": 6,  # all traffic signs map to 6
}

def map_class_name_to_id(class_name: str) -> int:
    """Map class name to BTL2 ID (0-6). Print log for debugging."""
    normalized = class_name.lower().strip()
    
    # Try exact match first
    if normalized in BTL2_CLASS_NAME_TO_ID:
        mapped_id = BTL2_CLASS_NAME_TO_ID[normalized]
        print(f"  Phát hiện: {class_name} -> Ép về ID: {mapped_id}")
        return mapped_id
    
    # Try partial match for traffic signs
    if "traffic" in normalized or "sign" in normalized or "light" in normalized:
        print(f"  Phát hiện: {class_name} -> Ép về ID: 6 (traffic sign)")
        return 6
    
    # Unknown class
    print(f"  ⚠️ Phát hiện: {class_name} -> Ép về ID: 0 (unknown, default)")
    return 0

ROBOFLOW_CLASS_ALIASES = {
    "motorcycle": "motorbike",
    "motor bike": "motorbike",
    "traffic light": "traffic_light",
    "traffic-light": "traffic_light",
    "trafficlight": "traffic_light",
    "stop sign": "traffic_sign",
    "traffic sign": "traffic_sign",
    "traffic-sign": "traffic_sign",
    "trafficsign": "traffic_sign",
}


@dataclass
class RoboflowPrediction:
    image_name: str
    class_id: int | str
    class_name: str
    confidence: float
    x: float
    y: float
    width: float
    height: float

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        half_w = self.width * 0.5
        half_h = self.height * 0.5
        return self.x - half_w, self.y - half_h, self.x + half_w, self.y + half_h


def _looks_like_prediction(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    keys = set(item.keys())
    return {"x", "y", "width", "height"}.issubset(keys) and (
        "class" in keys or "class_name" in keys or "class_id" in keys
    )


def normalize_class_name(class_name: Any) -> str:
    normalized = str(class_name or "").strip().lower().replace("-", " ").replace("_", " ")
    normalized = " ".join(normalized.split())
    return ROBOFLOW_CLASS_ALIASES.get(normalized, normalized.replace(" ", "_"))


def _prediction_lists(payload: Any) -> Iterable[list[dict[str, Any]]]:
    """Yield every nested ``predictions`` list that looks like detections."""
    if isinstance(payload, dict):
        predictions = payload.get("predictions")
        if isinstance(predictions, list) and any(_looks_like_prediction(p) for p in predictions):
            yield predictions
        for value in payload.values():
            yield from _prediction_lists(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from _prediction_lists(item)


def extract_predictions(payload: Any, image_name: str, min_confidence: float = 0.0) -> List[RoboflowPrediction]:
    """Extract Roboflow detections from either Workflow or model JSON output."""
    extracted: list[RoboflowPrediction] = []
    seen = set()
    
    print(f"\n=== Xử lý ảnh: {image_name} ===")
    
    for predictions in _prediction_lists(payload):
        for pred in predictions:
            if not _looks_like_prediction(pred):
                continue
            confidence = float(pred.get("confidence", pred.get("score", 0.0)))
            if confidence < min_confidence:
                continue
            # Lấy tên class từ API
            class_name = normalize_class_name(pred.get("class_name", pred.get("class", pred.get("label", ""))))
            
            # Bỏ qua class_id từ API, dùng mapping theo tên
            class_id = map_class_name_to_id(class_name)
            
            normalized = RoboflowPrediction(
                image_name=image_name,
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                x=float(pred["x"]),
                y=float(pred["y"]),
                width=float(pred["width"]),
                height=float(pred["height"]),
            )
            dedupe_key = (
                normalized.class_id,
                normalized.class_name,
                round(normalized.confidence, 5),
                round(normalized.x, 2),
                round(normalized.y, 2),
                round(normalized.width, 2),
                round(normalized.height, 2),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            extracted.append(normalized)
    
    print(f"Tổng cộng: {len(extracted)} đối tượng\n")
    return extracted


def translate_predictions(
    rows: Sequence[RoboflowPrediction],
    *,
    dx: float,
    dy: float,
    image_name: str,
) -> list[RoboflowPrediction]:
    return [
        RoboflowPrediction(
            image_name=image_name,
            class_id=row.class_id,
            class_name=row.class_name,
            confidence=row.confidence,
            x=row.x + dx,
            y=row.y + dy,
            width=row.width,
            height=row.height,
        )
        for row in rows
    ]


def _iou(a: RoboflowPrediction, b: RoboflowPrediction) -> float:
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0.0 else 0.0


def merge_predictions(
    rows: Sequence[RoboflowPrediction],
    *,
    iou_threshold: float = 0.55,
) -> list[RoboflowPrediction]:
    merged: list[RoboflowPrediction] = []
    for row in sorted(rows, key=lambda item: item.confidence, reverse=True):
        duplicate = any(row.class_name == kept.class_name and _iou(row, kept) >= iou_threshold for kept in merged)
        if not duplicate:
            merged.append(row)
    return merged


def save_json(payload: Any, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return output_path


def append_csv_rows(csv_path: Path, rows: Sequence[RoboflowPrediction], write_header: bool = False) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header or not file_exists:
            writer.writerow(
                [
                    "image_name",
                    "class_id",
                    "class_name",
                    "confidence",
                    "x",
                    "y",
                    "width",
                    "height",
                    "x_min",
                    "y_min",
                    "x_max",
                    "y_max",
                ]
            )
        for row in rows:
            x_min, y_min, x_max, y_max = row.xyxy
            writer.writerow(
                [
                    row.image_name,
                    row.class_id,
                    row.class_name,
                    round(row.confidence, 6),
                    round(row.x, 3),
                    round(row.y, 3),
                    round(row.width, 3),
                    round(row.height, 3),
                    round(x_min, 3),
                    round(y_min, 3),
                    round(x_max, 3),
                    round(y_max, 3),
                ]
            )
    return csv_path


def draw_predictions(image_path: Path, rows: Sequence[RoboflowPrediction], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    palette = [
        (0, 220, 255),
        (255, 80, 180),
        (255, 230, 0),
        (80, 255, 130),
        (255, 120, 40),
        (160, 110, 255),
        (255, 255, 255),
    ]
    for idx, row in enumerate(rows):
        color = palette[idx % len(palette)]
        x_min, y_min, x_max, y_max = row.xyxy
        draw.rectangle((x_min, y_min, x_max, y_max), outline=color, width=3)
        label = f"{row.class_name} {row.confidence:.2f}".strip()
        text_box = draw.textbbox((x_min, y_min), label)
        pad = 3
        bg = (text_box[0], text_box[1] - pad, text_box[2] + pad * 2, text_box[3] + pad)
        draw.rectangle(bg, fill=color)
        draw.text((x_min + pad, y_min - 1), label, fill=(20, 20, 20))
    image.save(output_path)
    return output_path


def _annotated_image_values(payload: Any) -> Iterable[str]:
    if isinstance(payload, dict):
        value = payload.get("annotated_image")
        if isinstance(value, str) and value.strip():
            yield value
        elif isinstance(value, dict):
            nested_value = value.get("value")
            if isinstance(nested_value, str) and nested_value.strip():
                yield nested_value
        for nested in payload.values():
            yield from _annotated_image_values(nested)
    elif isinstance(payload, list):
        for item in payload:
            yield from _annotated_image_values(item)


def save_first_annotated_image(payload: Any, output_path: Path) -> Path | None:
    """Save Roboflow's own annotated image, if the Workflow returned one."""
    for encoded in _annotated_image_values(payload):
        if "," in encoded and encoded.lstrip().startswith("data:image"):
            encoded = encoded.split(",", 1)[1]
        try:
            raw = base64.b64decode(encoded)
            image = Image.open(BytesIO(raw)).convert("RGB")
        except Exception:
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        return output_path
    return None


def run_inference(
    *,
    api_url: str,
    api_key: str,
    workspace_name: str,
    model_id: str,
    image_path: Path,
) -> Any:
    """Run Roboflow direct inference (not workflow). More stable."""
    try:
        from inference_sdk import InferenceHTTPClient
    except ImportError as exc:
        raise RuntimeError("Thieu package inference-sdk. Cai bang: pip install inference-sdk") from exc

    if api_key.strip() in {"rf_xxx", "YOUR_API_KEY", "MÃ_API_THẬT_CỦA_BẠN_ỞĐÂY"}:
        raise RuntimeError("Roboflow API key dang la placeholder.")

    client = InferenceHTTPClient(api_url=api_url, api_key=api_key)

    # Model ID format: workspace/model-name or project-id/model-version
    full_model_id = f"{workspace_name}/{model_id}" if "/" not in model_id else model_id

    try:
        result = client.infer(str(image_path), model_id=full_model_id)
        # Wrap in same format as workflow for compatibility
        return [{"detections": {"image": {"width": 1280, "height": 720}, "predictions": result.get("predictions", [])}}]
    except Exception as exc:
        message = str(exc)
        if "401" in message or "Unauthorized" in message:
            raise RuntimeError(f"Roboflow 401: API key invalid for model {full_model_id}") from exc
        if "403" in message:
            raise RuntimeError(f"Roboflow 403: Forbidden - API key may be revoked") from exc
        raise RuntimeError(f"Roboflow inference failed: {message}") from exc


def run_workflow(
    *,
    api_url: str,
    api_key: str,
    workspace_name: str,
    workflow_id: str,
    image_path: Path,
    use_cache: bool = False,
    workflow_version_id: str | None = None,
) -> Any:
    try:
        from inference_sdk import InferenceHTTPClient
    except ImportError as exc:
        raise RuntimeError(
            "Thieu package inference-sdk. Cai bang: pip install inference-sdk"
        ) from exc

    if api_key.strip() in {"rf_xxx", "YOUR_API_KEY", "MÃ_API_THẬT_CỦA_BẠN_Ở_ĐÂY"}:
        raise RuntimeError("Roboflow API key dang la placeholder. Hay thay bang key that.")

    client = InferenceHTTPClient(api_url=api_url, api_key=api_key)

    # inference-sdk 1.x expects images=..., while some Roboflow snippets/docs
    # show inputs=.... Try the installed SDK shape first, then fallback.
    try:
        try:
            return client.run_workflow(
                workspace_name=workspace_name,
                workflow_id=workflow_id,
                images={"image": str(image_path)},
                use_cache=use_cache,
                workflow_version_id=workflow_version_id or None,
            )
        except TypeError as exc:
            if "unexpected keyword argument 'images'" not in str(exc):
                raise
            return client.run_workflow(
                workspace_name=workspace_name,
                workflow_id=workflow_id,
                inputs={"image": str(image_path)},
            )
    except Exception as exc:
        status_code = getattr(exc, "status_code", None)
        api_message = getattr(exc, "api_message", "")
        message = str(exc)
        if status_code == 401 or "401" in message or "Unauthorized" in message:
            raise RuntimeError(
                "Roboflow 401 Unauthorized: API key khong hop le hoac khong co quyen "
                f"voi workspace '{workspace_name}' / workflow '{workflow_id}'. "
                "Hay copy key that tu Roboflow Workspace Settings > API Keys."
            ) from exc
        if status_code == 404 or "404" in message:
            raise RuntimeError(
                f"Roboflow 404 Not Found: khong tim thay workspace '{workspace_name}' "
                f"hoac workflow '{workflow_id}'. Kiem tra lai ten workspace/workflow id."
            ) from exc
        if api_message:
            if "custom Python code" in api_message or "dynamic blocks" in api_message:
                raise RuntimeError(
                    "Roboflow Workflow co custom Python/dynamic block nhung endpoint hien tai "
                    "khong cho chay custom code. Hay bo block custom Python trong Workflow, "
                    "hoac deploy Workflow tren moi truong Roboflow/self-host co bat custom code."
                ) from exc
            raise RuntimeError(f"Roboflow error: {api_message}") from exc
        raise
