"""YOLO11 instance-segmentation inference and smoke-specific post-processing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class SmokeInstance:
    """One accepted smoke mask expressed in original-frame coordinates."""

    class_id: int
    class_name: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]
    area_pixels: int
    area_ratio: float
    polygon: tuple[tuple[float, float], ...]

    def to_record(self) -> dict[str, object]:
        return {
            "cls_id": self.class_id,
            "cls": self.class_name,
            "conf": round(self.confidence, 6),
            "box_xyxy": [round(value, 2) for value in self.box_xyxy],
            "area_pixels": self.area_pixels,
            "area_ratio": round(self.area_ratio, 8),
            "polygon": [
                [round(point[0], 2), round(point[1], 2)] for point in self.polygon
            ],
        }


@dataclass
class SmokeDetection:
    smoke: bool
    max_confidence: float
    smoke_area_ratio: float
    instance_count: int
    frame_shape: tuple[int, ...]
    roi_xyxy: Optional[tuple[int, int, int, int]]
    annotated_frame: np.ndarray
    instances: tuple[SmokeInstance, ...] = ()

    def to_record(self) -> dict[str, object]:
        classification = "smoke" if self.smoke else "no_smoke"
        smoke_area_pixels = sum(instance.area_pixels for instance in self.instances)
        if self.roi_xyxy:
            roi_area = max(
                1,
                (self.roi_xyxy[2] - self.roi_xyxy[0])
                * (self.roi_xyxy[3] - self.roi_xyxy[1]),
            )
        else:
            roi_area = max(1, self.frame_shape[0] * self.frame_shape[1])
        return {
            "smoke": self.smoke,
            "classification": classification,
            "cls": classification,
            "conf": round(self.max_confidence, 6),
            "area": round(self.smoke_area_ratio, 8),
            "max_confidence": round(self.max_confidence, 6),
            "smoke_area_ratio": round(self.smoke_area_ratio, 8),
            "smoke_area_pixels": smoke_area_pixels,
            "roi_area_pixels": roi_area,
            "instance_count": self.instance_count,
            "frame_height": self.frame_shape[0],
            "frame_width": self.frame_shape[1],
            "roi_xyxy": list(self.roi_xyxy) if self.roi_xyxy else None,
            "instances": [instance.to_record() for instance in self.instances],
        }


def _choose_device(device: str) -> str | int:
    if device.lower() != "auto":
        try:
            return int(device)
        except ValueError:
            return device
    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class SmokeDetector:
    """Run a one-class YOLO11-seg model on a full frame or configured ROI."""

    def __init__(
        self,
        model_path: Path,
        imgsz: int = 640,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        min_smoke_area_ratio: float = 0.005,
        device: str = "auto",
        roi_xyxy: Optional[tuple[int, int, int, int]] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.min_smoke_area_ratio = min_smoke_area_ratio
        self.device = _choose_device(device)
        self.roi_xyxy = roi_xyxy
        logger.info("Loading model %s on device %s", self.model_path, self.device)
        self.model = YOLO(str(self.model_path))

    def _crop(self, frame: np.ndarray) -> tuple[np.ndarray, Optional[tuple[int, int, int, int]]]:
        if not self.roi_xyxy:
            return frame, None
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = self.roi_xyxy
        if x2 <= x1 or y2 <= y1:
            raise ValueError("ROI must satisfy x2>x1 and y2>y1")
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width, x2))
        y2 = max(y1 + 1, min(height, y2))
        actual_roi = (x1, y1, x2, y2)
        return frame[y1:y2, x1:x2].copy(), actual_roi

    @staticmethod
    def _as_numpy(value: object) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _class_name(result: object, class_id: int) -> str:
        names = getattr(result, "names", {})
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        try:
            return str(names[class_id])
        except (IndexError, KeyError, TypeError):
            return str(class_id)

    @staticmethod
    def _fallback_polygon(binary: np.ndarray) -> tuple[tuple[float, float], ...]:
        contours_result = cv2.findContours(
            (binary > 0).astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contours = contours_result[0] if len(contours_result) == 2 else contours_result[1]
        if not contours:
            return ()
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(contour, max(0.5, perimeter * 0.002), True)
        return tuple(
            (float(point[0][0]), float(point[0][1])) for point in simplified
        )

    @classmethod
    def _result_polygon(
        cls,
        masks: object,
        index: int,
        binary: np.ndarray,
    ) -> tuple[tuple[float, float], ...]:
        polygons = getattr(masks, "xy", None)
        if polygons is not None:
            try:
                polygon = polygons[index]
                points = tuple(
                    (float(point[0]), float(point[1]))
                    for point in polygon
                    if len(point) >= 2
                )
                if len(points) >= 3:
                    return points
            except (IndexError, TypeError, ValueError):
                pass
        return cls._fallback_polygon(binary)

    def detect(self, frame: np.ndarray) -> SmokeDetection:
        if frame is None or frame.size == 0:
            raise ValueError("Empty frame received")

        crop, actual_roi = self._crop(frame)
        results = self.model.predict(
            source=crop,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        combined_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        max_confidence = 0.0
        instances: list[SmokeInstance] = []

        if result.masks is not None and result.boxes is not None:
            raw_masks = getattr(result.masks, "data", None)
            masks = self._as_numpy(raw_masks) if raw_masks is not None else np.empty((0,))
            confidences = self._as_numpy(result.boxes.conf).reshape(-1)
            class_ids = self._as_numpy(result.boxes.cls).astype(int).reshape(-1)
            raw_boxes = getattr(result.boxes, "xyxy", None)
            if raw_boxes is None:
                boxes_xyxy = np.empty((0, 4), dtype=float)
            else:
                boxes_xyxy = self._as_numpy(raw_boxes)
                if boxes_xyxy.ndim == 1:
                    boxes_xyxy = boxes_xyxy.reshape(-1, 4)

            for index, (mask, confidence, class_id) in enumerate(
                zip(masks, confidences, class_ids)
            ):
                # The training dataset contains one class: smoke.
                if class_id != 0 or float(confidence) < self.conf_threshold:
                    continue
                binary = (mask > 0.5).astype(np.uint8)
                if binary.shape != crop.shape[:2]:
                    binary = cv2.resize(
                        binary,
                        (crop.shape[1], crop.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                combined_mask = np.maximum(combined_mask, binary)
                max_confidence = max(max_confidence, float(confidence))

                area_pixels = int(np.count_nonzero(binary))
                crop_height, crop_width = crop.shape[:2]
                if index < len(boxes_xyxy):
                    box_values = [float(value) for value in boxes_xyxy[index][:4]]
                else:
                    ys, xs = np.where(binary > 0)
                    box_values = (
                        [
                            float(xs.min()),
                            float(ys.min()),
                            float(xs.max() + 1),
                            float(ys.max() + 1),
                        ]
                        if len(xs)
                        else [0.0, 0.0, 0.0, 0.0]
                    )
                origin_x, origin_y = (
                    (actual_roi[0], actual_roi[1]) if actual_roi else (0, 0)
                )
                frame_height, frame_width = frame.shape[:2]
                full_box = (
                    max(0.0, min(float(frame_width), box_values[0] + origin_x)),
                    max(0.0, min(float(frame_height), box_values[1] + origin_y)),
                    max(0.0, min(float(frame_width), box_values[2] + origin_x)),
                    max(0.0, min(float(frame_height), box_values[3] + origin_y)),
                )
                crop_polygon = self._result_polygon(result.masks, index, binary)
                full_polygon = tuple(
                    (
                        max(0.0, min(float(frame_width - 1), point[0] + origin_x)),
                        max(0.0, min(float(frame_height - 1), point[1] + origin_y)),
                    )
                    for point in crop_polygon
                )
                instances.append(
                    SmokeInstance(
                        class_id=int(class_id),
                        class_name=self._class_name(result, int(class_id)),
                        confidence=float(confidence),
                        box_xyxy=full_box,
                        area_pixels=area_pixels,
                        area_ratio=float(area_pixels) / max(1, crop_width * crop_height),
                        polygon=full_polygon,
                    )
                )

        crop_area = max(1, crop.shape[0] * crop.shape[1])
        smoke_area_ratio = float(np.count_nonzero(combined_mask)) / crop_area
        smoke = bool(instances) and smoke_area_ratio >= self.min_smoke_area_ratio

        annotated = frame.copy()
        if np.any(combined_mask):
            color_layer = np.zeros_like(crop)
            color_layer[:, :] = (0, 0, 255)  # BGR red overlay
            blended = cv2.addWeighted(crop, 0.65, color_layer, 0.35, 0)
            crop_mask = combined_mask.astype(bool)
            annotated_crop = crop.copy()
            annotated_crop[crop_mask] = blended[crop_mask]
            if actual_roi:
                x1, y1, x2, y2 = actual_roi
                annotated[y1:y2, x1:x2] = annotated_crop
            else:
                annotated = annotated_crop

        if actual_roi:
            x1, y1, x2, y2 = actual_roi
            cv2.rectangle(annotated, (x1, y1), (x2 - 1, y2 - 1), (255, 180, 0), 2)

        for instance in instances:
            x1, y1, x2, y2 = (int(round(value)) for value in instance.box_xyxy)
            cv2.rectangle(
                annotated,
                (x1, y1),
                (max(x1, x2 - 1), max(y1, y2 - 1)),
                (0, 255, 255),
                2,
            )
            if len(instance.polygon) >= 3:
                polygon_array = np.asarray(instance.polygon, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(annotated, [polygon_array], True, (0, 128, 255), 2)
            instance_label = (
                f"cls={instance.class_name} conf={instance.confidence:.2f} "
                f"area={instance.area_ratio:.3f}"
            )
            cv2.putText(
                annotated,
                instance_label,
                (max(4, x1), min(annotated.shape[0] - 8, max(55, y1 - 6))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        status = "SMOKE" if smoke else "NO SMOKE"
        status_color = (0, 0, 255) if smoke else (0, 180, 0)
        label = (
            f"{status}  cls={'smoke' if smoke else 'no_smoke'}  "
            f"conf={max_confidence:.2f}  area={smoke_area_ratio:.3f}  "
            f"instances={len(instances)}"
        )
        label_right = max(8, min(annotated.shape[1] - 8, max(220, 14 + len(label) * 9)))
        cv2.rectangle(annotated, (8, 8), (label_right, 38), (0, 0, 0), -1)
        cv2.putText(
            annotated,
            label,
            (15, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            status_color,
            2,
            cv2.LINE_AA,
        )

        return SmokeDetection(
            smoke=smoke,
            max_confidence=max_confidence,
            smoke_area_ratio=smoke_area_ratio,
            instance_count=len(instances),
            frame_shape=frame.shape,
            roi_xyxy=actual_roi,
            annotated_frame=annotated,
            instances=tuple(instances),
        )
