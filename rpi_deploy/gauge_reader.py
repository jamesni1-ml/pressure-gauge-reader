#!/usr/bin/env python3
"""
Analog Pressure Gauge Reader — Raspberry Pi 4 Inference Script

Pipeline: YOLOv8n (ONNX) → Perspective Correction → EasyOCR → Needle Detection → Reading

Usage:
    python3 gauge_reader.py --input ./photos/ --output results.csv
    python3 gauge_reader.py --input ./photos/ --output results.csv --save-annotated
    python3 gauge_reader.py --single photo.jpg
"""

import argparse
import csv
import math
import sys
import time
from collections import namedtuple
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# EasyOCR is imported lazily to show a clear message if missing
try:
    import easyocr
except ImportError:
    print("ERROR: easyocr not installed. Run: pip install easyocr")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

ScaleMarking = namedtuple('ScaleMarking', ['value', 'angle', 'position'])


@dataclass
class GaugeScale:
    markings: list
    min_value: float = 0.0
    max_value: float = 0.0
    start_angle: float = 0.0
    end_angle: float = 0.0
    sweep: float = 0.0


@dataclass
class GaugeReading:
    value: float = None
    scale_min: float = None
    scale_max: float = None
    needle_angle: float = None
    confidence: float = 0.0
    num_scale_markings: int = 0
    status: str = 'OK'
    annotated_image: np.ndarray = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# YOLO ONNX inference via OpenCV DNN
# ---------------------------------------------------------------------------

class YOLODetector:
    """Lightweight YOLOv8 ONNX detector using OpenCV DNN — no PyTorch needed."""

    def __init__(self, model_path, input_size=640, conf_threshold=0.5, nms_threshold=0.45):
        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

    def detect(self, image):
        """Run detection on a BGR image. Returns list of (x1, y1, x2, y2, conf, class_id)."""
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (self.input_size, self.input_size),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward()

        # YOLOv8 output shape: (1, num_classes+4, num_detections)
        output = outputs[0]
        if output.shape[0] == 1:
            output = output[0]

        # Transpose if needed: (num_classes+4, N) -> (N, num_classes+4)
        if output.shape[0] < output.shape[1]:
            output = output.T

        boxes = []
        confidences = []
        class_ids = []

        x_scale = w / self.input_size
        y_scale = h / self.input_size

        for detection in output:
            # detection: [cx, cy, w, h, class_scores...]
            cx, cy, bw, bh = detection[:4]
            scores = detection[4:]
            class_id = int(np.argmax(scores))
            conf = float(scores[class_id])

            if conf < self.conf_threshold:
                continue

            x1 = int((cx - bw / 2) * x_scale)
            y1 = int((cy - bh / 2) * y_scale)
            x2 = int((cx + bw / 2) * x_scale)
            y2 = int((cy + bh / 2) * y_scale)

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(conf)
            class_ids.append(class_id)

        # NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, bw, bh = boxes[i]
                results.append((x, y, x + bw, y + bh, confidences[i], class_ids[i]))

        return results


# ---------------------------------------------------------------------------
# Perspective correction
# ---------------------------------------------------------------------------

def correct_perspective(gauge_crop):
    gray = cv2.cvtColor(gauge_crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gauge_crop

    valid_contours = [c for c in contours if len(c) >= 5]
    if not valid_contours:
        return gauge_crop

    largest = max(valid_contours, key=cv2.contourArea)
    ellipse = cv2.fitEllipse(largest)
    center, (axis_a, axis_b), angle = ellipse

    aspect_ratio = min(axis_a, axis_b) / max(axis_a, axis_b) if max(axis_a, axis_b) > 0 else 1
    if aspect_ratio > 0.9:
        return gauge_crop

    h, w = gauge_crop.shape[:2]
    target_size = int(max(axis_a, axis_b))

    M_rotate = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gauge_crop, M_rotate, (w, h))

    scale_x = max(axis_a, axis_b) / min(axis_a, axis_b) if axis_a < axis_b else 1.0
    scale_y = max(axis_a, axis_b) / min(axis_a, axis_b) if axis_b < axis_a else 1.0

    M_scale = np.float32([[scale_x, 0, center[0] * (1 - scale_x)],
                          [0, scale_y, center[1] * (1 - scale_y)]])
    corrected = cv2.warpAffine(rotated, M_scale, (int(w * scale_x), int(h * scale_y)))

    cx, cy = int(center[0] * scale_x), int(center[1] * scale_y)
    radius = target_size // 2
    x1 = max(0, cx - radius)
    y1 = max(0, cy - radius)
    x2 = min(corrected.shape[1], cx + radius)
    y2 = min(corrected.shape[0], cy + radius)

    cropped = corrected[y1:y2, x1:x2]
    return cropped if cropped.size > 0 else gauge_crop


# ---------------------------------------------------------------------------
# Gauge center detection
# ---------------------------------------------------------------------------

def find_gauge_center(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    h, w = gray.shape
    min_radius = min(h, w) // 6
    max_radius = min(h, w) // 2

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=min(h, w) // 3, param1=100, param2=50,
        minRadius=min_radius, maxRadius=max_radius
    )

    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        img_center = np.array([w // 2, h // 2])
        dists = [np.linalg.norm(np.array([c[0], c[1]]) - img_center) for c in circles]
        best = circles[np.argmin(dists)]
        return int(best[0]), int(best[1]), int(best[2])

    edges = cv2.Canny(blurred, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(largest)
        return int(cx), int(cy), int(radius)

    return w // 2, h // 2, min(h, w) // 3


# ---------------------------------------------------------------------------
# OCR scale reading
# ---------------------------------------------------------------------------

def read_scale_numbers(image, center, radius, reader):
    cx, cy = center
    h, w = image.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), int(radius * 0.92), 255, -1)
    cv2.circle(mask, (cx, cy), int(radius * 0.45), 0, -1)

    masked = cv2.bitwise_and(image, image, mask=mask)
    results = reader.readtext(masked, allowlist='0123456789.')

    markings = []
    for bbox, text, conf in results:
        text = text.strip()
        try:
            value = float(text)
        except ValueError:
            continue
        if conf < 0.3:
            continue

        bbox_cx = sum(pt[0] for pt in bbox) / 4
        bbox_cy = sum(pt[1] for pt in bbox) / 4

        dx = bbox_cx - cx
        dy = -(bbox_cy - cy)
        angle = math.degrees(math.atan2(dx, dy)) % 360

        markings.append(ScaleMarking(value=value, angle=angle, position=(bbox_cx, bbox_cy)))

    markings.sort(key=lambda m: m.angle)
    return markings


# ---------------------------------------------------------------------------
# Scale builder
# ---------------------------------------------------------------------------

def build_scale(markings):
    if len(markings) < 2:
        return None

    by_value = sorted(markings, key=lambda m: m.value)
    min_val = by_value[0].value
    max_val = by_value[-1].value
    start_angle = by_value[0].angle
    end_angle = by_value[-1].angle

    sweep = end_angle - start_angle
    if sweep < 0:
        sweep += 360

    return GaugeScale(
        markings=by_value,
        min_value=min_val,
        max_value=max_val,
        start_angle=start_angle,
        end_angle=end_angle,
        sweep=sweep,
    )


# ---------------------------------------------------------------------------
# Needle detection
# ---------------------------------------------------------------------------

def detect_needle(image, center, radius):
    cx, cy = center
    h, w = image.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), int(radius * 0.90), 255, -1)
    cv2.circle(mask, (cx, cy), int(radius * 0.15), 0, -1)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    dark_mask = cv2.inRange(gray, 0, 60)

    red_mask = (cv2.inRange(hsv, np.array([0, 100, 50]), np.array([10, 255, 255])) |
                cv2.inRange(hsv, np.array([170, 100, 50]), np.array([180, 255, 255])))

    needle_mask = cv2.bitwise_or(dark_mask, red_mask)
    needle_mask = cv2.bitwise_and(needle_mask, mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    needle_mask = cv2.morphologyEx(needle_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    needle_mask = cv2.morphologyEx(needle_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    lines = cv2.HoughLinesP(needle_mask, 1, np.pi / 180, 30,
                            minLineLength=int(radius * 0.25),
                            maxLineGap=int(radius * 0.1))

    if lines is None:
        return _needle_from_contours(needle_mask, cx, cy, radius)

    best_line = None
    best_score = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        line_len = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if line_len < 1:
            continue
        dist = abs((y2 - y1) * cx - (x2 - x1) * cy + x2 * y1 - y2 * x1) / line_len
        if dist < radius * 0.3:
            score = line_len / (dist + 1)
            if score > best_score:
                best_score = score
                best_line = (x1, y1, x2, y2)

    if best_line is None:
        return _needle_from_contours(needle_mask, cx, cy, radius)

    x1, y1, x2, y2 = best_line
    d1 = math.sqrt((x1 - cx) ** 2 + (y1 - cy) ** 2)
    d2 = math.sqrt((x2 - cx) ** 2 + (y2 - cy) ** 2)
    tip_x, tip_y = (x1, y1) if d1 > d2 else (x2, y2)

    dx = tip_x - cx
    dy = -(tip_y - cy)
    return math.degrees(math.atan2(dx, dy)) % 360


def _needle_from_contours(needle_mask, cx, cy, radius):
    contours, _ = cv2.findContours(needle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_contour = None
    best_length = 0
    for c in contours:
        arc_len = cv2.arcLength(c, closed=False)
        if arc_len > radius * 0.3 and arc_len > best_length:
            best_length = arc_len
            best_contour = c

    if best_contour is None:
        return None

    dists = [math.sqrt((pt[0][0] - cx) ** 2 + (pt[0][1] - cy) ** 2) for pt in best_contour]
    tip_idx = int(np.argmax(dists))
    tip_x, tip_y = best_contour[tip_idx][0]

    dx = tip_x - cx
    dy = -(tip_y - cy)
    return math.degrees(math.atan2(dx, dy)) % 360


# ---------------------------------------------------------------------------
# Angle to reading
# ---------------------------------------------------------------------------

def angle_to_reading(needle_angle, scale):
    if scale is None or len(scale.markings) < 2:
        return None

    needle_rel = needle_angle - scale.start_angle
    if needle_rel < 0:
        needle_rel += 360

    mark_angles = []
    for m in scale.markings:
        a = m.angle - scale.start_angle
        if a < 0:
            a += 360
        mark_angles.append(a)

    if needle_rel <= mark_angles[0]:
        return scale.markings[0].value
    if needle_rel >= mark_angles[-1]:
        return scale.markings[-1].value

    for i in range(len(mark_angles) - 1):
        if mark_angles[i] <= needle_rel <= mark_angles[i + 1]:
            t = (needle_rel - mark_angles[i]) / (mark_angles[i + 1] - mark_angles[i])
            value = scale.markings[i].value + t * (scale.markings[i + 1].value - scale.markings[i].value)
            return round(value, 2)

    t = needle_rel / scale.sweep if scale.sweep > 0 else 0
    return round(scale.min_value + t * (scale.max_value - scale.min_value), 2)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def read_gauge(image, detector, ocr_reader):
    reading = GaugeReading()
    annotated = image.copy()

    detections = detector.detect(image)
    if not detections:
        reading.status = 'NO_GAUGE_DETECTED'
        reading.annotated_image = annotated
        return reading

    # Best detection by confidence
    best = max(detections, key=lambda d: d[4])
    x1, y1, x2, y2, det_conf, _ = best

    pad_x = int((x2 - x1) * 0.1)
    pad_y = int((y2 - y1) * 0.1)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(image.shape[1], x2 + pad_x)
    y2 = min(image.shape[0], y2 + pad_y)

    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

    gauge_crop = image[y1:y2, x1:x2]
    corrected = correct_perspective(gauge_crop)

    center_result = find_gauge_center(corrected)
    if center_result is None:
        reading.status = 'CENTER_NOT_FOUND'
        reading.annotated_image = annotated
        return reading

    cx, cy, radius = center_result

    markings = read_scale_numbers(corrected, (cx, cy), radius, ocr_reader)
    reading.num_scale_markings = len(markings)

    if len(markings) < 2:
        reading.status = 'INSUFFICIENT_SCALE_MARKINGS'
        reading.annotated_image = annotated
        return reading

    scale = build_scale(markings)
    if scale is None:
        reading.status = 'SCALE_BUILD_FAILED'
        reading.annotated_image = annotated
        return reading

    reading.scale_min = scale.min_value
    reading.scale_max = scale.max_value

    needle_angle = detect_needle(corrected, (cx, cy), radius)
    if needle_angle is None:
        reading.status = 'NEEDLE_NOT_FOUND'
        reading.annotated_image = annotated
        return reading

    reading.needle_angle = needle_angle
    reading.value = angle_to_reading(needle_angle, scale)
    reading.confidence = det_conf
    reading.status = 'OK'

    label = f"{reading.value:.1f} (conf: {det_conf:.2f})"
    cv2.putText(annotated, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.circle(annotated, (x1 + cx, y1 + cy), 5, (255, 0, 0), -1)

    reading.annotated_image = annotated
    return reading


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def batch_process(image_folder, detector, ocr_reader, output_csv, save_annotated=False):
    image_folder = Path(image_folder)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_paths = sorted([p for p in image_folder.iterdir() if p.suffix.lower() in extensions])

    if not image_paths:
        print(f"No images found in {image_folder}")
        return

    annotated_dir = None
    if save_annotated:
        annotated_dir = output_csv.parent / 'annotated'
        annotated_dir.mkdir(parents=True, exist_ok=True)

    total = len(image_paths)
    ok_count = 0
    times = []

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'filename', 'reading', 'scale_min', 'scale_max',
            'needle_angle', 'confidence', 'num_markings', 'status', 'time_sec'
        ])
        writer.writeheader()

        for i, img_path in enumerate(image_paths, 1):
            t0 = time.time()
            img = cv2.imread(str(img_path))

            if img is None:
                writer.writerow({
                    'filename': img_path.name, 'reading': '', 'scale_min': '',
                    'scale_max': '', 'needle_angle': '', 'confidence': 0,
                    'num_markings': 0, 'status': 'IMAGE_READ_ERROR', 'time_sec': 0
                })
                continue

            result = read_gauge(img, detector, ocr_reader)
            elapsed = time.time() - t0
            times.append(elapsed)

            if result.status == 'OK':
                ok_count += 1

            writer.writerow({
                'filename': img_path.name,
                'reading': result.value if result.value is not None else '',
                'scale_min': result.scale_min if result.scale_min is not None else '',
                'scale_max': result.scale_max if result.scale_max is not None else '',
                'needle_angle': f"{result.needle_angle:.1f}" if result.needle_angle is not None else '',
                'confidence': f"{result.confidence:.3f}",
                'num_markings': result.num_scale_markings,
                'status': result.status,
                'time_sec': f"{elapsed:.2f}",
            })

            if save_annotated and annotated_dir and result.annotated_image is not None:
                cv2.imwrite(str(annotated_dir / f"annotated_{img_path.name}"), result.annotated_image)

            print(f"  [{i}/{total}] {img_path.name}: "
                  f"{'OK' if result.status == 'OK' else result.status} "
                  f"({elapsed:.1f}s)"
                  + (f" -> {result.value:.1f}" if result.value is not None else ""))

    avg_time = sum(times) / len(times) if times else 0

    print(f"\n{'=' * 50}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'=' * 50}")
    print(f"Total images:   {total}")
    print(f"Successful:     {ok_count} ({ok_count / total * 100:.1f}%)")
    print(f"Failed:         {total - ok_count}")
    print(f"Avg time/image: {avg_time:.2f} sec")
    print(f"Total time:     {sum(times):.1f} sec")
    print(f"Results saved:  {output_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Analog Pressure Gauge Reader — Raspberry Pi 4 Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 gauge_reader.py --input ./photos/ --output results.csv
  python3 gauge_reader.py --input ./photos/ --output results.csv --save-annotated
  python3 gauge_reader.py --single photo.jpg
        """)

    parser.add_argument('--model', default='gauge_detector.onnx',
                        help='Path to ONNX model (default: gauge_detector.onnx)')
    parser.add_argument('--input', type=str,
                        help='Directory of images to process (batch mode)')
    parser.add_argument('--output', type=str, default='results.csv',
                        help='Output CSV path (default: results.csv)')
    parser.add_argument('--single', type=str,
                        help='Process a single image')
    parser.add_argument('--save-annotated', action='store_true',
                        help='Save annotated images with detections')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Detection confidence threshold (default: 0.5)')

    args = parser.parse_args()

    if not args.input and not args.single:
        parser.error("Specify --input <folder> for batch mode or --single <image> for single image")

    # Load models
    model_path = Path(args.model)
    if not model_path.exists():
        # Try relative to script directory
        model_path = Path(__file__).parent / args.model
    if not model_path.exists():
        print(f"ERROR: Model not found at {args.model}")
        sys.exit(1)

    print(f"Loading YOLO model: {model_path}")
    detector = YOLODetector(model_path, conf_threshold=args.conf)

    print("Loading EasyOCR (first run downloads models ~100MB)...")
    reader = easyocr.Reader(['en'], gpu=False)

    if args.single:
        # Single image mode
        img = cv2.imread(args.single)
        if img is None:
            print(f"ERROR: Cannot read image {args.single}")
            sys.exit(1)

        t0 = time.time()
        result = read_gauge(img, detector, reader)
        elapsed = time.time() - t0

        if result.status == 'OK':
            print(f"\nReading: {result.value:.1f}")
            print(f"Scale:   {result.scale_min} - {result.scale_max}")
            print(f"Angle:   {result.needle_angle:.1f} degrees")
            print(f"Conf:    {result.confidence:.3f}")
        else:
            print(f"\nFailed: {result.status}")

        print(f"Time:    {elapsed:.2f} sec")

        if args.save_annotated and result.annotated_image is not None:
            out_path = f"annotated_{Path(args.single).name}"
            cv2.imwrite(out_path, result.annotated_image)
            print(f"Saved:   {out_path}")
    else:
        # Batch mode
        batch_process(args.input, detector, reader, args.output, args.save_annotated)


if __name__ == '__main__':
    main()
