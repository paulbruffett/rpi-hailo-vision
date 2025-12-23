"""
Headless variant of the detection pipeline that yields detections (label, bbox, prob)
from a Python loop instead of drawing overlays. This reuses the existing
`detection-pipeline.py` helpers but swaps the sink for a fakesink and routes
results through a queue.
"""

from __future__ import annotations

import importlib.util
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

# Optional: if the Hailo GI module is available we try to use it first.
try:  # pragma: no cover - platform dependent
    gi.require_version("Hailo", "1.0")
    from gi.repository import Hailo  # type: ignore
except (ValueError, ImportError):  # pragma: no cover - platform dependent
    Hailo = None  # type: ignore


# ---------------------------------------------------------------------------
# Load the existing detection-pipeline helpers without modifying that file.
# The filename contains a dash, so we import it via importlib.
# ---------------------------------------------------------------------------
_DETECTION_PIPELINE_PATH = Path(__file__).with_name("detection-pipeline.py")
_spec = importlib.util.spec_from_file_location("detection_pipeline", _DETECTION_PIPELINE_PATH)
if _spec is None or _spec.loader is None:  # pragma: no cover - defensive
    raise ImportError(f"Failed to load detection-pipeline.py from {_DETECTION_PIPELINE_PATH}")
_detection_pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_detection_pipeline)  # type: ignore


# Re-export the pieces we need so the rest of the file stays readable.
GStreamerDetectionApp = _detection_pipeline.GStreamerDetectionApp
SOURCE_PIPELINE = _detection_pipeline.SOURCE_PIPELINE
INFERENCE_PIPELINE = _detection_pipeline.INFERENCE_PIPELINE
USER_CALLBACK_PIPELINE = _detection_pipeline.USER_CALLBACK_PIPELINE
QUEUE = _detection_pipeline.QUEUE
app_callback_class = _detection_pipeline.app_callback_class


@dataclass
class Detection:
    label: str | None
    probability: float | None
    bbox: tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)


class DetectionResults(app_callback_class):
    """Collect detection outputs from the pad-probe callback."""

    def __init__(self, max_results: int = 50):
        super().__init__()
        self.results: queue.Queue[list[Detection]] = queue.Queue(maxsize=max_results)

    def push(self, detections: list[Detection]) -> None:
        try:
            self.results.put_nowait(detections)
        except queue.Full:
            # Drop the oldest if the consumer is slower than the pipeline.
            _ = self.results.get_nowait()
            self.results.put_nowait(detections)

    def next(self, timeout: float | None = None) -> list[Detection] | None:
        try:
            return self.results.get(timeout=timeout)
        except queue.Empty:
            return None


def _first_attr(obj, names: Iterable[str]):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _normalize_bbox(bbox_obj) -> tuple[float, float, float, float] | None:
    if bbox_obj is None:
        return None

    attr_sets = [
        ("x_min", "y_min", "x_max", "y_max"),
        ("xmin", "ymin", "xmax", "ymax"),
        ("left", "top", "right", "bottom"),
    ]
    for attrs in attr_sets:
        if all(hasattr(bbox_obj, attr) for attr in attrs):
            xmin, ymin, xmax, ymax = (float(getattr(bbox_obj, attr)) for attr in attrs)
            return xmin, ymin, xmax, ymax

    if all(hasattr(bbox_obj, attr) for attr in ("x", "y", "w", "h")):
        x, y, w, h = (float(getattr(bbox_obj, attr)) for attr in ("x", "y", "w", "h"))
        return x, y, x + w, y + h

    if isinstance(bbox_obj, (list, tuple)) and len(bbox_obj) == 4:
        xmin, ymin, xmax, ymax = (float(v) for v in bbox_obj)
        return xmin, ymin, xmax, ymax

    return None


def _normalize_detection(det_obj) -> Detection | None:
    bbox_obj = _first_attr(det_obj, ("bbox", "box", "rectangle", "roi", "region"))
    bbox = _normalize_bbox(bbox_obj if bbox_obj is not None else det_obj)
    if bbox is None:
        return None

    probability = _first_attr(det_obj, ("prob", "probability", "score", "confidence"))
    label = _first_attr(det_obj, ("label", "class_name", "name"))
    class_id = _first_attr(det_obj, ("class_id", "id"))

    if label is None and class_id is not None:
        label = str(class_id)

    prob_value = float(probability) if probability is not None else None
    return Detection(label=label, probability=prob_value, bbox=bbox)


def _hailo_meta_to_detections(buffer) -> list[Detection]:
    """Try to extract detections via the Hailo GI helper (if present)."""
    detections: list[Detection] = []
    if Hailo is None:
        return detections

    try:  # pragma: no cover - platform dependent
        meta = Hailo.get_meta(buffer)
    except Exception:
        meta = None

    if meta is None:
        return detections

    for collection_name in ("detections", "objects", "roi_list", "rois"):
        if not hasattr(meta, collection_name):
            continue
        collection = getattr(meta, collection_name)
        for det_obj in collection:
            normalized = _normalize_detection(det_obj)
            if normalized:
                detections.append(normalized)
    return detections


def _generic_meta_to_detections(buffer) -> list[Detection]:
    """Fallback path that walks any attached GstMeta looking for detection-like objects."""
    detections: list[Detection] = []
    meta_iter = buffer.iterate_meta()
    while True:
        result, meta = meta_iter.next()
        if result != Gst.IteratorResult.OK:
            break

        candidates = []
        for name in ("objects", "detections", "roi_list", "rois", "hailo_objects"):
            if hasattr(meta, name):
                candidates.append(getattr(meta, name))

        for candidate in candidates:
            if candidate is None:
                continue
            try:
                iterator = iter(candidate)
            except TypeError:
                continue
            for det_obj in iterator:
                normalized = _normalize_detection(det_obj)
                if normalized:
                    detections.append(normalized)
    return detections


def extract_detections_from_buffer(buffer) -> list[Detection]:
    detections = _hailo_meta_to_detections(buffer)
    if detections:
        return detections
    return _generic_meta_to_detections(buffer)


def detection_callback(pad, info, user_data: DetectionResults):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    detections = extract_detections_from_buffer(buffer)
    if detections:
        user_data.push(detections)
    return Gst.PadProbeReturn.OK


class HeadlessDetectionApp(GStreamerDetectionApp):
    """Reuse the detection pipeline but swap the sink for fakesink."""

    def __init__(self, app_callback, user_data, parser=None):
        super().__init__(app_callback, user_data, parser)
        # Suppress the "hailo_display not found" warning in the parent runner.
        setattr(self.options_menu, "ui", True)

    def get_pipeline_string(self):
        source_pipeline = SOURCE_PIPELINE(
            video_source=self.video_source,
            video_width=self.video_width,
            video_height=self.video_height,
            frame_rate=self.frame_rate,
            sync=self.sync,
            no_webcam_compression=True,
        )

        detection_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.post_function_name,
            batch_size=self.batch_size,
            config_json=self.labels_json,
            additional_params=self.thresholds_str,
        )

        user_callback_pipeline = USER_CALLBACK_PIPELINE()
        sink_pipeline = f'{QUEUE(name="headless_sink_q")} ! fakesink name=headless_sink sync=false async=false'

        pipeline_string = (
            f"{source_pipeline} ! "
            f"{detection_pipeline} ! "
            f"{user_callback_pipeline} ! "
            f"{sink_pipeline}"
        )
        print(pipeline_string)
        return pipeline_string


def run_detection_loop(max_frames: int | None = None, poll_timeout: float = 1.0) -> Iterator[list[Detection]]:
    """
    Start the pipeline in a background thread and yield detection results.

    Args:
        max_frames: stop after this many batches of detections (None = run forever).
        poll_timeout: how long to wait for detections before checking for shutdown.
    """
    user_data = DetectionResults()
    app = HeadlessDetectionApp(detection_callback, user_data)

    runner = threading.Thread(target=app.run, daemon=True)
    runner.start()

    frames_seen = 0
    try:
        while user_data.running:
            detections = user_data.next(timeout=poll_timeout)
            if detections is None:
                continue
            frames_seen += 1
            yield detections
            if max_frames is not None and frames_seen >= max_frames:
                break
    finally:
        user_data.running = False
        app.shutdown()
        runner.join(timeout=2.0)


if __name__ == "__main__":
    print("Starting headless detection loop (Ctrl-C to stop)...")
    try:
        for idx, detections in enumerate(run_detection_loop(), start=1):
            timestamp = time.strftime("%H:%M:%S")
            summary = ", ".join(
                f"{det.label or 'unknown'} {det.probability or 0:.2f} "
                f"[{det.bbox[0]:.0f},{det.bbox[1]:.0f},{det.bbox[2]:.0f},{det.bbox[3]:.0f}]"
                for det in detections
            )
            print(f"[{timestamp}] frame {idx}: {summary if summary else 'no detections'}")
    except KeyboardInterrupt:
        print("Stopping headless detection loop.")
