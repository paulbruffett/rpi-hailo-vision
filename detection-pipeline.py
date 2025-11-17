"""
Standalone copy of the simple detection pipeline with zero dependencies on the
`hailo_apps` package. Everything that used to be imported from the shared
library lives in this file so it can run as a drop-in script.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import queue
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import gi
import numpy as np
import setproctitle

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib, GObject  # noqa: E402


Picamera2 = None  # type: ignore


# -----------------------------------------------------------------------------------------------
# Minimal constants copied from hailo_apps.hailo_app_python.core.common.defines
# -----------------------------------------------------------------------------------------------

HAILO_FILE_EXTENSION = ".hef"

RESOURCES_ROOT_PATH_DEFAULT = "/usr/local/hailo/resources"
RESOURCES_MODELS_DIR_NAME = "models"
RESOURCES_VIDEOS_DIR_NAME = "videos"
RESOURCES_SO_DIR_NAME = "so"
RESOURCES_PATH_KEY = "resources_path"
HAILO_ARCH_KEY = "hailo_arch"

SIMPLE_DETECTION_APP_TITLE = "Hailo Simple Detection App"
SIMPLE_DETECTION_PIPELINE = "simple_detection"
SIMPLE_DETECTION_VIDEO_NAME = "example_640.mp4"
SIMPLE_DETECTION_MODEL_NAME = "yolov6n"
SIMPLE_DETECTION_POSTPROCESS_SO_FILENAME = "libyolo_hailortpp_postprocess.so"
SIMPLE_DETECTION_POSTPROCESS_FUNCTION = "filter"

HAILO_RGB_VIDEO_FORMAT = "RGB"
HAILO_NV12_VIDEO_FORMAT = "NV12"
HAILO_YUYV_VIDEO_FORMAT = "YUYV"
GST_VIDEO_SINK = "autovideosink"
BASIC_PIPELINES_VIDEO_EXAMPLE_NAME = "example.mp4"
USB_CAMERA = "usb"
RPI_NAME_I = "rpi"



def detect_hailo_arch() -> str | None:
    """Use hailortcli to identify the connected Hailo device architecture."""
    try:
        args = shlex.split("hailortcli fw-control identify")
        result = subprocess.run(args, capture_output=True, text=True, check=False)
    except (OSError, ValueError):
        return None

    if result.returncode != 0:
        return None

    for line in result.stdout.splitlines():
        if "HAILO8L" in line:
            return "hailo8l"
        if "HAILO8" in line:
            return "hailo8"
    return None


def get_default_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hailo App Help")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Input source (file, USB webcam, RPi camera, ximage).",
    )
    parser.add_argument("--use-frame", "-u", action="store_true", help="Use frame from the callback function.")
    parser.add_argument("--show-fps", "-f", action="store_true", help="Print FPS on sink.")
    parser.add_argument(
        "--arch",
        default=None,
        choices=["hailo8", "hailo8l"],
        help="Specify the Hailo architecture. Auto-detect when omitted.",
    )
    parser.add_argument("--hef-path", default=None, help="Path to HEF file.")
    parser.add_argument(
        "--disable-sync",
        action="store_true",
        help="Disable display sink sync (run as fast as possible).",
    )
    parser.add_argument(
        "--disable-callback",
        action="store_true",
        help="Run the pipeline without invoking the callback logic.",
    )
    parser.add_argument("--dump-dot", action="store_true", help="Dump the pipeline graph to pipeline.dot")
    parser.add_argument("--frame-rate", "-r", type=int, default=30, help="Frame rate of the video source.")
    return parser


def _get_resource_root() -> Path:
    override = os.environ.get("resources_path")
    if override:
        return Path(override)
    return Path("/usr/local/hailo/resources")


def _get_model_name(pipeline_name: str, arch: str) -> str:
    if pipeline_name != "simple_detection":
        raise ValueError(f"Unsupported pipeline '{pipeline_name}' in standalone script.")
    return "yolov6n"


def get_resource_path(pipeline_name: str, resource_type: str, model: str | None = None) -> Path:
    """
    Resolve resource paths using the same defaults as the original hailo_apps helpers.
    """
    root = _get_resource_root()

    if resource_type == RESOURCES_VIDEOS_DIR_NAME:
        filename = model or SIMPLE_DETECTION_VIDEO_NAME
        return root / RESOURCES_VIDEOS_DIR_NAME / filename

    if resource_type == RESOURCES_SO_DIR_NAME:
        filename = model or SIMPLE_DETECTION_POSTPROCESS_SO_FILENAME
        return root / RESOURCES_SO_DIR_NAME / filename

    if resource_type == RESOURCES_MODELS_DIR_NAME:
        arch = os.environ.get(HAILO_ARCH_KEY) or detect_hailo_arch()
        if not arch:
            raise ValueError("Failed to detect Hailo architecture for model lookup.")
        model_name = model or _get_model_name(pipeline_name, arch)
        return (root / RESOURCES_MODELS_DIR_NAME / arch / model_name).with_suffix(HAILO_FILE_EXTENSION)

    raise ValueError(f"Unsupported resource type '{resource_type}'.")


def get_usb_video_devices() -> list[str]:
    """Scan /dev for video devices; fall back to an empty list when none are found."""
    dev_dir = Path("/dev")
    if not dev_dir.exists():
        return []
    return [str(path) for path in sorted(dev_dir.glob("video*"))]


# endregion


# region buffer utilities ----------------------------------------------------------------------
def get_caps_from_pad(pad: Gst.Pad):
    caps = pad.get_current_caps()
    if not caps:
        return None, None, None
    structure = caps.get_structure(0)
    if not structure:
        return None, None, None
    video_format = structure.get_value("format")
    width = structure.get_value("width")
    height = structure.get_value("height")
    return video_format, width, height


def _handle_rgb(map_info, width, height):
    return np.ndarray(shape=(height, width, 3), dtype=np.uint8, buffer=map_info.data).copy()


def _handle_nv12(map_info, width, height):
    y_plane_size = width * height
    y_plane = np.ndarray(shape=(height, width), dtype=np.uint8, buffer=map_info.data[:y_plane_size]).copy()
    uv_plane = np.ndarray(
        shape=(height // 2, width // 2, 2),
        dtype=np.uint8,
        buffer=map_info.data[y_plane_size:],
    ).copy()
    return y_plane, uv_plane


def _handle_yuyv(map_info, width, height):
    return np.ndarray(shape=(height, width, 2), dtype=np.uint8, buffer=map_info.data).copy()


FORMAT_HANDLERS = {
    HAILO_RGB_VIDEO_FORMAT: _handle_rgb,
    HAILO_NV12_VIDEO_FORMAT: _handle_nv12,
    HAILO_YUYV_VIDEO_FORMAT: _handle_yuyv,
}


def get_numpy_from_buffer(buffer, video_format, width, height):
    success, map_info = buffer.map(Gst.MapFlags.READ)
    if not success:
        raise ValueError("Buffer mapping failed.")
    try:
        handler = FORMAT_HANDLERS.get(video_format)
        if handler is None:
            raise ValueError(f"Unsupported format: {video_format}")
        return handler(map_info, width, height)
    finally:
        buffer.unmap(map_info)


# endregion


# region pipeline helpers ----------------------------------------------------------------------
def get_source_type(input_source) -> str:
    input_source = str(input_source)
    if input_source.startswith("/dev/video"):
        return "usb"
    if input_source.startswith("rpi"):
        return "rpi"
    if input_source.startswith("libcamera"):
        return "libcamera"
    if input_source.startswith("0x"):
        return "ximage"
    return "file"


def QUEUE(name, max_size_buffers=3, max_size_bytes=0, max_size_time=0, leaky="no"):
    return (
        f"queue name={name} leaky={leaky} max-size-buffers={max_size_buffers} "
        f"max-size-bytes={max_size_bytes} max-size-time={max_size_time} "
    )


def get_camera_resolution(video_width=640, video_height=640):
    if video_width <= 640 and video_height <= 480:
        return 640, 480
    if video_width <= 1280 and video_height <= 720:
        return 1280, 720
    if video_width <= 1920 and video_height <= 1080:
        return 1920, 1080
    return 3840, 2160


def SOURCE_PIPELINE(
    video_source,
    video_width=640,
    video_height=640,
    name="source",
    no_webcam_compression=False,
    frame_rate=30,
    sync=True,
    video_format="RGB",
):
    source_type = get_source_type(video_source)

    if source_type == "usb":
        if no_webcam_compression:
            source_element = (
                f"v4l2src device={video_source} name={name} ! "
                f"video/x-raw, width=640, height=480 ! "
                "videoflip name=videoflip video-direction=horiz ! "
            )
        else:
            width, height = get_camera_resolution(video_width, video_height)
            source_element = (
                f"v4l2src device={video_source} name={name} ! image/jpeg, framerate=30/1, "
                f"width={width}, height={height} ! "
                f'{QUEUE(name=f"{name}_queue_decode")} ! '
                f"decodebin name={name}_decodebin ! "
                "videoflip name=videoflip video-direction=horiz ! "
            )
    elif source_type == "rpi":
        source_element = (
            "appsrc name=app_source is-live=true leaky-type=downstream max-buffers=3 ! "
            "videoflip name=videoflip video-direction=horiz ! "
            f"video/x-raw, format={video_format}, width={video_width}, height={video_height} ! "
        )
    elif source_type == "libcamera":
        source_element = (
            f"libcamerasrc name={name} ! "
            f"video/x-raw, format={video_format}, width=1536, height=864 ! "
        )
    elif source_type == "ximage":
        source_element = (
            f"ximagesrc xid={video_source} ! "
            f'{QUEUE(name=f"{name}queue_scale_")} ! '
            "videoscale ! "
        )
    else:
        source_element = (
            f'filesrc location="{video_source}" name={name} ! '
            f'{QUEUE(name=f"{name}_queue_decode")} ! '
            f"decodebin name={name}_decodebin ! "
        )

    fps_caps = f"video/x-raw, framerate={frame_rate}/1" if sync else "video/x-raw"

    return (
        f"{source_element} "
        f'{QUEUE(name=f"{name}_scale_q")} ! '
        f"videoscale name={name}_videoscale n-threads=2 ! "
        f'{QUEUE(name=f"{name}_convert_q")} ! '
        f"videoconvert n-threads=3 name={name}_convert qos=false ! "
        f"video/x-raw, pixel-aspect-ratio=1/1, format={video_format}, "
        f"width={video_width}, height={video_height} ! "
        f"videorate name={name}_videorate ! capsfilter name={name}_fps_caps caps=\"{fps_caps}\" "
    )


def INFERENCE_PIPELINE(
    hef_path,
    post_process_so=None,
    batch_size=1,
    config_json=None,
    post_function_name=None,
    additional_params="",
    name="inference",
    scheduler_timeout_ms=None,
    scheduler_priority=None,
    vdevice_group_id=1,
    multi_process_service=None,
):
    config_str = f" config-path={config_json} " if config_json else ""
    function_name_str = f" function-name={post_function_name} " if post_function_name else ""
    vdevice_group_id_str = f" vdevice-group-id={vdevice_group_id} "
    multi_process_service_str = (
        f" multi-process-service={str(multi_process_service).lower()} "
        if multi_process_service is not None
        else ""
    )
    scheduler_timeout_ms_str = f" scheduler-timeout-ms={scheduler_timeout_ms} " if scheduler_timeout_ms else ""
    scheduler_priority_str = f" scheduler-priority={scheduler_priority} " if scheduler_priority else ""

    hailonet_str = (
        f"hailonet name={name}_hailonet "
        f"hef-path={hef_path} "
        f"batch-size={batch_size} "
        f"{vdevice_group_id_str}"
        f"{multi_process_service_str}"
        f"{scheduler_timeout_ms_str}"
        f"{scheduler_priority_str}"
        f"{additional_params} "
        "force-writable=true "
    )

    pipeline = (
        f'{QUEUE(name=f"{name}_scale_q")} ! '
        f"videoscale name={name}_videoscale n-threads=2 qos=false ! "
        f'{QUEUE(name=f"{name}_convert_q")} ! '
        "video/x-raw, pixel-aspect-ratio=1/1 ! "
        f"videoconvert name={name}_videoconvert n-threads=2 ! "
        f'{QUEUE(name=f"{name}_hailonet_q")} ! '
        f"{hailonet_str} ! "
    )

    if post_process_so:
        pipeline += (
            f'{QUEUE(name=f"{name}_hailofilter_q")} ! '
            f"hailofilter name={name}_hailofilter so-path={post_process_so} {config_str} {function_name_str} qos=false ! "
        )

    pipeline += f'{QUEUE(name=f"{name}_output_q")} '
    return pipeline


def USER_CALLBACK_PIPELINE(name="identity_callback"):
    return f'{QUEUE(name=f"{name}_q")} ! identity name={name} '


def DISPLAY_PIPELINE(video_sink=GST_VIDEO_SINK, sync="true", show_fps="false", name="hailo_display"):
    return (
        f'{QUEUE(name=f"{name}_overlay_q")} ! '
        f"hailooverlay name={name}_overlay ! "
        f'{QUEUE(name=f"{name}_videoconvert_q")} ! '
        f"videoconvert name={name}_videoconvert n-threads=2 qos=false ! "
        f'{QUEUE(name=f"{name}_q")} ! '
        f"fpsdisplaysink name={name} video-sink={video_sink} sync={sync} "
        f"text-overlay={show_fps} signal-fps-measurements=true "
    )


# endregion


# region callback utilities --------------------------------------------------------------------
class app_callback_class:
    def __init__(self):
        self.frame_count = 0
        self.use_frame = False
        self.frame_queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=3)
        self.running = True

    def increment(self):
        self.frame_count += 1

    def get_count(self):
        return self.frame_count

    def set_frame(self, frame):
        if not self.frame_queue.full():
            self.frame_queue.put(frame)

    def get_frame(self):
        if not self.frame_queue.empty():
            return self.frame_queue.get()
        return None


def dummy_callback(pad, info, user_data):
    return Gst.PadProbeReturn.OK


# endregion


# region GStreamer application base -------------------------------------------------------------
class GStreamerApp:
    def __init__(self, args, user_data: app_callback_class):
        setproctitle.setproctitle("Hailo Python App")
        self.options_menu = args.parse_args()
        signal.signal(signal.SIGINT, self.shutdown)

        if self.options_menu.input is None:
            self.video_source = str(_get_resource_root() / RESOURCES_VIDEOS_DIR_NAME / BASIC_PIPELINES_VIDEO_EXAMPLE_NAME)
        else:
            self.video_source = self.options_menu.input

        if self.video_source == USB_CAMERA:
            devices = get_usb_video_devices()
            if not devices:
                raise RuntimeError('Input set to "usb" but no video devices found.')
            self.video_source = devices[0]

        self.source_type = get_source_type(self.video_source)
        self.frame_rate = self.options_menu.frame_rate
        self.user_data = user_data
        self.video_sink = GST_VIDEO_SINK
        self.pipeline = None
        self.loop = None
        self.threads: list[threading.Thread] = []
        self.error_occurred = False
        self.pipeline_latency = 300  # milliseconds
        self.batch_size = 1
        self.video_width = 1280
        self.video_height = 720
        self.video_format = HAILO_RGB_VIDEO_FORMAT
        self.hef_path = None
        self.app_callback = None

        user_data.use_frame = self.options_menu.use_frame

        self.sync = "false" if (self.options_menu.disable_sync or self.source_type != "file") else "true"
        self.show_fps = self.options_menu.show_fps

        if self.options_menu.dump_dot:
            os.environ["GST_DEBUG_DUMP_DOT_DIR"] = os.getcwd()

        self.webrtc_frames_queue = None

    def appsink_callback(self, appsink):
        sample = appsink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.OK
        buffer = sample.get_buffer()
        if buffer is None:
            return Gst.FlowReturn.OK
        video_format, width, height = get_caps_from_pad(appsink.get_static_pad("sink"))
        frame = get_numpy_from_buffer(buffer, video_format, width, height)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            self.webrtc_frames_queue.put(frame)
        except queue.Full:
            print("Frame queue is full. Dropping frame.")
        return Gst.FlowReturn.OK

    def on_fps_measurement(self, sink, fps, droprate, avgfps):
        print(f"FPS: {fps:.2f}, Droprate: {droprate:.2f}, Avg FPS: {avgfps:.2f}")
        return True

    def create_pipeline(self):
        Gst.init(None)
        pipeline_string = self.get_pipeline_string()
        try:
            self.pipeline = Gst.parse_launch(pipeline_string)
        except Exception as exc:  # pragma: no cover - Gst exceptions are opaque
            print(f"Error creating pipeline: {exc}", file=sys.stderr)
            sys.exit(1)

        if self.show_fps:
            hailo_display = self.pipeline.get_by_name("hailo_display")
            if hailo_display:
                hailo_display.connect("fps-measurements", self.on_fps_measurement)

        self.loop = GLib.MainLoop()

    def bus_call(self, bus, message, loop):
        msg_type = message.type
        if msg_type == Gst.MessageType.EOS:
            print("End-of-stream")
            self.on_eos()
        elif msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}, {debug}", file=sys.stderr)
            self.error_occurred = True
            self.shutdown()
        elif msg_type == Gst.MessageType.QOS:
            if not hasattr(self, "qos_count"):
                self.qos_count = 0
            self.qos_count += 1
            if self.qos_count > 50 and self.qos_count % 10 == 0:
                qos_element = message.src.get_name()
                print(f"\033[91mQoS message received from {qos_element}\033[0m")
                print(
                    "\033[91mLots of QoS messages received: "
                    f"{self.qos_count}, consider optimizing the pipeline or reducing the frame rate.\033[0m"
                )
        return True

    def on_eos(self):
        if self.source_type == "file":
            if self.sync == "false":
                print("Pausing pipeline for rewind... some warnings are expected.")
                self.pipeline.set_state(Gst.State.PAUSED)
            success = self.pipeline.seek_simple(Gst.Format.TIME, Gst.SeekFlags.FLUSH, 0)
            if success:
                print("Video rewound successfully. Restarting playback...")
            else:
                print("Error rewinding video.", file=sys.stderr)
            self.pipeline.set_state(Gst.State.PLAYING)
        else:
            self.shutdown()

    def shutdown(self, signum=None, frame=None):
        print("Shutting down... Hit Ctrl-C again to force quit.")
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        if self.pipeline is None:
            return
        self.pipeline.set_state(Gst.State.PAUSED)
        GLib.usleep(100000)
        self.pipeline.set_state(Gst.State.READY)
        GLib.usleep(100000)
        self.pipeline.set_state(Gst.State.NULL)
        if self.loop:
            GLib.idle_add(self.loop.quit)

    def update_fps_caps(self, new_fps=30, source_name="source"):
        videorate_name = f"{source_name}_videorate"
        capsfilter_name = f"{source_name}_fps_caps"
        videorate = self.pipeline.get_by_name(videorate_name)
        if videorate is None:
            print(f"Element {videorate_name} not found in the pipeline.")
            return
        current_max_rate = videorate.get_property("max-rate")
        print(f"Current videorate max-rate: {current_max_rate}")
        videorate.set_property("max-rate", new_fps)
        updated_max_rate = videorate.get_property("max-rate")
        print(f"Updated videorate max-rate to: {updated_max_rate}")
        capsfilter = self.pipeline.get_by_name(capsfilter_name)
        if capsfilter:
            new_caps_str = f"video/x-raw, framerate={new_fps}/1"
            new_caps = Gst.Caps.from_string(new_caps_str)
            capsfilter.set_property("caps", new_caps)
            print("Updated capsfilter caps to match new rate")
        self.frame_rate = new_fps

    def get_pipeline_string(self):
        raise NotImplementedError

    def dump_dot_file(self):
        print("Dumping dot file...")
        Gst.debug_bin_to_dot_file(self.pipeline, Gst.DebugGraphDetails.ALL, "pipeline")
        return False

    def run(self):
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, self.loop)

        if not self.options_menu.disable_callback:
            identity = self.pipeline.get_by_name("identity_callback")
            if identity is None:
                print("Warning: identity_callback element not found.")
            else:
                identity_pad = identity.get_static_pad("src")
                identity_pad.add_probe(Gst.PadProbeType.BUFFER, self.app_callback, self.user_data)

        hailo_display = self.pipeline.get_by_name("hailo_display")
        if hailo_display is None and not getattr(self.options_menu, "ui", False):
            print("Warning: hailo_display element not found.")

        disable_qos(self.pipeline)

        if self.options_menu.use_frame:
            display_process = multiprocessing.Process(target=display_user_data_frame, args=(self.user_data,))
            display_process.start()
        else:
            display_process = None

        if self.source_type == RPI_NAME_I:
            picam_thread = threading.Thread(
                target=picamera_thread, args=(self.pipeline, self.video_width, self.video_height, self.video_format)
            )
            self.threads.append(picam_thread)
            picam_thread.start()

        self.pipeline.set_state(Gst.State.PAUSED)
        new_latency = self.pipeline_latency * Gst.MSECOND
        self.pipeline.set_latency(new_latency)
        self.pipeline.set_state(Gst.State.PLAYING)

        if self.options_menu.dump_dot:
            GLib.timeout_add_seconds(3, self.dump_dot_file)

        self.loop.run()

        try:
            self.user_data.running = False
            self.pipeline.set_state(Gst.State.NULL)
            if display_process:
                display_process.terminate()
                display_process.join()
            for thread in self.threads:
                thread.join()
        except Exception as exc:  # pragma: no cover - cleanup best effort
            print(f"Error during cleanup: {exc}", file=sys.stderr)
        finally:
            if self.error_occurred:
                print("Exiting with error...", file=sys.stderr)
                sys.exit(1)
            print("Exiting...")
            sys.exit(0)


def picamera_thread(pipeline, video_width, video_height, video_format, picamera_config=None):
    if Picamera2 is None:
        raise RuntimeError("Picamera2 is not available on this platform.")
    appsrc = pipeline.get_by_name("app_source")
    appsrc.set_property("is-live", True)
    appsrc.set_property("format", Gst.Format.TIME)
    with Picamera2() as picam2:
        if picamera_config is None:
            main = {"size": (1280, 720), "format": "RGB888"}
            lores = {"size": (video_width, video_height), "format": "RGB888"}
            controls = {"FrameRate": 30}
            config = picam2.create_preview_configuration(main=main, lores=lores, controls=controls)
        else:
            config = picamera_config
        picam2.configure(config)
        lores_stream = config["lores"]
        format_str = "RGB" if lores_stream["format"] == "RGB888" else video_format
        width, height = lores_stream["size"]
        appsrc.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw, format={format_str}, width={width}, height={height}, "
                "framerate=30/1, pixel-aspect-ratio=1/1"
            ),
        )
        picam2.start()
        frame_count = 0
        start_time = time.time()
        print("picamera_process started")
        while True:
            frame_data = picam2.capture_array("lores")
            if frame_data is None:
                print("Failed to capture frame.")
                break
            frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            buffer = Gst.Buffer.new_wrapped(frame.tobytes())
            buffer_duration = Gst.util_uint64_scale_int(1, Gst.SECOND, 30)
            buffer.pts = frame_count * buffer_duration
            buffer.duration = buffer_duration
            ret = appsrc.emit("push-buffer", buffer)
            if ret == Gst.FlowReturn.FLUSHING:
                break
            if ret != Gst.FlowReturn.OK:
                print("Failed to push buffer:", ret)
                break
            frame_count += 1
        elapsed = time.time() - start_time
        print(f"Picamera thread stopped after {frame_count} frames ({elapsed:.2f}s).")


def disable_qos(pipeline):
    if not isinstance(pipeline, Gst.Pipeline):
        print("The provided object is not a GStreamer Pipeline")
        return
    iterator = pipeline.iterate_elements()
    while True:
        result, element = iterator.next()
        if result != Gst.IteratorResult.OK:
            break
        props = [prop.name for prop in GObject.list_properties(element)]
        if "qos" in props:
            element.set_property("qos", False)


def display_user_data_frame(user_data: app_callback_class):
    while user_data.running:
        frame = user_data.get_frame()
        if frame is not None:
            cv2.imshow("User Frame", frame)
        cv2.waitKey(1)
    cv2.destroyAllWindows()


# endregion


# -----------------------------------------------------------------------------------------------
# Detection-specific pipeline
# -----------------------------------------------------------------------------------------------
class GStreamerDetectionApp(GStreamerApp):
    def __init__(self, app_callback, user_data, parser=None):
        if parser is None:
            parser = get_default_parser()
        parser.add_argument("--labels-json", default=None, help="Path to custom labels JSON file.")
        super().__init__(parser, user_data)

        self.video_width = 640
        self.video_height = 640
        self.batch_size = 2
        nms_score_threshold = 0.3
        nms_iou_threshold = 0.45

        if self.options_menu.input is None:
            self.video_source = str(
                get_resource_path(
                    pipeline_name=SIMPLE_DETECTION_PIPELINE,
                    resource_type=RESOURCES_VIDEOS_DIR_NAME,
                    model=SIMPLE_DETECTION_VIDEO_NAME,
                )
            )

        if self.options_menu.arch is None:
            detected_arch = detect_hailo_arch()
            if detected_arch is None:
                raise ValueError("Could not auto-detect Hailo architecture. Please specify --arch manually.")
            self.arch = detected_arch
            print(f"Auto-detected Hailo architecture: {self.arch}")
        else:
            self.arch = self.options_menu.arch

        if self.options_menu.hef_path is not None:
            self.hef_path = self.options_menu.hef_path
        else:
            self.hef_path = str(
                get_resource_path(
                    pipeline_name=SIMPLE_DETECTION_PIPELINE,
                    resource_type=RESOURCES_MODELS_DIR_NAME,
                )
            )
        print(f"Using HEF path: {self.hef_path}")

        self.post_process_so = str(
            get_resource_path(
                pipeline_name=SIMPLE_DETECTION_PIPELINE,
                resource_type=RESOURCES_SO_DIR_NAME,
                model=SIMPLE_DETECTION_POSTPROCESS_SO_FILENAME,
            )
        )
        print(f"Using post-process shared object: {self.post_process_so}")

        self.post_function_name = SIMPLE_DETECTION_POSTPROCESS_FUNCTION
        self.labels_json = self.options_menu.labels_json
        self.app_callback = app_callback

        self.thresholds_str = (
            f"nms-score-threshold={nms_score_threshold} "
            f"nms-iou-threshold={nms_iou_threshold} "
            f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )

        setproctitle.setproctitle(SIMPLE_DETECTION_APP_TITLE)
        self.create_pipeline()

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
        display_pipeline = DISPLAY_PIPELINE(video_sink=self.video_sink, sync=self.sync, show_fps=self.show_fps)

        pipeline_string = (
            f"{source_pipeline} ! "
            f"{detection_pipeline} ! "
            f"{user_callback_pipeline} ! "
            f"{display_pipeline}"
        )
        print(pipeline_string)
        return pipeline_string


def main():
    user_data = app_callback_class()
    app_callback = dummy_callback
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()


if __name__ == "__main__":
    print("Starting Hailo Detection App...")
    main()
