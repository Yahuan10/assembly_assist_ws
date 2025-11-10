#!/usr/bin/env python3
"""
Subie workstation visualizer styled after the reference layout.

The script listens to the SMACH introspection topic published by `scheduler.py`
and renders a schematic workbench with highlighted areas, the current robot
state, the upcoming state, and a simple progress indicator.

Usage
-----
Run the scheduler stack first, then:
    $ python workspace_visualizer.py

Optional: if `shared_state.py` is available in the same directory the visualizer
will use `get_ready_for_next_step()` to colour the “Ready” indicator.
"""

import argparse
import copy
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import rospy
    from smach_msgs.msg import SmachContainerStatus
    from moveit_msgs.msg import RobotTrajectory, RobotState
    from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest
    from sensor_msgs.msg import JointState
except ImportError as exc:  # pragma: no cover - executed when ROS is missing
    raise SystemExit(
        "❌ workspace_visualizer requires ROS (rospy + smach_msgs). "
        "Please ensure you are inside a ROS environment."
    ) from exc

try:
    from shared_state import get_ready_for_next_step  # type: ignore[attr-defined]
except ImportError:
    get_ready_for_next_step = None  # type: ignore[assignment]

try:  # Prefer standalone module shipped with ur_rtde>=1.6
    from rtde_receive import RTDEReceiveInterface
except ImportError:
    try:  # Fallback: import via package namespace (ur_rtde.rtde_receive)
        from ur_rtde import rtde_receive as _rtde_receive_mod  # type: ignore
    except ImportError:
        RTDEReceiveInterface = None  # type: ignore[assignment]
        rtde_receive = None  # legacy alias
    else:
        RTDEReceiveInterface = _rtde_receive_mod.RTDEReceiveInterface  # type: ignore[attr-defined]
        rtde_receive = RTDEReceiveInterface
else:
    rtde_receive = RTDEReceiveInterface  # legacy alias for backward compatibility

# Canvas configuration
CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Workbench boundary (top-left x/y, bottom-right x/y)
WORKBENCH_RECT = (60, 60, 1040, 660)
BOUNDARY_COLOR = (0, 255, 0)  # green

# Zone layout template: rectangles defined as (x1, y1, x2, y2)
ZONE_LAYOUT_TEMPLATE: Dict[str, Dict[str, Tuple[int, ...]]] = {
    "Battery": {"rect": (100, 90, 620, 230), "label": "Battery"},
    "Motor": {"rect": (660, 90, 860, 260), "label": "Motor"},
    "MotorTray": {"rect": (660, 90, 860, 260), "label": "Motor"},
    "PCB2": {"rect": (320, 260, 620, 320), "label": "PCB2"},
    "PCB1": {"rect": (660, 260, 880, 320), "label": "PCB1"},
    "PCB3": {"rect": (100, 360, 280, 540), "label": "PCB3"},
    "Handover": {"rect": (100, 560, 280, 640), "label": "Handover", "visible": False},
}

def _ensure_zone_centres(layout: Dict[str, Dict[str, Tuple[int, ...]]]) -> None:
    for zone_cfg in layout.values():
        x1, y1, x2, y2 = zone_cfg["rect"]
        zone_cfg["center"] = ((x1 + x2) // 2, (y1 + y2) // 2)


_ensure_zone_centres(ZONE_LAYOUT_TEMPLATE)

# Target circle inside the handover area (center x, center y, radius)
TARGET_CIRCLE = (190, 620, 30)

# Robot base indicator
ROBOT_BASE_POINT = (600, 420)

# Default homography (robot XY metres → canvas pixels), derived from calibration scripts
DEFAULT_HOMOGRAPHY = np.array(
    [
        [-944.336237, -87.4102614, -4.95230579],
        [-62.4513781, 874.191558, 134.01638],
        [-0.00962922264, -0.142556057, 1.0],
    ],
    dtype=float,
)

# Approximate world (robot base) XY coordinates (metres) for key zones.
ZONE_WORLD_CENTERS: Dict[str, Tuple[float, float]] = {
    "PCB1": (0.714149, -0.139535),
    "PCB2": (0.435914, -0.144026),
    "PCB3": (0.174782, -0.337729),
    "Battery": (0.653287, 0.04155),
    "Motor": (0.264111, 0.115139),
    "MotorTray": (0.264111, 0.115139),
    "Handover": (0.05, -0.45),
}


def _project_point(homography: np.ndarray, x: float, y: float) -> Optional[Tuple[float, float]]:
    vec = np.array([x, y, 1.0], dtype=float)
    mapped = homography @ vec
    if abs(mapped[2]) < 1e-9:
        return None
    return mapped[0] / mapped[2], mapped[1] / mapped[2]


def _recenter_rect(rect: Tuple[int, int, int, int], cx: float, cy: float) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = rect
    width = x2 - x1
    height = y2 - y1
    nx1 = int(round(cx - width / 2))
    ny1 = int(round(cy - height / 2))
    nx2 = int(round(cx + width / 2))
    ny2 = int(round(cy + height / 2))
    return (nx1, ny1, nx2, ny2)


def build_zone_layout(homography: np.ndarray, shift_with_h: bool) -> Dict[str, Dict[str, Tuple[int, ...]]]:
    layout = copy.deepcopy(ZONE_LAYOUT_TEMPLATE)
    if shift_with_h:
        for name, (wx, wy) in ZONE_WORLD_CENTERS.items():
            if name not in layout:
                continue
            projected = _project_point(homography, wx, wy)
            if projected is None:
                continue
            px, py = projected
            layout[name]["rect"] = _recenter_rect(layout[name]["rect"], px, py)
            layout[name]["center"] = (int(round(px)), int(round(py)))
    _ensure_zone_centres(layout)
    return layout

# Map SMACH states to their primary zones (used for colouring)
STATE_PRIMARY_ZONE = {
    "Start": "Motor",
    "MPickUp": "Motor",
    "MHold": "Handover",
    "MHoldHD": "Handover",
    "MPositioning": "Motor",
    "PCB1PickUpAndPositioning": "PCB1",
    "PCB2PickUpAndPositioning": "PCB2",
    "PCB3PickUpAndPositioning": "PCB3",
    "BatteryPickUpAndPositioning": "Battery",
    "Test": "Motor",
    "Aborted": "Handover",
    "finished": "Motor",
}

# Define the path (sequence of zone names) that the robot follows per state.
STATE_PATH_TEMPLATE: Dict[str, Sequence[str]] = {
    "Start": ("Motor",),
    "MPickUp": ("Motor",),
    "MHold": ("Handover",),
    "MHoldHD": ("Handover",),
    "MPositioning": ("Motor",),
    "PCB1PickUpAndPositioning": ("Motor", "PCB1"),
    "PCB2PickUpAndPositioning": ("Motor", "PCB2"),
    "PCB3PickUpAndPositioning": ("Motor", "PCB3"),
    "BatteryPickUpAndPositioning": ("Motor", "Battery"),
    "Test": ("Motor",),
    "Aborted": ("Handover", "Motor"),
    "finished": ("Motor",),
}


def build_state_paths(
    zone_layout: Dict[str, Dict[str, Tuple[int, ...]]]
) -> Dict[str, List[Tuple[int, int]]]:
    """Convert zone sequences to pixel coordinates."""
    paths: Dict[str, List[Tuple[int, int]]] = {}
    for state, zones in STATE_PATH_TEMPLATE.items():
        points: List[Tuple[int, int]] = []
        for zone in zones:
            cfg = zone_layout.get(zone)
            if not cfg:
                continue
            points.append(cfg["center"])  # type: ignore[index]
        if points:
            paths[state] = points
    return paths


STATE_PATHS = build_state_paths(ZONE_LAYOUT_TEMPLATE)

# Possible outgoing transitions (used to display the most likely next state)
STATE_TRANSITIONS: Dict[str, List[str]] = {
    "Start": ["MPickUp", "MHoldHD", "MPositioning", "PCB1PickUpAndPositioning"],
    "MPickUp": ["MHold", "MHoldHD", "Aborted"],
    "MHold": ["MPositioning", "Aborted"],
    "MHoldHD": ["MPositioning", "finished", "Aborted"],
    "MPositioning": [
        "MPickUp",
        "PCB1PickUpAndPositioning",
        "PCB2PickUpAndPositioning",
        "BatteryPickUpAndPositioning",
    ],
    "PCB1PickUpAndPositioning": ["PCB2PickUpAndPositioning", "Aborted"],
    "PCB2PickUpAndPositioning": ["PCB3PickUpAndPositioning", "Aborted"],
    "PCB3PickUpAndPositioning": ["BatteryPickUpAndPositioning", "Aborted"],
    "BatteryPickUpAndPositioning": ["finished", "MPickUp", "Aborted"],
    "Test": ["finished", "Aborted"],
    "Aborted": ["Start", "MPickUp", "finished"],
    "finished": ["Start"],
}


def get_state_label(name: str) -> str:
    """Pretty-print helper for state names."""
    return name.replace("_", " ")


@dataclass(frozen=True)
class VisualizerOptions:
    rtde_host: Optional[str]
    rtde_port: int
    rtde_frequency: float
    homography_path: Optional[str]
    tcp_history: int
    tcp_max_age: float
    show_handover_zone: bool
    keep_schematic_layout: bool
    overlay_template_zones: bool
    enable_planned_overlay: bool
    planned_traj_topic: str
    fk_service: str
    fk_link: str
    world_frame: str
    planned_stride: int


def parse_cli_args() -> VisualizerOptions:
    """Parse ROS-friendly CLI arguments for the visualizer."""
    parser = argparse.ArgumentParser(description="Subie workstation visualizer")
    parser.add_argument("--rtde-host", type=str, default=None, help="UR controller IP for live TCP pose streaming.")
    parser.add_argument("--rtde-port", type=int, default=30004, help="RTDE port of the UR controller.")
    parser.add_argument(
        "--rtde-frequency",
        type=float,
        default=30.0,
        help="Polling frequency (Hz) used for getActualTCPPose sampling.",
    )
    parser.add_argument(
        "--homography-path",
        type=str,
        default=None,
        help="Path to a 3x3 homography matrix (.npy or plain text) mapping robot XY to canvas pixels.",
    )
    parser.add_argument(
        "--tcp-history",
        type=int,
        default=240,
        help="Number of TCP samples retained for drawing the live path.",
    )
    parser.add_argument(
        "--tcp-max-age",
        type=float,
        default=20.0,
        help="Maximum age (in seconds) of TCP samples that remain visible.",
    )
    parser.add_argument(
        "--show-handover-zone",
        action="store_true",
        help="Render the handover rectangle (hidden by default for confidentiality).",
    )
    parser.add_argument(
        "--keep-schematic-layout",
        action="store_true",
        help="Do not shift the zone rectangles using the homography calibration.",
    )
    parser.add_argument(
        "--hide-template-zones",
        action="store_true",
        help="Do not draw the schematic zone overlay (enabled by default).",
    )
    parser.add_argument(
        "--enable-planned-overlay",
        action="store_true",
        help="Overlay the next planned TCP path using MoveIt FK results.",
    )
    parser.add_argument(
        "--planned-traj-topic",
        type=str,
        default="/planned_trajectory",
        help="ROS topic publishing moveit_msgs/RobotTrajectory for upcoming motion.",
    )
    parser.add_argument(
        "--fk-service",
        type=str,
        default="/compute_fk",
        help="GetPositionFK service used to convert planned joint states to TCP.",
    )
    parser.add_argument(
        "--fk-link",
        type=str,
        default="tool0",
        help="End-effector link name for FK requests.",
    )
    parser.add_argument(
        "--world-frame",
        type=str,
        default="base",
        help="World frame passed to FK service.",
    )
    parser.add_argument(
        "--planned-stride",
        type=int,
        default=2,
        help="Sample every Nth joint waypoint when building the planned TCP overlay.",
    )

    parsed = parser.parse_args(rospy.myargv()[1:])
    return VisualizerOptions(
        rtde_host=parsed.rtde_host,
        rtde_port=parsed.rtde_port,
        rtde_frequency=max(1.0, parsed.rtde_frequency),
        homography_path=parsed.homography_path,
        tcp_history=max(8, parsed.tcp_history),
        tcp_max_age=max(1.0, parsed.tcp_max_age),
        show_handover_zone=parsed.show_handover_zone,
        keep_schematic_layout=parsed.keep_schematic_layout,
        overlay_template_zones=not parsed.hide_template_zones,
        enable_planned_overlay=parsed.enable_planned_overlay,
        planned_traj_topic=parsed.planned_traj_topic,
        fk_service=parsed.fk_service,
        fk_link=parsed.fk_link,
        world_frame=parsed.world_frame,
        planned_stride=max(1, parsed.planned_stride),
    )


def load_homography_matrix(path_str: Optional[str]) -> np.ndarray:
    """Load a 3x3 homography matrix from disk or fall back to identity."""
    if not path_str:
        return DEFAULT_HOMOGRAPHY.copy()

    path = Path(path_str).expanduser()
    if not path.exists():
        print(f"⚠️ Homography file {path} not found. Falling back to default calibration.")
        return DEFAULT_HOMOGRAPHY.copy()

    try:
        if path.suffix == ".npy":
            matrix = np.load(path)
        else:
            matrix = np.loadtxt(path, dtype=float)
    except Exception as exc:  # pragma: no cover - depends on operator input
        print(f"⚠️ Failed to load homography from {path}: {exc}. Using default calibration.")
        return DEFAULT_HOMOGRAPHY.copy()

    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3):
        print(f"⚠️ Homography at {path} is {matrix.shape}, expected 3x3. Using default calibration.")
        return DEFAULT_HOMOGRAPHY.copy()
    return matrix


class ProjectedTCPPath:
    """Maintain a deque of projected TCP positions."""

    def __init__(self, homography: np.ndarray, history: int, max_age: float) -> None:
        self._homography = homography
        self._points: Deque[Tuple[int, int]] = deque(maxlen=history)
        self._timestamps: Deque[float] = deque(maxlen=history)
        self._max_age = max_age
        self._lock = threading.Lock()

    def add_pose(self, pose: Sequence[float]) -> None:
        point = self._project_pose(pose)
        if point is None:
            return
        now = time.time()
        with self._lock:
            self._points.append(point)
            self._timestamps.append(now)
            self._discard_expired(now)

    def get_path(self) -> List[Tuple[int, int]]:
        with self._lock:
            self._discard_expired(time.time())
            return list(self._points)

    def get_latest(self) -> Optional[Tuple[int, int]]:
        with self._lock:
            self._discard_expired(time.time())
            return self._points[-1] if self._points else None

    def _discard_expired(self, now: float) -> None:
        while self._timestamps and now - self._timestamps[0] > self._max_age:
            self._timestamps.popleft()
            self._points.popleft()

    def _project_pose(self, pose: Sequence[float]) -> Optional[Tuple[int, int]]:
        if len(pose) < 2:
            return None
        vec = np.array([pose[0], pose[1], 1.0], dtype=float)
        mapped = self._homography @ vec
        if abs(mapped[2]) < 1e-9:
            return None
        x = mapped[0] / mapped[2]
        y = mapped[1] / mapped[2]
        px = int(round(x))
        py = int(round(y))
        if px < 0 or px >= CANVAS_WIDTH or py < 0 or py >= CANVAS_HEIGHT:
            return None
        return px, py


class URTCPWatcher(threading.Thread):
    """Background thread that polls the UR controller via RTDE."""

    def __init__(self, host: str, port: int, frequency: float, tracker: ProjectedTCPPath) -> None:
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.frequency = frequency
        self.tracker = tracker
        self._stop_event = threading.Event()
        self._receiver = None

    def run(self) -> None:  # pragma: no cover - requires live robot
        if RTDEReceiveInterface is None:
            print("⚠️ rtde_receive not available. Live TCP path disabled.")
            return

        sample_period = max(0.01, 1.0 / self.frequency)
        while not self._stop_event.is_set():
            if self._receiver is None:
                self._receiver = self._init_receiver()
                if self._receiver is None:
                    time.sleep(2.0)
                    continue

            try:
                pose = self._receiver.getActualTCPPose()
            except Exception as exc:
                print(f"⚠️ Lost RTDE connection: {exc}")
                self._receiver = None
                time.sleep(1.0)
                continue

            if pose:
                self.tracker.add_pose(pose)
            time.sleep(sample_period)

    def stop(self) -> None:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=1.5)

    def _init_receiver(self):
        """Instantiate RTDEReceiveInterface, handling signature differences across versions."""
        if RTDEReceiveInterface is None:
            return None

        tried_signatures = []

        def attempt(*args):
            try:
                return RTDEReceiveInterface(*args)
            except TypeError as exc:
                tried_signatures.append((args, exc))
                return None
            except Exception as exc:
                print(f"⚠️ Unable to connect to UR controller at {self.host}:{self.port}: {exc}")
                return None

        # Newer ur_rtde exposes (host, port, frequency)
        receiver = attempt(self.host, self.port, self.frequency)
        if receiver:
            return receiver

        # Some RTDE Python bindings expect (host, frequency)
        receiver = attempt(self.host, self.frequency)
        if receiver:
            return receiver

        # Fallback to just hostname
        receiver = attempt(self.host)
        if receiver:
            return receiver

        # Only log detailed signature info once
        if tried_signatures:
            print("⚠️ RTDEReceiveInterface signature mismatch. Tried:")
            for args, exc in tried_signatures:
                formatted = ", ".join(repr(arg) for arg in args)
                print(f"    RTDEReceiveInterface({formatted}) → {exc}")

        return None


class SchedulerVisualizer:
    """Render the workbench layout with state overlays."""

    def __init__(self, options: VisualizerOptions) -> None:
        try:
            rospy.init_node("scheduler_visualizer", anonymous=True)
        except rospy.ROSException as exc:
            raise SystemExit(
                "❌ Could not initialise ROS node. Please start roscore and the scheduler first."
            ) from exc

        self._options = options
        self.current_state = "Start"
        self.next_states: List[str] = []
        self.state_since = time.time()
        self.state_history: Deque[str] = deque(maxlen=10)
        self.state_history.append(self.current_state)
        self._lock = threading.Lock()
        self._tcp_tracker: Optional[ProjectedTCPPath] = None
        self._tcp_thread: Optional[URTCPWatcher] = None
        self._homography = load_homography_matrix(options.homography_path)
        shift_layout = not options.keep_schematic_layout
        self.zone_layout: Dict[str, Dict[str, Tuple[int, ...]]] = build_zone_layout(self._homography, shift_layout)
        self._state_paths: Dict[str, List[Tuple[int, int]]] = build_state_paths(self.zone_layout)
        self._planned_path: List[Tuple[int, int]] = []
        self._fk_srv: Optional[GetPositionFK] = None

        rospy.Subscriber(
            "server_name/smach/container_status", SmachContainerStatus, self._status_callback
        )

        cv2.namedWindow("Subie Workbench Overview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Subie Workbench Overview", CANVAS_WIDTH, CANVAS_HEIGHT)

        if options.rtde_host:
            self._tcp_tracker = ProjectedTCPPath(self._homography, options.tcp_history, options.tcp_max_age)
            if RTDEReceiveInterface is None:
                print("⚠️ rtde_receive module missing. TCP path will not be drawn.")
            else:
                self._tcp_thread = URTCPWatcher(
                    host=options.rtde_host,
                    port=options.rtde_port,
                    frequency=options.rtde_frequency,
                    tracker=self._tcp_tracker,
                )
                self._tcp_thread.start()

        if options.enable_planned_overlay:
            try:
                rospy.wait_for_service(options.fk_service, timeout=5.0)
                self._fk_srv = rospy.ServiceProxy(options.fk_service, GetPositionFK)
                rospy.Subscriber(
                    options.planned_traj_topic, RobotTrajectory, self._planned_traj_cb, queue_size=1
                )
            except rospy.ROSException:
                print(f"⚠️ FK service {options.fk_service} unavailable. Planned overlay disabled.")
            except rospy.ServiceException as exc:
                print(f"⚠️ Could not create FK proxy: {exc}")

    def _status_callback(self, msg: SmachContainerStatus) -> None:
        if not msg.active_states:
            return

        state_name = msg.active_states[-1]
        with self._lock:
            if state_name != self.current_state:
                self.state_since = time.time()
                self.state_history.append(state_name)
            self.current_state = state_name
            self.next_states = STATE_TRANSITIONS.get(state_name, [])

    def _planned_traj_cb(self, msg: RobotTrajectory) -> None:
        if not self._fk_srv or not msg.joint_trajectory.points:
            return

        joint_names = list(msg.joint_trajectory.joint_names)
        if not joint_names:
            return

        fk_req = GetPositionFKRequest()
        fk_req.header.frame_id = self._options.world_frame
        fk_req.fk_link_names = [self._options.fk_link]

        stride = self._options.planned_stride
        projected: List[Tuple[int, int]] = []

        for idx, point in enumerate(msg.joint_trajectory.points):
            if idx % stride != 0:
                continue
            js = JointState(name=joint_names, position=list(point.positions))
            fk_req.robot_state = RobotState(joint_state=js)
            try:
                resp = self._fk_srv(fk_req)
            except rospy.ServiceException as exc:
                print(f"⚠️ FK request failed: {exc}")
                return

            if resp.error_code.val != resp.error_code.SUCCESS or not resp.pose_stamped:
                continue

            pose = resp.pose_stamped[0].pose
            px_py = self._project_world_xy(pose.position.x, pose.position.y)
            if px_py is not None:
                projected.append(px_py)

        with self._lock:
            self._planned_path = projected

    # === Drawing helpers ==================================================

    def _draw_workbench(self, canvas: np.ndarray, current_state: str, next_state: Optional[str]) -> None:
        canvas[:] = 255  # white background

        x1, y1, x2, y2 = WORKBENCH_RECT
        cv2.rectangle(canvas, (x1, y1), (x2, y2), BOUNDARY_COLOR, thickness=4)
        cv2.putText(canvas, "Bound", (x1, y2 + 18), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        active_zone = STATE_PRIMARY_ZONE.get(current_state)
        next_zone = STATE_PRIMARY_ZONE.get(next_state) if next_state else None

        # Draw schematic zones
        if self._options.overlay_template_zones:
            self._draw_zones(canvas, ZONE_LAYOUT_TEMPLATE, active_zone=None, next_zone=None, faded=True)

        zone_layout = self.zone_layout
        self._draw_zones(canvas, zone_layout, active_zone, next_zone, faded=False)

        # Target circle inside handover zone
        cx, cy, radius = TARGET_CIRCLE
        cv2.circle(canvas, (cx, cy), radius, (0, 0, 255), 4, cv2.LINE_AA)
        cv2.putText(canvas, "TARGET", (cx - 38, cy - radius - 8), FONT, 0.55, (0, 0, 150), 2, cv2.LINE_AA)

        # Robot base indicator
        cv2.circle(canvas, ROBOT_BASE_POINT, 8, (0, 0, 255), -1)

        # Active path polyline
        state_paths = getattr(self, "_state_paths", STATE_PATHS)
        points = state_paths.get(current_state, [])
        if len(points) >= 2:
            self._draw_gradient_path(canvas, points)

        self._draw_planned_path(canvas)
        self._draw_tcp_trace(canvas)

    @staticmethod
    def _draw_gradient_path(canvas: np.ndarray, points: Sequence[Tuple[int, int]]) -> None:
        start_color = np.array((0, 0, 255))
        end_color = np.array((0, 255, 0))

        segments = len(points) - 1
        for idx in range(segments):
            p1 = points[idx]
            p2 = points[idx + 1]
            t = idx / max(segments - 1, 1)
            color_vec = (1 - t) * start_color + t * end_color
            color = tuple(int(c) for c in color_vec)
            cv2.line(canvas, p1, p2, color, 6, cv2.LINE_AA)

        cv2.circle(canvas, points[0], 6, (0, 0, 0), -1)
        cv2.circle(canvas, points[-1], 6, (0, 0, 0), -1)

    def _project_world_xy(self, x_m: float, y_m: float) -> Optional[Tuple[int, int]]:
        coords = _project_point(self._homography, x_m, y_m)
        if coords is None:
            return None
        x_px, y_px = coords
        ix = int(round(x_px))
        iy = int(round(y_px))
        if ix < 0 or iy < 0 or ix >= CANVAS_WIDTH or iy >= CANVAS_HEIGHT:
            return None
        return ix, iy

    def _draw_zones(
        self,
        canvas: np.ndarray,
        layout: Dict[str, Dict[str, Tuple[int, ...]]],
        active_zone: Optional[str],
        next_zone: Optional[str],
        faded: bool,
    ) -> None:
        base_colour = (160, 220, 160) if faded else (0, 255, 0)
        base_thickness = 1 if faded else 3

        for name, data in layout.items():
            if name == "Handover" and not self._options.show_handover_zone:
                continue

            rect = data["rect"]
            label = data.get("label", name)
            colour = base_colour
            thickness = base_thickness

            if not faded:
                if name == active_zone:
                    colour = (0, 0, 255)
                    thickness = 5
                elif name == next_zone:
                    colour = (0, 165, 255)

            cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), colour, thickness)

            if faded:
                label_colour = (120, 160, 120)
                font_scale = 0.5
                font_thickness = 1
            else:
                label_colour = colour if name == active_zone else (0, 150, 0)
                font_scale = 0.6
                font_thickness = 2 if name == active_zone else 1

            cv2.putText(
                canvas,
                label,
                (rect[0] + 6, rect[1] + 20),
                FONT,
                font_scale,
                label_colour,
                font_thickness,
                cv2.LINE_AA,
            )

    def _draw_tcp_trace(self, canvas: np.ndarray) -> None:
        if not self._tcp_tracker:
            return

        path = self._tcp_tracker.get_path()
        if len(path) >= 2:
            cv2.polylines(canvas, [np.array(path, dtype=np.int32)], False, (30, 30, 200), 3, cv2.LINE_AA)

        latest = self._tcp_tracker.get_latest()
        if latest:
            cv2.circle(canvas, latest, 10, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, latest, 16, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                canvas,
                "TCP",
                (latest[0] + 10, latest[1] - 10),
                FONT,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

    def _draw_planned_path(self, canvas: np.ndarray) -> None:
        with self._lock:
            path = list(self._planned_path)

        if len(path) < 2:
            return

        color = (255, 140, 0)
        cv2.polylines(canvas, [np.array(path, dtype=np.int32)], False, color, 2, cv2.LINE_AA)

        end = path[-1]
        cv2.circle(canvas, end, 6, color, -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "PLAN",
            (end[0] + 8, end[1] - 8),
            FONT,
            0.45,
            color,
            2,
            cv2.LINE_AA,
        )

    def _draw_status_panel(
        self, canvas: np.ndarray, current_state: str, upcoming_state: Optional[str], elapsed: float
    ) -> None:
        ready = False
        if callable(get_ready_for_next_step):
            try:
                ready = bool(get_ready_for_next_step())  # type: ignore[misc]
            except Exception:
                ready = False

        # Ready indicator
        ready_rect = (1060, 80, 1220, 130)
        cv2.rectangle(canvas, (ready_rect[0], ready_rect[1]), (ready_rect[2], ready_rect[3]), (0, 0, 0), 2)
        cv2.putText(canvas, "Ready:", (ready_rect[0] + 10, ready_rect[1] + 30), FONT, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        indicator_color = (0, 200, 0) if ready else (0, 0, 255)
        cv2.circle(canvas, (ready_rect[2] - 30, ready_rect[1] + 25), 16, indicator_color, -1, cv2.LINE_AA)

        # Current image placeholder / label
        panel_rect = (1060, 160, 1220, 320)
        cv2.rectangle(canvas, (panel_rect[0], panel_rect[1]), (panel_rect[2], panel_rect[3]), (60, 60, 60), 2)
        cv2.rectangle(canvas, (panel_rect[0], panel_rect[1]), (panel_rect[2], panel_rect[3]), (180, 180, 180), -1)

        cv2.putText(canvas, "Current", (panel_rect[0] + 10, panel_rect[1] + 25), FONT, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            get_state_label(current_state),
            (panel_rect[0] + 12, panel_rect[1] + 95),
            FONT,
            0.55,
            (150, 0, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            "Next:",
            (panel_rect[0] + 10, panel_rect[3] + 35),
            FONT,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        next_label = get_state_label(upcoming_state) if upcoming_state else "—"
        cv2.putText(
            canvas,
            next_label,
            (panel_rect[0] + 70, panel_rect[3] + 35),
            FONT,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        # Progress circle
        centre = (1135, 540)
        radius = 60
        cv2.circle(canvas, centre, radius, (0, 200, 0), 4, cv2.LINE_AA)

        progress = min(elapsed / 30.0, 1.0)  # assume nominal 30s per state
        angle = int(progress * 360)
        cv2.ellipse(canvas, centre, (radius, radius), -90, 0, angle, (0, 200, 0), 8, cv2.LINE_AA)

        elapsed_text = f"{int(elapsed):d}"
        cv2.putText(
            canvas,
            elapsed_text,
            (centre[0] - 15, centre[1] + 8),
            FONT,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    def spin(self) -> None:
        rate = rospy.Rate(12)
        try:
            while not rospy.is_shutdown():
                with self._lock:
                    current_state = self.current_state
                    elapsed = time.time() - self.state_since
                    next_state = self.next_states[0] if self.next_states else None

                canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
                self._draw_workbench(canvas, current_state, next_state)
                self._draw_status_panel(canvas, current_state, next_state, elapsed)

                cv2.imshow("Subie Workbench Overview", canvas)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break

                rate.sleep()
        except rospy.ROSInterruptException:
            pass
        finally:
            cv2.destroyAllWindows()
            if self._tcp_thread:
                self._tcp_thread.stop()


def main() -> None:
    options = parse_cli_args()
    visualizer = SchedulerVisualizer(options)
    visualizer.spin()


if __name__ == "__main__":
    main()
