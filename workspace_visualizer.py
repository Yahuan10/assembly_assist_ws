import math
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# === NEW: for trajectory + FK ===
from moveit_msgs.msg import RobotTrajectory, RobotState
from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped


try:
    import rospy
    from smach_msgs.msg import SmachContainerStatus
except ImportError as exc:  # pragma: no cover - executed when ROS is missing
    raise SystemExit(
        "❌ workspace_visualizer requires ROS (rospy + smach_msgs). "
        "Please ensure you are inside a ROS environment."
    ) from exc

try:
    from shared_state import get_ready_for_next_step  # type: ignore[attr-defined]
except ImportError:
    get_ready_for_next_step = None  # type: ignore[assignment]


# Canvas configuration
CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Workbench boundary (top-left x/y, bottom-right x/y)
WORKBENCH_RECT = (60, 60, 1040, 660)
BOUNDARY_COLOR = (0, 255, 0)  # green

# Zone layout: rectangles defined as (x1, y1, x2, y2)
ZONE_LAYOUT: Dict[str, Dict[str, Tuple[int, ...]]] = {
    "Battery": {"rect": (100, 90, 620, 230), "label": "Battery"},
    "Motor": {"rect": (660, 90, 860, 260), "label": "Motor"},
    "MotorTray": {"rect": (660, 90, 860, 260), "label": "Motor"},
    "Assembly": {"rect": (320, 360, 620, 540), "label": "Assembly"},
    "PCB2": {"rect": (320, 260, 620, 320), "label": "PCB2"},
    "PCB1": {"rect": (660, 260, 880, 320), "label": "PCB1"},
    "PCB3": {"rect": (100, 360, 280, 540), "label": "PCB3"},
    "Handover": {"rect": (100, 560, 280, 640), "label": "Handover"},
    "Home": {"rect": (480, 420, 520, 460), "label": "Home"},
}

# Compute rectangle centres and store them for later use
for zone_cfg in ZONE_LAYOUT.values():
    x1, y1, x2, y2 = zone_cfg["rect"]
    zone_cfg["center"] = ((x1 + x2) // 2, (y1 + y2) // 2)

# Target circle inside the handover area (center x, center y, radius)
TARGET_CIRCLE = (190, 620, 30)

# Robot base indicator
ROBOT_BASE_POINT = (600, 420)

# Map SMACH states to their primary zones (used for colouring)
STATE_PRIMARY_ZONE = {
    "Start": "Motor",
    "MPickUp": "Motor",
    "MHold": "Handover",
    "MHoldHD": "Handover",
    "MPositioning": "Assembly",
    "PCB1PickUpAndPositioning": "PCB1",
    "PCB2PickUpAndPositioning": "PCB2",
    "PCB3PickUpAndPositioning": "PCB3",
    "BatteryPickUpAndPositioning": "Battery",
    "Test": "Motor",
    "Aborted": "Handover",
    "finished": "Home",
}

# Define the path (sequence of zone names) that the robot follows per state.
STATE_PATH_TEMPLATE: Dict[str, Sequence[str]] = {
    "Start": ("Home",),
    "MPickUp": ("Motor", "Assembly"),
    "MHold": ("Handover", "Assembly"),
    "MHoldHD": ("Handover", "Assembly"),
    "MPositioning": ("Assembly",),
    "PCB1PickUpAndPositioning": ("Assembly", "PCB1"),
    "PCB2PickUpAndPositioning": ("Assembly", "PCB2"),
    "PCB3PickUpAndPositioning": ("Assembly", "PCB3"),
    "BatteryPickUpAndPositioning": ("Assembly", "Battery"),
    "Test": ("Motor",),
    "Aborted": ("Handover", "Home"),
    "finished": ("Home",),
}


def build_state_paths() -> Dict[str, List[Tuple[int, int]]]:
    """Convert zone sequences to pixel coordinates."""
    paths: Dict[str, List[Tuple[int, int]]] = {}
    for state, zones in STATE_PATH_TEMPLATE.items():
        points: List[Tuple[int, int]] = []
        for zone in zones:
            cfg = ZONE_LAYOUT.get(zone)
            if not cfg:
                continue
            points.append(cfg["center"])  # type: ignore[index]
        if points:
            paths[state] = points
    return paths


STATE_PATHS = build_state_paths()

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


class SchedulerVisualizer:
    """Render the workbench layout with state overlays."""

    def __init__(self) -> None:
        try:
            rospy.init_node("scheduler_visualizer", anonymous=True)
        except rospy.ROSException as exc:
            raise SystemExit(
                "❌ Could not initialise ROS node. Please start roscore and the scheduler first."
            ) from exc

        self.current_state = "Start"
        self.next_states: List[str] = []
        self.state_since = time.time()
        self.state_history: Deque[str] = deque(maxlen=10)
        self.state_history.append(self.current_state)
        self._lock = threading.Lock()

                # === NEW: planned TCP path cache (pixel points) and params ===
        self.tcp_path_px: List[Tuple[int, int]] = []
        self.show_tcp_path: bool = rospy.get_param("~show_tcp_path", True)

        # 固定比例映射（不缩放，不自适应）
        self.world_frame: str = rospy.get_param("~world_frame", "base")
        self.eef_link: str = rospy.get_param("~eef_link", "tool0")
        self.px_per_m: float = float(rospy.get_param("~px_per_m", 1000.0))  # 1m 对应多少像素（固定）
        rb_xy = rospy.get_param("~robot_base_world_xy", [0.0, 0.0])
        self.robot_base_world_xy: Tuple[float, float] = (float(rb_xy[0]), float(rb_xy[1]))

        # === NEW: FK service
        rospy.loginfo("等待 /compute_fk 服务 ...")
        try:
            rospy.wait_for_service('/compute_fk', timeout=10.0)
            self._fk_srv = rospy.ServiceProxy('/compute_fk', GetPositionFK)
            rospy.loginfo("FK 服务就绪。")
        except Exception as exc:
            rospy.logwarn("FK 服务不可用：%s ；TCP 轨迹将不会显示。", exc)
            self._fk_srv = None

        # === Subscribers ===
        rospy.Subscriber(
            "server_name/smach/container_status", SmachContainerStatus, self._status_callback
        )
        planned_topic = rospy.get_param("~planned_traj_topic", "/planned_trajectory")
        rospy.Subscriber(planned_topic, RobotTrajectory, self._planned_traj_cb, queue_size=1)

        # OpenCV window
        cv2.namedWindow("Subie Workbench Overview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Subie Workbench Overview", CANVAS_WIDTH, CANVAS_HEIGHT)


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
        """Receive planned trajectory, run FK to get TCP poses, project to canvas (no scaling), cache polyline."""
        if not self._fk_srv:
            return
        if not msg.joint_trajectory.points or not msg.joint_trajectory.joint_names:
            return

        fk_req = GetPositionFKRequest()
        fk_req.header.frame_id = self.world_frame
        fk_req.fk_link_names = [self.eef_link]

        joint_names = list(msg.joint_trajectory.joint_names)
        pts_px: List[Tuple[int, int]] = []

        for pt in msg.joint_trajectory.points:
            js = JointState(name=joint_names, position=list(pt.positions))
            fk_req.robot_state = RobotState(joint_state=js)
            try:
                resp = self._fk_srv.call(fk_req)
            except Exception as e:
                rospy.logwarn("FK 调用失败，跳过该点：%s", e)
                continue

            if not resp.error_code or resp.error_code.val != resp.error_code.SUCCESS:
                continue

            pose = resp.pose_stamped[0].pose
            x_m, y_m = pose.position.x, pose.position.y
            pts_px.append(self._world_to_canvas_xy_no_scale(x_m, y_m))

        with self._lock:
            self.tcp_path_px = pts_px


    # === Drawing helpers ==================================================

    def _draw_workbench(self, canvas: np.ndarray, current_state: str, next_state: Optional[str]) -> None:
        canvas[:] = 255  # white background

        x1, y1, x2, y2 = WORKBENCH_RECT
        cv2.rectangle(canvas, (x1, y1), (x2, y2), BOUNDARY_COLOR, thickness=4)
        cv2.putText(canvas, "Bound", (x1, y2 + 18), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        active_zone = STATE_PRIMARY_ZONE.get(current_state)
        next_zone = STATE_PRIMARY_ZONE.get(next_state) if next_state else None

        # Draw schematic zones
        for name, data in ZONE_LAYOUT.items():
            rect = data["rect"]
            label = data.get("label", name)
            colour = (0, 255, 0)
            thickness = 3

            if name == active_zone:
                colour = (0, 0, 255)
                thickness = 5
            elif name == next_zone:
                colour = (0, 165, 255)

            cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), colour, thickness)
            cv2.putText(
                canvas,
                label,
                (rect[0] + 6, rect[1] + 20),
                FONT,
                0.6,
                colour if name == active_zone else (0, 150, 0),
                2 if name == active_zone else 1,
                cv2.LINE_AA,
            )

        # Target circle inside handover zone
        cx, cy, radius = TARGET_CIRCLE
        cv2.circle(canvas, (cx, cy), radius, (0, 120, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, "TARGET", (cx - 38, cy - radius - 8), FONT, 0.55, (0, 120, 0), 2, cv2.LINE_AA)

        # Robot base indicator
        cv2.circle(canvas, ROBOT_BASE_POINT, 8, (0, 0, 255), -1)

        # Active path polyline (schematic path between zones)
        points = STATE_PATHS.get(current_state, [])
        if len(points) >= 2:
            self._draw_gradient_path(canvas, points)

        # === NEW: draw planned TCP path (if any, clipped to canvas, no scaling)
        if self.show_tcp_path:
            self._draw_tcp_planned_path(canvas)

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

    # --------------------------
    # World→Canvas (fixed mapping, no scaling/adaptation)
    # --------------------------
    def _world_to_canvas_xy_no_scale(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """
        固定比例映射：以 ROBOT_BASE_WORLD → ROBOT_BASE_POINT 为锚点。
        世界坐标 Y 轴向上为正；OpenCV 画布 Y 轴向下为正，所以这里对 Y 取反。
        """
        dx_m = (x_m - self.robot_base_world_xy[0])
        dy_m = (y_m - self.robot_base_world_xy[1])

        px = int(ROBOT_BASE_POINT[0] + dx_m * self.px_per_m)
        py = int(ROBOT_BASE_POINT[1] - dy_m * self.px_per_m)
        return (px, py)

    @staticmethod
    def _canvas_clip_rect() -> Tuple[int, int, int, int]:
        """Return canvas clip rect (x1, y1, x2, y2) for cv2.clipLine."""
        return (0, 0, CANVAS_WIDTH - 1, CANVAS_HEIGHT - 1)

    def _draw_tcp_planned_path(self, canvas: np.ndarray) -> None:
        """Draw planned TCP polyline with clipping, no auto-scaling."""
        with self._lock:
            pts = list(self.tcp_path_px)

        if len(pts) < 2:
            return

        rect = self._canvas_clip_rect()
        start_color = np.array((0, 0, 255))
        end_color = np.array((0, 255, 0))
        segs = len(pts) - 1

        # Draw line segments with gradient and clip to canvas
        for i in range(segs):
            p1 = pts[i]
            p2 = pts[i + 1]
            t = i / max(segs - 1, 1)
            color = tuple(int(v) for v in ((1 - t) * start_color + t * end_color))

            ok, q1, q2 = cv2.clipLine(rect, p1, p2)
            if ok:
                cv2.line(canvas, q1, q2, color, 3, cv2.LINE_AA)

        # Start and end markers (only if visible)
        x0, y0 = pts[0]
        if 0 <= x0 < CANVAS_WIDTH and 0 <= y0 < CANVAS_HEIGHT:
            cv2.circle(canvas, (x0, y0), 5, (50, 50, 50), -1, cv2.LINE_AA)

        x1, y1 = pts[-1]
        if 0 <= x1 < CANVAS_WIDTH and 0 <= y1 < CANVAS_HEIGHT:
            cv2.circle(canvas, (x1, y1), 6, (0, 0, 0), -1, cv2.LINE_AA)

        # Label near start (clamped to canvas)
        label_x = min(max(x0 + 8, 0), CANVAS_WIDTH - 80)
        label_y = min(max(y0 - 8, 0), CANVAS_HEIGHT - 10)
        cv2.putText(canvas, "TCP Plan", (label_x, label_y), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # --------------------------
    # Main loop
    # --------------------------
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


def main() -> None:
    visualizer = SchedulerVisualizer()
    visualizer.spin()


if __name__ == "__main__":
    main()