#!/usr/bin/env python3
"""Receive compressed sensor frames over HTTP and republish them as ROS topics.

The server-side half of the robot link. A lightweight sender (the robot, or
tools/percorso-feed replaying a bag) posts frame bundles here; this node decodes
them and publishes exactly what the perception pipeline already expects, so
nothing downstream knows the frames arrived over a network.

Deliberately NOT a DDS bridge: the robot runs a different ROS distro and sits
behind NAT on another network, and Berzelius compute nodes are unreachable for
DDS discovery. One outbound HTTP connection sidesteps all of that.

Publishes: /cam0/rgb_image, /cam0/depth_image, /cam0/camera_info, /tf,
/tf_static, /clock — the same set `ros2 bag play --clock` produces.

Run before the pipeline:
    ros2 run percorso_perception_ros percorso_ingest_node.py --ros-args -p port:=9000
"""
import io
import json
import struct
import threading
import zlib

import numpy as np
# Imported at MODULE level on purpose: FastAPI resolves handler annotations
# against module globals, so a function-local `Request` import cannot be found
# and the endpoint silently degrades to a query parameter (HTTP 422).
from fastapi import FastAPI, Request, Response
import rclpy  # type: ignore
from rclpy.node import Node  # type: ignore
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy  # type: ignore
from sensor_msgs.msg import CameraInfo, Image  # type: ignore
from rosgraph_msgs.msg import Clock  # type: ignore
from tf2_msgs.msg import TFMessage  # type: ignore
from geometry_msgs.msg import TransformStamped  # type: ignore

_HEADER_LEN_BYTES = 4


def _decode_bundle(body: bytes) -> tuple[dict, bytes, bytes]:
    """Split a wire bundle into (header, rgb_bytes, depth_bytes).

    Wire format, chosen so both ends need no extra dependencies and a frame can
    be inspected with plain tools:
        [4-byte big-endian header length][JSON header][rgb blob][depth blob]
    The header carries the blob lengths and encodings, so this stays
    self-describing rather than positional.
    """
    if len(body) < _HEADER_LEN_BYTES:
        raise ValueError("bundle shorter than its length prefix")
    (hlen,) = struct.unpack(">I", body[:_HEADER_LEN_BYTES])
    start = _HEADER_LEN_BYTES + hlen
    if len(body) < start:
        raise ValueError("bundle truncated inside the header")
    header = json.loads(body[_HEADER_LEN_BYTES:start].decode("utf-8"))
    rgb_len = int(header.get("rgb_len", 0))
    depth_len = int(header.get("depth_len", 0))
    rgb = body[start:start + rgb_len]
    depth = body[start + rgb_len:start + rgb_len + depth_len]
    if len(rgb) != rgb_len or len(depth) != depth_len:
        raise ValueError(
            f"bundle truncated: want rgb={rgb_len} depth={depth_len}, "
            f"got rgb={len(rgb)} depth={len(depth)}"
        )
    return header, rgb, depth


class PercorsoIngestNode(Node):
    def __init__(self) -> None:
        super().__init__("percorso_ingest_node")

        self.declare_parameter("host", "0.0.0.0")
        self.declare_parameter("port", 9000)
        self.declare_parameter("publish_clock", True)
        self.declare_parameter("rgb_topic", "/cam0/rgb_image")
        self.declare_parameter("depth_topic", "/cam0/depth_image")
        self.declare_parameter("camera_info_topic", "/cam0/camera_info")

        self.host = self.get_parameter("host").value
        self.port = int(self.get_parameter("port").value)
        self.publish_clock = bool(self.get_parameter("publish_clock").value)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        # tf_static must be transient_local or a late-joining subscriber never
        # sees the one message the bag contains.
        static_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.pub_rgb = self.create_publisher(
            Image, self.get_parameter("rgb_topic").value, sensor_qos)
        self.pub_depth = self.create_publisher(
            Image, self.get_parameter("depth_topic").value, sensor_qos)
        self.pub_info = self.create_publisher(
            CameraInfo, self.get_parameter("camera_info_topic").value, sensor_qos)
        self.pub_tf = self.create_publisher(TFMessage, "/tf", sensor_qos)
        self.pub_tf_static = self.create_publisher(TFMessage, "/tf_static", static_qos)
        self.pub_clock = self.create_publisher(Clock, "/clock", sensor_qos) \
            if self.publish_clock else None

        self.frames = 0
        self.bytes_in = 0
        self.errors = 0
        self._lock = threading.Lock()

        self._server = None
        self._thread: threading.Thread | None = None
        self._start_http()

        self.create_timer(5.0, self._report)
        self.get_logger().info(
            f"percorso ingest listening on http://{self.host}:{self.port}  "
            f"(publish_clock={self.publish_clock})"
        )

    # ── HTTP ──────────────────────────────────────────────────────────────────

    def _build_app(self):
        app = FastAPI(title="percorso ingest")

        @app.get("/status")
        def status() -> dict:
            with self._lock:
                return {
                    "frames": self.frames,
                    "bytes_in": self.bytes_in,
                    "errors": self.errors,
                }

        @app.post("/ingest/frame")
        async def ingest_frame(request: Request) -> Response:
            body = await request.body()
            try:
                header, rgb, depth = _decode_bundle(body)
                self._publish(header, rgb, depth)
            except Exception as exc:
                with self._lock:
                    self.errors += 1
                self.get_logger().warning(f"bad frame bundle: {exc}")
                return Response(content=str(exc), status_code=400)
            with self._lock:
                self.frames += 1
                self.bytes_in += len(body)
            return Response(status_code=204)

        return app

    def _start_http(self) -> None:
        import uvicorn

        config = uvicorn.Config(
            self._build_app(), host=self.host, port=self.port,
            log_level="warning", access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run, name="percorso_ingest_http", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    # ── publishing ────────────────────────────────────────────────────────────

    def _stamp(self, header: dict):
        stamp = rclpy.time.Time(
            seconds=int(header["stamp_sec"]),
            nanoseconds=int(header["stamp_nanosec"]),
        ).to_msg()
        return stamp

    def _publish(self, header: dict, rgb_blob: bytes, depth_blob: bytes) -> None:
        stamp = self._stamp(header)
        frame_id = header.get("frame_id", "cam0")

        # /clock first: downstream nodes run with use_sim_time, so time must
        # already be at this frame before the data arrives.
        if self.pub_clock is not None:
            clock = Clock()
            clock.clock = stamp
            self.pub_clock.publish(clock)

        if rgb_blob:
            self.pub_rgb.publish(self._to_image(
                rgb_blob, header, stamp, frame_id, kind="rgb"))
        if depth_blob:
            self.pub_depth.publish(self._to_image(
                depth_blob, header, stamp, frame_id, kind="depth"))

        info_d = header.get("camera_info")
        if info_d:
            self.pub_info.publish(self._to_camera_info(info_d, stamp, frame_id))

        for key, pub in (("tf", self.pub_tf), ("tf_static", self.pub_tf_static)):
            entries = header.get(key)
            if entries:
                pub.publish(self._to_tf(entries))

    def _to_image(self, blob: bytes, header: dict, stamp, frame_id: str, *, kind: str) -> Image:
        meta = header[kind]
        codec = meta["codec"]
        width, height = int(meta["width"]), int(meta["height"])
        encoding = meta["encoding"]

        if codec == "jpeg":
            from PIL import Image as PILImage
            arr = np.asarray(PILImage.open(io.BytesIO(blob)).convert("RGB"), dtype=np.uint8)
        elif codec == "zlib":
            raw = zlib.decompress(blob)
            dtype = np.uint16 if "16" in encoding else np.uint8
            arr = np.frombuffer(raw, dtype=dtype)
            channels = int(meta.get("channels", 1))
            arr = arr.reshape((height, width) if channels == 1 else (height, width, channels))
        elif codec == "raw":
            dtype = np.uint16 if "16" in encoding else np.uint8
            arr = np.frombuffer(blob, dtype=dtype)
            channels = int(meta.get("channels", 1))
            arr = arr.reshape((height, width) if channels == 1 else (height, width, channels))
        else:
            raise ValueError(f"unknown codec {codec!r}")

        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height, msg.width = int(arr.shape[0]), int(arr.shape[1])
        msg.encoding = encoding
        msg.is_bigendian = 0
        msg.step = int(arr.strides[0])
        msg.data = np.ascontiguousarray(arr).tobytes()
        return msg

    def _to_camera_info(self, d: dict, stamp, frame_id: str) -> CameraInfo:
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = d.get("frame_id", frame_id)
        msg.height = int(d["height"])
        msg.width = int(d["width"])
        msg.distortion_model = d.get("distortion_model", "plumb_bob")
        msg.d = [float(x) for x in d.get("d", [])]
        for field in ("k", "r", "p"):
            if d.get(field):
                setattr(msg, field, np.array([float(x) for x in d[field]], dtype=np.float64))
        return msg

    def _to_tf(self, entries: list) -> TFMessage:
        msg = TFMessage()
        for e in entries:
            t = TransformStamped()
            t.header.stamp = rclpy.time.Time(
                seconds=int(e["stamp_sec"]), nanoseconds=int(e["stamp_nanosec"])
            ).to_msg()
            t.header.frame_id = e["frame_id"]
            t.child_frame_id = e["child_frame_id"]
            tr = e["translation"]
            t.transform.translation.x = float(tr[0])
            t.transform.translation.y = float(tr[1])
            t.transform.translation.z = float(tr[2])
            r = e["rotation"]
            t.transform.rotation.x = float(r[0])
            t.transform.rotation.y = float(r[1])
            t.transform.rotation.z = float(r[2])
            t.transform.rotation.w = float(r[3])
            msg.transforms.append(t)
        return msg

    def _report(self) -> None:
        with self._lock:
            frames, byts, errs = self.frames, self.bytes_in, self.errors
        if frames or errs:
            self.get_logger().info(
                f"ingest: {frames} frames, {byts / 1e6:.1f} MB, {errs} error(s)")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PercorsoIngestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
