#!/usr/bin/env python3

"""ROS2 Node to visualize Odometry, Images, Scene Graph, and Mesh using Rerun."""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from cv_bridge import CvBridge
import traceback
import threading

from scipy.spatial.transform import Rotation

# ROS2 messages
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from hydra_msgs.msg import DsgUpdate
from tf2_ros import TransformListener, Buffer
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from geometry_msgs.msg import TransformStamped
import textwrap

from daaam.utils.vision import load_color_map
from daaam.utils.logging import get_default_logger

try:
	import spark_dsg
	from spark_dsg import (
		DynamicSceneGraph,
		DsgLayers,
		NodeSymbol,
		SceneGraphNode,
		LayerView,
		BoundingBoxType
	)

	SPARK_DSG_ENABLED = True
except ImportError as e:
	SPARK_DSG_ENABLED = False
	print(f"Warning: Failed to import spark_dsg bindings: {e}")
	print("DSG visualization will be disabled.")


def to_ns(stamp):
	"""Convert ROS2 time to nanoseconds."""
	return stamp.sec * 1_000_000_000 + stamp.nanosec


# layer ID to name mapping (similar to open3d_visualization)
LAYER_NAMES = {
	DsgLayers.name_to_layer_id(DsgLayers.OBJECTS): "objects_agents",
	DsgLayers.name_to_layer_id(DsgLayers.PLACES): "places",
	DsgLayers.name_to_layer_id(DsgLayers.ROOMS): "rooms",
	DsgLayers.name_to_layer_id(DsgLayers.BUILDINGS): "buildings",
	DsgLayers.name_to_layer_id(DsgLayers.AGENTS): "objects_agents",
}

LAYER_COLORS = {
	DsgLayers.name_to_layer_id(DsgLayers.OBJECTS): [255, 0, 0],
	DsgLayers.name_to_layer_id(DsgLayers.PLACES): [0, 255, 0],
	DsgLayers.name_to_layer_id(DsgLayers.ROOMS): [0, 0, 255],
	DsgLayers.name_to_layer_id(DsgLayers.BUILDINGS): [255, 255, 0],
	DsgLayers.name_to_layer_id(DsgLayers.AGENTS): [255, 0, 255],
}


class RerunVisualizerNode(Node):
	"""Visualizes Odometry, Images, Scene Graph, and Mesh using Rerun."""

	def __init__(self):
		"""Initialize the node, declare parameters, and set up subscribers."""
		super().__init__("rerun_visualizer_node")

		if not SPARK_DSG_ENABLED:
			self.get_logger().error("Spark DSG bindings not found. Exiting.")

			raise ImportError("Spark DSG bindings are required but not found.")

		# --- Rerun Initialization ---
		rr.init("rerun_visualizer", spawn=True)
		rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True) 

		# blueprint = self.make_rerun_blueprint()
		# rr.send_blueprint(blueprint)

		# --- params ---
		self.declare_parameter("depth_info_topic", "/dominic/forward/camera_info") # #/tesse/depth_cam/camera_info
		self.declare_parameter("depth_image_topic", "/dominic/forward/depth/image_rect_raw") # /tesse/depth_cam/mono/image_raw
		self.declare_parameter("rgb_info_topic", "/dominic/forward/camera_info") # /tesse/left_cam/camera_info
		self.declare_parameter("rgb_image_topic", "/dominic/forward/color/image_raw") # /tesse/left_cam/rgb/image_raw
		self.declare_parameter("semantic_info_topic", "/dominic/forward/camera_info") # /tesse/left_cam/camera_info
		self.declare_parameter("semantic_color_topic", "/segmentation/color/image_raw") # /segmentation/color/image_raw
		self.declare_parameter("odom_topic", "/dominic/forward/colmap_odom") # /tesse/odom
		self.declare_parameter("dsg_topic", "/hydra/backend/dsg") # Now from Hydra directly
		self.declare_parameter("semantic_updates_topic", "/daaam/semantic_updates")
		self.declare_parameter("background_objects_topic", "/daaam/background_objects_updates")
		self.declare_parameter("dsg_update_rate", 0.1)  # Hz for DSG processing
		self.declare_parameter("depth_scale", 1000.0) 
		self.declare_parameter("log_object_meshes", False) 
		self.declare_parameter("log_depth", False)

		# colormap
		self.declare_parameter("labelspace_colors", "/path/to/daaam_ros/config/labels_pseudo.csv")
		
		# temporal visualization parameters
		self.declare_parameter("enable_temporal_viz", True)
		self.declare_parameter("enable_retroactive_logging", True)
		self.declare_parameter("temporal_pulse_duration_ms", 500)
		self.declare_parameter("track_buffer", 30)  # Frames to keep track alive
		self.declare_parameter("show_temporal_labels", True)
		self.declare_parameter("show_observation_pulses", False)
		self.declare_parameter("track_buffer_opacity_min", 0.3)
		self.declare_parameter("track_buffer_opacity_max", 1.0)
		
		# tf2 parameters
		self.declare_parameter("use_tf2", True) 
		self.declare_parameter("world_frame", "map")  # map
		self.declare_parameter("camera_frame", "dominic/forward_link")  # left_cam
		self.declare_parameter("tf_timeout", 0.1) 
		self.declare_parameter("tf_update_rate", 30.0)  # Hz

		# --- state variables ---
		self.bridge = CvBridge()
		self.depth_intrinsics = None
		self.rgb_intrinsics = None
		self.semantic_intrinsics = None
		self.current_time_ns = None

		self.dsg = DynamicSceneGraph()
		self.pending_dsg_update = None  # Store pending DSG update
		self.dsg_update_lock = threading.Lock()
		self.semantic_labels = {}  # Store semantic labels from updates
		self.background_objects = {}  # Store background objects that are not in Hydra

		# temporal visualization state
		self.enable_temporal_viz = self.get_parameter("enable_temporal_viz").value
		self.enable_retroactive = self.get_parameter("enable_retroactive_logging").value
		self.pulse_duration_ms = self.get_parameter("temporal_pulse_duration_ms").value
		self.track_buffer = self.get_parameter("track_buffer").value
		self.show_temporal_labels = self.get_parameter("show_temporal_labels").value
		self.show_observation_pulses = self.get_parameter("show_observation_pulses").value
		self.opacity_min = self.get_parameter("track_buffer_opacity_min").value
		self.opacity_max = self.get_parameter("track_buffer_opacity_max").value
		self.current_frame_id = 0  # Track current frame for temporal calculations
		
		# tf2 setup
		self.use_tf2 = self.get_parameter("use_tf2").value
		if self.use_tf2:
			self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
			self.tf_listener = TransformListener(self.tf_buffer, self)
			self.world_frame = self.get_parameter("world_frame").value
			self.camera_frame = self.get_parameter("camera_frame").value
			self.tf_timeout = self.get_parameter("tf_timeout").value
			self.get_logger().info(f"tf2 enabled: {self.world_frame} -> {self.camera_frame}")

		# get color map:
		self.color_map = load_color_map(
			self.get_parameter("labelspace_colors").value, self.get_logger()
			)

		# --- qos profiles ---
		# best effort for visualization data, keep last for state
		sensor_qos = QoSProfile(
			reliability=QoSReliabilityPolicy.BEST_EFFORT,
			history=QoSHistoryPolicy.KEEP_LAST,
			depth=10,
		)
		# reliable for odometry and graph/mesh updates
		reliable_qos = QoSProfile(
			reliability=QoSReliabilityPolicy.RELIABLE,
			history=QoSHistoryPolicy.KEEP_LAST,
			depth=10,
		)
		reliable_qos_semantic_updates = QoSProfile(
			reliability=QoSReliabilityPolicy.RELIABLE,
			history=QoSHistoryPolicy.KEEP_LAST,
			depth=200,
		)

		# tracked timestamps (/hydra/reconstruction/mesh has two messages per timestamp)
		self.tracked_timestamps = set() 

		# --- subscribers ---
		self.create_subscription(
			CameraInfo,
			self.get_parameter("depth_info_topic").value,
			self._depth_info_cb,
			sensor_qos,
		)
		self.create_subscription(
			Image,
			self.get_parameter("depth_image_topic").value,
			self._depth_cb,
			sensor_qos,
		)
		self.create_subscription(
			CameraInfo,
			self.get_parameter("rgb_info_topic").value,
			self._rgb_info_cb,
			sensor_qos,
		)
		self.create_subscription(
			Image,
			self.get_parameter("rgb_image_topic").value,
			self._rgb_cb,
			sensor_qos,
		)
		self.create_subscription(
			CameraInfo,
			self.get_parameter("semantic_info_topic").value,
			self._semantic_info_cb,
			sensor_qos,
		)
		self.create_subscription(
			Image,
			self.get_parameter("semantic_color_topic").value,
			self._semantic_cb,
			sensor_qos,
		)
		self.create_subscription(
			DsgUpdate,
			self.get_parameter("dsg_topic").value,
			self._dsg_cb,
			reliable_qos, # graph updates are important state
		)
		self.create_subscription(
			String,
			self.get_parameter("semantic_updates_topic").value,
			self._semantic_updates_cb,
			reliable_qos_semantic_updates,
		)
		self.create_subscription(
			String,
			self.get_parameter("background_objects_topic").value,
			self._background_objects_cb,
			reliable_qos_semantic_updates,
		)

		# Timer for periodic DSG processing
		dsg_update_rate = self.get_parameter("dsg_update_rate").value
		if dsg_update_rate > 0:
			self.dsg_timer = self.create_timer(1.0 / dsg_update_rate, self._process_pending_dsg)
			self.get_logger().info(f"DSG processing timer set to {dsg_update_rate} Hz")

		self.get_logger().info("Rerun Visualizer Node Initialized.")
		self.get_logger().info(f"Subscribing to:")
		self.get_logger().info(f"  Depth Info: {self.get_parameter('depth_info_topic').value}")
		self.get_logger().info(f"  Depth Image: {self.get_parameter('depth_image_topic').value}")
		self.get_logger().info(f"  RGB Info: {self.get_parameter('rgb_info_topic').value}")
		self.get_logger().info(f"  RGB Image: {self.get_parameter('rgb_image_topic').value}")
		self.get_logger().info(f"  Semantic Info: {self.get_parameter('semantic_info_topic').value}")
		self.get_logger().info(f"  Semantic Color: {self.get_parameter('semantic_color_topic').value}")
		# self.get_logger().info(f"  Odometry: {self.get_parameter('odom_topic').value}")
		self.get_logger().info(f"  DSG Update: {self.get_parameter('dsg_topic').value}")
		self.get_logger().info(f"  Semantic Updates: {self.get_parameter('semantic_updates_topic').value}") 
		
		if self.use_tf2:
			self.get_logger().info(f"tf2 Configuration:")
			self.get_logger().info(f"  Use tf2: {self.use_tf2}")
			self.get_logger().info(f"  World Frame: {self.world_frame}")
			self.get_logger().info(f"  Camera Frame: {self.camera_frame}")
			self.get_logger().info(f"  Timeout: {self.tf_timeout}s")
			tf_update_rate = self.get_parameter("tf_update_rate").value
			if tf_update_rate > 0:
				self.get_logger().info(f"  TF Update Rate: {tf_update_rate} Hz")

	def _get_camera_transform_tf2(self, timestamp):
		"""Get camera transform using tf2."""
		try:
			# try with the exact timestamp
			transform = self.tf_buffer.lookup_transform(
				self.world_frame,
				self.camera_frame,
				timestamp,
				timeout=rclpy.duration.Duration(seconds=self.tf_timeout)
			)
			
			pos = transform.transform.translation
			quat = transform.transform.rotation

			time_delta = to_ns(timestamp) - to_ns(transform.header.stamp)


			self.get_logger().info(f"Lookup transform for {timestamp}. Got timestamp {transform.header.stamp.sec}.{transform.header.stamp.nanosec}. Time delta: {time_delta/9} s")
			self.get_logger().debug(f"  Translation: [{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}]")
			self.get_logger().debug(f"  Rotation (xyzw): [{quat.x:.3f}, {quat.y:.3f}, {quat.z:.3f}, {quat.w:.3f}]")

			return pos, quat, to_ns(transform.header.stamp), True

		except Exception as e:
			# if exact timestamp fails, try with latest available
			try:
				self.get_logger().debug(f"tf2 exact timestamp failed, trying latest: {e}")
				transform = self.tf_buffer.lookup_transform(
					self.world_frame,
					self.camera_frame,
					rclpy.time.Time(), 
					timeout=rclpy.duration.Duration(seconds=self.tf_timeout)
				)

				time_delta = to_ns(timestamp) - to_ns(transform.header.stamp)
				
				pos = transform.transform.translation
				quat = transform.transform.rotation

				self.get_logger().info(f"Lookup transform for {timestamp}. Got timestamp {transform.header.stamp.sec}.{transform.header.stamp.nanosec}. Time delta: {time_delta/9} s")
				self.get_logger().debug(f"  Translation: [{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}]")
				self.get_logger().debug(f"  Rotation (xyzw): [{quat.x:.3f}, {quat.y:.3f}, {quat.z:.3f}, {quat.w:.3f}]")

				return pos, quat, to_ns(transform.header.stamp), True

			except Exception as e2:
				self.get_logger().warning(f"tf2 transform lookup failed (both exact and latest): {e2}")
				return None, None, timestamp, False
		except Exception as e:
			self.get_logger().warning(f"tf2 transform lookup failed: {e}")
			return None, None, timestamp, False

	def _log_all_frames(self, stamp):
		"""Log all relevant frames for CODa debugging."""
		# Set time once at the start for all frame logging
		time_ns = to_ns(stamp)
		rr.set_time(timeline='main', timestamp=1e-9 * time_ns)

		# Define frames to visualize
		frames_to_log = [
			("os1", "world/frames/os1"),
			("cam0_optical_frame", "world/frames/cam0_optical"),
			("cam1_optical_frame", "world/frames/cam1_optical"),
			("base_link", "world/frames/base_link"),
			("map", "world/frames/map"),
		]

		for frame_id, entity_path in frames_to_log:
			try:
				# Get transform from world to this frame
				pos, quat, time_ns, success = self._get_camera_transform_tf2(stamp)

				# rr.set_time(timeline='main', timestamp=1e-9 * time_ns)

				# Log frame to rerun
				rr.log(
					entity_path,
					rr.Transform3D(
						translation=[pos.x, pos.y, pos.z],
						quaternion=[quat.x, quat.y, quat.z, quat.w],
						relation=rr.TransformRelation.ParentFromChild
					),
				)
				
				# Add a small coordinate axis visualization
				rr.log(
					f"{entity_path}/axes",
					rr.Arrows3D(
						origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
						vectors=[[0.4, 0, 0], [0, 0.4, 0], [0, 0, 0.4]],  # XYZ axes
						colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],  # RGB for XYZ
					)
				)
				
			except Exception as e:
				# Frame not available, skip silently
				pass

	def _log_camera(self, msg: CameraInfo, entity_path: str):
		"""Logs camera intrinsics to Rerun."""
		time_ns = to_ns(msg.header.stamp)
		rr.set_time(timeline='main', timestamp=1e-9 * time_ns)
		self.current_time_ns = time_ns

		intrinsics = np.array(msg.k).reshape(3, 3)
		rr.log(
			entity_path,
			rr.Pinhole(
				image_from_camera=intrinsics,
				width=msg.width,
				height=msg.height,
			),
			static=False # probably can be static, but keeping for now
		)
		return intrinsics, msg.width, msg.height

	def _depth_info_cb(self, msg: CameraInfo):
		"""Callback for depth camera info."""
		if not self.get_parameter("log_depth").value:
			return
		self.depth_intrinsics, _, _ = self._log_camera(msg, "world/camera/depth/image")

	def _rgb_info_cb(self, msg: CameraInfo):
		"""Callback for RGB camera info."""
		self.rgb_intrinsics, _, _ = self._log_camera(msg, "world/camera/rgb/image")

		time_ns = to_ns(msg.header.stamp)
		rr.set_time(timeline='main', timestamp=1e-9 * time_ns)

		# self._log_transform(msg.header.stamp)  # Moved to _rgb_cb for 30Hz updates instead of camera_info rate

	def _semantic_info_cb(self, msg: CameraInfo):
		"""Callback for RGB camera info."""
		self.semantic_intrinsics, _, _ = self._log_camera(msg, "world/camera/semantic/image")

	def _rgb_cb(self, msg: Image):
		"""Callback for RGB image messages."""
		if self.rgb_intrinsics is None:
			self.get_logger().warning("Received RGB image before intrinsics.", throttle_duration_sec=5)
			return

		time_ns = to_ns(msg.header.stamp)
		rr.set_time(timeline='main', timestamp=1e-9 * time_ns)

		self._log_transform(msg.header.stamp)
		# Track current frame for temporal calculations
		self.current_frame_id += 1

		try:
			cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
			rr.log("world/camera/rgb/image", rr.Image(cv_image))
		except Exception as e:
			self.get_logger().error(f"Failed to convert RGB image: {e}")

	def _semantic_cb(self, msg: Image):
		"""Callback for RGB image messages."""
		if self.rgb_intrinsics is None:
			self.get_logger().warning("Received Semantic image before intrinsics.", throttle_duration_sec=5)
			return

		time_ns = to_ns(msg.header.stamp)
		rr.set_time(timeline='main', timestamp=1e-9 * time_ns)

		try:
			cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
			rr.log("world/camera/semantic/image", rr.Image(cv_image))
		except Exception as e:
			self.get_logger().error(f"Failed to convert Semantic image")

	def _depth_cb(self, msg: Image):
		"""Callback for depth image messages."""
		if self.depth_intrinsics is None:
			self.get_logger().warning("Received depth image before intrinsics.", throttle_duration_sec=5)
			return

		time_ns = to_ns(msg.header.stamp)
		rr.set_time(timeline='main', timestamp=1e-9 * time_ns)

		try:
			# 16UC1 or 32FC1 encoding
			cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
			depth_scale = self.get_parameter("depth_scale").value
			rr.log("world/camera/depth/image", rr.DepthImage(cv_image, meter=depth_scale))
		except Exception as e:
			self.get_logger().error(f"Failed to convert depth image")

	def _dsg_cb(self, msg: DsgUpdate):
		"""Callback for Dynamic Scene Graph updates - now just stores for periodic processing."""
		with self.dsg_update_lock:
			self.pending_dsg_update = msg
			self.get_logger().debug(f"DSG CB: Stored update for periodic processing")

	def _process_pending_dsg(self):
		"""Process pending DSG update at lower rate."""
		with self.dsg_update_lock:
			if self.pending_dsg_update is None:
				self.get_logger().info("No DSG update received yet.")
				return
			msg = self.pending_dsg_update
			self.pending_dsg_update = None

		time_ns = to_ns(msg.header.stamp)
		rr.set_time(timeline='main', timestamp=1e-9 * time_ns)
		self.get_logger().debug(f"Processing DSG update at t={time_ns}")

		try:
			# update internal DSG state
			self.get_logger().debug(f"DSG: Updating graph from binary (size: {len(msg.layer_contents)} bytes)")

			layer_contents_bytes = bytes(msg.layer_contents)

			# Handle full vs incremental updates correctly
			if msg.full_update:
				# Full update: create new graph from binary
				self.get_logger().debug("DSG: Performing full update with from_binary()")
				self.dsg = DynamicSceneGraph.from_binary(layer_contents_bytes)
				update_successful = True
			else:
				# Incremental update: update existing graph
				self.get_logger().debug("DSG: Performing incremental update with update_from_binary()")
				update_successful = self.dsg.update_from_binary(layer_contents_bytes)

			if not update_successful:
				self.get_logger().error("DSG: update_from_binary failed!")
				return

			self.get_logger().debug(f"DSG: Graph update successful (full_update={msg.full_update}).")

			if msg.full_update:
				# clear previous DSG visualization if it's a full reset
				self.get_logger().debug("DSG: Full update detected. Clearing and re-logging.")
				rr.log("world/dsg/nodes", rr.Clear(recursive=True))
				rr.log("world/dsg/edges", rr.Clear(recursive=True))
				rr.log("world/dsg/mesh", rr.Clear(recursive=True))
				self.get_logger().info("DSG Full Update received, clearing previous visualization.")
				# re-log entire graph after clearing
				self._log_full_dsg(self.dsg)
				# Log mesh from DSG if available
				self._log_dsg_mesh()
			else:
				# handle deletions
				if msg.deleted_nodes:
					self.get_logger().debug(f"DSG: Handling {len(msg.deleted_nodes)} deleted nodes.")
				for node_id in msg.deleted_nodes:
					symbol = NodeSymbol(node_id)
					self.get_logger().debug(f"  Clearing node {symbol}")

					for layer_id, layer_name in LAYER_NAMES.items():
						rr.log(f"world/dsg/nodes/{layer_name}/{symbol.category}{symbol.categoryId}", rr.Clear(recursive=False))

				if msg.deleted_edges:
					self.get_logger().debug(f"DSG: {len(msg.deleted_edges)} deleted edges (handling not implemented).")

				# re-log the whole graph for partial updates
				self.get_logger().debug("DSG: Partial update successful. Re-logging full graph state.")
				self._log_full_dsg(self.dsg)
				# Update mesh visualization
				self._log_dsg_mesh()

			# Process temporal history if enabled
			if self.enable_temporal_viz and self.enable_retroactive:
				self._process_temporal_history_for_update(msg)

		except Exception as e:
			self.get_logger().error(f"Failed to process DSG update: {str(e)[:500]}.")

	def _semantic_updates_cb(self, msg: String):
		"""Callback for semantic updates from daaam."""
		try:
			import json
			update = json.loads(msg.data)

			# Store semantic labels
			semantic_labels = update.get('semantic_labels', {})
			for sem_id_str, label in semantic_labels.items():
				sem_id = int(sem_id_str)
				self.semantic_labels[sem_id] = label

			self.get_logger().info(f"Received semantic update with {len(semantic_labels)} labels")

			# Store temporal observations if available
			temporal_observations = update.get('temporal_observations', {})
			if temporal_observations:
				self.get_logger().debug(f"Received {len(temporal_observations)} temporal observations")

			# Update visualization if we have the DSG
			if self.dsg and self.dsg.num_nodes() > 0:
				self._update_node_labels()

		except Exception as e:
			self.get_logger().error(f"Failed to process semantic update: {e}")

	def _update_node_labels(self):
		"""Update node labels based on semantic updates."""
		try:
			# Get objects layer
			if not self.dsg.has_layer(DsgLayers.OBJECTS):
				self.get_logger().debug("No objects layer in DSG yet")
				return

			objects_layer = self.dsg.get_layer(DsgLayers.OBJECTS)
			updated_count = 0
			nodes_to_update = []

			for node in objects_layer.nodes:
				if node.attributes and hasattr(node.attributes, 'semantic_label'):
					sem_id = node.attributes.semantic_label
					if sem_id in self.semantic_labels:
						# Store node for batch update
						nodes_to_update.append(node)

						# Update the label in metadata if possible
						if hasattr(node.attributes, 'metadata'):
							try:
								metadata = node.attributes.metadata.get()
								if metadata is None:
									metadata = {}
								metadata['description'] = self.semantic_labels[sem_id]
								# Note: We can't set metadata back, but we use it for display
								updated_count += 1
							except:
								pass

			if updated_count > 0:
				self.get_logger().info(f"Updated {updated_count} node labels from semantic updates")
				# Re-visualize the updated nodes with their new labels
				if nodes_to_update:
					self._log_updated_dsg_nodes(self.dsg, nodes_to_update)

		except Exception as e:
			self.get_logger().error(f"Failed to update node labels: {e}")

	def _background_objects_cb(self, msg: String):
		"""Callback for incremental background objects updates from daaam."""
		try:
			import json
			update = json.loads(msg.data)

			# Check update type (incremental vs full)
			update_type = update.get('update_type', 'full')
			objects = update.get('objects', [])

			self.get_logger().info(f"Received {update_type} background objects update with {len(objects)} objects")

			# Only process objects not in Hydra (filtered objects)
			filtered_objects = [obj for obj in objects if not obj.get('in_hydra', False)]
			in_hydra_objects = [obj for obj in objects if obj.get('in_hydra', False)]

			if not filtered_objects:
				self.get_logger().info(f"No filtered background objects to visualize ({len(in_hydra_objects)} in Hydra)")
				return

			# For incremental updates, add to existing visualization
			# No need to clear as we're only adding new objects
			self.get_logger().info(f"Processing {len(filtered_objects)} filtered objects (not in Hydra) from {update_type} update")

			positions = []
			colors = []
			labels = []
			radii = []

			for obj in filtered_objects:
				# Get position and other data
				pos_world = obj.get('position_world')
				if pos_world:
					positions.append(pos_world)

					# Get semantic label
					semantic_id = obj.get('semantic_id', -1)
					label_text = obj.get('label', f'Unknown_{semantic_id}')

					# Determine color based on semantic ID
					if semantic_id in self.color_map:
						color = self.color_map[semantic_id]
					else:
						# Use a distinct color for background objects not in Hydra
						color = [255, 165, 0]  # Orange for visibility
					colors.append(color)

					# Create label with reason why it's filtered
					filtered_reason = obj.get('filtered_reason', 'unknown')
					label_with_reason = f"{label_text}\n[{filtered_reason}]"
					wrapped_label = "\n".join(textwrap.wrap(label_with_reason, width=30))
					labels.append(wrapped_label)

					# Size based on observation count
					obs_count = obj.get('observation_count', 1)
					radius = 0.1 * (1.0 + min(0.5, obs_count / 20.0))  # Larger radius for visibility
					radii.append(radius)

			if positions:
				# Log background objects that are not in Hydra
				rr.log(
					"world/background_objects/filtered",
					rr.Points3D(
						np.array(positions),
						colors=np.array(colors),
						labels=labels,
						radii=np.array(radii),
						show_labels=True
					)
				)

				self.get_logger().info(f"Visualized {len(positions)} background objects not in Hydra")

				# Log statistics
				total_objects = len(objects)
				in_hydra_count = sum(1 for obj in objects if obj.get('in_hydra', False))
				self.get_logger().debug(f"Background objects - Total: {total_objects}, In Hydra: {in_hydra_count}, Filtered: {len(filtered_objects)}")

		except Exception as e:
			self.get_logger().error(f"Failed to process background objects update: {e}")

	def _log_transform(self, timestamp: rclpy.time.Time) -> bool:
		"""Logs the camera transform to Rerun."""
		if self.use_tf2:
			pos, quat, time_ns, success = self._get_camera_transform_tf2(timestamp)
			if not success:
				self.get_logger().debug("tf2 failed, skipping camera pose logging (odometry frame is incorrect)")
				return True
		else:
			return False

		self.get_logger().debug(f"Odom CB: Logging transform to world/camera at t={timestamp}")
		self.get_logger().debug(f"  Translation: [{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}]")
		self.get_logger().debug(f"  Rotation (xyzw): [{quat.x:.3f}, {quat.y:.3f}, {quat.z:.3f}, {quat.w:.3f}]")

		# Set time atomically before logging to prevent race conditions with other callbacks
		# time_ns = to_ns(timestamp)
		rr.set_time(timeline='main', timestamp=1e-9 * time_ns)

		rr.log(
			"world/camera",
			rr.Transform3D(
				translation=[pos.x, pos.y, pos.z],
				quaternion=[quat.x, quat.y, quat.z, quat.w],
				relation=rr.TransformRelation.ParentFromChild
			),
		)

		# Log additional frames for CODa debugging
		self._log_all_frames(timestamp)


		return True


	def _log_full_dsg(self, graph: DynamicSceneGraph):
		"""Logs the entire DSG to Rerun."""
		self.get_logger().debug("_log_full_dsg: Logging full graph.")
		all_nodes = list(graph.nodes) # Get all nodes
		self._log_updated_dsg_nodes(graph, all_nodes)
		all_edges = list(graph.edges)
		self._log_dsg_edges(graph, all_edges)


	def _log_updated_dsg_nodes(self, graph: DynamicSceneGraph, nodes_to_log: list[SceneGraphNode]):
		"""Logs specified DSG nodes and their attributes."""
		self.get_logger().debug(f"_log_updated_dsg_nodes: Logging {len(nodes_to_log)} nodes.")
		nodes_by_layer = {}
		for node in nodes_to_log:
			layer_id = node.layer
			if layer_id not in nodes_by_layer:
				nodes_by_layer[layer_id] = []
			nodes_by_layer[layer_id].append(node)

		for layer_id, nodes in nodes_by_layer.items():
			layer_name = LAYER_NAMES.get(layer_id, f"layer_{layer_id}")
			positions = []
			colors = []
			labels = []
			node_ids = []
			bboxes = []
			bbox_colors = []

			for node in nodes:
				node_attrs = node.attributes
				if node_attrs is None:
					continue
				positions.append(node_attrs.position)
				node_ids.append(node.id)
				
				# Get temporal properties if enabled
				temporal_props = None
				if self.enable_temporal_viz:
					# Use actual current time, not message time
					current_time_ns = self.current_time_ns
					temporal_props = self._compute_temporal_visual_properties(node, current_time_ns)

				# semantic attributes
				base_color = None
				if hasattr(node_attrs, 'semantic_label') and node_attrs.semantic_label in self.color_map:
					base_color = np.array(self.color_map[node_attrs.semantic_label], dtype=np.uint8)
				elif layer_id in LAYER_COLORS:
					base_color = np.array(LAYER_COLORS[layer_id], dtype=np.uint8)
				elif hasattr(node_attrs, 'color'):
					base_color = np.array(node_attrs.color, dtype=np.uint8)
				else:
					base_color = np.array([200, 200, 200], dtype=np.uint8)
				
				# Apply temporal color saturation if enabled
				if temporal_props and 'color_saturation' in temporal_props:
					saturation = temporal_props['color_saturation']
					# Desaturate color by blending with gray
					gray = np.array([128, 128, 128], dtype=np.uint8)
					base_color = (base_color * saturation + gray * (1 - saturation)).astype(np.uint8)
				
				colors.append(base_color)

				# Build label text with semantic information
				label_text = str(node.id)

				# First check if we have a semantic label stored from semantic_updates
				if hasattr(node_attrs, 'semantic_label'):
					sem_id = node_attrs.semantic_label
					if sem_id in self.semantic_labels:
						wrapped_name = "\n".join(textwrap.wrap(self.semantic_labels[sem_id], width=30))
						label_text = f"{node.id}: {wrapped_name}"

				# Otherwise try to get from metadata
				elif hasattr(node_attrs, 'metadata'):
					try:
						metadata = node_attrs.metadata.get()
						if 'description' in metadata:
							wrapped_name = "\n".join(textwrap.wrap(str(metadata['description']), width=30))
							label_text = f"{node.id}: {wrapped_name}"
					except:
						pass

				# Add temporal information if enabled
				if self.enable_temporal_viz and self.show_temporal_labels:
					# Use actual current time, not message time
					current_time_ns = self.current_time_ns
					temporal_props = self._compute_temporal_visual_properties(node, current_time_ns)
					if temporal_props['temporal_label']:
						label_text += temporal_props['temporal_label']

				labels.append(label_text) 

				if (
					hasattr(node_attrs, 'bounding_box') 
					and node_attrs.bounding_box is not None 
					and node_attrs.bounding_box.type != BoundingBoxType.INVALID
				):
					bbox = node_attrs.bounding_box
					bboxes.append(bbox)
					bbox_colors.append(colors[-1])

				if (
					self.get_parameter("log_object_meshes").value
					and hasattr(node_attrs, 'mesh') 
					and node_attrs.mesh().num_vertices() > 4
				):
					self._log_object_mesh(
						f"world/dsg/nodes/{layer_name}_meshes/{node.id}",
						node_attrs
					)

			if bboxes:
				self._log_bounding_boxes(
					f"world/dsg/nodes/{layer_name}_bboxes",
					bboxes,
					bbox_colors
				)

			if positions:
				entity_path = f"world/dsg/nodes/{layer_name}"
				self.get_logger().debug(f"  Logging {len(positions)} points to {entity_path}")
				if len(positions) > 0:
					self.get_logger().debug(f"First point pos: {positions[0]}, color: {colors[0] if colors else 'N/A'}, label: {labels[0] if labels else 'N/A'}.")
				
				# Apply temporal point scaling if enabled
				radii = None
				if self.enable_temporal_viz:
					radii = []
					current_time_ns = self.current_time_ns
					for i, node in enumerate(nodes):
						temporal_props = self._compute_temporal_visual_properties(node, current_time_ns)
						scale = temporal_props.get('point_scale', 1.0)
						radii.append(0.05 * scale)  # Base radius of 0.05
				
				rr.log(
					entity_path,
					rr.Points3D(
						np.array(positions),
						colors=np.array(colors) if colors else None,
						labels=labels if labels else None,
						radii=np.array(radii) if radii else None,
						show_labels=True,
					)
				)
				
				# Add temporal ring indicators if enabled
				if self.enable_temporal_viz:
					ring_positions = []
					ring_colors = []
					ring_radii = []

					current_time_ns = self.current_time_ns
					for i, node in enumerate(nodes):
						temporal_props = self._compute_temporal_visual_properties(node, current_time_ns)
						ring_color = temporal_props.get('ring_color')
						
						if ring_color:
							ring_positions.append(positions[i])
							ring_colors.append(ring_color)
							ring_radii.append(0.08 * temporal_props.get('point_scale', 1.0))
					
					if ring_positions:
						rr.log(
							f"{entity_path}_rings",
							rr.Points3D(
								np.array(ring_positions),
								colors=np.array(ring_colors),
								radii=np.array(ring_radii)
							)
						)
		# re-log all edges (TODO: optimize this if needed)
		self._log_dsg_edges(graph, graph.edges)


	def _log_dsg_edges(self, graph: DynamicSceneGraph, edges_to_log):
		"""Logs DSG edges to Rerun."""

		layer_edges = {}
		interlayer_points = []
		interlayer_indices_pairs = []

		for edge in edges_to_log:
			source_node = graph.find_node(edge.source)
			target_node = graph.find_node(edge.target)
			if not source_node or not target_node:
				continue # skip edges if nodes don't exist (e.g., due to deletion race)

			source_pos = source_node.attributes.position
			target_pos = target_node.attributes.position

			if source_node.layer == target_node.layer:
				layer_id = source_node.layer
				if layer_id not in layer_edges:
					layer_edges[layer_id] = {"points": [], "indices": []}

				point_idx_start = len(layer_edges[layer_id]["points"])
				layer_edges[layer_id]["points"].extend([source_pos, target_pos])
				layer_edges[layer_id]["indices"].append([point_idx_start, point_idx_start + 1])
			else:
				# interlayer edge
				point_idx_start = len(interlayer_points)
				interlayer_points.extend([source_pos, target_pos])
				interlayer_indices_pairs.append([point_idx_start, point_idx_start + 1])


		# log intra-layer edges
		for layer_id, edge_data in layer_edges.items():
			if not edge_data["points"]: 
				continue
			layer_name = LAYER_NAMES.get(layer_id, f"layer_{layer_id}")
			entity_path = f"world/dsg/edges/{layer_name}"
			self.get_logger().debug(f"  Logging {len(edge_data['indices'])} intra-layer edges for layer '{layer_name}' to {entity_path}")
			rr.log(
				entity_path,
				rr.LineStrips3D(
					[(edge_data["points"][i], edge_data["points"][j]) for i,j in edge_data["indices"]],
				)
			)

		# Log inter-layer edges
		if interlayer_points:
			self.get_logger().debug(f"  Logging {len(interlayer_indices_pairs)} inter-layer edges to world/dsg/edges/interlayer")
			rr.log(
				"world/dsg/edges/interlayer",
				rr.LineStrips3D(
					[(interlayer_points[i], interlayer_points[j]) for i,j in interlayer_indices_pairs],
				)
			)


	def _log_dsg_mesh(self):
		"""Log the mesh from DSG if available (similar to static_visualizer.py)."""
		try:
			if not self.dsg.has_mesh():
				self.get_logger().debug("DSG has no mesh")
				return

			mesh = self.dsg.mesh
			vertices = mesh.get_vertices()
			faces = mesh.get_faces()

			if vertices.size == 0 or faces.size == 0:
				self.get_logger().debug("DSG mesh is empty")
				return

			# Extract vertex positions (first 3 rows) and colors (last 3 rows)
			vertex_positions = vertices[:3, :].T.astype(np.float32)

			# Check if we have color information
			vertex_colors = None
			if vertices.shape[0] >= 6:
				# Colors are in rows 3-5, scale from [0,1] to [0,255]
				vertex_colors = (vertices[3:6, :].T * 255).astype(np.uint8)

			# Transpose faces to get Nx3 array
			triangle_indices = faces.T.astype(np.uint32)

			# Validate triangle indices
			max_vertex_idx = len(vertex_positions) - 1
			valid_face_mask = np.all(triangle_indices <= max_vertex_idx, axis=1)
			valid_face_mask = valid_face_mask & np.any(triangle_indices != 0, axis=1)
			# Remove degenerate triangles
			valid_face_mask = valid_face_mask & ~((triangle_indices[:, 0] == triangle_indices[:, 1]) &
												   (triangle_indices[:, 1] == triangle_indices[:, 2]))

			num_original_triangles = len(triangle_indices)
			triangle_indices = triangle_indices[valid_face_mask]
			if len(triangle_indices) != num_original_triangles:
				self.get_logger().debug(f"Filtered {num_original_triangles - len(triangle_indices)} invalid triangles")

			self.get_logger().debug(f"Logging DSG mesh: {len(vertex_positions)} vertices, {len(triangle_indices)} faces")

			rr.log(
				"world/dsg/mesh",
				rr.Mesh3D(
					vertex_positions=vertex_positions,
					vertex_colors=vertex_colors,
					triangle_indices=triangle_indices,
				)
			)

		except Exception as e:
			self.get_logger().error(f"Failed to log DSG mesh: {e}\n{traceback.format_exc()}")

	def _log_bounding_boxes(self, entity_path: str, bboxes: list, colors: list):
		"""Logs bounding boxes to Rerun."""
		box_entities = []
		centers_list = []
		colors_list = []
		half_sizes_list = []
		rotations_list = []

		for bbox, color in zip(bboxes, colors):
			centers_list.append(bbox.world_P_center)
			half_sizes_list.append((bbox.dimensions * 0.5))
			quat = Rotation.from_matrix(
				bbox.world_R_center
				).as_quat(scalar_first=False) # xyzw
			rotations_list.append(quat)
			colors_list.append(color)

		if centers_list:
			rr.log(
				entity_path,
				rr.Boxes3D(
					centers=np.array(centers_list),
					half_sizes=np.array(half_sizes_list),
					colors=np.array(colors_list),
					quaternions=np.array(rotations_list),
				)
			)
			
	def _log_object_mesh(self, entity_path, node_attr):
		"""Logs individual object meshes from the DSG (if available)."""
		pos = node_attr.position
		mesh = node_attr.mesh()
		vertices = mesh.get_vertices() # 6xN (xyz + rgb)
		faces = mesh.get_faces() # 3xM

		rr.log(
			entity_path,
			rr.Mesh3D(
				vertex_positions=pos + vertices[:3, :].T,
				vertex_colors=vertices[3:, :].T,
				triangle_indices=faces.T,
			)
		)
	
	def _process_temporal_history_for_update(self, msg):
		"""Process temporal history from DSG update and inject retroactive visualizations."""
		try:
			# Get all nodes from the graph
			all_nodes = list(self.dsg.nodes)
			
			for node in all_nodes:
				if not hasattr(node.attributes, 'metadata'):
					continue
					
				try:
					metadata = node.attributes.metadata.get()
					temporal_history = metadata.get('temporal_history', {})
					
					if temporal_history and temporal_history.get('frame_ids'):
						self._process_temporal_history_retroactively(
							node, 
							temporal_history,
							msg.header.stamp
						)
				except Exception as e:
					self.get_logger().debug(f"Failed to process temporal history for node {node.id}: {e}")
					
		except Exception as e:
			self.get_logger().warning(f"Failed to process temporal history: {e}")
	
	def _process_temporal_history_retroactively(self, node, temporal_history, msg_time):
		"""Retroactively inject temporal visualizations at historical timestamps."""
		frame_ids = temporal_history.get('frame_ids', [])
		timestamps = temporal_history.get('timestamps', [])
		
		if not frame_ids or not timestamps:
			return
			
		# Current receipt time
		receipt_time_ns = to_ns(msg_time)
		semantic_id = node.id.value
		position = node.attributes.position
		
		# Log observation pulses at historical times
		if self.show_observation_pulses:
			for i, (frame_id, timestamp) in enumerate(zip(frame_ids, timestamps)):
				# Set time to historical moment
				historical_time_ns = int(timestamp * 1e9)
				rr.set_time(timeline='main', timestamp=1e-9 * historical_time_ns)
				
				# Log observation pulse (smaller and will be overwritten by fade)
				rr.log(
					f"world/temporal/observations/{semantic_id}",
					rr.Points3D(
						positions=[position],
						radii=[0.05],  # Much smaller radius
						colors=[[0, 255, 0]]  # Green for observation
					)
				)
				
				# Add fading effect between observations
				if i < len(frame_ids) - 1:
					next_timestamp = timestamps[i + 1]
					gap_duration = next_timestamp - timestamp
					fade_steps = min(5, int(gap_duration * 10))  # Max 5 fade steps
					
					for fade_step in range(1, fade_steps + 1):
						fade_ratio = fade_step / (fade_steps + 1)
						fade_time_ns = historical_time_ns + int(fade_ratio * gap_duration * 1e9)
						rr.set_time(timeline='main', timestamp=1e-9 * fade_time_ns)
						
						if fade_step <= fade_steps:
							opacity = int(255 * (1.0 - fade_ratio))
							radius = 0.05 * (1.0 + fade_ratio * 0.3)  # Smaller radius, less expansion
							
							rr.log(
								f"world/temporal/observations/{semantic_id}",
								rr.Points3D(
									positions=[position],
									radii=[radius],
									colors=[[0, 255, 0, opacity]]
								)
							)
						else:
							# Clear the observation pulse after fade completes
							rr.log(f"world/temporal/observations/{semantic_id}", rr.Clear(recursive=False))
		
		# Return to current time
		rr.set_time(timeline='main', timestamp=1e-9 * receipt_time_ns)
		
		# Log knowledge update indicator at current time
		rr.log(
			f"world/temporal/knowledge_update/{semantic_id}",
			rr.Points3D(
				positions=[position],
				radii=[0.08],
				colors=[[255, 128, 0]]  # Orange for knowledge update
			)
		)
	
	def _compute_temporal_visual_properties(self, node, current_time_ns=None):
		"""Compute visual properties based on temporal history."""
		default_props = {
			'opacity': 1.0,
			'color_saturation': 1.0,
			'point_scale': 1.0,
			'ring_color': None,
			'temporal_label': ''
		}
		
		if not hasattr(node.attributes, 'metadata'):
			return default_props
			
		try:
			metadata = node.attributes.metadata.get()
			temporal_history = metadata.get('temporal_history', {})
			
			if not temporal_history:
				return default_props
				
			frame_ids = temporal_history.get('frame_ids', [])
			timestamps = temporal_history.get('timestamps', [])
			observation_count = temporal_history.get('observation_count', 0)
			last_observed = temporal_history.get('last_observed')
			
			if not frame_ids or last_observed is None:
				return default_props
			
			# Calculate time since last observation
			# current_time_ns is the timestamp when we received the DSG update
			if current_time_ns is not None:
				current_time_s = current_time_ns / 1e9
				time_since_observation = current_time_s - last_observed
			else:
				# Fallback: estimate based on frame IDs if available
				if frame_ids and self.current_frame_id > 0:
					last_frame = frame_ids[-1]
					frames_since = max(0, self.current_frame_id - last_frame)
					time_since_observation = frames_since / 30.0  # Assume 30 fps
				else:
					# No time information available
					time_since_observation = 0.0
			
			# Ensure non-negative (could be negative if DSG arrives before the observation timestamp)
			time_since_observation = max(0.0, time_since_observation)
			
			# Determine visual properties based on temporal state
			if time_since_observation < 0.1:  # Currently observed
				opacity = self.opacity_max
				color_saturation = 1.0
				ring_color = [0, 255, 0]  # Green
			elif time_since_observation < 1.0:  # Recently observed
				opacity = self.opacity_max - (time_since_observation * 0.2)
				color_saturation = 0.9
				ring_color = [128, 255, 0]  # Yellow-green
			elif time_since_observation < self.track_buffer / 30.0:  # Within track buffer
				opacity = self.opacity_min + (self.opacity_max - self.opacity_min) * (1.0 - time_since_observation / (self.track_buffer / 30.0))
				color_saturation = 0.7
				ring_color = [255, 255, 0]  # Yellow
			else:  # Lost track
				opacity = self.opacity_min
				color_saturation = 0.5
				ring_color = [255, 0, 0]  # Red
			
			# Scale point based on observation count
			point_scale = 1.0 + min(1.0, observation_count / 100.0)
			
			# Create temporal label
			temporal_label = ""
			if self.show_temporal_labels:
				temporal_label = f"\n👁 Seen: {observation_count} times\n⏱ Last: {time_since_observation:.1f}s ago"
			
			return {
				'opacity': opacity,
				'color_saturation': color_saturation,
				'point_scale': point_scale,
				'ring_color': ring_color,
				'temporal_label': temporal_label
			}
			
		except Exception as e:
			self.get_logger().debug(f"Failed to compute temporal properties: {e}")
			return default_props

	def make_rerun_blueprint(self):
		blueprint = rrb.Horizontal(
			rrb.Spatial3DView(
				origin="world",
				name="3D Scene",
				background=[255, 255, 255],
				line_grid=None,
			),
			rrb.Vertical(
				rrb.Spatial2DView(
					origin="world/camera/rgb/image",
					name="RGB Camera",
				),
				rrb.Spatial2DView(
					origin="world/camera/depth/image",
					name="Depth Camera",
				),
				rrb.Spatial2DView(
					origin="world/camera/semantic/image",
					name="Semantic Camera",
				),
			)
		)

		return blueprint


def main(args=None):
	"""Node entry point."""
	rclpy.init(args=args)
	try:
		node = RerunVisualizerNode()
		rclpy.spin(node)
	except ImportError as e:
		print(f"Initialization failed due to missing import: {e}")
	except KeyboardInterrupt:
		pass
	except Exception as e:
		print(f"Unhandled exception in RerunVisualizerNode: {e}")
	finally:
		if 'node' in locals() and node:
			node.destroy_node()
		rclpy.shutdown()


if __name__ == "__main__":
	main()
