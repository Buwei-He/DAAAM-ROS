"""
ROS2 bag writer output handler.

Writes frame data to a ROS2 bag file.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from scipy.spatial.transform import Rotation as ScipyR

from sensor_msgs.msg import Image, CameraInfo  # type: ignore
from geometry_msgs.msg import TransformStamped  # type: ignore
from std_msgs.msg import Header, ColorRGBA  # type: ignore
from visualization_msgs.msg import Marker, MarkerArray  # type: ignore
from cv_bridge import CvBridge  # type: ignore
import rclpy  # type: ignore
from rclpy.serialization import serialize_message  # type: ignore
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata  # type: ignore
from rosgraph_msgs.msg import Clock  # type: ignore

from tf2_msgs.msg import TFMessage  # type: ignore

from daaam_ros.nodes.dataloader.output.interfaces import OutputHandler
from daaam_ros.nodes.dataloader.models import FrameData, CalibrationData, Transform, ImageData
from daaam_ros.nodes.dataloader.image_utils import (
	resize_rgb_image,
	resize_depth_image,
	resize_label_image,
	update_intrinsics_for_resize
)


class BagWriter(OutputHandler):
	"""Writes data to ROS2 bag file."""
	
	def __init__(self, config: Dict[str, Any], node):
		super().__init__(config)
		self.node = node
		self.bridge = CvBridge()
		
		# Bag configuration
		self.bag_path = Path(config.get('bag_path', '/tmp/dataloader_output.bag'))
		self.storage_id = config.get('storage_id', 'sqlite3')
		self.compression_mode = config.get('compression_mode', 'none')
		self.compression_format = config.get('compression_format', '')
		
		# Writer
		self.writer: Optional[SequentialWriter] = None
		self._is_ready = False
		
		# Track created topics
		self.created_topics = set()
		
		# Track if static transforms have been written
		self.static_transforms_written = False

		# Store reference to service for accessing full config
		self._service_ref = None
		
	def initialize(self) -> bool:
		"""Initialize bag writer."""
		try:
			# Create output directory if needed
			self.bag_path.parent.mkdir(parents=True, exist_ok=True)
			
			# Add timestamp to bag name if requested
			if self.config.get('add_timestamp', True):
				timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
				stem = self.bag_path.stem
				suffix = self.bag_path.suffix
				self.bag_path = self.bag_path.parent / f"{stem}_{timestamp}{suffix}"
			
			# Create writer
			self.writer = SequentialWriter()
			
			# Configure storage
			storage_options = StorageOptions(
				uri=str(self.bag_path),
				storage_id=self.storage_id
			)
			
			# Configure converter
			converter_options = ConverterOptions(
				input_serialization_format='cdr',
				output_serialization_format='cdr'
			)
			
			# Open bag for writing
			self.writer.open(storage_options, converter_options)
			
			self._is_ready = True
			self.node.get_logger().info(f"BagWriter initialized: {self.bag_path}")
			return True
			
		except Exception as e:
			self.node.get_logger().error(f"Failed to initialize BagWriter: {e}")
			import traceback
			traceback.print_exc()
			return False
	
	def publish_frame(self, frame_data: FrameData, calibration: CalibrationData) -> bool:
		"""Write frame data to bag file.
		
		Args:
			frame_data: Frame with all data
			calibration: Camera calibration (for the first camera)
			
		Returns:
			bool: True if successful
		"""
		try:
			# Dataset timestamp is required for simulated time
			if not frame_data.timestamp:
				raise ValueError("Frame timestamp is required for simulated time but not available")

			# Check if resizing is enabled
			updated_calibrations = {}
			if hasattr(self, '_service_ref') and self._service_ref and hasattr(self._service_ref, 'config'):
				config = self._service_ref.config
				if config.resize_images:
					# Resize all images and update calibrations
					frame_data, updated_calibrations = self._resize_frame_images(
						frame_data,
						calibration,
						config.target_width,
						config.target_height
					)
					# Log resizing on first frame
					if frame_data.frame_id == 0:
						self.node.get_logger().info(
							f"Resizing images from original to {config.target_width}x{config.target_height}"
						)

			# Convert Unix timestamp to ROS time (nanoseconds)
			timestamp_ns = int(frame_data.timestamp * 1e9)
			
			# IMPORTANT: Write transforms BEFORE clock to ensure they're available when time advances
			# Write static transforms once at the beginning
			if not self.static_transforms_written and frame_data.static_transforms:
				self._write_static_transforms(frame_data.static_transforms, timestamp_ns)
				self.static_transforms_written = True
			
			# Track if we wrote transforms for this frame
			wrote_transforms = False
			
			# Write dynamic transforms for this frame
			if frame_data.dynamic_transforms:
				self._write_dynamic_transforms(frame_data.dynamic_transforms, timestamp_ns)
				wrote_transforms = True
				# Debug logging for transform timing
				if frame_data.frame_id % 100 == 0:  # Log every 100th frame
					self.node.get_logger().debug(
						f"[TF DEBUG] Frame {frame_data.frame_id}: Wrote transforms at {timestamp_ns/1e9:.6f}s"
					)
			# Handle deprecated poses field for backward compatibility
			elif frame_data.poses:
				self._write_dynamic_transforms(frame_data.poses, timestamp_ns)
				wrote_transforms = True
			
			# Warn if no transforms were written
			if not wrote_transforms and not self.static_transforms_written:
				self.node.get_logger().warning(
					f"[TF WARNING] Frame {frame_data.frame_id}: No transforms written for timestamp {timestamp_ns/1e9:.6f}s"
				)
			
			# NOW write clock message to advance simulated time
			# This ensures transforms are available when Hydra processes at this timestamp
			self._write_clock(timestamp_ns)
			
			# Write data for each camera
			for camera_id, rgb_image in frame_data.rgb_images.items():
				# Create topics if not already created
				if camera_id not in self.created_topics:
					self._create_topics_for_camera(camera_id)
					self.created_topics.add(camera_id)
				
				# Create header
				header = Header()
				header.stamp.sec = timestamp_ns // 1_000_000_000
				header.stamp.nanosec = timestamp_ns % 1_000_000_000
				header.frame_id = f"{camera_id}_optical_frame"
				
				# Write RGB image
				rgb_topic = f"/{camera_id}/rgb_image"
				rgb_msg = self.bridge.cv2_to_imgmsg(rgb_image.data, encoding="bgr8")
				rgb_msg.header = header
				self.writer.write(
					rgb_topic,
					serialize_message(rgb_msg),
					timestamp_ns
				)
				
				# Write depth image if available
				if camera_id in frame_data.depth_images and frame_data.depth_images[camera_id]:
					depth_image = frame_data.depth_images[camera_id]
					depth_topic = f"/{camera_id}/depth_image"
					
					# Convert depth to mm (uint16) for visualization
					depth_mm = (depth_image.data * 1000).astype(np.uint16)
					depth_msg = self.bridge.cv2_to_imgmsg(depth_mm, encoding="16UC1")
					depth_msg.header = header
					self.writer.write(
						depth_topic,
						serialize_message(depth_msg),
						timestamp_ns
					)
				
				# Write camera info
				info_topic = f"/{camera_id}/camera_info"

				# Get calibration for this camera (use updated if resizing was applied)
				if camera_id in updated_calibrations:
					cam_calib = updated_calibrations[camera_id]
				else:
					from ..services import DataloaderService
					if hasattr(self, '_service_ref') and self._service_ref:
						cam_calib = self._service_ref.loader.get_calibration(camera_id)
					else:
						cam_calib = calibration if camera_id == calibration.camera_id else None

				if cam_calib:
					info_msg = self._create_camera_info_msg(cam_calib, header)
					self.writer.write(
						info_topic,
						serialize_message(info_msg),
						timestamp_ns
					)
				
				# Write label image if available
				if camera_id in frame_data.label_images and frame_data.label_images[camera_id]:
					label_image = frame_data.label_images[camera_id]
					label_topic = f"/{camera_id}/label_image"
					label_msg = self.bridge.cv2_to_imgmsg(label_image.data, encoding="16UC1")
					label_msg.header = header
					self.writer.write(
						label_topic,
						serialize_message(label_msg),
						timestamp_ns
					)

			# Write 3D bounding box markers if available
			if frame_data.bbox_annotations:
				self._write_bbox_markers(frame_data.bbox_annotations, timestamp_ns)

			return True
			
		except Exception as e:
			self.node.get_logger().error(f"Failed to write frame to bag: {e}")
			import traceback
			traceback.print_exc()
			return False
	
	def _create_topics_for_camera(self, camera_id: str) -> None:
		"""Create topic metadata for a specific camera."""
		# Track topic IDs (incremental)
		if not hasattr(self, '_next_topic_id'):
			self._next_topic_id = 0
		
		# RGB image topic
		rgb_topic = f"/{camera_id}/rgb_image"
		rgb_metadata = TopicMetadata(
			id=self._next_topic_id,
			name=rgb_topic,
			type="sensor_msgs/msg/Image",
			serialization_format="cdr"
		)
		self.writer.create_topic(rgb_metadata)
		self._next_topic_id += 1
		
		# Depth image topic
		depth_topic = f"/{camera_id}/depth_image"
		depth_metadata = TopicMetadata(
			id=self._next_topic_id,
			name=depth_topic,
			type="sensor_msgs/msg/Image",
			serialization_format="cdr"
		)
		self.writer.create_topic(depth_metadata)
		self._next_topic_id += 1
		
		# Camera info topic
		info_topic = f"/{camera_id}/camera_info"
		info_metadata = TopicMetadata(
			id=self._next_topic_id,
			name=info_topic,
			type="sensor_msgs/msg/CameraInfo",
			serialization_format="cdr"
		)
		self.writer.create_topic(info_metadata)
		self._next_topic_id += 1
		
		# Label image topic
		label_topic = f"/{camera_id}/label_image"
		label_metadata = TopicMetadata(
			id=self._next_topic_id,
			name=label_topic,
			type="sensor_msgs/msg/Image",
			serialization_format="cdr"
		)
		self.writer.create_topic(label_metadata)
		self._next_topic_id += 1
		
		self.node.get_logger().info(f"Created bag topics for camera {camera_id}")

	def _create_bbox_topic(self) -> None:
		"""Create topic metadata for bbox markers."""
		bbox_topic = "/ground_truth/bboxes"

		if bbox_topic not in self.created_topics:
			if not hasattr(self, '_next_topic_id'):
				self._next_topic_id = 0

			bbox_metadata = TopicMetadata(
				id=self._next_topic_id,
				name=bbox_topic,
				type="visualization_msgs/msg/MarkerArray",
				serialization_format="cdr"
			)
			self.writer.create_topic(bbox_metadata)
			self._next_topic_id += 1
			self.created_topics.add(bbox_topic)
			self.node.get_logger().info(f"Created bbox topic: {bbox_topic}")

	def _write_clock(self, timestamp_ns: int) -> None:
		"""Write clock message to bag.
		
		Args:
			timestamp_ns: Timestamp in nanoseconds
		"""
		# Create /clock topic if not exists
		clock_topic = "/clock"
		if clock_topic not in self.created_topics:
			if not hasattr(self, '_next_topic_id'):
				self._next_topic_id = 0
			
			clock_metadata = TopicMetadata(
				id=self._next_topic_id,
				name=clock_topic,
				type="rosgraph_msgs/msg/Clock",
				serialization_format="cdr"
			)
			self.writer.create_topic(clock_metadata)
			self._next_topic_id += 1
			self.created_topics.add(clock_topic)
		
		# Create and write clock message
		clock_msg = Clock()
		clock_msg.clock.sec = timestamp_ns // 1_000_000_000
		clock_msg.clock.nanosec = timestamp_ns % 1_000_000_000
		self.writer.write(
			clock_topic,
			serialize_message(clock_msg),
			timestamp_ns
		)
	
	def _write_static_transforms(self, static_transforms: Dict[str, Transform], timestamp_ns: int) -> None:
		"""Write static transforms to /tf_static topic (once at the beginning).
		
		Args:
			static_transforms: Dictionary of static transforms
			timestamp_ns: Timestamp in nanoseconds
		"""
		# Create /tf_static topic if not exists
		tf_static_topic = "/tf_static"
		if tf_static_topic not in self.created_topics:
			if not hasattr(self, '_next_topic_id'):
				self._next_topic_id = 0
			
			tf_static_metadata = TopicMetadata(
				id=self._next_topic_id,
				name=tf_static_topic,
				type="tf2_msgs/msg/TFMessage",
				serialization_format="cdr"
			)
			self.writer.create_topic(tf_static_metadata)
			self._next_topic_id += 1
			self.created_topics.add(tf_static_topic)
		
		# Create TF message for static transforms
		tf_msg = TFMessage()
		
		for transform_id, pose in static_transforms.items():
			if pose is None:
				continue
			
			transform = TransformStamped()
			transform.header.stamp.sec = timestamp_ns // 1_000_000_000
			transform.header.stamp.nanosec = timestamp_ns % 1_000_000_000
			transform.header.frame_id = pose.frame_id
			transform.child_frame_id = pose.child_frame_id
			
			# Set translation
			transform.transform.translation.x = pose.translation[0]
			transform.transform.translation.y = pose.translation[1]
			transform.transform.translation.z = pose.translation[2]
			
			# Set rotation
			if len(pose.rotation) == 4:
				# Quaternion already in ROS2 format: [x, y, z, w]
				transform.transform.rotation.x = pose.rotation[0]
				transform.transform.rotation.y = pose.rotation[1]
				transform.transform.rotation.z = pose.rotation[2]
				transform.transform.rotation.w = pose.rotation[3]
			else:
				# Convert rotation matrix to quaternion
				R = np.array(pose.rotation).reshape(3, 3)
				q = ScipyR.from_matrix(R).as_quat()  # [x, y, z, w]
				transform.transform.rotation.x = q[0]
				transform.transform.rotation.y = q[1]
				transform.transform.rotation.z = q[2]
				transform.transform.rotation.w = q[3]
			
			tf_msg.transforms.append(transform)
		
		if tf_msg.transforms:
			self.writer.write(
				tf_static_topic,
				serialize_message(tf_msg),
				timestamp_ns
			)
			self.node.get_logger().info(f"Wrote {len(tf_msg.transforms)} static transforms to /tf_static")
	
	def _write_dynamic_transforms(self, dynamic_transforms: Dict[str, Transform], timestamp_ns: int) -> None:
		"""Write dynamic transforms to /tf topic.
		
		Args:
			dynamic_transforms: Dictionary of dynamic transforms for this frame
			timestamp_ns: Timestamp in nanoseconds (frame timestamp)
		"""
		# Create /tf topic if not exists
		tf_topic = "/tf"
		if tf_topic not in self.created_topics:
			if not hasattr(self, '_next_topic_id'):
				self._next_topic_id = 0
			
			tf_metadata = TopicMetadata(
				id=self._next_topic_id,
				name=tf_topic,
				type="tf2_msgs/msg/TFMessage",
				serialization_format="cdr"
			)
			self.writer.create_topic(tf_metadata)
			self._next_topic_id += 1
			self.created_topics.add(tf_topic)
		
		# Create TF message for dynamic transforms
		tf_msg = TFMessage()
		
		for transform_id, pose in dynamic_transforms.items():
			if pose is None:
				continue
			
			transform = TransformStamped()
			transform.header.stamp.sec = timestamp_ns // 1_000_000_000
			transform.header.stamp.nanosec = timestamp_ns % 1_000_000_000
			transform.header.frame_id = pose.frame_id
			transform.child_frame_id = pose.child_frame_id
			
			# Set translation
			transform.transform.translation.x = pose.translation[0]
			transform.transform.translation.y = pose.translation[1]
			transform.transform.translation.z = pose.translation[2]
			
			# Set rotation
			if len(pose.rotation) == 4:
				# Quaternion already in ROS2 format: [x, y, z, w]
				transform.transform.rotation.x = pose.rotation[0]
				transform.transform.rotation.y = pose.rotation[1]
				transform.transform.rotation.z = pose.rotation[2]
				transform.transform.rotation.w = pose.rotation[3]
			else:
				# Convert rotation matrix to quaternion
				R = np.array(pose.rotation).reshape(3, 3)
				q = ScipyR.from_matrix(R).as_quat()  # [x, y, z, w]
				transform.transform.rotation.x = q[0]
				transform.transform.rotation.y = q[1]
				transform.transform.rotation.z = q[2]
				transform.transform.rotation.w = q[3]
			
			tf_msg.transforms.append(transform)

		if tf_msg.transforms:
			self.writer.write(
				tf_topic,
				serialize_message(tf_msg),
				timestamp_ns
			)

	def _write_bbox_markers(self, bbox_annotations: list, timestamp_ns: int) -> None:
		"""Write 3D bounding box markers to bag.

		Args:
			bbox_annotations: List of BoundingBox3D objects
			timestamp_ns: Timestamp in nanoseconds
		"""
		# Create topic if needed
		self._create_bbox_topic()

		# Create marker array
		marker_array = MarkerArray()

		for idx, bbox in enumerate(bbox_annotations):
			# Create CUBE marker for bbox
			marker = Marker()
			marker.header.stamp.sec = timestamp_ns // 1_000_000_000
			marker.header.stamp.nanosec = timestamp_ns % 1_000_000_000
			marker.header.frame_id = "os1"  # Bboxes are in os1 frame
			marker.ns = "ground_truth_bboxes"
			marker.id = idx
			marker.type = Marker.CUBE
			marker.action = Marker.ADD

			# Set position (center)
			marker.pose.position.x = bbox.center[0]
			marker.pose.position.y = bbox.center[1]
			marker.pose.position.z = bbox.center[2]

			# Set orientation (quaternion [x, y, z, w])
			marker.pose.orientation.x = bbox.orientation[0]
			marker.pose.orientation.y = bbox.orientation[1]
			marker.pose.orientation.z = bbox.orientation[2]
			marker.pose.orientation.w = bbox.orientation[3]

			# Set scale (size)
			marker.scale.x = bbox.size[0]  # length
			marker.scale.y = bbox.size[1]  # width
			marker.scale.z = bbox.size[2]  # height

			# Set color based on class (hash to consistent color)
			color = self._class_to_color(bbox.class_id)
			marker.color.r = color[0]
			marker.color.g = color[1]
			marker.color.b = color[2]
			marker.color.a = 0.5  # Semi-transparent

			marker_array.markers.append(marker)

			# Create TEXT marker for label
			text_marker = Marker()
			text_marker.header = marker.header
			text_marker.ns = "ground_truth_labels"
			text_marker.id = idx
			text_marker.type = Marker.TEXT_VIEW_FACING
			text_marker.action = Marker.ADD

			# Position text above bbox
			text_marker.pose.position.x = bbox.center[0]
			text_marker.pose.position.y = bbox.center[1]
			text_marker.pose.position.z = bbox.center[2] + bbox.size[2] / 2 + 0.3

			text_marker.text = f"{bbox.class_id}\n{bbox.instance_id}"
			text_marker.scale.z = 0.2  # Text height

			# White text with black outline
			text_marker.color.r = 1.0
			text_marker.color.g = 1.0
			text_marker.color.b = 1.0
			text_marker.color.a = 1.0

			marker_array.markers.append(text_marker)

		# Write to bag
		if marker_array.markers:
			self.writer.write(
				"/ground_truth/bboxes",
				serialize_message(marker_array),
				timestamp_ns
			)

	def _class_to_color(self, class_id: str) -> Tuple[float, float, float]:
		"""Hash class ID to consistent RGB color.

		Args:
			class_id: Class identifier string

		Returns:
			RGB color tuple (values in [0, 1])
		"""
		# Simple hash to RGB
		hash_val = hash(class_id)
		r = ((hash_val & 0xFF0000) >> 16) / 255.0
		g = ((hash_val & 0x00FF00) >> 8) / 255.0
		b = (hash_val & 0x0000FF) / 255.0
		return (r, g, b)

	def close(self) -> None:
		"""Close bag file."""
		if self.writer:
			self.writer.close()
			self._is_ready = False
			self.node.get_logger().info(f"Bag file written: {self.bag_path}")
	
	@property
	def is_ready(self) -> bool:
		"""Check if handler is ready to write."""
		return self._is_ready
	
	def set_service_ref(self, service) -> None:
		"""Set reference to dataloader service for accessing calibration."""
		self._service_ref = service

	def _resize_frame_images(
		self,
		frame_data: FrameData,
		calibration: CalibrationData,
		target_width: int,
		target_height: int
	) -> Tuple[FrameData, Dict[str, CalibrationData]]:
		"""Resize all images in frame data and update calibrations.

		Args:
			frame_data: Original frame data
			calibration: Original calibration (for first camera)
			target_width: Target width for resizing
			target_height: Target height for resizing

		Returns:
			Tuple of (resized_frame_data, updated_calibrations)
		"""
		# Create a copy of frame data to avoid modifying original
		resized_frame = FrameData(
			frame_id=frame_data.frame_id,
			timestamp=frame_data.timestamp,
			rgb_images={},
			depth_images={},
			label_images={},
			dynamic_transforms=frame_data.dynamic_transforms,
			static_transforms=frame_data.static_transforms,
			poses=frame_data.poses,
			sequence_name=frame_data.sequence_name,
			metadata=frame_data.metadata
		)

		updated_calibrations = {}

		# Process each camera's images
		for camera_id in frame_data.rgb_images.keys():
			# Get calibration for this camera
			if hasattr(self, '_service_ref') and self._service_ref:
				cam_calib = self._service_ref.loader.get_calibration(camera_id)
			else:
				cam_calib = calibration if camera_id == calibration.camera_id else None

			if not cam_calib:
				# If no calibration, skip resizing for this camera
				self.node.get_logger().warning(f"No calibration for camera {camera_id}, skipping resize")
				resized_frame.rgb_images[camera_id] = frame_data.rgb_images[camera_id]
				resized_frame.depth_images[camera_id] = frame_data.depth_images.get(camera_id)
				resized_frame.label_images[camera_id] = frame_data.label_images.get(camera_id)
				continue

			# Resize RGB image
			rgb_image = frame_data.rgb_images[camera_id]
			resized_rgb, crop_info = resize_rgb_image(
				rgb_image.data, target_width, target_height
			)
			resized_frame.rgb_images[camera_id] = ImageData(
				data=resized_rgb,
				encoding=rgb_image.encoding,
				timestamp=rgb_image.timestamp,
				frame_id=rgb_image.frame_id
			)

			# Resize depth image if available
			if camera_id in frame_data.depth_images and frame_data.depth_images[camera_id]:
				depth_image = frame_data.depth_images[camera_id]
				resized_depth, _ = resize_depth_image(
					depth_image.data, target_width, target_height
				)
				resized_frame.depth_images[camera_id] = ImageData(
					data=resized_depth,
					encoding=depth_image.encoding,
					timestamp=depth_image.timestamp,
					frame_id=depth_image.frame_id
				)
			else:
				resized_frame.depth_images[camera_id] = None

			# Resize label image if available
			if camera_id in frame_data.label_images and frame_data.label_images[camera_id]:
				label_image = frame_data.label_images[camera_id]
				resized_labels, _ = resize_label_image(
					label_image.data, target_width, target_height
				)
				resized_frame.label_images[camera_id] = ImageData(
					data=resized_labels,
					encoding=label_image.encoding,
					timestamp=label_image.timestamp,
					frame_id=label_image.frame_id
				)
			else:
				resized_frame.label_images[camera_id] = None

			# Update calibration for this camera
			original_size = (cam_calib.intrinsics.width, cam_calib.intrinsics.height)
			updated_intrinsics = update_intrinsics_for_resize(
				cam_calib.intrinsics,
				original_size,
				crop_info,
				(target_width, target_height)
			)

			# Create updated calibration
			updated_calib = CalibrationData(
				camera_id=cam_calib.camera_id,
				intrinsics=updated_intrinsics,
				extrinsics=cam_calib.extrinsics,
				stereo_baseline=cam_calib.stereo_baseline,
				stereo_transform=cam_calib.stereo_transform
			)
			updated_calibrations[camera_id] = updated_calib

		return resized_frame, updated_calibrations
	
	def _create_camera_info_msg(self, calibration: CalibrationData, header: Header) -> CameraInfo:
		"""Create CameraInfo message from calibration data.
		
		Args:
			calibration: Camera calibration data
			header: Message header
			
		Returns:
			CameraInfo message
		"""
		msg = CameraInfo()
		msg.header = header
		
		# Set dimensions
		msg.height = calibration.intrinsics.height
		msg.width = calibration.intrinsics.width
		
		# Set camera matrix (K)
		K = calibration.intrinsics.to_camera_matrix()
		msg.k = K.flatten().tolist()
		
		# Set distortion
		msg.distortion_model = calibration.intrinsics.distortion_model
		msg.d = calibration.intrinsics.distortion_coeffs
		
		# Set projection matrix (P)
		P = calibration.intrinsics.to_projection_matrix()
		msg.p = P.flatten().tolist()
		
		# Set rectification matrix (R) - identity if not stereo rectified
		msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
		
		# ROI (region of interest) - full image
		msg.binning_x = 0
		msg.binning_y = 0
		msg.roi.x_offset = 0
		msg.roi.y_offset = 0
		msg.roi.height = msg.height
		msg.roi.width = msg.width
		msg.roi.do_rectify = False
		
		return msg