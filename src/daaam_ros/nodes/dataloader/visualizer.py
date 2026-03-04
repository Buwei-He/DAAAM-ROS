"""
Rerun visualizer for dataloader system.

Visualizes frames, transforms, cameras, and images from the dataloader pipeline.
"""

import numpy as np
import rerun as rr
from typing import Optional, Dict, Any, Set
from pathlib import Path
import cv2

from scipy.spatial.transform import Rotation as ScipyR


from daaam_ros.nodes.dataloader.models import FrameData, CalibrationData, Transform


class DataloaderVisualizer:
	"""Visualizes dataloader output in Rerun."""
	
	def __init__(self, app_id: str = "dataloader_viz", recording_id: Optional[str] = None):
		"""Initialize the visualizer.
		
		Args:
			app_id: Rerun application ID
			recording_id: Optional recording ID for saving
		"""
		self.app_id = app_id
		self.recording_id = recording_id
		self.initialized = False
		self.logged_cameras: Set[str] = set()  # Track which cameras have been configured
		self.logged_static_transforms: Set[str] = set()  # Track which static transforms have been logged
		self.frame_count = 0
		self.calibrations: Optional[Dict[str, CalibrationData]] = None
		
	def initialize(self, calibrations: Optional[Dict[str, CalibrationData]] = None) -> bool:
		"""Initialize Rerun recording.
		
		Args:
			calibrations: Optional calibration data for static transforms
		"""
		try:
			# Initialize rerun
			rr.init(self.app_id, spawn=True)
			rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)  # Match test_coda_transforms.py 
			
			# Set up time
			rr.set_time("frame", sequence=0)
			
			# Log world frame as root
			rr.log("world", rr.Transform3D())
			
			# Store calibrations for later use
			if calibrations:
				self.calibrations = calibrations
				# Log static transforms from calibration
				self._log_static_transforms(calibrations)
			
			self.initialized = True
			print(f"Rerun visualizer initialized with app_id: {self.app_id}")
			return True
			
		except Exception as e:
			print(f"Failed to initialize Rerun: {e}")
			return False
	
	def _log_static_transforms(self, calibrations: Dict[str, CalibrationData]) -> None:
		"""Log static calibration transforms once at initialization.
		
		Args:
			calibrations: Camera calibration data containing static transforms
		"""
		# Log os1 frame placeholder (will be updated dynamically)
		rr.log("world/os1", rr.Transform3D())
		
		# Log static camera transforms relative to os1
		for camera_id, calib in calibrations.items():
			if calib.extrinsics:  # os1->camera transform
				transform = calib.extrinsics
				entity_path = f"world/os1/{transform.child_frame_id}"
				
				# Rotation is already in [x, y, z, w] format
				if len(transform.rotation) == 4:
					rotation = transform.rotation  # Already [x, y, z, w]
				else:
					print(f"Unexpected rotation format for {camera_id}")
					continue
				
				# Log static transform (os1->camera uses ChildFromParent)
				rr.log(
					entity_path,
					rr.Transform3D(
						translation=transform.translation,
						quaternion=rotation,
						relation=rr.TransformRelation.ParentFromChild  # Match test_coda_transforms.py
					),
				)
				self.logged_static_transforms.add(entity_path)
				print(f"Logged static transform: {entity_path} (os1 -> {transform.child_frame_id})")
	
	def log_frame(self, frame: FrameData, calibrations: Dict[str, CalibrationData]) -> None:
		"""Log a complete frame to Rerun.
		
		Args:
			frame: Frame data containing images, poses, etc.
			calibrations: Camera calibration data by camera ID
		"""
		if not self.initialized:
			# Initialize with calibrations for static transforms
			if not self.initialize(calibrations):
				return
		
		# Store/update calibrations if not already set
		if not self.calibrations:
			self.calibrations = calibrations
		
		# Set frame time
		self.frame_count += 1
		rr.set_time("frame", sequence=self.frame_count)
		if frame.timestamp:
			rr.set_time("timestamp", timestamp=frame.timestamp)
		
		# Log all transforms
		self._log_transforms(frame)

		# Log camera data (images, depth, pinhole)
		self._log_cameras(frame, calibrations)

		# Log 3D bounding boxes if available
		self._log_bboxes(frame)
		
	def _log_transforms(self, frame: FrameData) -> None:
		"""Log all frame transforms.
		
		Args:
			frame: Frame data containing pose transforms
		"""
		for pose_name, transform in frame.poses.items():
			if transform is None:
				continue
			
			# Build entity path from frame hierarchy
			entity_path = self._build_entity_path(transform)
			
			# Skip if this is a static transform that was already logged
			if entity_path in self.logged_static_transforms:
				continue
			
			# Determine the transform relation based on frame hierarchy
			if transform.frame_id == "world" and transform.child_frame_id == "os1":
				# World -> OS1 transform (dynamic pose)
				relation = rr.TransformRelation.ParentFromChild  # Match test_coda_transforms.py
			elif transform.frame_id == "os1" and "optical_frame" in transform.child_frame_id:
				# OS1 -> Camera transform (static calibration)
				relation = rr.TransformRelation.ChildFromParent  # Match test_coda_transforms.py
			else:
				# Default
				relation = rr.TransformRelation.ParentFromChild
			
			# Transform stores rotation in [x, y, z, w] format from ROS2
			if len(transform.rotation) == 4:
				# For Rerun, we need [x, y, z, w] format (same as ROS2)
				rotation = transform.rotation
			elif len(transform.rotation) == 9:
				# Rotation matrix - convert to quaternion
				rot = ScipyR.from_matrix(np.array(transform.rotation).reshape(3, 3))
				rotation = rot.as_quat().tolist()  # Returns [x, y, z, w]
			else:
				print(f"Unexpected rotation format for {pose_name}: {len(transform.rotation)} elements")
				continue

			# Debug logging for first frame
			if self.frame_count == 1:
				print(f"Logging transform for {pose_name} at {entity_path}:")
				print(f"  xyz: {transform.translation}")
				print(f"  quat[xyzw]: {rotation}")
				print(f"  relation: {relation}")
			
			# Log transform
			rr.log(
				entity_path,
				rr.Transform3D(
					translation=transform.translation,
					quaternion=rotation,
					relation=relation
				)
			)
			
			# Debug log
			if self.frame_count == 1:
				print(f"Logged transform: {entity_path} ({transform.frame_id} -> {transform.child_frame_id})")
	
	def _build_entity_path(self, transform: Transform) -> str:
		"""Build entity path from transform frames.
		
		Args:
			transform: Transform containing frame_id and child_frame_id
			
		Returns:
			Entity path string for Rerun
		"""
		# Parse the transform relationship
		parent = transform.frame_id
		child = transform.child_frame_id
		
		# Build hierarchical path based on parent frame
		if parent == "world":
			# Direct child of world (e.g., os1, base_link)
			return f"world/{child}"
		elif parent == "os1":
			# Camera/sensor relative to os1
			return f"world/os1/{child}"
		elif parent == "cam0":
			# Stereo pair (cam1 relative to cam0) - though CODa doesn't use this
			return f"world/os1/cam0/{child}"
		else:
			# Default: assume it's under world
			return f"world/{child}"
	
	def _log_cameras(self, frame: FrameData, calibrations: Dict[str, CalibrationData]) -> None:
		"""Log camera data including pinhole, images, and depth.
		
		Args:
			frame: Frame data with images
			calibrations: Camera calibrations
		"""
		# Process RGB images
		for camera_id, image_data in frame.rgb_images.items():
			if image_data is None:
				continue
			
			# Get calibration
			calib = calibrations.get(camera_id)
			if calib is None:
				print(f"No calibration for camera {camera_id}")
				continue
			
			# Determine entity path for this camera
			camera_entity = self._get_camera_entity_path(camera_id, frame)
			
			# Log pinhole camera (only once per camera)
			if camera_entity not in self.logged_cameras:
				self._log_pinhole(camera_entity, calib)
				self.logged_cameras.add(camera_entity)
			
			# Log RGB image
			self._log_rgb_image(f"{camera_entity}/image", image_data)
			
			# Log depth if available
			depth_data = frame.depth_images.get(camera_id)
			if depth_data is not None:
				self._log_depth_image(f"{camera_entity}/depth", depth_data)
	
	def _get_camera_entity_path(self, camera_id: str, frame: FrameData) -> str:
		"""Get entity path for a camera based on its pose.
		
		Args:
			camera_id: Camera identifier
			frame: Frame data with poses
			
		Returns:
			Entity path for the camera
		"""
		# Look for pose with matching camera ID
		for pose_name, transform in frame.poses.items():
			if transform is None:
				continue
			
			# Check if this pose is for the camera
			# Handle variations: cam0_optical, cam0_optical_frame, cam0, etc.
			if camera_id in pose_name or pose_name in camera_id:
				return self._build_entity_path(transform)
		
		# Fallback: use camera_id directly under world
		return f"world/{camera_id}"
	
	def _log_pinhole(self, entity_path: str, calib: CalibrationData) -> None:
		"""Log pinhole camera model.
		
		Args:
			entity_path: Entity path for the camera
			calib: Camera calibration data
		"""
		intrinsics = calib.intrinsics
		
		# Create camera matrix
		image_from_camera = np.array([
			[intrinsics.fx, 0, intrinsics.cx],
			[0, intrinsics.fy, intrinsics.cy],
			[0, 0, 1]
		])
		
		rr.log(
			entity_path,
			rr.Pinhole(
				image_from_camera=image_from_camera,
				width=intrinsics.width,
				height=intrinsics.height,
			)
		)
		
		print(f"Logged pinhole at {entity_path}: {intrinsics.width}x{intrinsics.height}, "
			  f"fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}")
	
	def _log_rgb_image(self, entity_path: str, image_data) -> None:
		"""Log RGB image.
		
		Args:
			entity_path: Entity path for the image
			image_data: ImageData containing the image
		"""
		img = image_data.data
		
		# Convert BGR to RGB if needed
		if image_data.encoding == "bgr8":
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		
		rr.log(entity_path, rr.Image(img))
	
	def _log_depth_image(self, entity_path: str, depth_data) -> None:
		"""Log depth image.
		
		Args:
			entity_path: Entity path for the depth
			depth_data: ImageData containing depth
		"""
		depth = depth_data.data
		
		# Handle different depth formats
		if depth_data.encoding == "32FC1":
			# Already in meters as float32
			depth_m = depth
		elif depth_data.encoding == "16UC1":
			# Convert from mm to meters
			depth_m = depth.astype(np.float32) / 1000.0
		else:
			depth_m = depth.astype(np.float32)
		
		# Log as depth image
		rr.log(entity_path, rr.DepthImage(depth_m, meter=1.0))

	def _log_bboxes(self, frame: FrameData) -> None:
		"""Log 3D bounding boxes to Rerun.

		Args:
			frame: Frame data containing bbox annotations
		"""
		if not frame.bbox_annotations:
			return

		# Prepare data for Boxes3D
		centers = []
		half_sizes = []
		quaternions = []
		colors = []
		class_ids = []

		for bbox in frame.bbox_annotations:
			centers.append(bbox.center)
			# Boxes3D uses half_sizes, bbox.size is full size
			half_sizes.append([s / 2.0 for s in bbox.size])
			quaternions.append(bbox.orientation)  # Already [x, y, z, w]

			# Hash class to color
			color = self._class_to_color(bbox.class_id)
			colors.append(color)
			class_ids.append(bbox.class_id)

		# Log all boxes at once under os1 frame
		rr.log(
			"world/os1/ground_truth_bboxes",
			rr.Boxes3D(
				centers=np.array(centers),
				half_sizes=np.array(half_sizes),
				quaternions=np.array(quaternions),
				colors=np.array(colors),
				class_ids=class_ids
			)
		)

	def _class_to_color(self, class_id: str) -> tuple:
		"""Hash class ID to consistent RGB color.

		Args:
			class_id: Class identifier string

		Returns:
			RGB color tuple (values in [0, 255])
		"""
		# Simple hash to RGB (0-255 range for Rerun)
		hash_val = hash(class_id)
		r = (hash_val & 0xFF0000) >> 16
		g = (hash_val & 0x00FF00) >> 8
		b = hash_val & 0x0000FF
		return (r, g, b)

	def close(self) -> None:
		"""Close the visualizer."""
		if self.initialized:
			# Optionally save recording
			if self.recording_id:
				print(f"Saving recording to {self.recording_id}.rrd")
				# rr.save(f"{self.recording_id}.rrd")  # Uncomment to save
			
			self.initialized = False
			print("Rerun visualizer closed")