"""
CODa dataset loader implementation.

Loads data from the CODa dataset structure:
- 2d_rect/cam{0,1}/sequence/*.png - RGB images
- calibrations/sequence/*.yaml - Camera calibration
- poses/dense_global/sequence/*.txt, fallback to poses/dense/sequence/*.txt - Pose data
- timestamps/sequence.txt - Frame timestamps
- 3d_raw_estimated/cam{0,1}/sequence/*.png - Estimated depth images (optional)
- 3d_semantic/os1/sequence/*.pcd - Semantic point clouds (optional)
"""

import numpy as np
import cv2
import yaml
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import re
from collections import defaultdict
from natsort import natsorted
from scipy.spatial.transform import Rotation as ScipyR

from daaam_ros.nodes.dataloader.interfaces import DataLoader
from daaam_ros.nodes.dataloader.models import FrameData, CalibrationData, DataloaderConfig, CameraIntrinsics, Transform, ImageData, BoundingBox3D


class CodaDataLoader(DataLoader):
	"""Dataloader for CODa dataset format."""
	
	def __init__(self, config: DataloaderConfig):
		super().__init__(config)
		self.root_path = config.dataset_path
		self.sequence = config.sequence or "0"
		self.selected_cameras = config.camera_ids
		
		# Data structures
		self.image_paths: Dict[str, List[Path]] = defaultdict(list)
		self.depth_paths: Dict[str, List[Path]] = defaultdict(list)
		self.bbox_paths_dict: Dict[int, Path] = {}  # Bbox annotation files by absolute frame number
		self.timestamps: Optional[List[float]] = None
		self.calibrations: Dict[str, CalibrationData] = {}
		self.pose_data: Optional[np.ndarray] = None  # Store all poses from file
		self.static_transforms: Dict[str, Transform] = {}  # Store static transforms (os1->cameras)
		
		self._num_frames = 0
		self._framerate: Optional[float] = None
		
	def initialize(self) -> bool:
		"""Initialize the CODa dataloader."""
		try:
			# Validate root path
			if not self.root_path.exists():
				print(f"Dataset path does not exist: {self.root_path}")
				return False
			
			# Load RGB image paths
			self._load_image_paths()
			
			# Load depth image paths if available
			self._load_depth_paths()
			
			# Load calibrations
			self._load_calibrations()
			
			# Load timestamps if available
			self._load_timestamps()
			
			# Load poses if available
			self._load_poses()

			# Load bbox annotations if available
			self._load_bbox_annotations()

			# Validate we have data
			if self._num_frames == 0:
				print(f"No frames found for sequence {self.sequence}")
				return False
			
			self._log_dataset_info()
			return True
			
		except Exception as e:
			print(f"Failed to initialize CODa dataloader: {e}")
			import traceback
			traceback.print_exc()
			return False
	
	def _load_image_paths(self) -> None:
		"""Load paths to RGB images."""
		rgb_base = self.root_path / "2d_rect"
		
		for camera_id in self.selected_cameras:
			camera_dir = rgb_base / camera_id / self.sequence
			
			if not camera_dir.exists():
				print(f"Warning: Camera directory not found: {camera_dir}")
				continue
			
			# Get all image files and sort them
			image_files = natsorted(camera_dir.glob("*.png"))
			
			if not image_files:
				print(f"Warning: No images found in {camera_dir}")
				continue
			
			self.image_paths[camera_id] = image_files
			
			# Update frame count (should be same for all cameras)
			if self._num_frames == 0:
				self._num_frames = len(image_files)
			elif self._num_frames != len(image_files):
				print(f"Warning: Camera {camera_id} has {len(image_files)} frames, expected {self._num_frames}")
				self._num_frames = min(self._num_frames, len(image_files))
		
		# Apply frame range from config
		if self.config.end_frame:
			self._num_frames = min(self._num_frames, self.config.end_frame - self.config.start_frame)
	
	def _load_calibrations(self) -> None:
		"""Load camera calibration data."""
		calib_dir = self.root_path / "calibrations" / self.sequence
		
		if not calib_dir.exists():
			print(f"Warning: Calibration directory not found: {calib_dir}")
			return
		
		# os1_to_cam0 and os1_to_cam1 transforms for CODa
		self.os1_to_cam_transforms = {}
		
		for camera_id in self.selected_cameras:
			# Use undistorted intrinsics for rectified images (2d_rect/)
			intrinsics_file = calib_dir / f"calib_{camera_id}_undist_intrinsics.yaml"
			if not intrinsics_file.exists():
				# Fallback to regular intrinsics if undist not available
				intrinsics_file = calib_dir / f"calib_{camera_id}_intrinsics.yaml"
				print(f"Warning: Using distorted calibration for {camera_id}, but images are rectified!")
				if not intrinsics_file.exists():
					print(f"Warning: Intrinsics not found for {camera_id}")
					continue
			
			with open(intrinsics_file, 'r') as f:
				calib_data = yaml.safe_load(f)
			
			# Parse intrinsics
			camera_matrix = np.array(calib_data['camera_matrix']['data']).reshape(3, 3)
			intrinsics = CameraIntrinsics(
				fx=camera_matrix[0, 0],
				fy=camera_matrix[1, 1],
				cx=camera_matrix[0, 2],
				cy=camera_matrix[1, 2],
				width=calib_data.get('image_width', 1224),
				height=calib_data.get('image_height', 1024),
				distortion_model=calib_data.get('distortion_model', 'plumb_bob'),
				distortion_coeffs=calib_data.get('distortion_coefficients', {}).get('data', [])
			)
			
			# Create calibration object
			self.calibrations[camera_id] = CalibrationData(
				camera_id=camera_id,
				intrinsics=intrinsics
			)
			
			# Load os1_to_cam transform for this camera
			# Try undistorted version first for rectified images
			os1_to_cam_file = calib_dir / f"calib_os1_to_{camera_id}_undist.yaml"
			if not os1_to_cam_file.exists():
				# Fallback to regular transform
				os1_to_cam_file = calib_dir / f"calib_os1_to_{camera_id}.yaml"
			
			if os1_to_cam_file.exists():
				with open(os1_to_cam_file, 'r') as f:
					os1_cam_data = yaml.safe_load(f)
				
				if 'extrinsic_matrix' in os1_cam_data:
					# Extract 4x4 matrix
					if 'data' in os1_cam_data['extrinsic_matrix']:
						# Flat format
						data = os1_cam_data['extrinsic_matrix']['data']
						T = np.array(data).reshape(4, 4)
					else:
						# Already a 4x4 matrix
						T = np.array(os1_cam_data['extrinsic_matrix']).reshape(4, 4)
					
					self.os1_to_cam_transforms[camera_id] = T
					
					# Store static transform in calibration data (inverted!!)
					cam_translation = (-T[:3, :3].T @ T[:3, 3]).tolist()
					cam_rot = ScipyR.from_matrix(T[:3, :3].T)
					cam_quaternion = cam_rot.as_quat().tolist()  # Already [x, y, z, w]
					
					self.calibrations[camera_id].extrinsics = Transform(
						translation=cam_translation,
						rotation=cam_quaternion,
						frame_id="os1",
						child_frame_id=f"{camera_id}_optical_frame"
					)
					
					# Store as static transform
					self.static_transforms[f"{camera_id}_optical_frame"] = Transform(
						translation=cam_translation,
						rotation=cam_quaternion,
						frame_id="os1",
						child_frame_id=f"{camera_id}_optical_frame"
					)
					
					print(f"Loaded os1_to_{camera_id} transform")
			
			# Load stereo calibration if cam0 and cam1
			if camera_id == "cam1" and "cam0" in self.selected_cameras:
				stereo_file = calib_dir / "calib_cam0_to_cam1.yaml"
				if stereo_file.exists():
					with open(stereo_file, 'r') as f:
						stereo_data = yaml.safe_load(f)
					
					# Parse transform - CODa format has extrinsic_matrix with R and T
					if 'extrinsic_matrix' in stereo_data:
						rotation = np.array(stereo_data['extrinsic_matrix']['R']['data']).reshape(3, 3)
						translation = stereo_data['extrinsic_matrix']['T']
						rotation_matrix = rotation.flatten().tolist()
					elif 'T_cn_cnm1' in stereo_data:
						# Alternative format (4x4 matrix)
						Trans = np.array(stereo_data['T_cn_cnm1']).reshape(4, 4)
						translation = Trans[:3, 3].tolist()
						rotation_matrix = Trans[:3, :3].flatten().tolist()
					else:
						print(f"Warning: Unknown stereo calibration format in {stereo_file}")
						continue
					
					self.calibrations[camera_id].stereo_transform = Transform(
						translation=translation,
						rotation=rotation_matrix,
						frame_id="cam0",
						child_frame_id="cam1"
					)
					
					# Calculate baseline
					self.calibrations[camera_id].stereo_baseline = np.linalg.norm(translation)
		
		# Note: os1_to_base is not needed - CODa poses are directly os1 poses in world frame
		# The test_coda_transforms.py script shows the correct interpretation
	
	def _load_timestamps(self) -> None:
		"""Load frame timestamps if available."""
		timestamp_file = self.root_path / "timestamps" / f"{self.sequence}.txt"
		
		if not timestamp_file.exists():
			print(f"Timestamps not found, will use target framerate")
			return
		
		with open(timestamp_file, 'r') as f:
			self.timestamps = [float(line.strip()) for line in f if line.strip()]
		
		# Calculate framerate from timestamps
		if len(self.timestamps) > 1:
			dt_values = np.diff(self.timestamps[:min(100, len(self.timestamps))])
			self._framerate = 1.0 / np.median(dt_values)
	
	def _load_depth_paths(self) -> None:
		"""Load paths to depth images if available."""
		# Check depth source from config
		depth_source = getattr(self.config, 'depth_source', 'none')
		
		if depth_source == '3d_raw':
			raise(ValueError("Raw depth from cam3 is in a different reference frame. We recommend running stereo depth on the frames of cam0/cam1 instead, which are rectified and in the same frame as the RGB images."))	
		
		elif depth_source == '3d_raw_estimated':
			# Estimated depth (pre-computed offline)
			depth_base = self.root_path / "3d_raw_estimated"
			
			for camera_id in self.selected_cameras:
				camera_depth_dir = depth_base / camera_id / self.sequence
				if camera_depth_dir.exists():
					depth_files = natsorted(camera_depth_dir.glob("*.png"))
					if depth_files:
						self.depth_paths[camera_id] = depth_files
		
		elif depth_source != 'none':
			print(f"Warning: Unknown depth source '{depth_source}'")	
	
	def _load_poses(self) -> None:
		"""Load poses from CODa format (single file with all poses)."""
		# Try dense_global first (globally consistent), then fall back to dense
		pose_file = self.root_path / "poses" / "dense_global" / f"{self.sequence}.txt"
		
		if not pose_file.exists():
			# Try dense directory as fallback
			pose_file = self.root_path / "poses" / "dense" / f"{self.sequence}.txt"
			if not pose_file.exists():
				print(f"Pose file not found in dense_global or dense directories for sequence {self.sequence}")
				return
			else:
				print(f"Using dense poses (may have drift): {pose_file}")
		else:
			print(f"Using dense_global poses (globally consistent): {pose_file}")
			
		try:
			# load all poses: [timestamp, x, y, z, qw, qx, qy, qz]
			self.pose_data = np.loadtxt(pose_file)
			print(f"Loaded {len(self.pose_data)} poses from {pose_file}")
			
			# Verify first pose for debugging
			if len(self.pose_data) > 0:
				first = self.pose_data[0]
				print(f"First pose: ts={first[0]:.6f}, xyz=[{first[1]:.3f}, {first[2]:.3f}, {first[3]:.3f}], quat=[{first[4]:.3f}, {first[5]:.3f}, {first[6]:.3f}, {first[7]:.3f}]")
		except Exception as e:
			print(f"Failed to load poses: {e}")

	def _load_bbox_annotations(self) -> None:
		"""Load 3D bbox annotation file paths if available."""
		bbox_dir = self.root_path / "3d_bbox" / "os1" / self.sequence

		if not bbox_dir.exists():
			print(f"Bbox annotation directory not found: {bbox_dir}")
			# Initialize dict to store bbox paths by absolute frame number
			self.bbox_paths_dict = {}
			return

		# Get all bbox annotation files
		bbox_files = natsorted(bbox_dir.glob("3d_bbox_os1_*.json"))

		if not bbox_files:
			print(f"No bbox annotations found in {bbox_dir}")
			self.bbox_paths_dict = {}
			return

		# Create dictionary mapping absolute frame numbers to bbox paths
		self.bbox_paths_dict = {}

		for bbox_file in bbox_files:
			# Extract frame number from filename: 3d_bbox_os1_{sequence}_{frame}.json
			filename = bbox_file.stem
			parts = filename.split('_')
			if len(parts) >= 4:
				try:
					frame_num = int(parts[-1])
					self.bbox_paths_dict[frame_num] = bbox_file
				except ValueError:
					print(f"Warning: Could not parse frame number from {filename}")

		# Count how many frames have annotations
		print(f"Loaded {len(self.bbox_paths_dict)} bbox annotation files")

	def get_frame(self, index: Optional[int] = None) -> Optional[FrameData]:
		"""Get a frame at specified index or next frame."""
		if index is None:
			index = self._current_index
			self._current_index += 1
		
		# Apply stride and start offset
		actual_index = self.config.start_frame + index * self.config.stride
		
		if actual_index >= self._num_frames:
			return None
		
		# Create frame data
		frame = FrameData(
			frame_id=actual_index,
			sequence_name=self.sequence
		)
		
		# Add timestamp if available
		if self.timestamps and actual_index < len(self.timestamps):
			frame.timestamp = self.timestamps[actual_index]
		
		# Load RGB images for each camera
		for camera_id, image_paths in self.image_paths.items():
			if actual_index < len(image_paths):
				img_path = image_paths[actual_index]
				img = cv2.imread(str(img_path))
				
				if img is not None:
					frame.rgb_images[camera_id] = ImageData(
						data=img,
						encoding="bgr8",
						timestamp=frame.timestamp,
						frame_id=camera_id
					)
		
		# Load depth images if available
		for camera_id, depth_paths in self.depth_paths.items():
			if actual_index < len(depth_paths):
				depth_path = depth_paths[actual_index]
				# Load depth as 16-bit PNG (millimeters)
				depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
				
				if depth_mm is not None:
					# Convert from mm to meters
					depth_m = depth_mm.astype(np.float32) / 1000.0
					
					frame.depth_images[camera_id] = ImageData(
						data=depth_m,
						encoding="32FC1",
						timestamp=frame.timestamp,
						frame_id=f"{camera_id}_optical_frame"
					)
		
		# Load pose if available
		if self.pose_data is not None and frame.timestamp:
			# Find closest pose by timestamp (should be exact match based on our verification)
			pose_idx = np.argmin(np.abs(self.pose_data[:, 0] - frame.timestamp))
			pose = self.pose_data[pose_idx]
			
			# Use pose timestamp as authoritative (they're identical anyway)
			pose_timestamp = pose[0]
			original_timestamp = frame.timestamp
			time_diff = abs(pose_timestamp - original_timestamp)
			
			# Debug logging for any discrepancies (shouldn't happen with CODa)
			if time_diff > 0.001:  # More than 1ms difference
				print(f"[TIMESTAMP DEBUG] Frame {actual_index}: Image ts={original_timestamp:.6f}, "
					  f"Pose ts={pose_timestamp:.6f}, diff={time_diff*1000:.2f}ms")
			
			# Check for pose gaps (for debugging)
			if pose_idx > 0:
				prev_pose_ts = self.pose_data[pose_idx - 1, 0]
				pose_gap = pose_timestamp - prev_pose_ts
				if pose_gap > 0.15:  # More than 150ms gap
					print(f"[POSE GAP WARNING] Frame {actual_index}: Gap of {pose_gap*1000:.1f}ms between poses")
			
			# Use exact pose timestamp to ensure perfect sync
			frame.timestamp = pose_timestamp
			
			# Extract translation and quaternion for os1 frame
			os1_translation = pose[1:4].tolist()  # x, y, z
			# CODa format: [qw, qx, qy, qz], convert to ROS2 [x, y, z, w]
			quat_coda = pose[4:8]  # [qw, qx, qy, qz]
			os1_quaternion = [quat_coda[1], quat_coda[2], quat_coda[3], quat_coda[0]]  # [x, y, z, w]
			
			# Store world -> os1 transform as dynamic (changes each frame)
			frame.dynamic_transforms["os1"] = Transform(
				translation=os1_translation,
				rotation=os1_quaternion,  # Quaternion in ROS2 format [x, y, z, w]
				frame_id="world",
				child_frame_id="os1"
				# No separate timestamp - will use frame.timestamp
			)
			
			# Reference to static transforms (loaded once in _load_calibrations)
			frame.static_transforms = self.static_transforms
			
			# For backward compatibility, also populate deprecated poses field
			frame.poses["os1"] = frame.dynamic_transforms["os1"]
			for key, transform in self.static_transforms.items():
				frame.poses[key] = transform

		# Load bbox annotations if available (use absolute frame index)
		if actual_index in self.bbox_paths_dict:
			bbox_file = self.bbox_paths_dict[actual_index]
			try:
				with open(bbox_file, 'r') as f:
					bbox_data = json.load(f)

				# Parse each bbox annotation
				for bbox_obj in bbox_data.get("3dbbox", []):
					# Extract bbox parameters
					center = [bbox_obj["cX"], bbox_obj["cY"], bbox_obj["cZ"]]
					size = [bbox_obj["l"], bbox_obj["w"], bbox_obj["h"]]

					# Convert roll, pitch, yaw to quaternion
					roll = bbox_obj["r"]
					pitch = bbox_obj["p"]
					yaw = bbox_obj["y"]

					# Create rotation from RPY (intrinsic XYZ order)
					rot = ScipyR.from_euler('xyz', [roll, pitch, yaw])
					quaternion = rot.as_quat().tolist()  # [x, y, z, w]

					# Create BoundingBox3D object
					bbox = BoundingBox3D(
						center=center,
						size=size,
						orientation=quaternion,
						class_id=bbox_obj["classId"],
						instance_id=bbox_obj["instanceId"],
						occlusion=bbox_obj.get("labelAttributes", {}).get("isOccluded", "None"),
						roll=roll,
						pitch=pitch,
						yaw=yaw
					)

					frame.bbox_annotations.append(bbox)

			except Exception as e:
				print(f"Warning: Failed to load bbox annotations for frame {actual_index}: {e}")

		return frame
	
	def get_calibration(self, camera_id: str) -> Optional[CalibrationData]:
		"""Get calibration for a specific camera."""
		return self.calibrations.get(camera_id)
	
	def has_next(self) -> bool:
		"""Check if more frames are available."""
		next_index = self.config.start_frame + self._current_index * self.config.stride
		return next_index < self._num_frames
	
	def reset(self) -> None:
		"""Reset to the beginning of the dataset."""
		self._current_index = 0
	
	@property
	def num_frames(self) -> int:
		"""Total number of frames in dataset."""
		return (self._num_frames - self.config.start_frame) // self.config.stride
	
	@property
	def camera_ids(self) -> List[str]:
		"""List of available camera IDs."""
		return list(self.image_paths.keys())
	
	@property
	def has_depth(self) -> bool:
		"""Whether dataset provides depth data."""
		# Check if we loaded any depth paths
		return len(self.depth_paths) > 0
	
	@property
	def has_semantics(self) -> bool:
		"""Whether dataset provides semantic labels."""
		semantic_dir = self.root_path / "3d_semantic" / "os1" / self.sequence
		return semantic_dir.exists()
	
	@property
	def has_poses(self) -> bool:
		"""Whether dataset provides pose data."""
		return self.pose_data is not None

	@property
	def has_bboxes(self) -> bool:
		"""Whether dataset provides 3D bbox annotations."""
		return len(self.bbox_paths_dict) > 0
	
	@property
	def framerate(self) -> Optional[float]:
		"""Native framerate if available from timestamps."""
		return self._framerate
	
	def _log_dataset_info(self) -> None:
		"""Log information about loaded dataset."""
		print(f"\n{'='*50}")
		print(f"CODa Dataset Loaded:")
		print(f"  Path: {self.root_path}")
		print(f"  Sequence: {self.sequence}")
		print(f"  Cameras: {', '.join(self.camera_ids)}")
		print(f"  Frames: {self.num_frames}")
		print(f"  Has timestamps: {self.timestamps is not None}")
		print(f"  Has depth: {self.has_depth}")
		if self.has_depth:
			for cam_id in self.depth_paths.keys():
				print(f"    - {cam_id}: {len(self.depth_paths[cam_id])} depth frames")
		print(f"  Has poses: {self.has_poses}")
		print(f"  Has semantics: {self.has_semantics}")
		print(f"  Has bboxes: {self.has_bboxes}")
		if self.has_bboxes:
			print(f"    - Bbox files: {len(self.bbox_paths_dict)}")
			frame_range = f"{min(self.bbox_paths_dict.keys())}-{max(self.bbox_paths_dict.keys())}"
			print(f"    - Frame range: {frame_range}")
		if self._framerate:
			print(f"  Native framerate: {self._framerate:.2f} Hz")
		print(f"{'='*50}\n")