"""
Data models for the dataloader system using Pydantic.

Following daaam pattern of schema definitions.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from pathlib import Path
from datetime import datetime

from scipy.spatial.transform import Rotation as ScipyR


class BoundingBox3D(BaseModel):
	"""3D bounding box annotation."""
	center: List[float] = Field(description="Center position [x, y, z] in meters")
	size: List[float] = Field(description="Size [length, width, height] in meters")
	orientation: List[float] = Field(description="Orientation as quaternion [x, y, z, w]")
	class_id: str = Field(description="Semantic class name")
	instance_id: str = Field(description="Unique instance identifier")
	occlusion: str = Field(default="None", description="Occlusion level")
	confidence: float = Field(default=1.0, description="Detection confidence")

	# Original RPY values (optional, for reference)
	roll: Optional[float] = Field(default=None, description="Roll in radians")
	pitch: Optional[float] = Field(default=None, description="Pitch in radians")
	yaw: Optional[float] = Field(default=None, description="Yaw in radians")

	class Config:
		arbitrary_types_allowed = True


class CameraIntrinsics(BaseModel):
	"""Camera intrinsic parameters."""
	fx: float = Field(description="Focal length x")
	fy: float = Field(description="Focal length y")
	cx: float = Field(description="Principal point x")
	cy: float = Field(description="Principal point y")
	width: int = Field(description="Image width")
	height: int = Field(description="Image height")
	distortion_model: str = Field(default="plumb_bob", description="Distortion model type")
	distortion_coeffs: List[float] = Field(default_factory=list, description="Distortion coefficients")
	
	class Config:
		arbitrary_types_allowed = True
		
	def to_camera_matrix(self) -> np.ndarray:
		"""Convert to 3x3 camera matrix."""
		return np.array([
			[self.fx, 0, self.cx],
			[0, self.fy, self.cy],
			[0, 0, 1]
		])
	
	def to_projection_matrix(self) -> np.ndarray:
		"""Convert to 3x4 projection matrix."""
		P = np.zeros((3, 4))
		P[:3, :3] = self.to_camera_matrix()
		return P


class Transform(BaseModel):
	"""6DOF transformation (position and orientation)."""
	translation: List[float] = Field(description="Translation [x, y, z]")
	rotation: List[float] = Field(description="Rotation as quaternion [x, y, z, w] or matrix")
	frame_id: str = Field(description="Reference frame")
	child_frame_id: str = Field(description="Target frame")
	timestamp: Optional[float] = Field(default=None, description="Transform timestamp (for poses)")
	
	class Config:
		arbitrary_types_allowed = True
	
	def to_matrix(self) -> np.ndarray:
		"""Convert to 4x4 transformation matrix."""
		T = np.eye(4)
		T[:3, 3] = self.translation
		
		if len(self.rotation) == 4:
			# Quaternion [x, y, z, w]
			R = ScipyR.from_quat(self.rotation)
			T[:3, :3] = R.as_matrix()
		elif len(self.rotation) == 9:
			# Rotation matrix as flat list
			T[:3, :3] = np.array(self.rotation).reshape(3, 3)
		
		return T


class CalibrationData(BaseModel):
	"""Complete calibration data for a camera."""
	camera_id: str = Field(description="Camera identifier")
	intrinsics: CameraIntrinsics = Field(description="Camera intrinsics")
	extrinsics: Optional[Transform] = Field(default=None, description="Camera extrinsics (pose)")
	stereo_baseline: Optional[float] = Field(default=None, description="Stereo baseline if applicable")
	stereo_transform: Optional[Transform] = Field(default=None, description="Transform to stereo pair")
	
	class Config:
		arbitrary_types_allowed = True


class ImageData(BaseModel):
	"""Image data wrapper."""
	data: Any = Field(description="Image as numpy array")  # Will be np.ndarray
	encoding: str = Field(default="bgr8", description="Image encoding")
	timestamp: Optional[float] = Field(default=None, description="Image timestamp")
	frame_id: str = Field(default="", description="TF frame ID")
	
	class Config:
		arbitrary_types_allowed = True


class FrameData(BaseModel):
	"""Complete frame data from dataloader."""
	frame_id: int = Field(description="Frame index/ID")
	timestamp: Optional[float] = Field(default=None, description="Frame timestamp")
	
	# Image data for each camera
	rgb_images: Dict[str, ImageData] = Field(default_factory=dict, description="RGB images by camera ID")
	depth_images: Dict[str, Optional[ImageData]] = Field(default_factory=dict, description="Depth images by camera ID")
	label_images: Dict[str, Optional[ImageData]] = Field(default_factory=dict, description="Semantic labels by camera ID")

	# 3D bounding box annotations
	bbox_annotations: List[BoundingBox3D] = Field(default_factory=list, description="3D bounding box ground truth")
	
	# Transform data - separated for efficiency
	dynamic_transforms: Dict[str, Optional[Transform]] = Field(default_factory=dict, description="Dynamic transforms (e.g., world->os1)")
	static_transforms: Dict[str, Optional[Transform]] = Field(default_factory=dict, description="Static transforms (e.g., os1->cameras)")
	# Deprecated: Use dynamic_transforms and static_transforms instead
	poses: Dict[str, Optional[Transform]] = Field(default_factory=dict, description="DEPRECATED: Use dynamic_transforms and static_transforms")
	
	# Metadata
	sequence_name: Optional[str] = Field(default=None, description="Sequence identifier")
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
	
	class Config:
		arbitrary_types_allowed = True


class DataloaderConfig(BaseModel):
	"""Configuration for dataloader system."""
	
	# Dataset configuration
	dataset_type: str = Field(default="coda", description="Type of dataset (coda, generic, etc.)")
	dataset_path: Path = Field(description="Root path to dataset")
	sequence: Optional[str] = Field(default=None, description="Sequence to load")
	camera_ids: List[str] = Field(default_factory=lambda: ["cam0"], description="Cameras to load")
	
	# Frame selection
	start_frame: int = Field(default=0, description="Starting frame index")
	end_frame: Optional[int] = Field(default=None, description="Ending frame index")
	stride: int = Field(default=1, description="Frame stride for subsampling")
	
	# Depth configuration
	depth_method: str = Field(default="provided", description="Depth method: provided, sgbm, raft, none")
	depth_source: str = Field(default="none", description="Depth source for provided method: 3d_raw_estimated, none")
	depth_params: Dict[str, Any] = Field(default_factory=dict, description="Depth estimation parameters")
	
	# Output configuration (bag writing only)
	output_params: Dict[str, Any] = Field(default_factory=dict, description="Bag writer parameters")
	
	# Timing
	target_framerate: float = Field(default=10.0, description="Target framerate for bag writing (set high for max speed)")
	
	# Processing
	num_workers: int = Field(default=2, description="Number of worker threads for loading")
	prefetch_frames: int = Field(default=10, description="Number of frames to prefetch")
	
	# Visualization
	enable_visualizer: bool = Field(default=False, description="Enable Rerun visualization")

	# Image resizing configuration
	resize_images: bool = Field(default=False, description="Enable image resizing")
	target_width: int = Field(default=640, description="Target image width after resize")
	target_height: int = Field(default=480, description="Target image height after resize")
	resize_method: str = Field(default="crop_resize", description="Resize method: crop_resize or pad_resize")
	preserve_aspect_ratio: bool = Field(default=True, description="Preserve aspect ratio during resize")

	class Config:
		arbitrary_types_allowed = True