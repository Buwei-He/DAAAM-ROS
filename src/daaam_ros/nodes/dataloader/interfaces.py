"""
Abstract base classes for dataloader components.

Following the daaam pattern of defining interfaces separately.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from pathlib import Path
from daaam_ros.nodes.dataloader.models import FrameData, CalibrationData, DataloaderConfig


class DataLoader(ABC):
	"""Abstract base class for dataset loaders."""
	
	def __init__(self, config: DataloaderConfig):
		self.config = config
		self._current_index = 0
		
	@abstractmethod
	def initialize(self) -> bool:
		"""Initialize the dataloader, validate paths, load metadata.
		
		Returns:
			bool: True if initialization successful
		"""
		pass
	
	@abstractmethod
	def get_frame(self, index: Optional[int] = None) -> Optional[FrameData]:
		"""Get a frame at specified index or next frame.
		
		Args:
			index: Frame index, None for next frame
			
		Returns:
			FrameData or None if no more frames
		"""
		pass
	
	@abstractmethod
	def get_calibration(self, camera_id: str) -> Optional[CalibrationData]:
		"""Get calibration for a specific camera.
		
		Args:
			camera_id: Camera identifier (e.g., 'cam0', 'cam1')
			
		Returns:
			CalibrationData or None if not available
		"""
		pass
	
	@abstractmethod
	def has_next(self) -> bool:
		"""Check if more frames are available."""
		pass
	
	@abstractmethod
	def reset(self) -> None:
		"""Reset to the beginning of the dataset."""
		pass
	
	@property
	@abstractmethod
	def num_frames(self) -> int:
		"""Total number of frames in dataset."""
		pass
	
	@property
	@abstractmethod
	def camera_ids(self) -> List[str]:
		"""List of available camera IDs."""
		pass
	
	@property
	@abstractmethod
	def has_depth(self) -> bool:
		"""Whether dataset provides depth data."""
		pass
	
	@property
	@abstractmethod
	def has_semantics(self) -> bool:
		"""Whether dataset provides semantic labels."""
		pass
	
	@property
	@abstractmethod
	def has_poses(self) -> bool:
		"""Whether dataset provides pose data."""
		pass
	
	@property
	@abstractmethod
	def framerate(self) -> Optional[float]:
		"""Native framerate if available from timestamps."""
		pass

