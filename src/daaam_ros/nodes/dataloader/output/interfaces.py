from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from daaam_ros.nodes.dataloader.models import FrameData, CalibrationData

class OutputHandler(ABC):
	"""Abstract base class for output handlers."""
	
	def __init__(self, config: Dict[str, Any]):
		self.config = config
		
	@abstractmethod
	def initialize(self) -> bool:
		"""Initialize output handler (e.g., create publishers, open bag).
		
		Returns:
			bool: True if initialization successful
		"""
		pass
	
	@abstractmethod
	def publish_frame(self, frame_data: FrameData, calibration: CalibrationData) -> bool:
		"""Publish/write frame data.
		
		Args:
			frame_data: Frame with all data
			calibration: Camera calibration
			
		Returns:
			bool: True if successful
		"""
		pass
	
	@abstractmethod
	def close(self) -> None:
		"""Clean up resources (close bag, destroy publishers, etc.)."""
		pass
	
	@property
	@abstractmethod
	def is_ready(self) -> bool:
		"""Check if handler is ready to publish."""
		pass