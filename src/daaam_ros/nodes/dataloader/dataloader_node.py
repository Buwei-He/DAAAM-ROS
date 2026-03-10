#!/usr/bin/env python3
"""
ROS2 Dataloader Node

Main node that loads dataset data and publishes to ROS2 topics or bags.
Following daaam patterns for ROS2 nodes.
"""

import rclpy  # type: ignore
from rclpy.node import Node  # type: ignore
from pathlib import Path
import signal
import sys
from typing import Optional, Dict, Any
import yaml

from daaam_ros.nodes.dataloader.config import ConfigManager
from daaam_ros.nodes.dataloader.services import DataloaderService
from daaam_ros.nodes.dataloader.output.bag_writer import BagWriter


class DataloaderNode(Node):
	"""ROS2 node for dataset loading and publishing."""
	
	def __init__(self):
		super().__init__('dataloader_node')
		
		# Declare parameters
		self._declare_parameters()
		
		# Load configuration
		self.config = self._load_configuration()
		
		# Initialize service
		self.service = DataloaderService(self.config, self)
		
		# Output handler
		self.output_handler = None
		
		# Setup signal handlers
		signal.signal(signal.SIGINT, self._signal_handler)
		signal.signal(signal.SIGTERM, self._signal_handler)
		
		self.get_logger().info("DataloaderNode initialized")
	
	def _declare_parameters(self) -> None:
		"""Declare ROS2 parameters."""
		
		# Main configuration file
		self.declare_parameter('config_file', '')
		
		# Dataset parameters
		self.declare_parameter('dataset_type', 'coda')
		self.declare_parameter('dataset_path', '/path/to/coda/dir/')
		
		# Sequence - declare as integer (ROS2 will parse command line as int)
		self.declare_parameter('sequence', 0)
		
		self.declare_parameter('camera_ids', ['cam0', 'cam1'])
		
		# Frame selection
		self.declare_parameter('start_frame', 0)
		self.declare_parameter('end_frame', -1)  # -1 means all frames
		self.declare_parameter('stride', 1)
		self.declare_parameter('frame_skip', 0)  # 0=keep all, 1=skip every other, 2=skip 2 of 3, etc.
		
		# Depth configuration
		self.declare_parameter('depth_method', 'provided')  # provided, sgbm, none
		self.declare_parameter('depth_source', '3d_raw_estimated')  # 3d_raw_estimated, none
		self.declare_parameter('depth_params.min_disparity', 0)
		self.declare_parameter('depth_params.num_disparities', 128)
		self.declare_parameter('depth_params.block_size', 11)
		
		# Output configuration (bag writing only)
		self.declare_parameter('bag_path', '/tmp/dataloader_output.bag')
		
		# Timing
		# Timestamps from dataset are always required for simulated time
		self.declare_parameter('target_framerate', 10.0)
		
		# Processing
		self.declare_parameter('num_workers', 2)
		self.declare_parameter('prefetch_frames', 10)
		
		# Auto-start
		self.declare_parameter('auto_start', True)
		
		# Auto-shutdown after completion (useful for bag writing)
		self.declare_parameter('auto_shutdown', True)
		
		# Visualization
		self.declare_parameter('enable_visualizer', True)

		# Image resizing configuration
		self.declare_parameter('resize_images', False)
		self.declare_parameter('target_width', 640)
		self.declare_parameter('target_height', 480)
		self.declare_parameter('resize_method', 'crop_resize')
		self.declare_parameter('preserve_aspect_ratio', True)
	
	def _load_configuration(self) -> Any:
		"""Load configuration from file and parameters."""
		# Get config file path
		config_file_param = self.get_parameter('config_file').value
		config_path = Path(config_file_param) if config_file_param else None
		
		# Build overrides from ROS parameters
		overrides = {}
		
		# Dataset config
		overrides['dataset_type'] = self.get_parameter('dataset_type').value
		overrides['dataset_path'] = self.get_parameter('dataset_path').value
		# Handle sequence as either string or int
		sequence_param = self.get_parameter('sequence').value
		overrides['sequence'] = str(sequence_param)  # Convert to string if int
		overrides['camera_ids'] = self.get_parameter('camera_ids').value
		
		# Frame selection
		overrides['start_frame'] = self.get_parameter('start_frame').value
		end_frame = self.get_parameter('end_frame').value
		if end_frame > 0:
			overrides['end_frame'] = end_frame
		
		# Handle frame_skip parameter - converts to stride
		frame_skip = self.get_parameter('frame_skip').value
		stride = self.get_parameter('stride').value
		
		# If frame_skip is specified and non-zero, use it to calculate stride
		# frame_skip=1 means skip every other frame, so stride=2
		if frame_skip > 0:
			overrides['stride'] = frame_skip + 1
			self.get_logger().info(f"Frame skip {frame_skip} -> using stride {frame_skip + 1} (keeping every {frame_skip + 1} frames)")
		else:
			overrides['stride'] = stride
		
		# Depth params
		overrides['depth_method'] = self.get_parameter('depth_method').value
		overrides['depth_source'] = self.get_parameter('depth_source').value
		overrides['depth_params'] = {
			'min_disparity': self.get_parameter('depth_params.min_disparity').value,
			'num_disparities': self.get_parameter('depth_params.num_disparities').value,
			'block_size': self.get_parameter('depth_params.block_size').value,
		}
		
		# Output params (bag writing only)
		overrides['output_params'] = {
			'bag_path': self.get_parameter('bag_path').value,
			'publish_tf': True,
		}
		
		# Timing
		# Timestamps are always used from dataset (required for simulated time)
		overrides['target_framerate'] = self.get_parameter('target_framerate').value
		
		# Processing
		overrides['num_workers'] = self.get_parameter('num_workers').value
		overrides['prefetch_frames'] = self.get_parameter('prefetch_frames').value
		
		# Visualization
		overrides['enable_visualizer'] = self.get_parameter('enable_visualizer').value

		# Image resizing configuration
		overrides['resize_images'] = self.get_parameter('resize_images').value
		overrides['target_width'] = self.get_parameter('target_width').value
		overrides['target_height'] = self.get_parameter('target_height').value
		overrides['resize_method'] = self.get_parameter('resize_method').value
		overrides['preserve_aspect_ratio'] = self.get_parameter('preserve_aspect_ratio').value

		# Load config with overrides
		config = ConfigManager.load_config(config_path, overrides)
		
		self._log_configuration(config)
		
		return config
	
	def initialize(self) -> bool:
		"""Initialize the node components."""
		try:
			# Initialize dataloader service
			if not self.service.initialize():
				self.get_logger().error("Failed to initialize dataloader service")
				return False
			
			# Create bag writer (only output mode now)
			self.output_handler = BagWriter(self.config.output_params, self)
			
			# Initialize output handler
			if not self.output_handler.initialize():
				self.get_logger().error("Failed to initialize output handler")
				return False
			
			# Set output handler in service
			self.service.set_output_handler(self.output_handler)
			
			# Set service reference in output handler (for calibration access)
			if hasattr(self.output_handler, 'set_service_ref'):
				self.output_handler.set_service_ref(self.service)
			
			self.get_logger().info("DataloaderNode initialized successfully")
			return True
			
		except Exception as e:
			self.get_logger().error(f"Failed to initialize: {e}")
			import traceback
			traceback.print_exc()
			return False
	
	def start(self) -> None:
		"""Start processing dataset."""
		self.get_logger().info("Starting dataloader...")
		self.service.start()
	
	def stop(self) -> None:
		"""Stop processing."""
		self.get_logger().info("Stopping dataloader...")
		self.service.stop()
		
		if self.output_handler:
			self.output_handler.close()
	
	def _signal_handler(self, signum, frame) -> None:
		"""Handle shutdown signals."""
		self.get_logger().info(f"Received signal {signum}, shutting down...")
		self.stop()
		# Force exit without waiting for rclpy cleanup
		sys.exit(0)
	
	def _log_configuration(self, config) -> None:
		"""Log the active configuration."""
		self.get_logger().info("=" * 60)
		self.get_logger().info("Configuration:")
		self.get_logger().info(f"  Dataset: {config.dataset_type}")
		self.get_logger().info(f"  Path: {config.dataset_path}")	
		self.get_logger().info(f"  Sequence: {config.sequence}")
		self.get_logger().info(f"  Cameras: {config.camera_ids}")
		if config.stride > 1:
			frame_skip = config.stride - 1
			keep_ratio = 100.0 / config.stride
			self.get_logger().info(f"  Frame skip: {frame_skip} (keeping {keep_ratio:.1f}% of frames)")
			self.get_logger().info(f"  Stride: {config.stride}")
		self.get_logger().info(f"  Depth method: {config.depth_method}")
		self.get_logger().info(f"  Output: Writing to bag")
		self.get_logger().info(f"  Target FPS: {config.target_framerate}")
		if config.resize_images:
			self.get_logger().info(f"  Resizing: Enabled ({config.target_width}x{config.target_height})")
			self.get_logger().info(f"  Resize method: {config.resize_method}")
		self.get_logger().info("=" * 60)


def main(args=None):
	"""Main entry point."""
	rclpy.init(args=args)
	
	node = DataloaderNode()
	
	# Initialize node
	if not node.initialize():
		node.get_logger().error("Initialization failed")
		rclpy.shutdown()
		return 1
	
	# Start processing if auto_start is enabled
	if node.get_parameter('auto_start').value:
		node.start()
	
	# Check if auto-shutdown is enabled
	auto_shutdown = node.get_parameter('auto_shutdown').value
	
	try:
		if auto_shutdown:
			# Monitor service and shutdown when complete
			import threading
			import time
			
			shutdown_initiated = False
			
			def monitor_completion():
				nonlocal shutdown_initiated
				while node.service.is_running:
					time.sleep(1.0)
				# Service finished, initiate shutdown
				node.get_logger().info("Processing complete, shutting down...")
				shutdown_initiated = True
				rclpy.shutdown()
			
			monitor_thread = threading.Thread(target=monitor_completion, daemon=True)
			monitor_thread.start()
		
		# Spin to keep node alive
		try:
			rclpy.spin(node)
		except RuntimeError:
			# ROS2 shutdown was called, exit cleanly
			pass
	except KeyboardInterrupt:
		node.get_logger().info("Keyboard interrupt received")
	finally:
		node.stop()
		try:
			node.destroy_node()
		except:
			pass
		# Only shutdown if not already done by monitor thread
		if not (auto_shutdown and 'shutdown_initiated' in locals() and shutdown_initiated):
			try:
				rclpy.shutdown()
			except:
				pass
	
	return 0


if __name__ == '__main__':
	sys.exit(main())