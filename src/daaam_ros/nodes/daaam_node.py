#!/usr/bin/env python3
# python 3.10
import rclpy # type: ignore
from rclpy.node import Node # type: ignore
from cv_bridge import CvBridge  # type: ignore
from sensor_msgs.msg import Image, CameraInfo  # type: ignore
from std_msgs.msg import String  # type: ignore
from hydra_msgs.msg import DsgUpdate  # type: ignore
import message_filters  # type: ignore
import threading
from spark_dsg import DsgLayers
from functools import partial
import cv2
import numpy as np
import multiprocessing as mp
import queue
from collections import deque
import json 
import os 
import yaml 
from pathlib import Path 
from datetime import datetime 
import traceback
import sys
from concurrent.futures import ThreadPoolExecutor
import time
from tf2_ros import TransformListener, Buffer
from typing import Tuple, Optional
import signal
import subprocess


from daaam.pipeline import PipelineOrchestrator, PipelineConfig
from daaam.pipeline.models import Frame
from daaam.utils.logging import setup_main_logging
from daaam.utils.performance import performance_measure, time_execution_sync 
from daaam.utils.vision import BoundingBox
from daaam.utils.transform import compute_ema_velocity
from daaam import ROOT_DIR

class DaaamNode(Node):
	"""
	ROS2 node using modular pipeline architecture.
	In/Out topic names handled via ROS2 remaps in launch file.:
	remap:
		- {from: '~/rgb_image', to: $(var rgb_topic)}
		- {from: '~/depth_image', to: $(var depth_topic)}
		- {from: '~/camera_info', to: $(var camera_info_topic)}
		- {from: '~/dsg_updates', to: $(var dsg_update_topic)}
		- {from: '~/label_image', to: $(var segmentation_topic)}
		- {from: '~/colored_label_image', to: $(var segmentation_color_topic)}

	"""

	def __init__(self) -> None:
		super().__init__("daaam_node")

		self._setup_logging()
		self.logger.info("Starting MMLLM Grounded SAM Node")

		# CUDA diagnostics
		import torch
		cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')
		self.logger.info(f"CUDA_VISIBLE_DEVICES={cvd}")
		if torch.cuda.is_available():
			self.logger.info(f"torch.cuda.device_count={torch.cuda.device_count()}, device_name(0)={torch.cuda.get_device_name(0)}")

		# ros 
		self._declare_parameters()
		self._get_parameters()

		# load pipeline self.config, override with ROS parameters
		self._load_pipeline_config()

		# Initialize pipeline orchestrator with self.config
		self._initialize_pipeline()

		self._initialize_ros_components()
		
		# For velocity computation using EMA
		self.transforms_history = deque(maxlen=10)
		self.timestamps_history = deque(maxlen=10)

		self._start_pipeline()

		self.logger.info(f"MMLLM Grounded SAM Node initialized successfully with config:\n{self.config}")

	def _setup_logging(self) -> None:
		"""Setup unified logging system."""
		output_dir = Path(ROOT_DIR) / "output" / "logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
		
		# Control console log level (all messages still saved to files)
		# Options: logging.DEBUG, logging.INFO (default), logging.WARNING, logging.ERROR
		# Example: To only see warnings and errors on console:
		# self.logging_manager = setup_main_logging(output_dir, console_level=logging.WARNING)
		self.logging_manager = setup_main_logging(output_dir)
		
		self.logger = self.logging_manager.get_logger("daaam_node")
		self.log_dir = str(output_dir)  # Store for passing to config

	def _declare_parameters(self) -> None:
		"""Declare ROS parameters."""
		# configs
		self.declare_parameter("pipeline_config", "config/pipeline_config.yaml")
		self.declare_parameter("semantic_config", "config/labels_pseudo.yaml")
		self.declare_parameter("labelspace_colors", "config/labels_pseudo.csv")
		
		# model params
		self.declare_parameter("sam_model", "fastsam/FastSAM-s.pt")
		self.declare_parameter("sam_model_config_path", "fastsam/fastsam_config.yaml")
		# Use string parameter for sam_imgsz, will parse it to list
		self.declare_parameter("sam_imgsz", "")  # Empty string means None/auto, e.g., "1248,1024" for TensorRT (height,width)
		self.declare_parameter("sentence_embedding_model", "sentence-transformers/sentence-t5-large")

		# processing parameters
		self.declare_parameter("query_interval_frames", 60)
		self.declare_parameter("num_assignment_workers", 1)
		self.declare_parameter("num_grounding_workers", 4)
		self.declare_parameter("assignment_worker", "min_frames_max_size")  # Options: "min_frames", "min_frames_max_size"
		self.declare_parameter("grounding_worker", "dam_multi_image")  # Options: "dam_multi_image"
		self.declare_parameter("min_mask_region_area", 300)
		self.declare_parameter("max_mask_region_area", 640*480)
		self.declare_parameter("polygon_epsilon_factor", 0.001)
		
		# worker-specific parameters (commonly overridden)
		self.declare_parameter("agent_model_name", "gpt-4.1-mini")
		self.declare_parameter("min_obs_per_track", 6)
		self.declare_parameter("save_debug_data", False)
		self.declare_parameter("multi_image_min_n_masks", 32)
		self.declare_parameter("dataset_name", "unknown")
		self.declare_parameter("compute_full_image_description", False)
		self.declare_parameter("save_grounding_images", False)
		self.declare_parameter("save_plain_grounding_images", False)
		self.declare_parameter("save_object_images", False)
		self.declare_parameter("selectframe_clip_backend", "openclip")  # "openclip" or "pe" (Perception Encoder)
		self.declare_parameter("selectframe_clip_model_name", "ViT-L-14")  # Model name for CLIP backend
		self.declare_parameter("cuda_device", "")  # CUDA device for grounding workers ("" = inherit env, "0", "1", etc.)

		# depth filtering
		self.declare_parameter("depth_scale", 1.0) # used within node to convert depth values to m
		self.declare_parameter("depth_lb", 0.25)
		self.declare_parameter("depth_ub", 5.0)
		self.declare_parameter("sync_tolerance", 0.033)  # sync tolerance in seconds (default 33ms for 30Hz cameras)

		# tf2
		self.declare_parameter("world_frame", "map")  # map
		self.declare_parameter("camera_frame", "dominic/forward_link")  # left_cam
		self.declare_parameter("tf_timeout", 0.01) 
		self.declare_parameter("tf_update_rate", 30.0)  # Hz
		
		# scene graph processing
		self.declare_parameter("defer_dsg_processing", False)

		# tracking parameters
		self.declare_parameter("reid_weights", "checkpoints/reid_weights/clip_general.engine")
		self.declare_parameter("with_reid", True)
		self.declare_parameter("reid_half", False)  # FP16 for ReID (False for CLIP models)

		# debug and output
		self.declare_parameter("enable_debug_output", True)
		self.declare_parameter("output_dir", "output")

	def _get_parameters(self) -> None:
		"""Get parameter values."""
		self.pipeline_config_path = self.get_parameter("pipeline_config").get_parameter_value().string_value
		self.semantic_config_path = self.get_parameter("semantic_config").get_parameter_value().string_value
		self.labelspace_colors_path = self.get_parameter("labelspace_colors").get_parameter_value().string_value
		
		# processing parameters
		self.sam_model = self.get_parameter("sam_model").get_parameter_value().string_value
		self.sam_model_config_path = self.get_parameter("sam_model_config_path").get_parameter_value().string_value
		# Parse sam_imgsz from string (e.g., "1024,1248" -> [1024, 1248])
		sam_imgsz_str = self.get_parameter("sam_imgsz").get_parameter_value().string_value
		if sam_imgsz_str:
			try:
				self.sam_imgsz = [int(x.strip()) for x in sam_imgsz_str.split(',')]
			except ValueError:
				self.logger.warning(f"Invalid sam_imgsz format: {sam_imgsz_str}. Expected 'height,width' (e.g., '1248,1024')")
				self.sam_imgsz = []
		else:
			self.sam_imgsz = []
		self.sentence_embedding_model = self.get_parameter("sentence_embedding_model").get_parameter_value().string_value
		self.query_interval_frames = self.get_parameter("query_interval_frames").get_parameter_value().integer_value
		self.num_assignment_workers = self.get_parameter("num_assignment_workers").get_parameter_value().integer_value
		self.num_grounding_workers = self.get_parameter("num_grounding_workers").get_parameter_value().integer_value
		self.assignment_worker = self.get_parameter("assignment_worker").get_parameter_value().string_value
		self.grounding_worker = self.get_parameter("grounding_worker").get_parameter_value().string_value
		self.min_mask_region_area = self.get_parameter("min_mask_region_area").get_parameter_value().integer_value
		self.max_mask_region_area = self.get_parameter("max_mask_region_area").get_parameter_value().integer_value
		self.polygon_epsilon_factor = self.get_parameter("polygon_epsilon_factor").get_parameter_value().double_value
		
		# worker-specific parameters
		self.agent_model_name = self.get_parameter("agent_model_name").get_parameter_value().string_value
		self.min_obs_per_track = self.get_parameter("min_obs_per_track").get_parameter_value().integer_value
		self.save_debug_data = self.get_parameter("save_debug_data").get_parameter_value().bool_value
		self.multi_image_min_n_masks = self.get_parameter("multi_image_min_n_masks").get_parameter_value().integer_value
		self.dataset_name = self.get_parameter("dataset_name").get_parameter_value().string_value
		self.compute_full_image_description = self.get_parameter("compute_full_image_description").get_parameter_value().bool_value
		self.save_grounding_images = self.get_parameter("save_grounding_images").get_parameter_value().bool_value
		self.save_plain_grounding_images = self.get_parameter("save_plain_grounding_images").get_parameter_value().bool_value
		self.save_object_images = self.get_parameter("save_object_images").get_parameter_value().bool_value
		self.selectframe_clip_backend = self.get_parameter("selectframe_clip_backend").get_parameter_value().string_value
		self.selectframe_clip_model_name = self.get_parameter("selectframe_clip_model_name").get_parameter_value().string_value
		cuda_device_str = self.get_parameter("cuda_device").get_parameter_value().string_value
		self.cuda_device = cuda_device_str if cuda_device_str else None

		# depth parameters
		self.depth_scale = self.get_parameter("depth_scale").get_parameter_value().double_value # used within node to convert depth values to m
		self.depth_lb = self.get_parameter("depth_lb").get_parameter_value().double_value
		self.depth_ub = self.get_parameter("depth_ub").get_parameter_value().double_value
		self.sync_tolerance = self.get_parameter("sync_tolerance").get_parameter_value().double_value
		
		# scene graph processing
		self.defer_dsg_processing = self.get_parameter("defer_dsg_processing").get_parameter_value().bool_value

		# tracking parameters
		self.reid_weights = self.get_parameter("reid_weights").get_parameter_value().string_value
		self.with_reid = self.get_parameter("with_reid").get_parameter_value().bool_value
		self.reid_half = self.get_parameter("reid_half").get_parameter_value().bool_value

		# debug
		self.enable_debug_output = self.get_parameter("enable_debug_output").get_parameter_value().bool_value
		self.output_dir = Path(self.get_parameter("output_dir").get_parameter_value().string_value)

	def _load_pipeline_config(self) -> None:
		"""Load and customize pipeline configuration."""
		try:
			# config file if it exists, otherwise create from parameters
			config_path = Path(ROOT_DIR) / self.pipeline_config_path
			
			if config_path.exists():
				self.logger.info(f"Loading pipeline config from {config_path}")
				self.config = PipelineConfig.from_yaml(str(config_path))
			else:
				self.logger.warning(f"Pipeline config file not found: {config_path}, creating from parameters")
				self.config = self._create_config_from_parameters()
			
			# override with ROS parameters
			self._override_config_with_parameters()

		except Exception as e:
			self.logger.error(f"Failed to load pipeline config: {e}")
			self.logger.info("Creating default configuration from ROS parameters")
			self.config = self._create_config_from_parameters()

	def _create_config_from_parameters(self) -> PipelineConfig:
		"""Create pipeline configuration from ROS parameters."""
		from daaam.config import (
			SegmentationConfig, TrackingConfig, GroundingConfig, 
			WorkerConfig, DepthConfig, SceneGraphConfig
		)
		
		return PipelineConfig(
			segmentation=SegmentationConfig(
				model_name=self.sam_model,
				model_config_path=self.sam_model_config_path,
				min_mask_region_area=self.min_mask_region_area,
				polygon_epsilon_factor=self.polygon_epsilon_factor,
				imgsz=tuple(self.sam_imgsz) if self.sam_imgsz and len(self.sam_imgsz) == 2 else None
			),
			tracking=TrackingConfig(
				reid_weights=self.reid_weights,
				with_reid=self.with_reid,
				reid_half=self.reid_half,
			),
			grounding=GroundingConfig(
				agent_model_name=self.agent_model_name,
				query_interval_frames=self.query_interval_frames,
				sentence_embedding_model=self.sentence_embedding_model,
			),
			workers=WorkerConfig(
				num_assignment_workers=self.num_assignment_workers,
				num_grounding_workers=self.num_grounding_workers,
				assignment_worker=self.assignment_worker,
				grounding_worker=self.grounding_worker
			),
			depth=DepthConfig(
				depth_lb=self.depth_lb,
				depth_ub=self.depth_ub
			),
			scene_graph=SceneGraphConfig(
				defer_dsg_processing=self.defer_dsg_processing
			),
			semantic_config_path=self.semantic_config_path,
			labelspace_colors_path=self.labelspace_colors_path,
			output_dir=str(self.output_dir)
		)

	def _override_config_with_parameters(self) -> None:
		"""Override configuration with ROS parameters."""
		self.config.segmentation.model_name = self.sam_model
		self.config.segmentation.model_config_path = self.sam_model_config_path
		self.config.segmentation.min_mask_region_area = self.min_mask_region_area
		self.config.segmentation.polygon_epsilon_factor = self.polygon_epsilon_factor
		# Set imgsz if provided (for TensorRT models)
		if self.sam_imgsz and len(self.sam_imgsz) == 2:
			self.config.segmentation.imgsz = tuple(self.sam_imgsz)
		self.config.grounding.agent_model_name = self.agent_model_name
		self.config.grounding.query_interval_frames = self.query_interval_frames

		self.config.grounding.sentence_embedding_model = self.sentence_embedding_model
		self.config.workers.dam_grounding_config.sentence_embedding_model_name = self.sentence_embedding_model
		self.config.workers.dam_grounding_config.compute_full_image_description = self.compute_full_image_description

		self.config.workers.num_assignment_workers = self.num_assignment_workers
		self.config.workers.num_grounding_workers = self.num_grounding_workers
		self.config.workers.assignment_worker = self.assignment_worker
		self.config.workers.grounding_worker = self.grounding_worker
		self.config.depth.depth_lb = self.depth_lb
		self.config.depth.depth_ub = self.depth_ub
		self.config.semantic_config_path = self.semantic_config_path
		self.config.labelspace_colors_path = self.labelspace_colors_path
		self.config.output_dir = str(self.output_dir)
		
		self.config.log_dir = self.log_dir
		
		# worker-specific parameters if provided
		self.config.workers.assignment_config.min_obs_per_track = self.min_obs_per_track
		self.config.workers.assignment_config.min_mask_region_area = self.min_mask_region_area
		self.config.workers.assignment_config.max_mask_region_area = self.max_mask_region_area
		
		self.config.workers.dam_grounding_config.multi_image_min_n_masks = self.multi_image_min_n_masks
		self.config.workers.dam_grounding_config.save_grounding_images = self.save_grounding_images
		self.config.workers.dam_grounding_config.save_plain_grounding_images = self.save_plain_grounding_images
		self.config.workers.dam_grounding_config.save_object_images = self.save_object_images
		self.config.workers.dam_grounding_config.selectframe_clip_backend = self.selectframe_clip_backend
		self.config.workers.dam_grounding_config.selectframe_clip_model_name = self.selectframe_clip_model_name

		# CUDA device for grounding workers
		if self.cuda_device is not None:
			self.config.grounding.cuda_device = self.cuda_device
			self.logger.info(f"Set cuda_device to {self.cuda_device}")

		self.logger.info(f"Set multi_image_min_n_masks to {self.multi_image_min_n_masks}")
		self.logger.info(f"Set save_grounding_images to {self.save_grounding_images}")
		self.logger.info(f"Set save_plain_grounding_images to {self.save_plain_grounding_images}")
		self.logger.info(f"Set save_object_images to {self.save_object_images}")
		self.logger.info(f"Set selectframe_clip_backend to {self.selectframe_clip_backend}")
		self.logger.info(f"Set selectframe_clip_model_name to {self.selectframe_clip_model_name}")

		self.config.scene_graph.defer_dsg_processing = self.defer_dsg_processing
		self.logger.info(f"Set defer_dsg_processing to {self.defer_dsg_processing}")

		# Tracking config
		self.config.tracking.reid_weights = self.reid_weights
		self.config.tracking.with_reid = self.with_reid
		self.config.tracking.reid_half = self.reid_half
		self.logger.info(f"Set reid_weights to {self.reid_weights}, with_reid to {self.with_reid}, reid_half to {self.reid_half}")

	def _initialize_pipeline(self) -> None:
		"""Initialize the pipeline orchestrator."""
		try:
			self.orchestrator = PipelineOrchestrator(
				config=self.config,
				logger=self.logger
			)
			self.logger.info("Pipeline orchestrator initialized")
		except Exception as e:
			self.logger.error(f"Failed to initialize pipeline orchestrator: {e}")
			raise

	def _initialize_ros_components(self) -> None:
		"""Initialize ROS publishers, subscribers, and other components."""
		# msg -> img
		self.bridge = CvBridge()
		
		# state vars 
		self.latest_camera_info = None
		self.processing_lock = threading.Lock()
		
		# create message_filters subscribers for synchronization
		self.rgb_subscriber = message_filters.Subscriber(
			self,
			Image,
			"~/rgb_image"
		)
		
		self.depth_subscriber = message_filters.Subscriber(
			self,
			Image,
			"~/depth_image"
		)
		
		# regular subscriber for camera info (doesn't need sync)
		self.camera_info_subscriber = self.create_subscription(
			CameraInfo,
			"~/camera_info",
			self.camera_info_callback,
			10
		)
		
		# synchronize RGB and depth images
		self.synchronizer = message_filters.ApproximateTimeSynchronizer(
			[self.rgb_subscriber, self.depth_subscriber],
			queue_size=10,
			slop=self.sync_tolerance
		)
		self.synchronizer.registerCallback(self.synchronized_callback)
		
		self.logger.info(f"Configured RGB-depth synchronization with tolerance: {self.sync_tolerance}s")
		
		self.dsg_subscriber = self.create_subscription(
			DsgUpdate,
			"~/dsg_updates",
			self.dsg_callback,
			10
		)
		
		# publishers (segmentation and colored segmentation in main thread (image cb), 
		# corrected DSG async (every 1s))
		self.segmentation_publisher = self.create_publisher(
			Image,
			"~/label_image",
			10
		)
		
		self.segmentation_color_publisher = self.create_publisher(
			Image,
			"~/colored_label_image",
			10
		)
		
		self.corrected_dsg_publisher = self.create_publisher(
			DsgUpdate,
			"~/corrected_dsg",
			10
		)
		
		# health monitoring timer (grounding and assignment worker health)
		self.health_timer = self.create_timer(10.0, self.health_check_callback)
		
		# Timer for asynchronous DSG publishing (1 Hz is sufficient)
		self.dsg_publish_timer = self.create_timer(1.0, self._publish_corrected_dsg_async)

		# Publisher for semantic updates
		self.semantic_updates_publisher = self.create_publisher(
			String,
			"~/semantic_updates",
			200
		)

		# Register semantic update callback with orchestrator
		if hasattr(self.orchestrator, 'set_semantic_update_callback'):
			self.orchestrator.set_semantic_update_callback(self._publish_semantic_update)
			self.logger.info("Registered semantic update callback with orchestrator")
		
		# scene graph service async processing. DSG update_from_binary is expensive, so
		# we process it asynchronously. Gets set to sync upon shutdown.
		if hasattr(self.orchestrator, 'scene_graph_service'):
			self.orchestrator.scene_graph_service.enable_async_processing()
			self.logger.info("Enabled async DSG processing in scene graph service")


		# TF2 
		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)
		self.world_frame = self.get_parameter("world_frame").value
		self.camera_frame = self.get_parameter("camera_frame").value
		self.tf_timeout = self.get_parameter("tf_timeout").value
		self.logger.info(f"tf2 enabled: {self.world_frame} -> {self.camera_frame}")

		
		self.logger.info("ROS components initialized")

	def _start_pipeline(self) -> None:
		"""Start the pipeline orchestrator."""
		try:
			self.orchestrator.start()
		except Exception as e:
			self.logger.error(f"Failed to start pipeline orchestrator: {e}")
			raise

	def synchronized_callback(self, rgb_msg: Image, depth_msg: Image) -> None:
		"""Process synchronized RGB and depth image pairs."""
		try:
			# timestamp
			timestamp: float = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9
			
			# tf2 transform (optional - can fail)
			transform, success = self._get_camera_transform_tf2(rgb_msg.header.stamp)
			if not success:
				self.logger.debug("tf2 failed, continuing without transform (odometry frame unavailable)")
				transform = None  # Will proceed without transform
			
			# Compute velocities if transform available
			if transform is not None:
				transform = np.array(transform)  # Ensure it's a numpy array
				lin_vel, ang_vel = self._compute_frame_velocity(transform, timestamp)
			else:
				self.logger.warning("No transform available, setting velocities to zero")
				lin_vel, ang_vel = np.zeros(3), np.zeros(3)

			# RGB image
			cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
			
			# depth image
			depth_img = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
			depth_img = depth_img / self.depth_scale  # scale to meters

			# Extract camera intrinsics if available
			camera_intrinsics = None
			if self.latest_camera_info is not None:
				K = self.latest_camera_info.k  # 3x3 camera matrix as flat array
				camera_intrinsics = {
					'fx': K[0], 'fy': K[4],
					'cx': K[2], 'cy': K[5]
				}

			# Create Frame object with all available data including velocities
			frame = Frame(
				frame_id=0,  # Will be updated by orchestrator
				timestamp=timestamp,
				rgb_image=cv_image,
				depth_image=depth_img,
				transform=transform,
				lin_vel=lin_vel,
				ang_vel=ang_vel,
				camera_intrinsics=camera_intrinsics
			)
			
			# Process frame through pipeline (include in performance tracking)
			tracker = self.orchestrator.performance_tracker if hasattr(self.orchestrator, 'performance_tracker') else None
			with performance_measure("process_frame", self.logger.info, tracker):
				label_image, color_image = self.orchestrator.process_frame(frame)

			self._publish_segmentation_results(label_image, color_image, rgb_msg.header)

			self.logger.debug(f"Processed synchronized frame with {len(frame.tracks)} tracks")

		except Exception as e:
			self.logger.error(f"Error in synchronized callback: {e}")
			traceback.print_exc()

	def camera_info_callback(self, msg: CameraInfo) -> None:
		"""Handle camera info updates."""
		self.latest_camera_info = msg

	def dsg_callback(self, msg: DsgUpdate) -> None:
		"""Process DSG updates using scene graph service async interface."""
		try:
			if hasattr(self.orchestrator, 'scene_graph_service'):
				# async
				success = self.orchestrator.scene_graph_service.update_scene_graph_async(
					bytes(msg.layer_contents),
					msg.full_update,
					msg.deleted_nodes,
					msg.header
				)
				if not success:
					self.logger.warning("Failed to queue DSG update")
			else:
				self.logger.warning("Scene graph service not available")
				
		except Exception as e:
			self.logger.error(f"Error in DSG callback: {e}")
			traceback.print_exc()

	def _publish_segmentation_results(
		self, 
		label_image: np.ndarray, 
		color_image: np.ndarray, 
		header
	) -> None:
		"""Publish segmentation results.
		publishes both label image (16UC1) and color image (rgb8).
		"""
		try:
			# label image
			label_msg = self.bridge.cv2_to_imgmsg(label_image.astype(np.uint16), "16UC1")
			label_msg.header = header
			self.segmentation_publisher.publish(label_msg)
			
			# color image
			color_msg = self.bridge.cv2_to_imgmsg(color_image, "rgb8")
			color_msg.header = header
			self.segmentation_color_publisher.publish(color_msg)
			
		except Exception as e:
			self.logger.error(f"Error publishing segmentation results: {e}")

	def health_check_callback(self) -> None:
		"""Periodic health check of the pipeline."""
		try:
			health_status = self.orchestrator.get_health_status()
			
			# log health information
			self.logger.info(f"Pipeline health: {health_status['orchestrator']['frame_count']} frames processed")
			
			# check if workers are alive
			for service_name, service_health in health_status.items():
				if service_name == "orchestrator":
					continue
				
				if isinstance(service_health, dict) and "workers" in service_health:
					for worker in service_health["workers"]:
						if not worker.get("is_alive", True):
							self.logger.warning(f"Dead worker detected: {worker}")
			
		except Exception as e:
			self.logger.error(f"Error in health check: {e}")

	def _publish_semantic_update(self, update) -> None:
		"""Publish semantic update as JSON string."""
		try:
			# Convert update to JSON string
			update_json = update.model_dump_json()

			# Create and publish message
			msg = String()
			msg.data = update_json
			self.semantic_updates_publisher.publish(msg)

			self.logger.debug(f"Published semantic update for {len(update.semantic_labels)} labels")

		except Exception as e:
			self.logger.error(f"Error publishing semantic update: {e}")

	def _publish_corrected_dsg_async(self) -> None:
		"""Publish the latest corrected DSG at 1Hz."""
		try:
			if not hasattr(self.orchestrator, 'scene_graph_service'):
				return
			
			latest_dsg = self.orchestrator.scene_graph_service.get_latest_corrected_dsg()
			if latest_dsg is None:
				return
			
			correction_msg = DsgUpdate()
			correction_msg.header = latest_dsg['header']
			correction_msg.layer_contents = latest_dsg['binary']
			correction_msg.full_update = True
			correction_msg.deleted_nodes = latest_dsg['deleted_nodes']
			
			self.corrected_dsg_publisher.publish(correction_msg)
			self.logger.debug("Published corrected DSG asynchronously")
			
		except Exception as e:
			self.logger.error(f"Error publishing corrected DSG: {e}")
	
	def destroy_node(self) -> None:
		"""Cleanup when node is destroyed."""
		print("[Shutdown] Shutting down MMLLM Grounded SAM Node")

		try:
			# Publish final semantic update before stopping
			# if hasattr(self, 'orchestrator') and hasattr(self, 'semantic_updates_publisher'):
				# try:
				# 	# Get final semantic update from scene graph service
				# 	scene_graph_service = self.orchestrator.scene_graph_service
				# 	if scene_graph_service:
				# 		final_update = scene_graph_service.create_final_semantic_update()
				# 		if final_update:
				# 			self._publish_semantic_update(final_update)
				# 			print(f"[Shutdown] Published final semantic update with {len(final_update.semantic_labels)} labels")
				# 		else:
				# 			print("[Shutdown] No semantic labels to publish in final update")
				# except Exception as e:
				# 	print(f"[Shutdown] Error publishing final semantic update: {e}")

			# stop pipeline orchestrator


			if hasattr(self, 'orchestrator'):
				print("[Shutdown] Stopping orchestrator...")
				self.orchestrator.stop()  # Exports performance statistics and stops workers
				print("[Shutdown] Orchestrator stopped")

			import time
			time.sleep(1.0)  # Grace period for final log writes

			if hasattr(self, 'logging_manager'):
				print("[Shutdown] Flushing log buffers...")
				self.logging_manager.flush_all_handlers()
				print("[Shutdown] Log buffers flushed")

			if hasattr(self, 'logging_manager'):
				print("[Shutdown] Stopping logging system...")
				self.logging_manager.stop()
				print("[Shutdown] Logging system stopped")

		except Exception as e:
			print(f"[Shutdown] Error during shutdown: {e}")
			import traceback
			traceback.print_exc()

		super().destroy_node()

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
			
			self.logger.debug(f"tf2: Got transform {self.world_frame} -> {self.camera_frame}")
			self.logger.debug(f"  Translation: [{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}]")
			self.logger.debug(f"  Rotation (xyzw): [{quat.x:.3f}, {quat.y:.3f}, {quat.z:.3f}, {quat.w:.3f}]")

			tf_ = [pos.x, pos.y, pos.z, quat.x, quat.y, quat.z, quat.w]

			return tf_, True

		except Exception as e:
			# if exact timestamp fails, try with latest available
			try:
				self.logger.debug(f"tf2 exact timestamp failed, trying latest: {e}")
				transform = self.tf_buffer.lookup_transform(
					self.world_frame,
					self.camera_frame,
					rclpy.time.Time(), 
					timeout=rclpy.duration.Duration(seconds=self.tf_timeout)
				)
				
				pos = transform.transform.translation
				quat = transform.transform.rotation
				
				self.logger.debug(f"tf2: Got latest transform {self.world_frame} -> {self.camera_frame}")
				self.logger.debug(f"  Translation: [{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}]")
				self.logger.debug(f"  Rotation (xyzw): [{quat.x:.3f}, {quat.y:.3f}, {quat.z:.3f}, {quat.w:.3f}]")
				
				tf_ = [pos.x, pos.y, pos.z, quat.x, quat.y, quat.z, quat.w]
				return tf_, True

			except Exception as e2:
				self.logger.warning(f"tf2 transform lookup failed (both exact and latest): {e2}")
				return None, False
		except Exception as e:
			self.logger.warning(f"tf2 transform lookup failed: {e}")
			return None, False
	
	def _compute_frame_velocity(self, transform: np.ndarray, timestamp: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
		"""Compute linear and angular velocities from transform history."""
		# Add current transform and timestamp to history
		self.transforms_history.append(transform)
		self.timestamps_history.append(timestamp)
		
		# Need at least 2 frames to compute velocity
		if len(self.transforms_history) < 2:
			return np.zeros(3), np.zeros(3)
		
		# Convert deques to numpy arrays for velocity computation
		transforms = np.array(list(self.transforms_history))
		timestamps = np.array(list(self.timestamps_history))
		
		# Compute velocity using EMA
		velocity = compute_ema_velocity(transforms, timestamps, alpha=0.4)
		
		if velocity is not None:
			lin_vel = velocity[1:4]  # [vx, vy, vz]
			ang_vel = velocity[4:7]  # [wx, wy, wz]
			self.logger.debug(f"Computed velocities - Linear: {lin_vel}, Angular: {ang_vel}")
			return lin_vel, ang_vel

		self.logger.debug("Insufficient data for velocity computation, returning zeros")
		return np.zeros(3), np.zeros(3)

def _shutdown_watchdog(timeout: float) -> None:
	"""Force process exit if shutdown exceeds timeout."""
	time.sleep(timeout)
	print(f"[WATCHDOG] Shutdown exceeded {timeout}s — forcing exit", file=sys.stderr)
	os._exit(1)

def main(args=None):
	"""Main function for the ROS2 node."""
	rclpy.init(args=args)

	try:
		node = DaaamNode()
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	except Exception as e:
		print(f"Error starting node: {e}")
		traceback.print_exc()
	finally:
		# Start watchdog — guarantees process exit even if shutdown path blocks
		watchdog = threading.Thread(target=_shutdown_watchdog, args=(55.0,), daemon=True)
		watchdog.start()

		print("Destroying node...")
		node.destroy_node()
		rclpy.shutdown()


if __name__ == "__main__":
	main()