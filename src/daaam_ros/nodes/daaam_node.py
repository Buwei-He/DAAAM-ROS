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
from concurrent.futures import ThreadPoolExecutor, wait
import time
from tf2_ros import TransformListener, Buffer
from typing import Tuple, Optional
import signal

from daaam.pipeline import PipelineOrchestrator, PipelineConfig
from daaam.pipeline.models import Frame
from daaam.utils.logging import setup_main_logging
from daaam.utils.performance import performance_measure, time_execution_sync 
from daaam.utils.vision import BoundingBox
from daaam.utils.transform import compute_ema_velocity
from daaam import ROOT_DIR
from daaam.human_reason.human_clip_recorder import (
	ClipTrackObservation,
	HumanClipArtifact,
	HumanClipRecorder,
	HumanClipRecorderConfig,
)
from daaam.human_reason import process_clip as _hoi_process_clip
from daaam.human_reason.live_query import LiveQueryBridge
from daaam.human_reason import refresh_event_outputs as _refresh_event_outputs

_HUMAN_REASON_PARAM_DEFAULTS = {
	"save_human_clips": False,
	"human_clip_detector_weights": "yolo11n.pt",
	"human_clip_detector_conf": 0.25,
	"human_clip_detector_device": None,
	"human_clip_min_frames": 4,
	"human_clip_output_fps": 4.0,
	"egocentric": False,
	"enable_cosmos_hoi_processing": False,
	"cosmos_hoi_base_url": "http://localhost:8000/v1",
	"cosmos_hoi_model": "cosmos-reason2",
	"cosmos_hoi_api_key": "EMPTY",
	"cosmos_hoi_media_root": "",
	"cosmos_hoi_fps": 4.0,
	"cosmos_hoi_match_iou_threshold": 0.1,
	"cosmos_hoi_semantic_reranking": False,
	"enable_semantic_event_post_processing": True,
	"semantic_event_output_name": "events_semantic.yaml",
	"semantic_event_neighbor_window_sec": 8.0,
	"semantic_event_confidence_threshold": 0.65,
	"event_grouper_model": "gpt-5.4-mini",
}

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
		self._initialize_optional_recorders()

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
		
		# model params - empty string means "use pipeline_config.yaml value"
		self.declare_parameter("sam_model", "")
		self.declare_parameter("sam_model_config_path", "")
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
		self.declare_parameter("output_run_prefix", "")
		self.declare_parameter("save_human_clips", False)
		self.declare_parameter("human_clip_detector_weights", "yolo11n.pt")
		self.declare_parameter("human_clip_detector_conf", 0.25)
		self.declare_parameter("human_clip_detector_device", "")
		self.declare_parameter("human_clip_min_frames", 4)
		self.declare_parameter("human_clip_output_fps", 4.0)
		# The one clip length. Clips are cut here with the person still in frame
		# and reopen on the next one, so a clip is exactly what Pass 1 sees —
		# there is no separate "chunk" any more. 8s matches the June egg runs.
		# 0.0 leaves clips unbounded, which is what the paper recorder did.
		self.declare_parameter("human_clip_max_clip_sec", 8.0)
		self.declare_parameter("egocentric", False)
		self.declare_parameter("enable_cosmos_hoi_processing", False)
		self.declare_parameter("cosmos_hoi_base_url", "http://localhost:8000/v1")
		self.declare_parameter("cosmos_hoi_model", "cosmos-reason2")
		self.declare_parameter("cosmos_hoi_api_key", "EMPTY")
		self.declare_parameter("cosmos_hoi_media_root", "")
		self.declare_parameter("cosmos_hoi_fps", 4.0)
		self.declare_parameter("cosmos_hoi_match_iou_threshold", 0.1)
		self.declare_parameter("cosmos_hoi_semantic_reranking", False)
		self.declare_parameter("enable_semantic_event_post_processing", True)
		self.declare_parameter("semantic_event_output_name", "events_semantic.yaml")
		self.declare_parameter("semantic_event_neighbor_window_sec", 8.0)
		self.declare_parameter("semantic_event_confidence_threshold", 0.65)
		self.declare_parameter("event_grouper_model", "gpt-5.4-mini")

		# live HOI query (demo): reason on a rolling recent-history buffer on demand,
		# independent of save_human_clips/enable_cosmos_hoi_processing's finalize-on-
		# absence lifecycle. Off by default; does not affect the dataset workflows.
		self.declare_parameter("enable_live_hoi_query", False)
		# Whole multiple of the clip length: 5 x 8s. See the check below.
		self.declare_parameter("live_buffer_sec", 40.0)
		self.declare_parameter("live_snapshot_keep", 5)
		self.declare_parameter("live_bridge_host", "0.0.0.0")
		self.declare_parameter("live_bridge_port", 8100)
		# Must stay below the agent's own HTTP timeout (ToolConfig.live_bridge_timeout_sec,
		# 120s) so a slow Cosmos call fails here rather than being abandoned by the
		# caller while it keeps holding one of the two concurrency slots.
		self.declare_parameter("live_query_timeout_sec", 45.0)

		# Periodic mid-run rebuild of events.yaml / events_semantic.yaml, which are
		# otherwise shutdown-only artifacts. The interval arms the refresh; the next
		# DSG update fires it. Off by default — dataset runs keep shutdown-only behaviour.
		self.declare_parameter("enable_live_event_refresh", False)
		self.declare_parameter("event_refresh_interval_sec", 120.0)

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
		self.output_run_prefix = self.get_parameter("output_run_prefix").get_parameter_value().string_value
		self.save_human_clips = self.get_parameter("save_human_clips").get_parameter_value().bool_value
		self.human_clip_detector_weights = self.get_parameter("human_clip_detector_weights").get_parameter_value().string_value
		self.human_clip_detector_conf = self.get_parameter("human_clip_detector_conf").get_parameter_value().double_value
		human_clip_detector_device = self.get_parameter("human_clip_detector_device").get_parameter_value().string_value
		self.human_clip_detector_device = human_clip_detector_device if human_clip_detector_device else None
		self.human_clip_min_frames = self.get_parameter("human_clip_min_frames").get_parameter_value().integer_value
		self.human_clip_output_fps = self.get_parameter("human_clip_output_fps").get_parameter_value().double_value
		self.human_clip_max_clip_sec = self.get_parameter(
			"human_clip_max_clip_sec"
		).get_parameter_value().double_value
		self.egocentric = self.get_parameter("egocentric").get_parameter_value().bool_value
		self.enable_cosmos_hoi_processing = self.get_parameter(
			"enable_cosmos_hoi_processing"
		).get_parameter_value().bool_value
		self.cosmos_hoi_base_url = self.get_parameter("cosmos_hoi_base_url").get_parameter_value().string_value
		self.cosmos_hoi_model = self.get_parameter("cosmos_hoi_model").get_parameter_value().string_value
		self.cosmos_hoi_api_key = self.get_parameter("cosmos_hoi_api_key").get_parameter_value().string_value
		if self.enable_cosmos_hoi_processing:
			self.logger.info(f"Cosmos HOI: url={self.cosmos_hoi_base_url} model={self.cosmos_hoi_model}")
		self.cosmos_hoi_media_root = self.get_parameter("cosmos_hoi_media_root").get_parameter_value().string_value
		self.cosmos_hoi_fps = self.get_parameter("cosmos_hoi_fps").get_parameter_value().double_value
		self.cosmos_hoi_match_iou_threshold = self.get_parameter(
			"cosmos_hoi_match_iou_threshold"
		).get_parameter_value().double_value
		self.cosmos_hoi_semantic_reranking = self.get_parameter(
			"cosmos_hoi_semantic_reranking"
		).get_parameter_value().bool_value
		self.enable_semantic_event_post_processing = self.get_parameter(
			"enable_semantic_event_post_processing"
		).get_parameter_value().bool_value
		self.semantic_event_output_name = self.get_parameter(
			"semantic_event_output_name"
		).get_parameter_value().string_value
		self.semantic_event_neighbor_window_sec = self.get_parameter(
			"semantic_event_neighbor_window_sec"
		).get_parameter_value().double_value
		self.semantic_event_confidence_threshold = self.get_parameter(
			"semantic_event_confidence_threshold"
		).get_parameter_value().double_value
		self.event_grouper_model = self.get_parameter("event_grouper_model").get_parameter_value().string_value

		self.enable_live_hoi_query = self.get_parameter("enable_live_hoi_query").get_parameter_value().bool_value
		self.live_buffer_sec = self.get_parameter("live_buffer_sec").get_parameter_value().double_value
		self.live_snapshot_keep = self.get_parameter("live_snapshot_keep").get_parameter_value().integer_value
		self.live_bridge_host = self.get_parameter("live_bridge_host").get_parameter_value().string_value
		self.live_bridge_port = self.get_parameter("live_bridge_port").get_parameter_value().integer_value
		self.live_query_timeout_sec = self.get_parameter(
			"live_query_timeout_sec"
		).get_parameter_value().double_value
		self.enable_live_event_refresh = self.get_parameter(
			"enable_live_event_refresh"
		).get_parameter_value().bool_value
		self.event_refresh_interval_sec = self.get_parameter(
			"event_refresh_interval_sec"
		).get_parameter_value().double_value
		self._last_event_refresh = time.time()
		self._event_refresh_running = False

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
			
			# Use explicit human/HOI config defaults before applying ROS overrides.
			self._apply_human_reason_config_defaults()
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
			WorkerConfig, DepthConfig, SceneGraphConfig,
			HumanReasonConfig, HumanClipConfig, CosmosHOIConfig, SemanticEventConfig,
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
			human_reason=HumanReasonConfig(
				human_clips=HumanClipConfig(
					enabled=self.save_human_clips,
					detector_weights=self.human_clip_detector_weights,
					detector_conf=self.human_clip_detector_conf,
					detector_device=self.human_clip_detector_device,
					min_frames=self.human_clip_min_frames,
					output_fps=self.human_clip_output_fps,
				),
				cosmos_hoi=CosmosHOIConfig(
					enabled=self.enable_cosmos_hoi_processing,
					base_url=self.cosmos_hoi_base_url,
					model=self.cosmos_hoi_model,
					api_key=self.cosmos_hoi_api_key,
					media_root=self.cosmos_hoi_media_root,
					fps=self.cosmos_hoi_fps,
					match_iou_threshold=self.cosmos_hoi_match_iou_threshold,
				),
				semantic_events=SemanticEventConfig(
					enabled=self.enable_semantic_event_post_processing,
					output_name=self.semantic_event_output_name,
					neighbor_window_sec=self.semantic_event_neighbor_window_sec,
					confidence_threshold=self.semantic_event_confidence_threshold,
				),
			),
			semantic_config_path=self.semantic_config_path,
			labelspace_colors_path=self.labelspace_colors_path,
			output_dir=str(self.output_dir),
			output_run_prefix=str(self.output_run_prefix)
		)

	def _use_config_default(self, attr_name: str, config_value) -> None:
		if getattr(self, attr_name) == _HUMAN_REASON_PARAM_DEFAULTS[attr_name]:
			setattr(self, attr_name, config_value)

	def _apply_human_reason_config_defaults(self) -> None:
		"""Use pipeline_config human_reason values unless ROS params override them."""
		human_reason = getattr(self.config, "human_reason", None)
		if human_reason is None:
			return

		self._use_config_default("egocentric", human_reason.egocentric)

		human_clips = human_reason.human_clips
		self._use_config_default("save_human_clips", human_clips.enabled)
		self._use_config_default("human_clip_detector_weights", human_clips.detector_weights)
		self._use_config_default("human_clip_detector_conf", human_clips.detector_conf)
		self._use_config_default("human_clip_detector_device", human_clips.detector_device)
		self._use_config_default("human_clip_min_frames", human_clips.min_frames)
		self._use_config_default("human_clip_output_fps", human_clips.output_fps)

		cosmos_hoi = human_reason.cosmos_hoi
		self._use_config_default("enable_cosmos_hoi_processing", cosmos_hoi.enabled)
		self._use_config_default("cosmos_hoi_base_url", cosmos_hoi.base_url)
		self._use_config_default("cosmos_hoi_model", cosmos_hoi.model)
		self._use_config_default("cosmos_hoi_api_key", cosmos_hoi.api_key)
		self._use_config_default("cosmos_hoi_media_root", cosmos_hoi.media_root)
		self._use_config_default("cosmos_hoi_fps", cosmos_hoi.fps)
		self._use_config_default("cosmos_hoi_match_iou_threshold", cosmos_hoi.match_iou_threshold)
		self._use_config_default("cosmos_hoi_semantic_reranking", cosmos_hoi.semantic_reranking)

		semantic_events = human_reason.semantic_events
		self._use_config_default("enable_semantic_event_post_processing", semantic_events.enabled)
		self._use_config_default("semantic_event_output_name", semantic_events.output_name)
		self._use_config_default("semantic_event_neighbor_window_sec", semantic_events.neighbor_window_sec)
		self._use_config_default("semantic_event_confidence_threshold", semantic_events.confidence_threshold)
		self._use_config_default("event_grouper_model", semantic_events.event_grouper_model)

	def _override_config_with_parameters(self) -> None:
		"""Override configuration with ROS parameters."""
		if self.sam_model:
			self.config.segmentation.model_name = self.sam_model
		if self.sam_model_config_path:
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
		self.config.output_run_prefix = str(self.output_run_prefix)
		
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

		# Human reasoning config
		self.config.human_reason.human_clips.enabled = self.save_human_clips
		self.config.human_reason.human_clips.detector_weights = self.human_clip_detector_weights
		self.config.human_reason.human_clips.detector_conf = self.human_clip_detector_conf
		self.config.human_reason.human_clips.detector_device = self.human_clip_detector_device
		self.config.human_reason.human_clips.min_frames = self.human_clip_min_frames
		self.config.human_reason.human_clips.output_fps = self.human_clip_output_fps
		self.config.human_reason.egocentric = self.egocentric
		self.config.human_reason.cosmos_hoi.enabled = self.enable_cosmos_hoi_processing
		self.config.human_reason.cosmos_hoi.base_url = self.cosmos_hoi_base_url
		self.config.human_reason.cosmos_hoi.model = self.cosmos_hoi_model
		self.config.human_reason.cosmos_hoi.api_key = self.cosmos_hoi_api_key
		self.config.human_reason.cosmos_hoi.media_root = self.cosmos_hoi_media_root
		self.config.human_reason.cosmos_hoi.fps = self.cosmos_hoi_fps
		self.config.human_reason.cosmos_hoi.match_iou_threshold = self.cosmos_hoi_match_iou_threshold
		self.config.human_reason.cosmos_hoi.semantic_reranking = self.cosmos_hoi_semantic_reranking
		self.config.human_reason.semantic_events.enabled = self.enable_semantic_event_post_processing
		self.config.human_reason.semantic_events.output_name = self.semantic_event_output_name
		self.config.human_reason.semantic_events.neighbor_window_sec = self.semantic_event_neighbor_window_sec
		self.config.human_reason.semantic_events.confidence_threshold = self.semantic_event_confidence_threshold

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

	def _initialize_optional_recorders(self) -> None:
		"""Initialize optional video recorders."""
		self.human_clip_recorder = None
		self.human_clip_recording_enabled = (
			self.save_human_clips or self.enable_cosmos_hoi_processing or self.enable_live_hoi_query
		)
		if not self.human_clip_recording_enabled:
			return
		if self.enable_live_hoi_query and not (self.save_human_clips or self.enable_cosmos_hoi_processing):
			self.logger.info("Enabling human clip recording because live HOI query is active")
		elif self.enable_cosmos_hoi_processing and not self.save_human_clips:
			self.logger.info("Enabling human clip recording because Cosmos HOI processing is active")

		recorder_config = HumanClipRecorderConfig(
			enabled=self.human_clip_recording_enabled,
			detector_weights=self.human_clip_detector_weights,
			detector_conf=self.human_clip_detector_conf,
			detector_device=self.human_clip_detector_device,
			min_clip_frames=self.human_clip_min_frames,
			output_fps=self.human_clip_output_fps,
			max_clip_sec=self.human_clip_max_clip_sec,
			egocentric=self.egocentric,
			enable_live_buffer=self.enable_live_hoi_query,
			live_buffer_sec=self.live_buffer_sec,
			live_snapshot_keep=self.live_snapshot_keep,
		)
		# The live buffer is sized in clips, not in loose seconds: at 6s clips and a
		# 30s window it holds exactly the last 5, so "what just happened" and "what
		# Pass 1 saw" cover the same material.
		if self.enable_live_hoi_query and self.human_clip_max_clip_sec > 0:
			clips = self.live_buffer_sec / self.human_clip_max_clip_sec
			if abs(clips - round(clips)) > 1e-6:
				self.logger.warning(
					f"live_buffer_sec={self.live_buffer_sec:.1f} is not a whole number of "
					f"{self.human_clip_max_clip_sec:.1f}s clips ({clips:.2f}); the newest "
					"snapshot will straddle a clip boundary"
				)
			else:
				self.logger.info(
					f"Live buffer holds {round(clips)} x {self.human_clip_max_clip_sec:.0f}s "
					f"clips ({self.live_buffer_sec:.0f}s)"
				)

		self.human_clip_recorder = HumanClipRecorder(
			config=recorder_config,
			output_dir=self.orchestrator.output_dir,
			logger=self.logger,
			on_clip_finalized=self._handle_human_clip_finalized,
		)

	def _initialize_ros_components(self) -> None:
		"""Initialize ROS publishers, subscribers, and other components."""
		# msg -> img
		self.bridge = CvBridge()
		
		# state vars
		self.latest_camera_info = None
		self.processing_lock = threading.Lock()
		# Executor for HOI background tasks. Shutdown gives these tasks a
		# bounded grace period after the core DAAAM outputs are saved.
		from concurrent.futures import ThreadPoolExecutor as _TPE
		self._hoi_executor = _TPE(max_workers=4, thread_name_prefix="cosmos_hoi")
		self._hoi_futures = []
		self._runtime_stats = {
			"start_wall": time.time(),
			"last_health_wall": time.time(),
			"last_health_frame_count": 0,
			"callback_count": 0,
			"callback_wall_ms": deque(maxlen=300),
			"callback_wall_period_ms": deque(maxlen=300),
			"source_period_ms": deque(maxlen=300),
			"source_to_wall_age_sec": deque(maxlen=300),
			"last_callback_start_wall": None,
			"last_source_stamp": None,
		}
		
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

		# live HOI query (demo): HTTP surface over the rolling recent-history buffer
		# and the latest corrected DSG, so an out-of-process scene-understanding
		# agent can reach both without speaking ROS.
		self.live_query_bridge = None
		if self.enable_live_hoi_query:
			self._start_live_query_bridge()

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
		callback_start_wall = time.time()
		timestamp: Optional[float] = None
		try:
			# timestamp
			timestamp = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9
			self._record_callback_start(timestamp, callback_start_wall)
			
			# tf2 transform (optional - can fail)
			transform, success = self._get_camera_transform_tf2(rgb_msg.header.stamp)
			if not success:
				self.logger.debug("tf2 failed, continuing without transform (odometry frame unavailable)")
				transform = None  # Will proceed without transform
			
			# Compute velocities if transform available
			if transform is not None:
				transform = np.array(transform)  # Ensure it's a numpy array
				lin_vel, ang_vel = self._compute_frame_velocity(transform, timestamp)
				self._tf_missing_since = None
			else:
				# Rate-limited: on a robot that never publishes map->camera this
				# fires on every frame, which at camera rate buries every other
				# line in the log — exactly when the log is what you need.
				self._log_missing_transform()
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

			if self.human_clip_recorder is not None:
				try:
					with performance_measure("human_clip_recorder", self.logger.debug, tracker):
						track_observations = self._build_clip_track_observations(frame)
						self.human_clip_recorder.process_frame(
							cv_image,
							timestamp,
							frame.frame_id,
							track_observations,
						)
				except Exception as recorder_error:
					self.logger.error(f"Human clip recording failed on frame {frame.frame_id}: {recorder_error}")

			with performance_measure("publish_segmentation_results", self.logger.debug, tracker):
				self._publish_segmentation_results(label_image, color_image, rgb_msg.header)

			self.logger.debug(f"Processed synchronized frame with {len(frame.tracks)} tracks")

		except Exception as e:
			self.logger.error(f"Error in synchronized callback: {e}")
			traceback.print_exc()
		finally:
			self._record_callback_end(callback_start_wall)

	def _record_callback_start(self, timestamp: float, callback_start_wall: float) -> None:
		stats = getattr(self, "_runtime_stats", None)
		if not stats:
			return
		stats["callback_count"] += 1
		last_wall = stats.get("last_callback_start_wall")
		if last_wall is not None:
			stats["callback_wall_period_ms"].append((callback_start_wall - last_wall) * 1000.0)
		last_source = stats.get("last_source_stamp")
		if last_source is not None and timestamp >= last_source:
			stats["source_period_ms"].append((timestamp - last_source) * 1000.0)
		# With rosbag/sim-time, source stamps may not share wall-clock epoch. Only
		# record source age when it is plausibly a wall-clock timestamp.
		age_sec = callback_start_wall - timestamp
		if -3600.0 <= age_sec <= 3600.0:
			stats["source_to_wall_age_sec"].append(age_sec)
		stats["last_callback_start_wall"] = callback_start_wall
		stats["last_source_stamp"] = timestamp

	def _record_callback_end(self, callback_start_wall: float) -> None:
		stats = getattr(self, "_runtime_stats", None)
		if not stats:
			return
		duration_s = time.time() - callback_start_wall
		stats["callback_wall_ms"].append(duration_s * 1000.0)
		tracker = getattr(getattr(self, "orchestrator", None), "performance_tracker", None)
		if tracker is not None:
			tracker.record("ros_callback_total", int(duration_s * 1_000_000_000))

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
					# Fire a due event refresh here so regrouping runs against a
					# just-updated graph. Returns immediately unless one is due.
					self._maybe_refresh_events()
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
			orch = health_status["orchestrator"]
			queues = health_status.get("queues", {})
			frames = int(orch["frame_count"])
			now = time.time()
			stats = getattr(self, "_runtime_stats", {})
			last_wall = float(stats.get("last_health_wall", now))
			last_frames = int(stats.get("last_health_frame_count", frames))
			dt = max(now - last_wall, 1e-6)
			window_fps = (frames - last_frames) / dt
			stats["last_health_wall"] = now
			stats["last_health_frame_count"] = frames

			callback_summary = self._series_summary_ms(stats.get("callback_wall_ms", []))
			callback_period = self._series_summary_ms(stats.get("callback_wall_period_ms", []))
			source_period = self._series_summary_ms(stats.get("source_period_ms", []))
			age_summary = self._series_summary_sec(stats.get("source_to_wall_age_sec", []))
			hoi_status = self._hoi_status()
			perf = self.orchestrator.performance_tracker.get_statistics()
			process_frame = perf.get("process_frame", {})
			segment_frame = perf.get("segment_frame", {})
			grounding_workers = health_status.get("grounding_service", {}).get("workers", [])
			ready_grounding = sum(1 for worker in grounding_workers if worker.get("is_ready"))
			self.logger.info(
				"Pipeline health: "
				f"frames={frames} window_fps={window_fps:.2f} "
				f"active_tracks={orch['active_tracks']} pending_tracks={orch.get('pending_track_ids')} "
				f"snapshots={orch.get('frame_snapshots')} queues={queues} "
				f"callback_ms={callback_summary} callback_period_ms={callback_period} "
				f"source_period_ms={source_period} source_age_sec={age_summary} "
				f"process_frame_p95_ms={process_frame.get('p95_ms', 0.0):.1f} "
				f"segment_p95_ms={segment_frame.get('p95_ms', 0.0):.1f} "
				f"grounding_ready={ready_grounding}/{len(grounding_workers)} "
				f"cosmos_hoi={hoi_status} "
				f"scene_graph={health_status.get('scene_graph_service', {})}"
			)
			
			# check if workers are alive
			for service_name, service_health in health_status.items():
				if service_name in {"orchestrator", "queues", "scene_graph_service", "worker_health"}:
					continue
				
				if isinstance(service_health, dict) and "workers" in service_health:
					for worker in service_health["workers"]:
						if not worker.get("is_alive", True):
							self.logger.warning(f"Dead worker detected: {worker}")
			
		except Exception as e:
			self.logger.error(f"Error in health check: {e}")

	def _series_summary_ms(self, values) -> str:
		values = list(values or [])
		if not values:
			return "n=0"
		arr = np.asarray(values, dtype=float)
		return (
			f"n={len(arr)} mean={float(np.mean(arr)):.1f} "
			f"p95={float(np.percentile(arr, 95)):.1f} max={float(np.max(arr)):.1f}"
		)

	def _series_summary_sec(self, values) -> str:
		values = list(values or [])
		if not values:
			return "n=0"
		arr = np.asarray(values, dtype=float)
		return (
			f"n={len(arr)} mean={float(np.mean(arr)):.2f} "
			f"p95={float(np.percentile(arr, 95)):.2f} max={float(np.max(arr)):.2f}"
		)

	def _hoi_status(self) -> dict:
		futures = list(getattr(self, "_hoi_futures", []))
		done = sum(1 for future in futures if future.done())
		running = sum(1 for future in futures if future.running())
		cancelled = sum(1 for future in futures if future.cancelled())
		failed = 0
		for future in futures:
			if future.done() and not future.cancelled():
				try:
					if future.exception() is not None:
						failed += 1
				except Exception:
					failed += 1
		return {
			"total": len(futures),
			"done": done,
			"running": running,
			"pending": len(futures) - done - running,
			"cancelled": cancelled,
			"failed": failed,
		}

	def _publish_semantic_update(self, update) -> None:
		"""Publish semantic update as JSON string."""
		# After SIGINT the rclpy context is invalidated before destroy_node runs;
		# corrections regenerated during shutdown would otherwise spam publish errors.
		# The final state is persisted to corrections.yaml / dsg.json regardless.
		if not self.context.ok():
			return
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
		if not self.context.ok():
			return
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

	def _build_clip_track_observations(self, frame: Frame) -> list[ClipTrackObservation]:
		"""Collect DAAAM track/semantic state for a clip sidecar frame."""
		state_lock = getattr(self.orchestrator, "_state_lock", None)
		if state_lock is not None:
			with state_lock:
				object_labels = dict(self.orchestrator.object_labels)
		else:
			object_labels = dict(self.orchestrator.object_labels)

		observations: list[ClipTrackObservation] = []
		for track in frame.tracks:
			track_id = int(track.id)
			bbox_xyxy = [float(v) for v in np.asarray(track.bbox).reshape(-1)[:4]]
			if len(bbox_xyxy) != 4:
				continue
			segmentation_contours: list[list[list[int]]] = []
			for contour in getattr(track, "segmentation_contours", []) or []:
				points = np.asarray(contour).reshape(-1, 2)
				if len(points) >= 3:
					segmentation_contours.append(
						[[int(round(x)), int(round(y))] for x, y in points]
					)

			observations.append(
				ClipTrackObservation(
					track_id=track_id,
					semantic_id=int(object_labels.get(track_id, -1)),
					bbox_xyxy=bbox_xyxy,
					segmentation_contours=segmentation_contours,
				)
			)
		return observations

	def _handle_human_clip_finalized(self, artifact: HumanClipArtifact) -> None:
		"""Spawn HOI processing for a finalized clip in a background thread."""
		if not self.enable_cosmos_hoi_processing:
			return

		clip_path = artifact.clip_path
		metadata_path = artifact.metadata_path
		self.logger.info(f"Launching Cosmos HOI processing for {clip_path.name}")

		def _run() -> None:
			start_wall = time.time()
			try:
				out = _hoi_process_clip(
					clip_path=clip_path,
					metadata_path=metadata_path,
					base_url=self.cosmos_hoi_base_url,
					model=self.cosmos_hoi_model,
					api_key=self.cosmos_hoi_api_key,
					fps=self.cosmos_hoi_fps,
					match_iou_threshold=self.cosmos_hoi_match_iou_threshold,
					media_root=self.cosmos_hoi_media_root,
					egocentric=self.egocentric,
				)
				duration = time.time() - start_wall
				self.logger.info(f"Cosmos HOI done in {duration:.1f}s: {out}")
			except Exception:
				duration = time.time() - start_wall
				import traceback as _tb
				self.logger.error(
					f"Cosmos HOI failed for {clip_path.name} after {duration:.1f}s:\n{_tb.format_exc()}"
				)

		future = self._hoi_executor.submit(_run)
		self._hoi_futures.append(future)

	def _refresh_event_outputs(self) -> dict:
		"""Run the HOI post-processing chain against current outputs.

		Shared by shutdown and the periodic mid-run refresh; every stage rewrites
		its output from scratch, so repeated calls are safe.
		"""
		from pathlib import Path as _Path
		import os as _os
		return _refresh_event_outputs(
			_Path(self.orchestrator.output_dir),
			cosmos_hoi_enabled=getattr(self, 'enable_cosmos_hoi_processing', False),
			llm_base_url=self.config.grounding.llm_base_url or "https://api.openai.com/v1",
			llm_model=self.event_grouper_model,
			llm_api_key=_os.environ.get("OPENAI_API_KEY", ""),
			match_iou_threshold=self.cosmos_hoi_match_iou_threshold,
			semantic_reranking=self.cosmos_hoi_semantic_reranking,
			semantic_enabled=getattr(self, 'enable_semantic_event_post_processing', True),
			semantic_output_name=self.semantic_event_output_name,
			neighbor_window_sec=self.semantic_event_neighbor_window_sec,
			confidence_threshold=self.semantic_event_confidence_threshold,
			logger=self.logger,
		)

	def _maybe_refresh_events(self) -> None:
		"""Timer-armed, DSG-update-fired refresh of events.yaml.

		Atomic HOI interactions have to accumulate before they aggregate into
		meaningful high-level events, so the interval arms the refresh and the
		next DSG update actually fires it — that way the regrouping always runs
		against a scene graph and corrections set that have just been updated.
		Single-flighted and run off the callback thread: the chain makes an LLM
		call and must never block DSG processing.
		"""
		if not self.enable_live_event_refresh:
			return
		if self._event_refresh_running:
			return
		if time.time() - self._last_event_refresh < self.event_refresh_interval_sec:
			return

		self._event_refresh_running = True

		def _run() -> None:
			start_wall = time.time()
			try:
				# Post-processing reads labels off disk, so publish the current
				# corrections before regrouping — without this the refresh would
				# keep using whatever corrections.yaml existed at the last save.
				self.orchestrator.scene_graph_service.save_corrections(
					self.orchestrator.output_dir
				)
				result = self._refresh_event_outputs()
				if result.get("events_path"):
					self.logger.info(
						f"Event refresh done in {time.time() - start_wall:.1f}s: "
						f"{result['events_path']}"
					)
			except Exception:
				import traceback as _tb
				self.logger.error(f"Event refresh failed:\n{_tb.format_exc()}")
			finally:
				self._last_event_refresh = time.time()
				self._event_refresh_running = False

		self._hoi_executor.submit(_run)

	def _start_live_query_bridge(self) -> None:
		"""Expose the live buffer + latest corrected DSG over HTTP.

		Snapshotting is cheap (the buffer is already in RAM) and the Cosmos call
		runs on the bridge's own request thread, so the camera callback is never
		blocked. Unlike _handle_human_clip_finalized this needs no clip to
		finalize, so it also answers while a person stays continuously in frame.
		"""
		if self.human_clip_recorder is None:
			self.logger.warning("Live query bridge disabled: no human clip recorder")
			return

		self.live_query_bridge = LiveQueryBridge(
			host=self.live_bridge_host,
			port=self.live_bridge_port,
			snapshot_fn=self.human_clip_recorder.snapshot_live_buffer,
			# Deliberately not get_latest_corrected_dsg: that one expires anything
			# older than 2s, which is right for the 1Hz republisher and wrong for a
			# reader answering questions about a scene the robot is standing still in.
			dsg_fn=self.orchestrator.scene_graph_service.get_corrected_dsg_snapshot,
			buffer_stats_fn=self.human_clip_recorder.live_buffer_stats,
			cosmos_base_url=self.cosmos_hoi_base_url,
			cosmos_model=self.cosmos_hoi_model,
			cosmos_api_key=self.cosmos_hoi_api_key,
			cosmos_media_root=self.cosmos_hoi_media_root,
			query_timeout_sec=self.live_query_timeout_sec,
			output_dir=self.orchestrator.output_dir,
		)
		self.live_query_bridge.start()
		self.logger.info(
			f"Live query bridge on {self.live_bridge_host}:{self.live_bridge_port} "
			f"(buffer={self.live_buffer_sec:.0f}s)"
		)
		if self.defer_dsg_processing:
			self.logger.warning(
				"defer_dsg_processing=true: /dsg will stay empty until shutdown. "
				"Set defer_dsg_processing:=false for live scene-graph queries."
			)

	def destroy_node(self) -> None:
		"""Cleanup when node is destroyed."""
		print("[Shutdown] Shutting down MMLLM Grounded SAM Node")

		# Stop live publishing before shutdown post-processing. orchestrator.stop()
		# and _save_all_data() regenerate corrections that fire the semantic-update
		# callback; after SIGINT the publisher's context is already invalid, and
		# any live consumer is shutting down too. Final state is on disk regardless.
		if hasattr(self, 'orchestrator') and hasattr(self.orchestrator, 'set_semantic_update_callback'):
			self.orchestrator.set_semantic_update_callback(None)
		if hasattr(self, 'dsg_publish_timer'):
			self.dsg_publish_timer.cancel()
		if getattr(self, 'live_query_bridge', None) is not None:
			print("[Shutdown] Stopping live query bridge...")
			self.live_query_bridge.stop()

		try:
			if hasattr(self, 'human_clip_recorder') and self.human_clip_recorder is not None:
				print("[Shutdown] Finalizing human clip recorder...")
				self.human_clip_recorder.close()
				print("[Shutdown] Human clip recorder finalized")

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

			# Checkpoint: save corrections.yaml / dsg.json now so they survive
			# even if SIGKILL fires during the HOI wait below.
			if hasattr(self, 'orchestrator'):
				try:
					print("[Shutdown] Saving checkpoint (corrections + DSG)...")
					self.orchestrator._save_all_data()
					print("[Shutdown] Checkpoint saved")
				except Exception as _e:
					print(f"[Shutdown] Checkpoint save failed (non-fatal): {_e}")

			# Wait for in-flight HOI tasks BEFORE stopping the orchestrator so
			# that the multiprocessing Manager (used by query_with_wait) stays
			# alive until every future has finished. query_with_wait has its own
			# internal timeout (90 s) so this blocks for at most that long.
			if hasattr(self, '_hoi_executor'):
				print("[Shutdown] Waiting for in-flight Cosmos HOI tasks to finish...")
				futures = list(getattr(self, '_hoi_futures', []))
				if futures:
					wait(futures)  # no timeout - internal timeouts in query_with_wait bound this
					print(f"[Shutdown] All {len(futures)} Cosmos HOI task(s) complete")
				self._hoi_executor.shutdown(wait=False, cancel_futures=True)
				print("[Shutdown] Cosmos HOI task wait complete")

			# Now safe to stop the orchestrator (final save + worker teardown).
			if hasattr(self, 'orchestrator'):
				print("[Shutdown] Stopping orchestrator...")
				self.orchestrator.stop()  # Exports performance statistics and stops workers
				print("[Shutdown] Orchestrator stopped")

			# corrections.yaml is now written - revalidate HOI matches before event grouping
			if hasattr(self, 'orchestrator') and hasattr(self.orchestrator, 'output_dir'):
				self.logger.info("[Shutdown] Refreshing event outputs...")
				self._refresh_event_outputs()

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

	def _log_missing_transform(self) -> None:
		"""Warn once when tf2 starts failing, then at most every 10s while it stays down."""
		now = time.time()
		since = getattr(self, "_tf_missing_since", None)
		last = getattr(self, "_tf_missing_last_log", 0.0)
		if since is None:
			self._tf_missing_since = now
			self._tf_missing_last_log = now
			self.logger.warning(
				f"No tf2 transform {self.world_frame} -> {self.camera_frame}; "
				"continuing with zero velocities. Geometry-dependent outputs "
				"(Hydra placement) will be wrong until it is published."
			)
			return
		if now - last >= 10.0:
			self._tf_missing_last_log = now
			self.logger.warning(
				f"Still no tf2 transform {self.world_frame} -> {self.camera_frame} "
				f"after {now - since:.0f}s"
			)

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
				# Caller reports this to the user via _log_missing_transform(),
				# rate-limited; logging it here too would just double the spam.
				self.logger.debug(f"tf2 transform lookup failed (both exact and latest): {e2}")
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
	print(f"[WATCHDOG] Shutdown exceeded {timeout}s - forcing exit", file=sys.stderr)
	os._exit(1)

def main(args=None):
	"""Main function for the ROS2 node."""
	rclpy.init(args=args)
	node = None

	try:
		node = DaaamNode()
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	except Exception as e:
		print(f"Error starting node: {e}")
		traceback.print_exc()
	finally:
		# Start watchdog - guarantees process exit even if shutdown path blocks
		watchdog = threading.Thread(target=_shutdown_watchdog, args=(240.0,), daemon=True)
		watchdog.start()

		print("Destroying node...")
		if node is not None:
			node.destroy_node()
		if rclpy.ok():
			rclpy.shutdown()


if __name__ == "__main__":
	main()
