"""
Configuration management for dataloader system.

Following daaam pattern for configuration.
"""

import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from daaam_ros.nodes.dataloader.models import DataloaderConfig
import os


class ConfigManager:
	"""Manages dataloader configuration loading and validation."""
	
	@staticmethod
	def load_config(config_path: Optional[Path] = None, 
				   overrides: Optional[Dict[str, Any]] = None) -> DataloaderConfig:
		"""Load configuration from file with optional overrides.
		
		Args:
			config_path: Path to YAML config file
			overrides: Dictionary of parameter overrides
			
		Returns:
			DataloaderConfig: Validated configuration
		"""
		config_dict = {}
		
		# Load from file if provided
		if config_path and config_path.exists():
			with open(config_path, 'r') as f:
				config_dict = yaml.safe_load(f) or {}
		
		# Apply overrides
		if overrides:
			config_dict = ConfigManager._deep_merge(config_dict, overrides)
		
		# Ensure dataset_path is Path object
		if 'dataset_path' in config_dict:
			config_dict['dataset_path'] = Path(config_dict['dataset_path'])
		
		return DataloaderConfig(**config_dict)
	
	@staticmethod
	def save_config(config: DataloaderConfig, path: Path) -> None:
		"""Save configuration to YAML file.
		
		Args:
			config: Configuration to save
			path: Output path
		"""
		path.parent.mkdir(parents=True, exist_ok=True)
		
		# Convert to dict and handle Path objects
		config_dict = config.dict()
		if 'dataset_path' in config_dict:
			config_dict['dataset_path'] = str(config_dict['dataset_path'])
		
		with open(path, 'w') as f:
			yaml.safe_dump(config_dict, f, default_flow_style=False)
	
	@staticmethod
	def _deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
		"""Deep merge two dictionaries.
		
		Args:
			base: Base dictionary
			updates: Updates to apply
			
		Returns:
			Merged dictionary
		"""
		result = base.copy()
		
		for key, value in updates.items():
			if key in result and isinstance(result[key], dict) and isinstance(value, dict):
				result[key] = ConfigManager._deep_merge(result[key], value)
			else:
				result[key] = value
		
		return result
	
	@staticmethod
	def get_default_config() -> DataloaderConfig:
		"""Get default configuration."""
		return DataloaderConfig(
			dataset_type="coda",
			dataset_path=Path("/path/to/coda/dir/"),
			sequence="0",
			camera_ids=["cam0", "cam1"],
			depth_method="sgbm",
			depth_source="none",
			depth_params={
				"min_disparity": 0,
				"num_disparities": 128,
				"block_size": 11,
				"p1": 8 * 3 * 11**2,
				"p2": 32 * 3 * 11**2,
				"disp12_max_diff": 1,
				"prefilter_cap": 63,
				"uniqueness_ratio": 10,
				"speckle_window_size": 100,
				"speckle_range": 32
			},
			output_mode="ros_topics",
			output_params={
				"queue_size": 10,
				"latch": False
			},
			use_timestamps=True,
			target_framerate=10.0,
			num_workers=2,
			prefetch_frames=10
		)