# Parameter Hierarchy and Configuration Guide

This document explains how configuration parameters are prioritized and why there are multiple configuration locations in the MMLLM Grounded SAM system.

## Parameter Priority Hierarchy

**ROS Launch Parameters > YAML Config File > Code Defaults**

```
1. Code Defaults (lowest priority)
   ↓
2. YAML Config File (pipeline_config.yaml)
   ↓  
3. ROS Launch Parameters (highest priority)
```

### Implementation Details

The parameter loading happens in this order:

1. **Initialize with code defaults** (in dataclass definitions)
2. **Load YAML config file** (if exists) - `PipelineConfig.from_yaml()`
3. **Override with ROS parameters** - `_override_config_with_parameters()`

```python
# 1. Start with YAML (or code defaults if YAML missing)
config = PipelineConfig.from_yaml("config/pipeline_config.yaml")

# 2. Override with ROS parameters (highest priority)
config.workers.assignment_config.min_obs_per_track = self.min_obs_per_track
config.workers.dam_grounding_config.save_debug_data = self.save_debug_data
```

## Parameter Categories

### Always in ROS Launch Parameters
- **Topics**: `rgb_topic`, `depth_topic`, `camera_info_topic`
- **Output settings**: `output_dir`, `enable_debug_output`
- **Model selection**: `sam_model`, `agent_model_name`

### Usually in YAML Config
- **Worker-specific settings**: Full `dam_grounding_config`
- **Complex structures**: `color_map`, nested configurations
- **Prompt templates**: `group_prompt_file`,

### Can Be in Either (Commonly Overridden)
- **Debug settings**: `save_debug_data`
- **Performance tuning**: `min_obs_per_track`, `query_interval_frames`
- **Batch processing**: `multi_image_min_n_masks`