# Launch File Integration Guide

This guide explains how to integrate the MMLLM Grounded SAM node into existing launch files and create new ones.

## Launch File Structure

The launch files follow the ROS2 convention with explicit argument declarations and parameter passing.

### Base Launch File

`daaam_node.launch.yaml` is the base launch file that:

1. **Declares all arguments** with defaults
2. **Starts the node** with parameter passing
3. **Sets up topic remapping** for integration flexibility

```yaml
---
launch:
  # Argument declarations
  - arg: {name: rgb_topic, default: '/tesse/left_cam/rgb/image_raw'}
  - arg: {name: agent_model_name, default: 'gpt-4.1'}
  # ... more args ...
  
  # Node definition
  - node:
      pkg: daaam_ros
      exec: daaam_node
      name: daaam
      param:
        - {name: agent_model_name, value: $(var agent_model_name)}
        # ... more params ...
      remap:
        - {from: '~/rgb_image', to: $(var rgb_topic)}
        # ... more remaps ...
```

## Integration Patterns

### Pattern 1: Direct Launch
```bash
# Launch with defaults
ros2 launch daaam_ros daaam_node.launch.yaml

# Launch with overrides
ros2 launch daaam_ros daaam_node.launch.yaml \
  agent_model_name:=gpt-4.1-mini \
  save_debug_data:=true
```

### Pattern 2: Include in Other Launch Files
```yaml
# Include with specific configuration
- include:
    file: $(find-pkg-share daaam_ros)/launch/daaam_node.launch.yaml
    arg:
      - {name: rgb_topic, value: '/my_robot/camera/rgb'}
      - {name: grounding_worker, value: 'dam_multi_image'}
      - {name: save_debug_data, value: 'true'}
```

### Pattern 3: Specialized Launch Files
Create specialized launch files that include the base with specific configurations:

```yaml
# debug_mode.launch.yaml
---
launch:
  - include:
      file: $(find-pkg-share daaam_ros)/launch/daaam_node.launch.yaml
      arg:
        - {name: save_debug_data, value: 'true'}
        - {name: min_obs_per_track, value: '3'}
```

## Available Arguments

### Topic Configuration
- `rgb_topic`: RGB camera topic
- `depth_topic`: Depth camera topic  
- `camera_info_topic`: Camera info topic
- `dsg_update_topic`: DSG updates output topic
- `segmentation_topic`: Segmentation output topic
- `segmentation_color_topic`: Colored segmentation output topic

### Model Configuration
- `agent_model_name`: MMLLM model (e.g., 'gpt-4.1', 'gpt-4.1-mini', 'gemini-2.5-flash')
- `sam_model`: SAM model path (e.g., 'fastsam/FastSAM-s.pt', 'fastsam/FastSAM-x.pt')
- `sam_model_config_path`: SAM model configuration (e.g., 'fastsam/fastsam_config.yaml)

### Processing Parameters
- `query_interval_frames`: Frames between grounding queries
- `num_assignment_workers`: Number of assignment worker processes
- `num_grounding_workers`: Number of grounding worker processes
- `assignment_worker`: Assignment worker type ('min_frames')
- `grounding_worker`: Grounding worker type ('dam_single_image', 'dam_multi_image')

### Worker-Specific Parameters
- `min_obs_per_track`: Minimum observations before track becomes eligible
- `save_debug_data`: Enable debug data saving
- `multi_image_min_n_masks`: Minimum masks for DAM batch processing (-1 = use YAML default)

### Depth Filtering
- `depth_lb`: Lower depth bound
- `depth_ub`: Upper depth bound

### Debug and Output
- `enable_debug_output`: Enable debug output
- `output_dir`: Output directory for logs and debug data

