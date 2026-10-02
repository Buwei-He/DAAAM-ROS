#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
from rosbag2_py import SequentialReader, SequentialWriter, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import CameraInfo
from daaam.utils.colmap import read_cameras_binary, camera_to_camera_info
from tqdm import tqdm


class ColmapCameraInfoProcessor(Node):
    def __init__(self):
        """
        Initialize the ROS2 node for COLMAP camera info processing.
        """
        super().__init__('colmap_camera_info_processor')
        
        # Declare parameters
        self.declare_parameter('colmap_cameras_path', '')
        self.declare_parameter('input_bag_path', '')
        self.declare_parameter('output_bag_path', '')
        self.declare_parameter('image_topic', '')
        self.declare_parameter('camera_info_topic', '')
        self.declare_parameter('camera_frame_id', '')
        
        # Get parameters
        self.colmap_cameras_path = self.get_parameter('colmap_cameras_path').value
        self.input_bag_path = self.get_parameter('input_bag_path').value
        self.output_bag_path = self.get_parameter('output_bag_path').value
        self.image_topic = self.get_parameter('image_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.camera_frame_id = self.get_parameter('camera_frame_id').value
        
        # Validate parameters
        if not self.colmap_cameras_path or not self.input_bag_path:
            self.get_logger().error("Missing required parameters")
            example_usage = """
            Usage: ros2 run percorso_perception_ros colmap_camera_info_node 
                --ros-args 
                -p colmap_cameras_path:=<path>
                -p input_bag_path:=<path>
                -p output_bag_path:=<path>
                -p image_topic:=<topic>
                -p camera_info_topic:=<topic>
                -p camera_frame_id:=<frame_id>
            """
            self.get_logger().error(example_usage)
            return
            
        if not self.output_bag_path:
            if self.input_bag_path.endswith('/'):
                input_base = self.input_bag_path[:-1]
            else:
                input_base = self.input_bag_path
            self.output_bag_path = f"{input_base}_with_camera_info"
            
        # Process the bag
        try:
            self.init_camera_info()
            self.process_bag()
        except Exception as e:
            self.get_logger().error(f"Error processing bag: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def init_camera_info(self):
        """Initialize camera info from COLMAP."""
        # Read COLMAP cameras
        self.cameras = read_cameras_binary(self.colmap_cameras_path)

        if not self.cameras:
            self.get_logger().error("No cameras found in COLMAP data")
            raise ValueError("No cameras found in COLMAP data")

        self.get_logger().info(f"Found {len(self.cameras)} cameras in COLMAP data")

        # Use the first camera in the COLMAP data
        self.camera = list(self.cameras.values())[0]
        self.get_logger().info(
            f"Using camera {self.camera.id}, model {self.camera.model}, "
            f"{self.camera.width}x{self.camera.height}"
        )

        # Convert COLMAP camera to ROS CameraInfo
        self.camera_info_params = camera_to_camera_info(self.camera)

    def process_bag(self):
        """Process the bag file, adding CameraInfo messages."""
        self.get_logger().info(f"Processing bag file: {self.input_bag_path}")
        self.get_logger().info(f"Output will be saved to: {self.output_bag_path}")

        # Setup reader
        reader_options = StorageOptions(uri=self.input_bag_path)
        reader_converter_options = ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr')
        reader = SequentialReader()
        reader.open(reader_options, reader_converter_options)

        # Get topic information
        topic_types = reader.get_all_topics_and_types()
        image_topic_type = None
        
        for topic_info in topic_types:
            if topic_info.name == self.image_topic:
                image_topic_type = topic_info.type
                break
                
        if not image_topic_type:
            self.get_logger().error(f"Image topic {self.image_topic} not found in bag")
            return False

        # Setup writer
        writer_options = StorageOptions(uri=self.output_bag_path)
        writer_converter_options = ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr')
        writer = SequentialWriter()
        writer.open(writer_options, writer_converter_options)

        # Create topics in the output bag
        for topic_info in topic_types:
            writer.create_topic(topic_info)
            
        # Create camera_info topic if not already in the bag
        camera_info_topic_exists = any(topic.name == self.camera_info_topic for topic in topic_types)
        if not camera_info_topic_exists:
            from rosidl_runtime_py.utilities import get_message_type_from_string
            camera_info_type = 'sensor_msgs/msg/CameraInfo'
            topic_info = type('TopicInfo', (), {
                'name': self.camera_info_topic,
                'type': camera_info_type,
                'serialization_format': 'cdr'
            })
            writer.create_topic(topic_info)

        # Create a base CameraInfo message
        base_camera_info = CameraInfo()
        base_camera_info.height = self.camera_info_params["height"]
        base_camera_info.width = self.camera_info_params["width"]
        base_camera_info.distortion_model = self.camera_info_params["distortion_model"]
        base_camera_info.d = self.camera_info_params["D"]
        base_camera_info.k = self.camera_info_params["K"]
        base_camera_info.r = self.camera_info_params["R"]
        base_camera_info.p = self.camera_info_params["P"]
        base_camera_info.binning_x = 0
        base_camera_info.binning_y = 0
        base_camera_info.roi.x_offset = 0
        base_camera_info.roi.y_offset = 0
        base_camera_info.roi.height = 0
        base_camera_info.roi.width = 0
        base_camera_info.roi.do_rectify = False
        base_camera_info.header.frame_id = self.camera_frame_id

        # Process all messages
        image_count = 0
        while reader.has_next():
            topic_name, data, timestamp = reader.read_next()
            
            # Write the original message
            writer.write(topic_name, data, timestamp)
            
            # If this is an image message, create and write a camera_info message
            if topic_name == self.image_topic:
                # Deserialize the image message to get the header
                image_msg_type = get_message(image_topic_type)
                image_msg = deserialize_message(data, image_msg_type)
                
                # Update camera_info header
                camera_info = CameraInfo()
                camera_info.header.stamp = image_msg.header.stamp
                camera_info.header.frame_id = self.camera_frame_id
                camera_info.height = base_camera_info.height
                camera_info.width = base_camera_info.width
                camera_info.distortion_model = base_camera_info.distortion_model
                camera_info.d = base_camera_info.d
                camera_info.k = base_camera_info.k
                camera_info.r = base_camera_info.r
                camera_info.p = base_camera_info.p
                camera_info.binning_x = base_camera_info.binning_x
                camera_info.binning_y = base_camera_info.binning_y
                camera_info.roi = base_camera_info.roi
                
                # Serialize and write camera_info
                camera_info_ser = serialize_message(camera_info)
                writer.write(self.camera_info_topic, camera_info_ser, timestamp)
                image_count += 1
                
                if image_count % 100 == 0:
                    self.get_logger().info(f"Processed {image_count} images")

        self.get_logger().info(f"Processing complete. Added {image_count} camera_info messages.")
        self.get_logger().info(f"Output saved to {self.output_bag_path}")


def main(args=None):
    rclpy.init(args=args)
    node = ColmapCameraInfoProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()