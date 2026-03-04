"""
Service layer for dataloader orchestration.

Following daaam pattern of service-based architecture.
"""

import asyncio
import threading
from typing import Optional, Dict, Any, List
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
import time
from tqdm import tqdm


from daaam_ros.nodes.dataloader.interfaces import DataLoader
from daaam_ros.nodes.dataloader.output.interfaces import OutputHandler
from daaam_ros.nodes.dataloader.models import DataloaderConfig
from daaam_ros.nodes.dataloader.loaders.coda_loader import CodaDataLoader
from daaam_ros.nodes.dataloader.models import ImageData
from daaam_ros.nodes.dataloader.visualizer import DataloaderVisualizer

class DataloaderService:
	"""Main service orchestrating data loading, processing, and output."""
	
	def __init__(self, config: DataloaderConfig, node=None):
		self.config = config
		self.node = node  # ROS2 node reference
		
		# Initialize components
		self.loader: Optional[DataLoader] = None
		self.output_handler: Optional[OutputHandler] = None
		self.visualizer: Optional[DataloaderVisualizer] = None
		
		# Threading and queues
		self.frame_queue = Queue(maxsize=config.prefetch_frames)
		self.executor = ThreadPoolExecutor(max_workers=config.num_workers)
		self.is_running = False
		self.loader_thread: Optional[threading.Thread] = None
		self.writer_thread: Optional[threading.Thread] = None
		
		# Statistics
		self.frames_processed = 0
		self.start_time = None
		
	def initialize(self) -> bool:
		"""Initialize all components."""
		try:
			# Create dataloader based on type
			if self.config.dataset_type == "coda":
				self.loader = CodaDataLoader(self.config)
			else:
				print(f"Unknown dataset type: {self.config.dataset_type}")
				return False
			
			# Initialize dataloader
			if not self.loader.initialize():
				print("Failed to initialize dataloader")
				return False
			
			# Depth handling
			if self.config.depth_method == "provided":
				# Depth will be loaded from dataset
				print(f"Using provided depth from source: {self.config.depth_source}")
			elif self.config.depth_method == "none":
				print("No depth processing")
			else:
				print(f"Note: Depth method '{self.config.depth_method}' is deprecated. Use 'provided' or 'none'.")
				print("Defaulting to 'provided' depth method.")
				self.config.depth_method = "provided"
			
			# Output handler will be created by ROS node
			
			# Create visualizer if enabled
			enable_viz = getattr(self.config, 'enable_visualizer', False)
			if enable_viz:
				self.visualizer = DataloaderVisualizer(app_id="coda_dataloader")
				
				# Collect calibrations for static transforms
				calibrations = {}
				for camera_id in self.config.camera_ids:
					calib = self.loader.get_calibration(camera_id)
					if calib:
						calibrations[camera_id] = calib
				
				# Initialize with calibrations for static transforms
				if not self.visualizer.initialize(calibrations):
					print("Warning: Failed to initialize visualizer, continuing without it")
					self.visualizer = None
				else:
					print("Visualizer initialized successfully with static transforms")
			
			self._log_service_info()
			return True
			
		except Exception as e:
			print(f"Failed to initialize DataloaderService: {e}")
			import traceback
			traceback.print_exc()
			return False
	
	def start(self) -> None:
		"""Start the dataloader service."""
		if self.is_running:
			print("DataloaderService already running")
			return
		
		self.is_running = True
		self.start_time = time.time()
		
		# Start loader thread
		self.loader_thread = threading.Thread(target=self._loader_loop, daemon=True)
		self.loader_thread.start()
		
		# Start writer thread
		self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
		self.writer_thread.start()
		
		print("DataloaderService started")
	
	def stop(self) -> None:
		"""Stop the dataloader service."""
		self.is_running = False
		
		# Wait for threads to finish
		if self.loader_thread:
			self.loader_thread.join(timeout=2.0)
		if self.writer_thread:
			self.writer_thread.join(timeout=2.0)
		
		# Clean up
		self.executor.shutdown(wait=False)
		
		if self.output_handler:
			self.output_handler.close()
		
		if self.visualizer:
			self.visualizer.close()
		
		# Print final statistics
		if self.start_time:
			duration = time.time() - self.start_time
			fps = self.frames_processed / duration if duration > 0 else 0
			print(f"\n{'='*50}")
			print(f"DataloaderService Summary:")
			print(f"  Frames processed: {self.frames_processed}")
			print(f"  Duration: {duration:.2f} seconds")
			print(f"  Average FPS: {fps:.2f}")
			print(f"{'='*50}")
	
	def _loader_loop(self) -> None:
		"""Background thread for loading frames."""
		# Create progress bar for loading
		total_frames = self.loader.num_frames
		with tqdm(total=total_frames, desc="Loading frames", unit="frames", position=0) as pbar:
			while self.is_running and self.loader.has_next():
				try:
					# Load next frame
					frame = self.loader.get_frame()
					if frame is None:
						break
					
					# Depth is already loaded from dataset if available (no processing needed)
					
					# Add to queue (blocks if queue is full)
					self.frame_queue.put(frame, timeout=0.1)
					pbar.update(1)
					
				except Exception as e:
					print(f"Error in loader loop: {e}")
					if not self.is_running:
						break
	
	def _writer_loop(self) -> None:
		"""Background thread for writing frames to bag."""
		# No rate limiting - write as fast as possible
		# Playback speed is controlled by /clock messages in the bag
		
		# Create progress bar for writing
		total_frames = self.loader.num_frames
		
		with tqdm(total=total_frames, desc="Writing to bag", unit="frames", position=1) as pbar:
			while self.is_running:
				try:
					# Get frame from queue
					frame = self.frame_queue.get(timeout=0.1)
					
					# Write frame to bag (transforms are written before clock in bag_writer)
					if self.output_handler and self.output_handler.is_ready:
						# Get calibration for the first camera (publish_frame only needs one)
						for camera_id in frame.rgb_images.keys():
							calib = self.loader.get_calibration(camera_id)
							if calib:
								self.output_handler.publish_frame(frame, calib)
								break  # Only need to call once
					
					# Visualize frame if enabled
					if self.visualizer:
						calibrations = {}
						for camera_id in frame.rgb_images.keys():
							calib = self.loader.get_calibration(camera_id)
							if calib:
								calibrations[camera_id] = calib
						self.visualizer.log_frame(frame, calibrations)
					
					self.frames_processed += 1
					pbar.update(1)
					
					# Update progress bar description with FPS
					if self.frames_processed % 10 == 0:
						duration = time.time() - self.start_time
						fps = self.frames_processed / duration if duration > 0 else 0
						pbar.set_postfix(fps=f"{fps:.1f}")
					
				except Empty:
					# No frame available
					if not self.loader.has_next() and self.frame_queue.empty():
						break
				except Exception as e:
					print(f"Error in publisher loop: {e}")
					if not self.is_running:
						break
		
		# Finished processing all frames - stop the service
		print(f"\nFinished processing all {self.frames_processed} frames")
		self.is_running = False
	
	
	def set_output_handler(self, handler: OutputHandler) -> None:
		"""Set the output handler (called by ROS node)."""
		self.output_handler = handler
	
	def _log_service_info(self) -> None:
		"""Log service configuration."""
		print(f"\nDataloaderService initialized:")
		print(f"  Dataset type: {self.config.dataset_type}")
		print(f"  Depth method: {self.config.depth_method}")
		print(f"  Output: Writing to bag")
		print(f"  Target framerate: {self.config.target_framerate} Hz")
		print(f"  Worker threads: {self.config.num_workers}")
		print(f"  Prefetch frames: {self.config.prefetch_frames}")