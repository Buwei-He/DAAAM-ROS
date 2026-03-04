"""
Image processing utilities for dataloader.

Handles image resizing with aspect ratio preservation.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from daaam_ros.nodes.dataloader.models import CameraIntrinsics


def crop_and_resize(
	image: np.ndarray,
	target_width: int,
	target_height: int,
	interpolation: int = cv2.INTER_LINEAR
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
	"""Crop image to target aspect ratio then resize.

	Args:
		image: Input image array
		target_width: Target width after resize
		target_height: Target height after resize
		interpolation: OpenCV interpolation method

	Returns:
		Tuple of (resized_image, crop_info)
		crop_info: (crop_x, crop_y, crop_width, crop_height) - original crop region
	"""
	h, w = image.shape[:2]
	target_aspect = target_width / target_height
	current_aspect = w / h

	# Determine crop dimensions to match target aspect ratio
	if current_aspect > target_aspect:
		# Image is wider than target - crop width
		crop_height = h
		crop_width = int(h * target_aspect)
	else:
		# Image is taller than target - crop height
		crop_width = w
		crop_height = int(w / target_aspect)

	# Center crop
	crop_x = (w - crop_width) // 2
	crop_y = (h - crop_height) // 2

	# Perform crop
	cropped = image[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]

	# Resize to target dimensions
	resized = cv2.resize(cropped, (target_width, target_height), interpolation=interpolation)

	return resized, (crop_x, crop_y, crop_width, crop_height)


def resize_rgb_image(
	image: np.ndarray,
	target_width: int,
	target_height: int
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
	"""Resize RGB image with bilinear interpolation.

	Args:
		image: RGB image array
		target_width: Target width
		target_height: Target height

	Returns:
		Tuple of (resized_image, crop_info)
	"""
	return crop_and_resize(image, target_width, target_height, cv2.INTER_LINEAR)


def resize_depth_image(
	depth: np.ndarray,
	target_width: int,
	target_height: int
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
	"""Resize depth image preserving depth values.

	Args:
		depth: Depth image array (meters as float32)
		target_width: Target width
		target_height: Target height

	Returns:
		Tuple of (resized_depth, crop_info)
	"""
	# Use bilinear interpolation for depth to preserve continuity
	# Depth values are preserved during interpolation
	return crop_and_resize(depth, target_width, target_height, cv2.INTER_LINEAR)


def resize_label_image(
	labels: np.ndarray,
	target_width: int,
	target_height: int
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
	"""Resize label image preserving label IDs.

	Args:
		labels: Label image array (uint16)
		target_width: Target width
		target_height: Target height

	Returns:
		Tuple of (resized_labels, crop_info)
	"""
	# Use nearest neighbor to preserve exact label IDs
	return crop_and_resize(labels, target_width, target_height, cv2.INTER_NEAREST)


def update_intrinsics_for_resize(
	intrinsics: CameraIntrinsics,
	original_size: Tuple[int, int],
	crop_info: Tuple[int, int, int, int],
	target_size: Tuple[int, int]
) -> CameraIntrinsics:
	"""Update camera intrinsics after crop and resize.

	Args:
		intrinsics: Original camera intrinsics
		original_size: (width, height) of original image
		crop_info: (crop_x, crop_y, crop_width, crop_height) from crop operation
		target_size: (width, height) of final resized image

	Returns:
		Updated CameraIntrinsics
	"""
	crop_x, crop_y, crop_width, crop_height = crop_info
	target_width, target_height = target_size

	# Step 1: Adjust principal point for crop
	# Principal point shifts by the crop offset
	cx_cropped = intrinsics.cx - crop_x
	cy_cropped = intrinsics.cy - crop_y

	# Step 2: Scale for resize
	scale_x = target_width / crop_width
	scale_y = target_height / crop_height

	# Apply scaling to intrinsics
	new_fx = intrinsics.fx * scale_x
	new_fy = intrinsics.fy * scale_y
	new_cx = cx_cropped * scale_x
	new_cy = cy_cropped * scale_y

	# Create updated intrinsics
	return CameraIntrinsics(
		fx=new_fx,
		fy=new_fy,
		cx=new_cx,
		cy=new_cy,
		width=target_width,
		height=target_height,
		distortion_model=intrinsics.distortion_model,
		distortion_coeffs=intrinsics.distortion_coeffs  # Distortion coeffs remain unchanged
	)