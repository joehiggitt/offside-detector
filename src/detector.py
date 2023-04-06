"""
A module for detecting offsides in images of broadcast football matches.
Written by Joe Higgitt.
"""

from __future__ import annotations
from typing import Any

import numpy as np
import cv2 as cv
import image as im
import utils as ut
from sklearn.cluster import DBSCAN
from copy import deepcopy


DEFAULT_PARAMS: ut.Params_Type = {
	"dominant colour": {
		"k": 0.2,
		"c": 0.6,
		"n": 3,
		"bounds": (
			np.array([20, 0, 0], "uint8"),
			np.array([80, 255, 255], "uint8"),
		),
	},
	"lines blur": {
		"sigma": 15,
		"kernel_size": 15,
	},
	"player blur": {
		"kernel_size": 3,
	},
	"grass mask": {
		"deviations": (4, 6, 6),
	},
	"grass mask open": {
		"kernel_size": 3,
		"iterations": 8,
	},
	"grass mask close": {
		"kernel_size": 3,
		"iterations": 2,
	},
	"pitch mask close": {
		"kernel_size": 11,
		"iterations": 5,
	},
	"object mask erode": {
		"kernel_size": 3,
		"iterations": 1,
	},
}


class OffsideDetector:
	"""
	A class that, given an broadcast football image, can identify the offside players in the image.
	
	Class Methods
	-------------
	`OffsideDetector(params: dict)` : `OffsideDetector`
		Creates an offside detector object for an image of a football
		match.
	
	Methods
	-------
	`param(operation: str, param: str)` : `dict` or `Any`
		Returns a parameter or parameter list required by a certain
		function.
	`get_offsides(image: image.Image)` : ``
		Finds the offsides in a football image.
	"""

	def __init__(self, params: ut.Params_Type = DEFAULT_PARAMS):
		"""
		Creates an offside detector object for an image of a football
		match.

		Parameters
		----------
		`params` : `dict` of `str: dict`, of `str: Any`, optional
			A dictionary of parameters for the various operations in the
			detection process. See `DEFAULT_PARAMS` for a list of the
			required parameters. If not provided, `DEFAULT_PARAMS` is
			used.

		Returns
		-------
		`OffsideDetector`
			An offside detector object.

		Raises
		------
		`ValueError`
			If `params` doesn't have an identical structure to
			`DEFAULT_PARAMS`.
		"""
		OffsideDetector.__verify_params(params)
		self.__params: dict     = params

	@classmethod
	def __verify_params(cls, params: ut.Params_Type):
		"""
		Validates whether a provided parameter dictionary is valid or
		not. To be valid, the dictionary must have an identical 
		structure to `DEFAULT_PARAMS`.

		Parameters
		----------
		`params` : `dict` of `str: dict`, of `str: Any`
			A dictionary of parameters for the various operations in the
			detection process. See `DEFAULT_PARAMS` for a list of the
			required parameters.

		Raises
		------
		`ValueError`
			If `params` doesn't have an identical structure to
			`DEFAULT_PARAMS`.
		"""
		if (DEFAULT_PARAMS.keys() != params.keys()):
			raise ValueError("Provided parameter dictionary is invalid.")

		for key in DEFAULT_PARAMS.keys():
			if (DEFAULT_PARAMS[key].keys() != params[key].keys()):
				raise ValueError(f"Parameters provided for operation '{key}'" +
					+ "are invalid.")

	def param(self, operation: str, param: str=None) -> dict[str, Any] | Any:
		"""
		Returns a parameter or parameter list required by a certain
		function.

		Parameters
		----------
		`operation` : `str`
			The name of the operation the parameter is for.
		`param` : `str`
			The name of the parameter.

		Returns
		-------
		`dict` of `str: Any`
			If `param` not provided, the keyword argument dictionary to
			pass into the function.
		`Any`
			If `param` provided, the parameter value.

		Raises
		------
		`ValueError`
			If `operation` or `param` is not recognised.
		"""
		if (operation.lower() not in self.__params.keys()):
			raise ValueError(f"Operation '{operation}' not recognised.")

		if param is None:
			return self.__params[operation.lower()]
		
		if (param.lower() not in self.__params[operation.lower()].keys()):
			raise ValueError(f"Parameter '{param}' not recognised for " +
				f"operation '{operation}'.")
		
		return self.__params[operation.lower()][param.lower()]
	
	def __get_grass_mask(self, blur_image: im.Image, grass_colour: 
		ut.Mutiple_Colour_Type = None, grass_sigma: ut.Mutiple_Colour_Type = None
		) -> im.Image:
		"""
		Creates a mask of the grass in a football image.

		Parameters
		----------
		`blur_image` : `image.Image`
			The blurred football image to mask the grass in.
		`grass_colour` : `tuple` of `numpy.ndarray`, optional
			The colour of the grass. The tuple must contain a `ndarray`
			for each channel in `blur_image`. The length of this array
			is arbitrary.
		`grass_sigma` : `tuple` of `numpy.ndarray`, optional
			The standard deviation of the colour of the grass. Must be
			of an identical shape to `grass_colour`.

		Returns
		-------
		`image.Image`
			The mask containing just the grass-coloured pixels.
		"""
		if (grass_colour is None):
			grass_colour = blur_image.dominant_colour(**self.param(
				"dominant colour"))
		if (grass_sigma is None):
			grass_sigma = blur_image.dominant_colour_deviation(grass_colour, 
				**self.param("dominant colour"))
		
		# Creates a mask of the pitch on the grass colour
		grass_mask = blur_image.create_mask(grass_colour, grass_sigma, 
			**self.param("grass mask"))
	
		# Open to remove crowd noise, Close to fill in some of the pitch noise
		grass_mask = grass_mask.morphology("open", 
			**self.param("grass mask open"))
		grass_mask = grass_mask.morphology("close", 
			**self.param("grass mask close"))
		return grass_mask

	def __get_pitch_mask(self, grass_mask: im.Image) -> im.Image:
		"""
		Creates a mask of the playing area in a football image using
		morphological operations.

		Parameters
		----------
		`grass_mask` : `image.Image`
			The mask of the grass in the football image. See 
			`get_grass_mask` for creating this.
		
		Returns
		-------
		`image.Image`
			The mask containing the playing area.
		"""
		# Close to fill in gaps caused by players
		pitch_image = grass_mask.morphology("close", **self.param(
			"pitch mask close"))
		return pitch_image

	def __get_object_mask(self, grass_mask: im.Image, pitch_mask: im.Image
		) -> im.Image:
		"""
		Creates a mask of the non-grass features in a football image by
		subtracting a grass mask from a mask of the whole playing area.

		Parameters
		----------
		`grass_mask` : `image.Image`
			The mask of the grass in the football image. See 
			`get_grass_mask` for creating this.
		`pitch_mask` : `image.Image`
			The mask of the playing area in the football image. See 
			`get_pitch_mask` for creating this.

		Returns
		-------
		`image.Image`
			The mask containing the non-grass pixels in the image.
		"""
		# Finds objects as differences between pixels in the pitch mask but not
		# in the grass mask
		object_mask  = np.bitwise_and(np.bitwise_not(grass_mask.get()),
			pitch_mask.get())
		object_image = im.Image(object_mask, "BINARY")
		object_image = object_image.morphology("erode",
			**self.param("object mask erode"))
		return object_image
	
	
	def __get_line_mask(self, grey_image: im.Image, pitch_mask: im.Image
		) -> im.Image:
		"""
		Creates a mask of the pitch lines in a football image using the
		morphological tophat operator.

		Parameters
		----------
		`grey_image` : `image.Image`
			The greyscale blurred image of the football pitch
		`pitch_mask` : `image.Image`
			The mask of the playing area in the football image. See 
			`__get_pitch_mask` for creating this.

		Returns
		-------
		`image.Image`
			The mask containing the pitch line pixels in the image.
		"""
		# Tophat operator extracts the light features in an image
		tophat_image = grey_image.morphology("tophat", **self.param(
			"lines top hat"))
		tophat_image = tophat_image.threshold(**self.param("lines threshold"))
		line_mask    = np.bitwise_and(tophat_image.get(), pitch_mask.get())
		line_image   = im.Image(line_mask, "BINARY")

		# Erosion to remove noise and thin found lines
		line_image   = line_image.morphology("erode", 
			**self.param("lines erode"))
		return line_image
	
	def __get_lines(self, line_mask: im.Image) -> (np.ndarray | None):
		"""
		Finds lines in the image, using DBSCAN to refine the results.

		Parameters
		----------
		`line_mask` : `image.Image`
			The mask of the pitch lines in the football image. See 
			`__get_line_mask` for creating this.
		
		Returns
		-------
		`numpy.ndarray` of shape `(k, 2)`
			If lines were found, an an array of the lines found, in the
			form '(rho, theta)', where 'rho' is the distance from the
			origin and 'theta' is the line normal from the origin. theta
			must be in the interval [0, π) radians (or [0, 180) 
			degrees).
		`None`
			If no lines were found.

		"""
		lines = line_mask.hough(**self.param("lines hough"))
		if (lines is None):
			return None

		# Removes lines which lie outside a bound
		rho_bound, theta_bound = self.param("lines bounds", "rho"), self.param(
			"lines bounds", "theta")
		bounded_lines = lines[
			(rho_bound[0] <= lines[:, 0]) & (lines[:, 0] <= rho_bound[1]) & 
			(theta_bound[0] <= lines[:, 1]) & (lines[:, 1] <= theta_bound[1])
		]

		# Clusters lines into distinct groups
		X = np.dstack((bounded_lines[:, 0] * self.param("lines normalise", 
			"scale"), bounded_lines[:, 1]))[0]
		model = DBSCAN(**self.param("lines dbscan")).fit(X)
		labels = model.labels_

		means = []
		for i in range(np.amax(labels) + 1):
			means.append(np.mean(bounded_lines[labels == i], axis=0))
		mean_lines = np.array(means)
		if (mean_lines.shape[0] == 0):
			return None

		return mean_lines[np.argsort(mean_lines[:, 1]) < self.param(
			"lines number", "number")]

	def get_offsides(self, image: im.Image):
		"""
		Finds the offsides in a football image.

		Parameters
		----------
		`image` : `image.Image`
			The football image.
		
		Returns
		-------
		`x`
			x
		"""
		# Blur images for line and player detection
		line_image = image.blur(**self.param("lines blur"))
		player_image = image.blur(**self.param("player blur"))

		# Get the colour characteristics of the current image
		grass_colour = line_image.dominant_colour(**self.param(
			"dominant colour"))
		grass_sigma = line_image.dominant_colour_deviation(grass_colour, 
			**self.param("dominant colour"))

		# Get the grass mask of the image using the colours
		grass_mask = self.__get_grass_mask(player_image, grass_colour, 
			grass_sigma)
		pitch_mask = self.__get_pitch_mask(grass_mask)
		object_mask = self.__get_object_mask(grass_mask, pitch_mask)

		# Create the main line and player masks
		grey_line_image = line_image.convert("BGR").convert("GREY")
		line_mask = self.__get_line_mask(grey_line_image, pitch_mask)

		# Find pitch lines in the line mask
		pitch_lines = self.__get_lines(line_mask)

		# If the results found are poor, attempts to reverse the image
		reverse = False
		if (pitch_lines is None):
			pitch_lines = self.__get_lines(line_mask.flip("h"))
			if (pitch_lines is None):
				return None
			reverse = True

		if reverse:
			player_mask = player_mask.flip("h")
			player_image = player_image.flip("h")
	
