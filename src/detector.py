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
from parameters import DEFAULT_PARAMS
from sklearn.cluster import DBSCAN
from copy import deepcopy


class OffsideDetector:
	"""
	A class that, given an broadcast football image, can identify the offside players in the image.
	
	Class Methods
	-------------
	`OffsideDetector(params: dict)` : `OffsideDetector`
		Creates an offside detector object, used to detect offsides in
		broadcast football matches.
	
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

	def param(self, operation: str, param: str=None) -> (dict[str, Any] | Any):
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
	
	
	@classmethod
	def __get_intersecting_point(cls, line1: ut.Line_Type, line2: ut.Line_Type,
		mode: str = "RAD") -> (ut.Point_Type | None):
		"""
		Finds the intersection between two lines defined with polar
		coordinates.

		Parameters
		----------
		`line1`, `line2` : `tuple` of two `float`
			The lines to find the intersection between. Lines must be
			in the form '(rho, theta)', where 'rho' is the distance from
			the origin and 'theta' is the line normal from the origin.
			theta must be in the interval [0, π) radians (or
			[0, 180) degrees).
		`mode`: `["RAD", "DEG"]`, optional, default=`"RAD"`
			The units used by the inputted theta values (can be either
			radians or degrees). By default, radians are used.
			
		Returns
		-------
		`tuple` of `float`
			If the lines intersect, the coordinates of the point where
			they intersect (in the form `(x, y)`).
		`None`
			If the lines don't intersect (because they're parallel).
		"""
		ALLOWED_MODES = ["RAD", "DEG"]
		if (mode.upper() not in ALLOWED_MODES):
			raise ValueError(f"Unrecognised mode '{mode}' inputted. " +
				"Supported modes are radians ('RAD') and degrees ('DEG').")
		
		rho1, theta1 = line1
		rho2, theta2 = line2

		# Converts to radians if degrees provided (since numpy trig functions
		# use radians)
		if (mode.upper() == "DEG"):
			theta1, theta2 = np.deg2rad(theta1), np.deg2rad(theta2)

		# If theta values are equal, then the lines are parallel
		if (theta1 == theta2):
			return None

		x, y = None, None

		# Checks edge cases where theta is 0 or 90 deg (meaning intersection
		# equation is undefined)
		is_theta1 = None
		if (theta1 == 0):
			x = rho1
			is_theta1 = False
		elif (theta1 == (np.pi / 2)):
			y = rho1
			is_theta1 = False
		if (theta2 == 0):
			x = rho2
			is_theta1 = True
		elif (theta2 == (np.pi / 2)):
			y = rho2
			is_theta1 = True

		# Checks rare edge case where both x and y have be found without a
		# calculation needed (where one is 0 deg and the other is 90 deg)
		if ((x is not None) and (y is not None)):
			return x, y

		# Equations for calculating x or y from the other
		get_x = lambda y, tan_t, cos_t: (rho1 / cos_t) - (y * tan_t)
		get_y = lambda x, tan_t, sin_t: (rho1 / sin_t) - (x / tan_t)

		# Checks if either x or y have been set
		if (x is not None):
			if is_theta1:
				theta = theta1
			else:
				theta = theta2
			return x, get_y(x, np.tan(theta), np.sin(theta))
		if (y is not None):
			if is_theta1:
				theta = theta1
			else:
				theta = theta2
			return get_x(y, np.tan(theta), np.cos(theta)), y

		# If no edge cases have occured, calculates the intersection
		sin_theta1, sin_theta2 = np.sin(theta1), np.sin(theta2)
		tan_theta1, tan_theta2 = np.tan(theta1), np.tan(theta2)

		a = rho1 * sin_theta2 - rho2 * sin_theta1
		b = sin_theta1 * sin_theta2
		c = tan_theta1 * tan_theta2
		d = tan_theta2 - tan_theta1

		u = a / b
		v = c / d
			
		x = u * v
		y = get_y(x, tan_theta1, sin_theta1)
		# y = get_y(x, tan_theta2, sin_theta2)
		return x, y
		
	@classmethod
	def __get_intersecting_angle(cls, point1: ut.Point_Type, point2: 
		ut.Point_Type, mode: str = "RAD") -> (float | None):
		"""
		Finds the orientation of one point defined with polar
		coordinates with another.

		Parameters
		----------
		`point1`, `point2` : `tuple` of two `float`
			The points to find the intersection between. Points must be
			in the form '(x, y)'.
		`mode`: `{"RAD", "DEG"}`, optional, default=`"RAD"`
			The units used by the inputted theta values (can be either
			radians or degrees). By default, radians are used.
			
		Returns
		-------
		`float`
			The angle between the points.
		`None`
			If the two points are the same
		"""
		ALLOWED_MODES = ["RAD", "DEG"]
		if (mode.upper() not in ALLOWED_MODES):
			raise ValueError(f"Unrecognised mode '{mode}' inputted. " +
				"Supported modes are radians ('RAD') and degrees ('DEG').")

		if (np.allclose(point1, point2)):
			return None

		x1, y1 = point1
		x2, y2 = point2

		# Get theta angle of intersecting line
		if (x1 == x2):
			theta = 0
		elif (y1 == y2):
			theta = np.pi / 2
		else:
			tan_theta = (x1 - x2) / (y2 - y1)
			theta = np.arctan(tan_theta)  # TODO consider using arctan2

			# Ensures theta lies in [0, pi) interval
			if (theta < 0):
				theta += np.pi

		# Converts to degrees if mode selected
		if (mode.upper() == "DEG"):
			return np.rad2deg(theta)
		return theta

	@classmethod
	def __get_intersecting_line(cls, point1: ut.Point_Type, point2: 
		ut.Point_Type, mode: str = "RAD") -> (ut.Line_Type | None):
		"""
		Finds the line that intersects two points defined with polar
		coordinates.

		Parameters
		----------
		`point1`, `point2` : `tuple` of two `float`
			The points to find the intersection between. Points must be
			in the form '(x, y)'.
		`mode`: `["RAD", "DEG"]`, optional, default=`"RAD"`
			The units used by the inputted theta values (can be either
			radians or degrees). By default, radians are used.
			
		Returns
		-------
		`tuple` of `float`
			The line that intersects the points in the form '(rho,
			theta)', where 'rho' is the distance from the origin and
			'theta' is the line normal from the origin. theta must be in
			the interval [0, π) radians (or [0, 180) degrees).
		`None`
			If the two points are the same
		"""
		ALLOWED_MODES = ["RAD", "DEG"]
		if (mode.upper() not in ALLOWED_MODES):
			raise ValueError(f"Unrecognised mode '{mode}' inputted. " +
				"Supported modes are radians ('RAD') and degrees ('DEG').")

		theta = OffsideDetector.__get_intersecting_angle(point1, point2, "RAD")
		if (theta is None):
			return None

		x1, y1 = point1
		# x2, y2 = point2

		# Get rho distance from origin of intersecting line
		get_rho = lambda x, y, t: np.sin(t) * y + np.cos(t) * x
		rho = get_rho(x1, y1, theta)
		# rho = get_rho(x2, y2, theta)

		# Converts to degrees if mode selected
		if (mode.upper() == "DEG"):
			theta = np.rad2deg(theta)
		return rho, theta


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

	def __get_player_mask(self, object_mask: im.Image, line_mask: im.Image
		) -> im.Image:
		"""
		Creates a mask of the players in a football image by subtracting
		a pitch line mask from a mask of the objects.

		Parameters
		----------
		`object_mask` : `image.Image`
			The mask of the objects in the football image. See 
			`__get_object_mask` for creating this.
		`line_mask` : `image.Image`
			The mask of the pitch lines in the football image. See 
			`__get_line_mask` for creating this.

		Returns
		-------
		`image.Image`
			The mask containing the player pixels in the image.
		"""
		# Subtracts the line mask from object mask to leave mostly players
		player_mask = object_mask.get() - line_mask.get()
		player_image = im.Image(player_mask, "BINARY")

		# Open to remove noise
		player_image = player_image.morphology("open", 
			**self.param("player mask open"))
		return player_image

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

		if ((bounded_lines is None) or (len(bounded_lines) < 2)):
			return None

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
	
	
	def __get_offside_point(self, lines: list[ut.Line_Type]) -> (ut.Point_Type
		| None):
		"""
		Finds the vanishing point used to calculate offsides around. If
		two lines are provided, the vanishing point is where they
		intersect. If more than two lines are provided, the point is the
		average of all line intersections. It is not recommended to
		provide more than three lines. 
		
		Parameters
		----------
		`lines`: `list` of `tuple`, of two `float`
			A list of lines to be used to find the vanishing point.
			Lines must be in polar form, '(rho, theta)', where 'rho' is the
			distance from the origin and 'theta' is the line normal from
			the origin. theta must be in the interval [0, π) radians.

		Returns
		-------
		`tuple` of two `float`
			The vanishing point.
		"""
		if (len(lines) < 2):
			return None
		points = []
		for i in range(len(lines)):
			for j in range(i + 1, len(lines)):
				point = OffsideDetector.__get_intersecting_point(lines[i], 
					lines[j])
				if (point is not None):
					points.append(point)

		if (len(points) > 1):
			return tuple(np.average(np.array(points), axis=0))
		return points[0]

	def __get_offside_lines(self, offside_point: ut.Point_Type, player_points: 
		np.ndarray) -> np.ndarray:
		"""
		Finds the angle between the vanishing (offside) point and each
		player point.
		
		Parameters
		----------
		`offside_point`: `tuple` of two `float`
			The vanishing point to compare each player angle to. Points
			must be in the form '(x, y)'.
		`player_points`: `numpy.ndarray` of shape `(k, 2)`
			A list of `k` player points. Points must be in the form 
			'(x, y)'.

		Returns
		-------
		`numpy.ndarray` of shape `(k, 2)`
			A list of the `k` offside lines for each player. Lines in 
			polar form, '(rho, theta)', where 'rho' is the distance from 
			the origin and 'theta' is the line normal from the origin. 
			theta must be in the interval [0, π) radians.
		"""
		lines = []
		for point in player_points:
			line = OffsideDetector.__get_intersecting_line(offside_point, 
				tuple(point))
			if (line is not None):
				lines.append(line)
		return np.array(lines)

	def __get_candidate_players(self, player_mask: im.Image) -> np.ndarray:
		"""
		Finds candidate blobs where players may be in the image.
		Identifies the blobs then removes blobs which don't lie within
		the height and width bounds.

		Parameters
		----------
		`player_mask` : `image.Image`
			The mask of the candidate player blobs in the football
			image. See `__get_player_mask` for creating this.

		Returns
		-------
		`numpy.ndarray` of shape `(k, 4)`
			The list of player bounding boxes. Each box is in the form
			'(x, y, w, h)' where x and y are the location of the upper
			left corner of the box in the image and w and h are the
			width and height of the box respectively.
		"""
		# Find the bounding boxes for the player blobs
		contours = player_mask.contours()
		boxes = im.Image.contour_boxes(contours)

		w_bound = self.param("player size bounds", "width")
		h_bound = self.param("player size bounds", "height")
		c = (w_bound[0] <= boxes[:, 2]) & (boxes[:, 2] <= w_bound[1]) & (
			h_bound[0] <= boxes[:, 3]) & (boxes[:, 3] <= h_bound[1])
		bounded_boxes = boxes[c]

		return bounded_boxes

	
	def __get_players(self, blur_image: im.Image, player_mask: im.Image,
		candidate_boxes: np.ndarray, grass_colour: np.ndarray) -> np.ndarray:
		"""
		Refines the player bounding boxes found from 
		`__get_candidate_players`. For each candidate player box, the
		player's dominant colour is calculated. If this colour is too
		similar to the dominant grass colour, it is removed. If another
		sufficiently different dominant colour is found, the box is
		found to contain two players.

		Parameters
		----------
		`blur_image` : `image.Image`
			The blurred football image to find players in, in HSV colour
			space.
		`player_mask` : `image.Image`
			The mask of the candidate player blobs in the football
			image. See `__get_player_mask` for creating this.
		`candidate_boxes` : `numpy.ndarray` of shape `(k, 4)`
			The list of candidate player bounding boxes. Each box is in
			the form '(x, y, w, h)' where x and y are the location of 
			the upper left corner of the box in the image and w and h
			are the width and height of the box respectively.
		`grass_colour` : `numpy.ndarray` of shape `(3,)`
			The dominant HSV colour of the grass. If the grass has
			multiple dominant colours, they should be averaged for each
			channel.

		Returns
		-------
		`numpy.ndarray` of shape `(k, 4)`
			The list of player bounding boxes.
		"""
		# Gets the mask and blur image arrays (which is converted to HSV)
		mask = player_mask.get()
		image = blur_image.convert("HSV")
		if (image is None):
			image = blur_image.get()
		else:
			image = image.get()

		# Fetches the parameters used during the function loop
		dc_params = self.param("player dominant colour")
		colour_thresh = self.param("player colour similarity", "threshold")
		grass_thresh = self.param("player grass similarity", "threshold")
		mask_params = self.param("player boxes chromatic")
		mask_params_achromatic = self.param("player boxes achromatic")
		open_params = self.param("player boxes open")
		dilate_params = self.param("player boxes dilate")
		# w_bound = self.param("player size bounds", "width")
		# h_bound = self.param("player size bounds", "height")
		hsv_dims = im.COLOUR_SPACES["HSV"]["dimensions"]

		player_colours, player_boxes = [], []
		for i in range(len(candidate_boxes)):
			# Gets the current box's mask and image region
			x, y, w, h = candidate_boxes[i]
			player_mask = im.Image(mask[y: y + h, x: x + w], "BINARY")
			player_image = im.Image(image[y: y + h, x: x + w], 
			   blur_image.colour_space())

			# Finds the player's dominant colour, using the mask to ignore 
			# grass
			player_colour = player_image.dominant_colour(mask=player_mask, 
				**dc_params)
	
			# Splits the two dominant colours in the player box
			colours = [[], []]
			for j in range(len(player_colour)):
				colours[0].append(player_colour[j][0])
				if (player_colour[j].shape[0] > 1):
					colours[1].append(player_colour[j][1])
				else:
					colours[1].append(player_colour[j][0])
			colours = np.array(colours)

			# Compares the two dominant colours in the player box - if they're
			# too similar they're averaged and counted as one
			if not np.allclose(colours[0], colours[1]):
				colour_similarity = im.Image.hsv_distance(colours[0], 
					colours[1])
				if (colour_similarity < colour_thresh):
					if (colours[0][0] >= hsv_dims[0] // 2):
						colours[0][0] -= hsv_dims[0]
					if (colours[1][0] >= hsv_dims[0] // 2):
						colours[1][0] -= hsv_dims[0]
					colours = np.array([np.mean(colours, axis=0)])
					if (colours[0][0] < 0):
						colours[0][0] += hsv_dims[0]

			else:
				colours = colours[0:1]

			# Compares the player colours with the grass colour - if they're
			# too similar the blob is assumed to be noise
			length = colours.shape[0]
			for j in range(length):
				J = length - 1 - j
				grass_similarity = im.Image.hsv_distance(colours[J], 
					grass_colour)
				if (grass_similarity < grass_thresh):
					colours = np.delete(colours, J, axis=0)
			
			if (len(colours) > 1):
				new_boxes = []
				for j in range(len(colours)):
					J = length - 1 - j
					colour_tuple = [colours[J][0:1], colours[J][1:2], colours[J
						][2:3]]
					if (colours[J][1] < im.COLOUR_SPACES["HSV"][
						"achromatic threshold"]):
						mp = mask_params_achromatic
					else:
						mp = deepcopy(mask_params)
						if (colour_tuple[0][0] - mp["deviations"][0] < 0):
							colour_tuple[0] = np.append(colour_tuple[0], 
				   				(hsv_dims[0] + colour_tuple[0][0]))
							mp["sigma"][0] = np.append(mp["sigma"][0], (1))
						elif (colour_tuple[0][0] + mp["deviations"][0] >= 
							hsv_dims[0]):
							colour_tuple[0] = np.append(colour_tuple[0], (
								-colour_tuple[0][0]))
							mp["sigma"][0] = np.append(mp["sigma"][0], (1))

					new_player_mask = player_image.create_mask(colour_tuple, 
						**mp)
					new_player_mask = new_player_mask.morphology("open", 
						**open_params)
					new_player_mask = new_player_mask.morphology("dilate", 
						**dilate_params)
					new_player_mask = np.bitwise_and(player_mask.get(), 
						new_player_mask.get())
					new_player_mask = im.Image(new_player_mask, "BINARY")
					contours = new_player_mask.contours()
					if (len(contours) != 0):
						boxes = im.Image.contour_boxes(contours) + [x, y, 0, 0]
						box_areas = boxes[:, 2] * boxes[:, 3]
						box = boxes[box_areas.argmax()]
						new_boxes.append(box)
						player_colours.append(colours[J])

				player_boxes += new_boxes
			elif (len(colours) == 1):
				player_boxes.append(candidate_boxes[i])
				player_colours.append(colours[0])

		return np.array(player_colours), np.array(player_boxes)

	def __get_team_classifications(self, player_colours: np.ndarray) -> tuple[
		np.ndarray, tuple[np.ndarray, np.ndarray]]:
		"""
		Classfies the players into teams using their dominant HSV colour
		and the DBSCAN Clustering Algorithm.

		Parameters
		----------
		`player_colours` : `numpy.ndarray` of shape `(k, 3)`
			A list of the `k` dominant HSV colours of each player.

		Returns
		-------
		`tuple` of:
			`numpy.ndarray` of shape `(c, 3)`
				The average colour of each of the `c` classes.
			`numpy.ndarray` of shape `(k,)`
				The classification of each player.
		"""
		# Clusters player colours into distinct groups
		model = DBSCAN(**self.param("player dbscan"), metric=
			im.Image.hsv_distance).fit(player_colours)

		# Finds the average team colours
		means = []
		for i in range(np.amax(model.labels_) + 1):
			means.append(np.mean(player_colours[model.labels_ == i], axis=0))
		mean_colours = np.array(means)

		return mean_colours, model.labels_

	def __get_defending_team(self, player_angles: np.ndarray, player_teams: 
		np.ndarray) -> int:
		"""
		Picks the label which best suits the defending team, based on a
		sample of the players closest to the goal.

		Parameters
		----------
		`player_angles` : `numpy.ndarray` of shape `(k,)`
			A list of the offside angle for each player.
		`player_teams` : `numpy.ndarray` of shape `(k,)`
			A list of the team classifications of each player.

		Returns
		-------
		`int`
			The label of the defending team.
		"""
		sorted_angles = np.argsort(player_angles)

		# Tallies up the number of players for each team closest to the goal,
		# assuming the defending team will have more players close to the goal
		team_count = np.zeros(np.amax(player_teams) + 1)
		players_to_sample = self.param("defending team", "sample number")
		for i in range(np.minimum(players_to_sample, sorted_angles.shape[0])):
			team = player_teams[sorted_angles[i]]
			if (team != -1):
				team_count[team] += 1

		# If the two largest classes have an equal number of players, expands
		# the search until one has more
		sorted_teams = team_count.argsort()
		while ((team_count[sorted_teams[0]] == team_count[sorted_teams[1]]) and
			(i < sorted_angles.shape[0] - 1)):
			i += 1
			team = player_teams[sorted_angles[i]]
			if (team != -1):
				team_count[team] += 1
			sorted_teams = team_count.argsort()

		if (team_count[sorted_teams[0]] == team_count[sorted_teams[1]]):
			return 0

		return team_count.argmax()


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
		player_mask = self.__get_player_mask(object_mask, line_mask)

		# Find pitch lines in the line mask
		pitch_lines = self.__get_lines(line_mask)

		# If the results found are poor, attempts to reverse the image
		reverse = False
		if ((pitch_lines is None) or (len(pitch_lines) < 2)):
			pitch_lines = self.__get_lines(line_mask.flip("h"))
			if ((pitch_lines is None) or (len(pitch_lines) < 2)):
				return None
			reverse = True

		if reverse:
			player_mask = player_mask.flip("h")
			player_image = player_image.flip("h")
	
		# Finds candidate player blobs
		candidate_player_boxes = self.__get_candidate_players(player_mask)

		# Refines these blobs to detect occlusisons, and finds each players colour
		grass_colour_tuple = np.array((np.mean(grass_colour[0]), np.mean(
			grass_colour[1]), np.mean(grass_colour[2])))
		player_colours, player_boxes = self.__get_players(player_image.convert(
			"HSV"), player_mask, candidate_player_boxes, grass_colour_tuple)
		if (len(player_boxes) < 1):
			return None
		player_points = np.stack((player_boxes[:, 0], player_boxes[:, 1] + 
			player_boxes[:, 3])).T

		# Finds the offside line for each player
		offside_point = self.__get_offside_point(pitch_lines)
		player_lines = self.__get_offside_lines(offside_point, player_points)

		# Classifies the players on kit colour
		team_colours, player_teams = self.__get_team_classifications(
			player_colours)
		if (len(np.bincount(player_teams[player_teams != -1])) < 2):
			return None
		
		# Assigns a defending and attacking team
		classes = {}
		classes["defending"] = self.__get_defending_team(player_lines[:, 1],
			player_teams)
		non_defenders = player_teams[player_teams != classes["defending"]]
		classes["attacking"] = np.bincount(non_defenders + 1)[1:].argmax()
