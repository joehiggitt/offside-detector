from __future__ import annotations
from typing import Any
import cv2 as cv
import numpy as np


COLOUR_SPACES = {
	"BGR": {
		"dimensions": [256, 256, 256],
		"convert": {
			"RGB": cv.COLOR_BGR2RGB,
			"HSV": cv.COLOR_BGR2HSV,
			"GREY": cv.COLOR_BGR2GRAY,
		}
	},
	"RGB": {
		"dimensions": [256, 256, 256],
		"convert": {
			"BGR": cv.COLOR_RGB2BGR,
			"HSV": cv.COLOR_RGB2HSV,
			"GREY": cv.COLOR_RGB2GRAY,
		}
	},
	"HSV": {
		"dimensions": [180, 256, 256],
		"convert": {
			"BGR": cv.COLOR_HSV2BGR,
			"RGB": cv.COLOR_HSV2RGB,
		}
	},
	"GREY": {
		"dimensions": [256],
		"convert": {
			"BGR": cv.COLOR_GRAY2BGR,
			"RGB": cv.COLOR_GRAY2RGB,
		}
	},
}

class Image:
	"""
	A class to encapsulate OpenCV images (stored as `numpy.ndarray`) and
	save the colour space. Methods performing operations on the current
	image return a new image object containing the altered image.

	Class Methods
	-------
	`Image(image: numpy.ndarray, colour_space: str)` : `Image`
		Creates an Image object from an image array.
	`Image.open(filepath: str)` : `Image`
		Creates an Image object from a filepath.
	`Image.display_images(*images: tuple)`
		Displays multiple images in multiple windows.

	Methods	
	-------
	`save(filepath: str)` : `bool`
		Saves the image in the filepath provided.
	`get()` : `numpy.ndarray`
		Returns the image array.
	`channels(channel_num: int)` : `numpy.ndarray` or 
	`tuple[numpy.ndarray]`
		Returns the separate colour channels in the image array.
	`histogram(channel_num: int, normalise: bool)` : `numpy.ndarray`
	or `tuple[numpy.ndarray]`
		Returns the histograms of the separate colour channels in the
		image.
	`colour_space()` : `str`
		Returns the colour space of the image.
	`convert(colour_space: str)` : `bool`
		Converts the image from one colour space to another
	`display(**kwargs: dict)`
		Displays the image in a window.
	`dominant_colour(k: float)` : `numpy.ndarray`
		Finds the dominant colour of the image using a tight bound about
		the peak colour.
	`standard_deviation(means: numpy.ndarray, k: float)` : 
	`numpy.ndarray`
		Finds the standard deviation in the colour in the image.
	`create_mask(colour: numpy.ndarray, sigma: numpy.ndarray,
	deviations: float)` : `numpy.ndarray`
		Creates a mask of the image, where pixels are included if
		they're within a defined number of standard deviations from the
		colour.
	`apply_mask(mask: numpy.ndarray)` : `Image`
		Masks the image.
	"""
	def __init__(self, image: np.ndarray, colour_space: str) -> Image:
		"""
		Creates an Image object from an image array.

		Parameters
		----------
		`image` : `numpy.ndarray`
			An image array.
		`colour_space` : `{"BGR", "RGB", "HSV", "GREY"}`
			The colour space the image is represented by.

		Returns
		-------
		`Image`
			An Image object.

		Raises
		------
		`ValueError`
			If the value of `colour_space` is not recognised.
		"""
		if (colour_space.upper() not in COLOUR_SPACES.keys()):
			raise ValueError(
				f"""Invalid colour space '{colour_space}' provided for 
				image."""
			)
		
		self.__image:  np.ndarray = image
		self.__colour: str        = colour_space.upper()

	@classmethod
	def open(cls, filepath: str) -> Image:
		"""
		Creates an Image object from a filepath.

		Parameters
		----------
		`filepath` : `str`
			The filepath to the image being opened.

		Returns
		-------
		`Image`
			An image object.

		Raises
		------
		`FileNotFoundError`
			If the file provided doesn't exist or contain a valid image.
		"""
		image_in = cv.imread(filepath, cv.IMREAD_UNCHANGED)
		if (image_in is None):
			raise FileNotFoundError("The file provided could not be opened.")
		return Image(image_in, "BGR")

	def save(self, filepath: str) -> bool:
		"""
		Saves the image in the filepath provided.

		Parameters
		----------
		`filepath` : `str`
			The filepath to the image being opened.

		Returns
		-------
		`bool`
			`True` if the file was successfully saved, `False`
			otherwise.
		"""
		try:
			result = cv.imwrite(filepath, self.get())
		except cv.error:
			raise FileNotFoundError(
				"The filepath provided could not be written to."
			)
		return result

	def get(self) -> np.ndarray:
		"""
		Returns the image array.

		Returns
		-------
		`numpy.ndarray`
			The image array.
		"""
		return self.__image

	def channels(self,
			channel_num: int=None
		) -> tuple[np.ndarray] | np.ndarray:
		"""
		Returns the separate colour channels in the image array.

		Parameters
		----------
		`channel_num`: `int`, optional
			The specific channel to return.

		Returns
		-------
		`tuple` of `numpy.ndarray`
			If `channel_num` not provided, a tuple of the colour
			channel arrays.
		`numpy.ndarray`
			If `channel_num` provided, the colour channel array.

		Raises
		------
		`ValueError`
			If `channel_num` is out of bounds
		"""
		channels = cv.split(self.__image)

		if (channel_num is not None):
			if ((channel_num >= 0) and (channel_num < len(channels))):
				channels = channels[channel_num]
			else:
				raise ValueError(
					f"""Invalid channel number '{channel_num}' provided. 
					Channel number should be between 0 and 
					{len(channels) - 1}."""
				)
		
		return channels

	def histogram(self,
	    	channel_num: int=None,
			normalise: bool=True
		) -> tuple[np.ndarray] | np.ndarray:
		"""
		Returns the histograms of the separate colour channels in the
		image.

		Parameters
		----------
		`channel_num`: `int`, optional
			The specific channel histogram to return.
		`normalise`: `bool`, optional, default = True
			`True` if the histogram should be normalised, so that the sum
			is 1, `False` otherwise.

		Returns
		-------
		`tuple` of `numpy.ndarray`
			If `channel_num` not provided, a tuple of the histogram
			arrays.
		`numpy.ndarray`
			If `channel_num` provided, the histogram array.

		Raises
		------
		`ValueError`
			If `channel_num` is out of bounds
		"""
		channels = self.channels()
		dimensions = COLOUR_SPACES[self.colour_space()]["dimensions"]

		# If a channel number is provided
		if (channel_num is not None):
			if ((channel_num >= 0) and (channel_num < len(channels))):
				channels = channels[channel_num]
				hist = cv.calcHist(channels, [channel_num], None,
		       		[dimensions[channel_num]], (0, dimensions[channel_num]),
					accumulate=False
				)
				if normalise:
					cv.normalize(hist, hist, 1, 0, cv.NORM_L2)
				return hist
			else:
				raise ValueError(
					f"""Invalid channel number '{channel_num}' provided. 
					Channel number should be between 0 and 
					{len(channels) - 1}."""
				)
		
		hists = []
		for i in range(len(channels)):
			hists.append(cv.calcHist(channels, [i], None, [dimensions[i]], (0, dimensions[i]), accumulate=False))
			if normalise:
				cv.normalize(hists[i], hists[i], 1, 0, cv.NORM_L2)
		return tuple(hists)

	def colour_space(self) -> str:
		"""
		Returns the colour space of the image.

		Returns
		-------
		`{"BGR", "RGB", "HSV", "GREY"}`
			The colour space of the image.
		"""
		return self.__colour

	def convert(self, colour_space: str) -> bool:
		"""
		Converts the image from one colour space to another.

		Parameters
		----------
		`colour_space` : `{"BGR", "RGB", "HSV", "GREY"}`
			The new colour space of the image.

		Returns
		-------
		`bool`
			`True` if the conversion was successful, `False` otherwise.

		Raises
		------
		`ValueError`
			If the value of `colour_space` is not recognised.
		"""
		colour_space = colour_space.upper()
		if (colour_space not in COLOUR_SPACES.keys()):
			raise ValueError(
				f"""Invalid colour space '{colour_space}' provided for
				conversion. See list of supported colour spaces.""")
		
		try:
			code = COLOUR_SPACES[self.__colour]["convert"][colour_space]
		except KeyError:
			return False

		self.__image = cv.cvtColor(self.__image, code)
		self.__colour = colour_space
		return True

	@classmethod
	def __handle_kwargs(cls,
			inputted_kwargs: dict[str, Any],
			allowed_kwargs: dict[str, list[type]]
		):
		"""
		Function that handles variable length keyword parameter lists.
		Ensures that the kwargs passed into the function match the
		allowed kwargs and their types.

		Parameters
		----------
		`inputted_kwargs` : `dict` of `str: Any`
			Keyword argument dictionary passed into some function.
		`allowed_kwargs` : `dict` of `str: list` (of `type`)
			Dictionary containing allowed keyword arguments and their
			allowed types.

		Raises
		------
		`AttributeError`
			If an unrecognised keyword argument passed in.
		`TypeError`
			If a keyword argument passed in has the incorrect type.
		"""
		if (len(inputted_kwargs) > 0):
			for key, val in inputted_kwargs.items():
				if (key not in allowed_kwargs):
					raise AttributeError(
						f"An unexpected keyword argument '{key}' was received."
					)
				if (type(val) not in allowed_kwargs[key]):
					raise TypeError(
						f"""Keyword argument '{key}' was expecting a value of
						type {allowed_kwargs[key]}, but received
						{type(val)}."""
					)

	def __create_display(self,
			**kwargs: (str | tuple[int, int])
		) -> tuple[function, str]:
		"""
		Creates a `cv2.imshow` function call for use when displaying a single window or multiple windows.

		Parameters
		----------
		`**kwargs` : `dict`, optional
			Extra arguments for the display window:

			`title` : `str`
				Title of the window.
			`size` : `tuple` of `int`
				Size of the window.
			`location` : `tuple` of `int`
				Screen location of the window.

		Returns
		-------
		`tuple` of `function` and `str`
			A tuple containing the `cv2.imshow` function, which creates custom window, and its title.
		"""
		ALLOWED_KWARGS = {"title": [str], "size": [tuple], "location": [tuple]}
		Image.__handle_kwargs(kwargs, ALLOWED_KWARGS)
		
		# Creates window with title
		if ("title" in kwargs):
			title = kwargs["title"]
		else:
			title = "Image"
		cv.namedWindow(title, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)

		# Modifies window with inputted parameters if provided
		if ("size" in kwargs):
			cv.resizeWindow(title, kwargs["size"][0], kwargs["size"][1])
		if ("location" in kwargs):
			cv.moveWindow(title, kwargs["location"][0], kwargs["location"][1])

		# Returns display function image and waits
		return cv.imshow, title

	def display(self, **kwargs: (str | tuple[int, int])):
		"""
		Displays the image in a window.

		Parameters
		----------
		`**kwargs` : `dict`, optional
			Extra arguments for the display window:

			`title` : `str`
				Title of the window.
			`size` : `tuple` of `int`
				Size of the window.
			`location` : `tuple` of `int`
				Screen location of the window.
		"""
		# Converts the image to the desired colour space
		ALLOWED_SPACES = ["BGR", "GREY", "BINARY"]
		colour_space = self.colour_space()
		if (colour_space not in ALLOWED_SPACES):
			self.convert("BGR")
		
		show, title = self.__create_display(**kwargs)
		show(winname=title, mat=self.get())

		cv.waitKey(0)

		# Converts image back to original colour space
		self.convert(colour_space)

	@classmethod
	def display_images(cls, *images: tuple[Image]):
		"""
		Displays multiple images in multiple windows.

		Parameters
		----------
		`*images` : `tuple` of `Image`, optional
			Image objects to display.
		"""
		ALLOWED_SPACES = ["BGR", "GREY", "BINARY"]		
		colour_spaces = []
		for i in range(len(images)):
			# Converts the image to the BGR colour space for display
			colour_space = images[i].colour_space()
			if (colour_space not in ALLOWED_SPACES):
				images[i].convert("BGR")
			colour_spaces.append(colour_space)
			
			show, title = images[i].__create_display(title=f"Image {i + 1}")
			show(winname=title, mat=images[i].get())

		cv.waitKey(0)

		# Converts images back to original colour space
		for i in range(len(images)):
			images[i].convert(colour_spaces[i])

	@classmethod
	def __find_peak_bounds(cls,
			histogram: np.ndarray,
			k: float
		) -> tuple[int, int]:
		"""
		Finds the lower and upper bounds of an image channel around the
		dominant colour. Algorithm proposed by Ekin et al (2003)
		"Automatic Soccer Video Analysis and Summarization".

		Parameters
		----------
		`histogram` : `numpy.ndarray`
			The image channel histogram to find the bounds of.
		`k` : `float`, `0 < k < 1`
			Parameter describing how different two adjacent channels
			must be to be considered a bound. (Recommended value of 0.2)

		Returns
		-------
		`tuple` of two `int`
			The dominant colour lower and upper bound.
		"""
		peak_candidates = np.argsort(histogram.flatten())
		is_found = False
		while (not is_found and (len(peak_candidates) > 0)):
			peak, peak_candidates = peak_candidates[-1], peak_candidates[:-1]

			range_vals = [peak, len(histogram) - peak]
			mults = [-1, 1]
			bounds = []
			for i in range(2):
				bound = peak
				for x in range(range_vals[i] + 1):
					bound = peak + (x * mults[i])
					if ((bound == 0) or (bound == len(histogram) - 1)):
						break
					if ((histogram[bound] >= k * histogram[peak]) and 
	 					(histogram[bound + mults[i]] < k * histogram[peak])):
						break
				bounds.append(bound)

			if ((bounds[0] != peak) or (bounds[1] != peak)):
				is_found = True

		return tuple(bounds)

	@classmethod
	def __get_mean(cls,
			histogram: np.ndarray,
			bounds: tuple[int, int]=None
		) -> float:
		"""
		Finds the mean colour in an image channel using a tight bound
		about the peak colour. Algorithm proposed by Ekin et al (2003)
		"Automatic Soccer Video Analysis and Summarization".

		Parameters
		----------
		`histogram` : `numpy.ndarray`
			The image channel's histogram to find the bound of.
		`bounds` : `tuple` of two `int`, optional
			The lower and upper bound of the mean calculation. If not
			provided, uses the entire histogram.
		
		Returns
		-------
		`float`
			The mean of the image channel.
		"""
		if (bounds is None):
			bounds = (0, len(histogram) - 1)

		hist = histogram[bounds[0]: bounds[1] + 1].flatten()
		n = np.sum(hist * np.arange(bounds[0], bounds[1] + 1))
		d = np.sum(hist)
		return (n / d)
	
	@classmethod
	def __get_standard_deviation(cls,
			channel: np.ndarray,
			mean: float,
			bounds: tuple[int, int]=None
		) -> float:
		"""
		Finds the standard deviation in the colour in an image channel.

		Parameters
		----------
		`channel` : `numpy.ndarray`
			The image channel to find the standard deviation of.
		`mean` : `float`
			The mean of the channel.
		`bounds` : `tuple` of two `int`, optional
			The lower and upper bound of the mean calculation. If not
			provided, uses the entire channel.

		Returns
		-------
		`float`
			The standard deviation of the image channel.
		"""
		flat_channel = channel.flatten()

		if (bounds is None):
			bounds = (0, flat_channel.max())

		bounded_channel = flat_channel[
			(bounds[0] <= flat_channel) & (flat_channel <= bounds[1])
		]

		total = np.sum(np.square((-1 * bounded_channel) + mean))
		return np.sqrt(total / bounded_channel.size)

	def dominant_colour(self, k: float) -> np.ndarray:
		"""
		Finds the dominant colour of the image using a tight bound about
		the peak colour. Algorithm proposed by Ekin et al (2003)
		"Automatic Soccer Video Analysis and Summarization".

		Parameters
		----------
		`k` : `float`, `0 < k < 1`
			Parameter describing how different two adjacent channels
			must be to be considered a bound. (Recommended value of 0.2)

		Returns
		-------
		`numpy.ndarray`
			The three colour channels of the dominant colour, or an
			empty list if the provided colour space is invalid.
		"""
		# Gets image histograms
		hists = self.histogram()

		# Calculates means for each image channel
		means = []
		for hist in hists:
			bounds = Image.__find_peak_bounds(hist, k)
			means.append(Image.__get_mean(hist, bounds))

		return np.array(means)
	
	def standard_deviation(self,
		means: np.ndarray,
		k: float
	) -> np.ndarray:
		"""
		Finds the standard deviation in the colour in the image.

		Parameters
		----------
		`means` : `numpy.ndarray`
			The mean colour in the image to find the standard deviation
			around.
		`k` : `float`, `0 < k < 1`
			Parameter describing how different two adjacent channels
			must be to be considered a bound. (Recommended value of 0.2)

		Returns
		-------
		`numpy.ndarray`
			The standard deviations of the image.

		Raises
		------
		`ValueError`
			If a mean isn't provided for each channel in the image.
		"""
		channels = self.channels()
		hists = self.histogram()
		if (len(channels) != len(means)):
			return ValueError(
				"A mean must be provided for each channel in the image."
			)
		
		standard_deviations = []
		for i in range(len(channels)):
			bounds = Image.__find_peak_bounds(hists[i], k)
			std_dev = Image.__get_standard_deviation(channels[i],
				means[i], bounds
			)
			standard_deviations.append(std_dev)
		return np.array(standard_deviations)

	def create_mask(self,
			colour: np.ndarray,
			sigma: np.ndarray,
			deviations: float=2
		) -> np.ndarray:
		"""
		Creates a mask of the image, where pixels are included if
		they're within a defined number of standard deviations from the
		colour.

		Parameters
		----------
		`colour` : `numpy.ndarray`
			The mean colour to mask on.
		`sigma` : `numpy.ndarray`
			The standard deviation around the mean colour.
		`deviations` : `float`, optional, default = 2
			The number of standard deviations from the mean to include
			in the mask.
		
		Returns
		-------
		`numpy.ndarray`
			An image mask.

		Raises
		------
		`ValueError`
			If the size of the mean colour array is different to that of
			the sigma array.
		"""
		if (len(colour) != len(sigma)):
			return ValueError(
				"The provided colour and sigma must have equal dimensions."
			)
		
		# TODO check values for red issues (360 -> 0 degree wrap issues)
		lower_bound = colour - (deviations * sigma)
		upper_bound = colour + (deviations * sigma)

		return cv.inRange(self.get(), lower_bound, upper_bound)

	def apply_mask(self, mask: np.ndarray) -> Image:
		"""
		Masks the image.

		Parameters
		----------
		`mask` : `numpy.ndarray`
			The mask to apply to the image.

		Returns
		-------
		`Image`
			The masked image.
		"""
		return Image(cv.bitwise_and(self.get(), self.get(), mask=mask), self.colour_space())

