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
	A class to encapsulate OpenCV images and store the colour space.

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
	`channels(channel_num: int)` : `tuple[np.ndarray] | np.ndarray`
		Returns the separate colour channels in the image array.
	`histogram(channel_num: int, normalise: bool)` : `tuple[np.ndarray] | np.ndarray`
		Returns the histograms of the separate colour channels in the image.
	`colour_space()` : `str`
		Returns the colour space of the image.
	`convert(colour_space: str)` : `bool`
		Converts the image from one colour space to another
	`display(**kwargs: dict)`
		Displays the image in a window.

	"""
	def __init__(self, image: np.ndarray, colour_space: str):
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
			raise ValueError(f"Invalid colour space '{colour_space}' provided for image.")
		
		self.__image:  np.ndarray = image
		self.__colour: str        = colour_space.upper()

	@classmethod
	def open(cls, filepath: str):
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
			`True` if the file was successfully saved, `False` otherwise.
		"""
		try:
			result = cv.imwrite(filepath, self.get())
		except cv.error:
			raise FileNotFoundError("The filepath provided could not be written to.")
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

	def channels(self, channel_num: int=None) -> tuple[np.ndarray] | np.ndarray:
		"""
		Returns the separate colour channels in the image array.

		Parameters
		----------
		`channel_num`: `int`, optional
			The specific channel to return.

		Returns
		-------
		`tuple` of `numpy.ndarray`
			If `channel_num` not provided, a tuple of the colour channel arrays.
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
				raise ValueError(f"Invalid channel number '{channel_num}' provided. Channel number should be between 0 and {len(channels) - 1}.")
		
		return channels

	def histogram(self, channel_num: int=None, normalise: bool=True) -> tuple[np.ndarray] | np.ndarray:
		"""
		Returns the histograms of the separate colour channels in the image.

		Parameters
		----------
		`channel_num`: `int`, optional
			The specific channel histogram to return.
		`normalise`: `bool`, optional, default = True
			Whether the histogram should be normalised (so that the sum is 1) (True), or not (False).

		Returns
		-------
		`tuple` of `numpy.ndarray`
			If `channel_num` not provided, a tuple of the histogram arrays.
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
				hist = cv.calcHist(channels, [channel_num], None, [dimensions[channel_num]], (0, dimensions[channel_num]), accumulate=False)
				if normalise:
					cv.normalize(hist, hist, 1, 0, cv.NORM_L2)
				return hist
			else:
				raise ValueError(f"Invalid channel number '{channel_num}' provided. Channel number should be between 0 and {len(channels) - 1}.")
		
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
		Converts the image from one colour space to another

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
			raise ValueError(f"Invalid colour space '{colour_space}' provided for conversion. See list of supported colour spaces.")
		
		try:
			code = COLOUR_SPACES[self.__colour]["convert"][colour_space]
		except KeyError:
			return False

		self.__image = cv.cvtColor(self.__image, code)
		self.__colour = colour_space
		return True

	@classmethod
	def __handle_kwargs(cls, kwargs: dict[str, any], allowed_kwargs: dict[str, list[type]]):
		"""
		Function that handles variable length keyword parameter lists.
		Ensures that the kwargs passed into the function match the allowed kwargs and their types.

		Parameters
		----------
		`kwargs` : `dict` of `str: Any`
			Keyword argument dictionary passed into some function.
		`allowed_kwargs` : `dict` of `str: list` (of `type`)
			Dictionary containing allowed keyword arguments and their allowed types.

		Raises
		------
		`AttributeError`
			If an unrecognised keyword argument passed in.
		`TypeError`
			If a keyword argument passed in has the incorrect type.
		"""
		if (len(kwargs) > 0):
			for key, val in kwargs.items():
				if (key not in allowed_kwargs):
					raise AttributeError(f"An unexpected keyword argument '{key}' was received.")
				if (type(val) not in allowed_kwargs[key]):
					raise TypeError(f"Keyword argument '{key}' was expecting a value of type {allowed_kwargs[key]}, but received {type(val)}.")

	def __create_display(self, **kwargs: (str | tuple[int, int])) -> tuple[function, str]:
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
			`wait` : `bool`
				Whether the program should pause for the display to be shown.
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

	