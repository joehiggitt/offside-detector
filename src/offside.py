"""
A module for detecting offsides in images of broadcast football matches.
Written by Joe Higgitt.
"""

from __future__ import annotations
from typing import Any
from copy import deepcopy
import numpy as np
import cv2 as cv

import image as im


Params_Dict = dict[str, dict[str, Any]]
DEFAULT_PARAMS: Params_Dict = {
	
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

	"""
	def __init__(self, params: Params_Dict=DEFAULT_PARAMS) -> OffsideDetector:
		"""
		Creates an offside detector object for an image of a football
		match.

		Parameters
		----------
		`filepath` : `str`
			The filepath of the image.
		`params` : `dict` of `str: dict` of `str: int` or `str: float`
			The type of bound

		Returns
		-------
		`OffsideDetector`
			An offside detector object.

		Raises
		------
		`FileNotFoundError`
			If the file provided doesn't exist or contain a valid image.
		`ValueError`
			If the provided parameter dictionary doesn't contain the
			required parameters or isn't in the right format.
		"""
		self.__params: dict     = params

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
			If `param` not provided, the keyword argument dictionary to pass into the function.
		`Any`
			If `param` provided, the parameter value.

		Raises
		------
		`KeyError`
			If `operation` or `param` is not recognised.
		"""
		if (operation.lower() not in self.__params.keys()):
			raise ValueError(f"Operation '{operation}' not recognised.")

		if param is None:
			return self.__params[operation.lower()]
		
		if (param.lower() not in self.__params[operation.lower()].keys()):
			raise ValueError(f"Parameter '{param}' not recognised for operation '{operation}'.")
		
		return self.__params[operation.lower()][param.lower()]
	