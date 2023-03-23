from __future__ import annotations
from typing import Any

import numpy as np

Params_Type = dict[str, dict[str, Any]]
Point_Type = tuple[float, float]
Line_Type = tuple[float, float]
Colour_Type = tuple[int, int, int] | np.ndarray
Mutiple_Colour_Type = tuple[np.ndarray, np.ndarray, np.ndarray]
Bounds_Type = tuple[int, int]
Multiple_Bounds_Type = tuple[np.ndarray, np.ndarray]
Kernel_Size_Type = int | tuple[int, int]

def handle_kwargs(inputted_kwargs: dict[str, Any], allowed_kwargs: dict[str, 
    list[type]]):
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
                raise AttributeError(f"An unexpected keyword argument " +
                    f"'{key}' was received.")
            if (type(val) not in allowed_kwargs[key]):
                raise TypeError(f"Keyword argument '{key}' was expecting" +
                    f" a value of type {allowed_kwargs[key]}, but " +
                    f"received {type(val)}.")
