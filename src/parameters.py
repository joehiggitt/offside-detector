import numpy as np
import utils as ut

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
	"lines top hat": {
		"kernel_size": (3, 19),
		"iterations": 1,
	},
	"lines erode": {
		"kernel_size": 3,
		"iterations": 2,
	},
	"lines threshold": {
		"threshold": 20,
	},
	"lines hough": {
		"threshold": 200,
	},
	"lines bounds": {
		"rho": (0, 1400),
		"theta": (0.9, 1.4),
	},
	"lines normalise": {
		"scale": 1 / 1000,
	},
	"lines dbscan": {
		"eps": 0.03,
		"min_samples": 2,
	},
	"lines number": {
		"number": 2,
	},
	"player mask open": {
		"kernel_size": 3,
		"iterations": 2,
	},
	"player mask close": {
		"kernel_size": 5,
		"iterations": 10,
	},
	"player size bounds": {
		"width": (20, 200),
		"height": (50, 250),
	},
	"player box increase": {
		"width": 20,
		"height": (50, 20),
	},
	"player dominant colour": {
		"k": 0.2,
		"c": 0.4,
		"n": 2,
	},
	"player colour similarity": {
		"threshold": 140,
	},
	"player grass similarity": {
		"threshold": 140,
	},
	"player boxes chromatic": {
		"sigma": [np.ones(1)] * 3,
		"deviations": (10, 5, 30),
	},
	"player boxes achromatic": {
		"sigma": [np.ones(1)] * 3,
		"deviations": (255, 20, 30),
	},
	"player boxes open": {
		"kernel_size": 3,
		"iterations": 2,
	},
	"player boxes dilate": {
		"kernel_size": 3,
		"iterations": 5,
	},
	"player dbscan": {
		"eps": 70,
		"min_samples": 2,
	},
	"defending team": {
		"sample number": 5,
	},
    "goalkeeper": {
		"angle": 1.4,
	}
}
