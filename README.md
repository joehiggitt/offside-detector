# Tracking Football Players - Offside Detector
A project aiming to identify football players in broadcast images of football matches and classify them as offside or not. I completed this work for my third-year dissertation project, as part of my MEng (Hons) Computer Science degree at the University of Manchester.

This README provides instructions on how to install and use the offside detector.

## Report & Screencast
In addition to writing the project code, this project involved writing a 10,000 word report and creating an 8 minute video explaining the project. The report has been included in this repository (`offside-detector/Report.pdf`). Both the report and screencast can also be accessed through the following link: https://1drv.ms/f/s!Ast9SE_pAWg69AaCPK7Xhe58O0f7?e=eeohWP.

## Version
This tool is currently at version 1.0.

## Requirements
- Python 3.11
- OpenCV 4.6
- NumPy 1.23.4
- Sklearn 1.1.3

## Installation
Install the latest versions of Python and PIP, instructions for which can be found at the following websites:

- Python installation: https://www.python.org/downloads/
- PIP installation: https://pip.pypa.io/en/stable/installation/

Clone the git repository onto the server or computer you want to run the detector on.
```
$ git clone <your git clone address>
$ cd offside-detector
```
Once cloned, install the Python requirements with PIP.
```
$ pip install -r requirements.txt
```

## Project Structure
This project repository contains the following:

    offside-detector/
    ├── src/
    │   ├── image.py        - code for storing images
    │   ├── detector.py     - code for detecting offsides
    │   ├── utils.py        - code containing generic functions
    │   ├── parameters.py   - code containing parameter dictionary
    │   └── run.py          - code for running CLI
    ├── .gitignore
    ├── README.md
    ├── Report.pdf          - project report
    └── requirements.txt    - list of project requirements


The code to run the offside detector is located in the `/src` directory. There are five Python files: `image.py`, `detector.py`, `utils.py`, `parameters.py` and `run.py`.

- `image.py` contains code which encapsulates OpenCV images into a data structure that also keeps track of colour space information. This class provides many methods which call OpenCV operations on the image.

- `detector.py` contains code which performs the offside detection. The main class for this, `OffsideDetector`, is initialised with a parameter dictionary and can be used on images by calling `get_offsides(image)`. The result of this is stored in a separate class, `OffsideResult`, which contains methods to visualise the data found.

- `utils.py` contains generic code used by both the other files, including a function to handle kwargs lists and some type aliases.

- `parameters.py` contains a dictionary called `DEFAULT_PARAMS`, which is used as the default parameter data structure in `OffsideDetector`. To use different parameter values, either create a new parameter dictionary and pass that in to `OffsideDetector` (this must have the same format as `DEFAULT_PARAMS`) or directly edit the values in `parameters.py`.

- `run.py` provides a command line interface to run the project with.

## Running the Code
### CLI Usage
To run the program using `run.py`, use the following command:

    $ python run.py -i "input_image" [-o "output_image"] [-p "parameter_dictionary"]

The command line interface has the following options:

    -i, --file_in "input_image"
        The path to the input image file. Can be ".jpg" or ".png". Required.
    -o, --file_out "output_image"
        The path to the output image file. Can be ".jpg" or ".png". Optional.
    -p, --params "parameter_dictionary"
        Name of parameter dictionary to use, defined inside parameters.py. Default is 'DEFAULT_PARAMS'.
    -h, --help
        Shows a help message and exits.


### Using with Other Code
The following code shows how an offside can be detected using the code.

```
import image as im
import detector as dt

filepath = "path/to/image/file"
image = im.Image.open(filepath)
detector = dt.OffsideDetector()
result = detector.get_offsides(image)
output_image = result.create_visuals()
output_image.display()
```
