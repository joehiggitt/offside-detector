import argparse

import detector as dt
import image as im
import parameters as pm

parser = argparse.ArgumentParser(prog="run.py", description="Finds" +
    " offside players in football images.")

parser.add_argument("-i", "--file_in", required=True, help="Input filepath.")
parser.add_argument("-p", "--params", default="DEFAULT_PARAMS", help="Name of"
    + " parameter dictionary to use, defined inside parameters.py. Default is"
    + " 'DEFAULT_PARAMS'.")
parser.add_argument("-o", "--file_out", help="Optional output filepath.")

args = parser.parse_args()

if (args.file_in is None):
    print("No file provided.")
    exit()
print(args.params)
exec(f"params = pm.{args.params}")

detector = dt.OffsideDetector(params)
img = im.Image.open(args.file_in)
result = detector.get_offsides(img)
if (result is not None):
    out = result.create_visual()
    out.display(title = "Offsides")

    if (args.file_out is not None):
        out.save(args.file_out)
