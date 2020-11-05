import argparse
import logging
import os
import sys
from enum import Enum
try:
	import cv2 as cv
	import numpy as np
except:
	print("Could not import dependencies. " +
	      "Make sure you have them installed")
	print("Either do -")
	print(">python -m pip install --user numpy opencv-contrib-python")
	print(f">python {' '.join(sys.argv)}")
	print("OR")
	print(">python -m pip install poetry")
	print(">poetry install")
	print(f">poetry run python {' '.join(sys.argv)}")
	sys.exit()

dbg = logging.debug
err = logging.error
warn = logging.warning

########## CONFIG ##########
# Minimum dimensions allowed for the image
MIN_DIMENSIONS = 20, 20

# Logging level. Can be one of -
# logging.DEBUG=Verbose
# logging.WARNING=Important
# logging.ERROR=Errors
LOGGING_LEVEL = logging.DEBUG

# Step Length for image traversal
STEP_LENGTH = 10

# Ratio of black pixels to white pixels
BLACK_TO_WHITE_RATIO = 0.8

# Ratio of text segments to non-text segments
TEXT_TO_ROWS_RATIO = 0.4

# Binarization Thresholds
K1, K2, K3 = 50, 178, 229

# Intensity thresholds for black, gray and white pixels
B0, B1, G0, G1, W0, W1 = 0, 0, 0, 0, 0, 0
########## END CONFIG ##########


class EdgeType(Enum):
	BLACK = 1
	GRAY = 2
	NON_EDGE = 3


class ImageType(Enum):
	TEXT = 1
	NON_TEXT = 2


def open_image(filename):
	img = cv.imread(filename)
	if img is None:
		warn(f"'{filename}' not found")
		return None
	rows, cols = img.shape[:2]
	if rows < MIN_DIMENSIONS[0] or cols < MIN_DIMENSIONS[1]:
		warn(f"'{filename}' does not have enough rows/columns")
		return None
	dbg(f"Image has {rows} rows and {cols} columns")
	return img


def save_image(img_name, img):
	cv.imwrite(img_name, img)
	dbg(f"Deskewed image saved as '{img_name}'")


def classify_image_type(img):
	rows, cols = img.shape[:2]
	text_lines = 0
	dbg(f"Using Step length of {STEP_LENGTH} pixels")
	total_lines = rows // STEP_LENGTH
	for i in range(0, rows, STEP_LENGTH):
		black_acc, gray_acc, white_acc = 0, 0, 0
		for pixel in img[i]:
			if B0 <= pixel <= B1:
				black_acc += 1
			elif G0 <= pixel <= G1:
				gray_acc += 1
			elif W0 <= pixel <= W1:
				white_acc += 1
		if white_acc != 0 and (black_acc +
		                       gray_acc) / white_acc >= BLACK_TO_WHITE_RATIO:
			text_lines += 1
	dbg(f"Found {text_lines} text rows out of {total_lines} traversal rows")
	if text_lines / total_lines >= TEXT_TO_ROWS_RATIO:
		return ImageType.TEXT
	return ImageType.NON_TEXT


# todo
def classify_edge_type(img):
	return EdgeType.BLACK


def binarize(img, threshold):
	return cv.threshold(img, threshold, 255, cv.THRESH_BINARY)


#todo
def estimate_skew_angle(img):
	return 10


def deskew(img, skew_angle):
	(h, w) = img.shape[:2]
	# Maybe it would be better to get centroid of page instead of image center
	(cX, cY) = (w / 2, h / 2)

	M = cv.getRotationMatrix2D((cX, cY), skew_angle, 1.0)
	cos = np.abs(M[0, 0])
	sin = np.abs(M[0, 1])

	nW = int((h * sin) + (w * cos))
	nH = int((h * cos) + (w * sin))
	M[0, 2] += (nW / 2) - cX
	M[1, 2] += (nH / 2) - cY

	return cv.warpAffine(img, M, (nW, nH), borderMode=cv.BORDER_REPLICATE)


#todo
def crop(img):
	return img


def main(input_filename, output_filename):
	img = open_image(input_filename)
	gray = None
	if img is None:
		err(f"'{input_filename}' not found or can not be processed")
		return
	if len(img.shape) == 3 and img.shape[2] != 1:
		gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	else:
		gray = np.copy(img)
	img_type = classify_image_type(gray)
	dbg(f"Image classified as {img_type.name}")
	threshold = K2
	if img_type == ImageType.TEXT:
		img_edge_type = classify_edge_type(gray)
		dbg(f"Edges classified as {img_edge_type.name}")
		if (img_edge_type == EdgeType.BLACK):
			threshold = K1
		elif img_edge_type == EdgeType.GRAY:
			threshold = K2
		elif img_edge_type == EdgeType.NON_EDGE:
			threshold = K3
	binarized_img = binarize(gray, threshold)
	skew_angle = estimate_skew_angle(binarized_img)
	deskewed_img = deskew(img, skew_angle)
	cropped_img = crop(deskewed_img)
	save_image(output_filename, cropped_img)
	dbg("Done")


def init():
	logging.basicConfig(level=LOGGING_LEVEL,
	                    format="[%(levelname)s] %(message)s")


if __name__ == "__main__":
	init()
	ap = argparse.ArgumentParser(
	    description="Adaptive cropping and deskewing of scanned documents " +
	    "based on high accuracy estimation of skew angle and cropping value")
	ap.add_argument("-i", "--input", required=True, help="Input image file")
	ap.add_argument("-o", "--output", required=False, help="Output image file")
	args = ap.parse_args()
	root, ext = os.path.splitext(args.input)
	if args.output is None or args.output.strip() == '':
		args.output = root + "_deskewed" + ext
	if '.' not in args.output:
		args.output = f"{args.output}{ext}"
	main(args.input, args.output)
