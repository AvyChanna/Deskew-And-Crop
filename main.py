import logging
import sys
from enum import Enum
try:
	import cv2 as cv
	import numpy as np
except:
	print("run python3 -m pip install --user numpy opencv-contrib-python")

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

# Color thresholds
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
		warn(f"{filename} not found")
		return None
	rows, cols = img.shape[:2]
	if rows < MIN_DIMENSIONS[0] or cols < MIN_DIMENSIONS[1]:
		warn(f"{filename} does not have enough rows/columns")
		return None
	dbg(f"Image has {rows} rows and {cols} columns")
	return img


def save_image(img_name, img):
	cv.imwrite(img_name, img)
	dbg(f"Deskewed image saved as {img_name}")


def classify_image_type(img):
	rows, cols = img.shape[:2]
	text_lines = 0
	total_lines = rows / STEP_LENGTH
	for k in range(0, rows, STEP_LENGTH):
		black_acc, gray_acc, white_acc = 0, 0, 0
		for i in range(k, min(k + STEP_LENGTH, rows)):
			for j in range(cols):
				pixel = img[i][j]
				if pixel >= B0 and pixel <= B1:
					black_acc += 1
				elif pixel >= G0 and pixel <= G1:
					gray_acc += 1
				elif pixel >= W0 and pixel <= W1:
					white_acc += 1
		if white_acc != 0 and (black_acc +
		                       gray_acc) / white_acc >= BLACK_TO_WHITE_RATIO:
			text_lines += 1
	dbg("Found {text_lines} text segments out of {total_lines} total segments")
	if text_lines / total_lines >= TEXT_TO_ROWS_RATIO:
		return ImageType.TEXT
	return ImageType.NON_TEXT


# todo
def classify_edge_type(img):
	pass


def binarize(img, threshold):
	return cv.threshold(img, threshold, 255, cv.THRESH_BINARY)


#todo
def estimate_skew_angle(img):
	pass


#todo
def deskew_and_crop(img, skew_angle):
	pass


def main(image_filename):
	img = open_image(image_filename)
	gray = None
	if img is None:
		err(f"{image_filename} not found or can not be processed")
		return
	if len(img.shape) == 3 and img.shape[2] != 1:
		gray = cv.cvtColor(img,  cv.COLOR_BGR2GRAY)
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
	deskewed_img = deskew_and_crop(img, gray, skew_angle)
	save_image("_deskewed.".join(image_filename.split(".")), deskewed_img)


if __name__ == "__main_":
	logging.basicConfig(level=LOGGING_LEVEL,
	                    format="[%(levelname)s] %(message)s")
	if len(sys.argv) < 2:
		err("No file input")
		sys.exit()
	main(sys.argv[1])
