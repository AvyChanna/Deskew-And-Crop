import argparse
import logging
import os
import sys
from enum import Enum
try:
	import cv2 as cv
	import numpy as np
except:
	print("Could not import dependencies. Make sure you have them installed")
	print(">python -m pip install --user -r requirements.txt")
	sys.exit()

dbg = logging.debug
err = logging.error
warn = logging.warning

########## CONFIG ##########
# Minimum dimensions allowed for the image
MIN_DIMENSIONS = 20, 20

# Step Length for image traversal
STEP_LENGTH = 10

# Ratio of black pixels to white pixels
BLACK_TO_WHITE_RATIO = 0.8

# Ratio of text segments to non-text segments
TEXT_TO_ROWS_RATIO = 0.4

# Binarization Thresholds
K1, K2, K3 = 50, 178, 229

# Intensity thresholds for black, gray and white pixels
B0, B1, G0, G1, W0, W1 = 10, 60, 120, 190, 200, 250
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


def classify_edge_type(img):
	rows, cols = img.shape[:2]
	dbg(f"Using Step length of {STEP_LENGTH} pixels")
	b, g, r = 0, 0, 0
	for i in range(0, rows, STEP_LENGTH):
		b, g, r = 0, 0, 0
		pixelL = img[i][0]
		pixelR = img[i][cols - 1]
		if B0 <= pixelL <= B1 and B0 <= pixelR <= B1:
			b += 1
		elif G0 <= pixelL <= G1 and G0 <= pixelR <= G1:
			g += 1
		else:
			r += 1
	if b > max(g, r):
		return EdgeType.BLACK
	elif g > max(b, r):
		return EdgeType.GRAY
	else:
		return EdgeType.NON_EDGE


def binarize(img, threshold):
	return cv.threshold(img, threshold, 255, cv.THRESH_BINARY)[-1]


#estimate skew angle and return most apropriate skew angle
def estimate_skew_angle(img):
	# grab the (x, y) coordinates of all pixel values that
	# are greater than zero, then use these coordinates to
	# compute a rotated bounding box that contains all
	# coordinates
	coords = np.column_stack(np.where(img > 0))
	angle = cv.minAreaRect(coords)[-1]
	# the `cv.minAreaRect` function returns values in the
	# range [-90, 0); as the rectangle rotates clockwise the
	# returned angle trends to 0 -- in this special case we
	# need to add 90 degrees to the angle
	if angle < -45:
		angle = -(90 + angle)
	# otherwise, just take the inverse of the angle to make
	# it positive
	else:
		angle = -angle
	return angle


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


def order_rect(points):
	res = np.zeros((4, 2), dtype=np.float32)
	s = np.sum(points, axis=1)
	d = np.diff(points, axis=1)
	res[0] = points[np.argmin(s)]
	res[1] = points[np.argmin(d)]
	res[2] = points[np.argmax(s)]
	res[3] = points[np.argmax(d)]
	return res


def trans(img, points):
	rect = order_rect(points)
	(tl, tr, br, bl) = rect
	widthA = np.sqrt(((br[0] - bl[0])**2) + ((br[1] - bl[1])**2))
	widthB = np.sqrt(((tr[0] - tl[0])**2) + ((tr[1] - tl[1])**2))
	maxWidth = max(int(widthA), int(widthB))
	heightA = np.sqrt(((tr[0] - br[0])**2) + ((tr[1] - br[1])**2))
	heightB = np.sqrt(((tl[0] - bl[0])**2) + ((tl[1] - bl[1])**2))
	maxHeight = max(int(heightA), int(heightB))
	dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1],
	                [0, maxHeight - 1]],
	               dtype=np.float32)
	M = cv.getPerspectiveTransform(rect, dst)
	warped = cv.warpPerspective(img, M, (maxWidth, maxHeight))
	return warped


def crop_impl(img, gray, threshold):
	found = False
	loop = False
	old_val = 0
	i = 0

	im_h, im_w = img.shape[:2]
	while not found:
		if threshold >= 255 or threshold == 0 or loop:
			break

		_, thresh = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)
		contours = cv.findContours(thresh, cv.RETR_LIST,
		                           cv.CHAIN_APPROX_NONE)[0]
		im_area = im_w * im_h

		for cnt in contours:
			area = cv.contourArea(cnt)
			if area > (im_area / 100) and area < (im_area / 1.01):
				epsilon = 0.1 * cv.arcLength(cnt, True)
				approx = cv.approxPolyDP(cnt, epsilon, True)
				if len(approx) == 4:
					found = True
				elif len(approx) > 4:
					threshold = threshold - 1
					dbg(f"Adjust Threshold: {threshold}")
					if threshold == old_val + 1:
						loop = True
					break
				elif len(approx) < 4:
					threshold = threshold + 5
					dbg(f"Adjust Threshold: {threshold}")
					if threshold == old_val - 5:
						loop = True
					break

				rect = np.zeros((4, 2), dtype=np.float32)
				rect[0] = approx[0]
				rect[1] = approx[1]
				rect[2] = approx[2]
				rect[3] = approx[3]

				dst = trans(img, rect)
				dst_h, dst_w = dst.shape[:2]
				img = dst[0:dst_h, 0:dst_w]
			else:
				if i > 100:
					threshold = threshold + 5
					if threshold > 255:
						break
					dbg(f"Adjust Threshold: {threshold}")
					if threshold == old_val - 5:
						loop = True

	return found, img


def crop(img, threshold):
	old_img = img.copy()
	img = cv.copyMakeBorder(
	    img,
	    100,
	    100,
	    100,
	    100,
	    # cv.BORDER_CONSTANT,
	    cv.BORDER_REPLICATE,
	    value=[255, 255, 255])
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	found, img = crop_impl(img, gray, threshold)
	if found:
		dbg("Successfully cropped image")
		return img
	else:
		dbg("Image seems to be already cropped")
		return old_img


def show(img):
	cv.imshow("", img)
	if cv.waitKey(0) & 0xFF == 'q':
		sys.exit()
	cv.destroyAllWindows()


# takes image matrix as input, returns processed image matrix as output
def algo(img):
	assert img is not None
	gray = None
	if len(img.shape) == 3 and img.shape[2] != 1:
		gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	else:
		gray = np.copy(img)
		img = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
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
	dbg(f"Skew Angle = {skew_angle}")
	deskewed_img = deskew(img, skew_angle)
	cropped_img = crop(deskewed_img, threshold)
	return cropped_img


def main(input_filename, output_filename):
	img = open_image(input_filename)
	if img is None:
		err(f"'{input_filename}' not found or can not be processed")
		return
	cropped_img = algo(img)
	save_image(output_filename, cropped_img)
	dbg("Done")


def init_logging(level):
	logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


if __name__ == "__main__":
	ap = argparse.ArgumentParser(
	    description="Adaptive cropping and deskewing of scanned documents " +
	    "based on high accuracy estimation of skew angle and cropping value")
	ap.add_argument("-i", "--input", required=True, help="Input image file")
	ap.add_argument("-o", "--output", required=False, help="Output image file")
	ap.add_argument("-d",
	                "--debug",
	                action="store_const",
	                const=logging.DEBUG,
	                default=logging.WARNING)
	args = ap.parse_args()
	init_logging(args.debug)
	root, ext = os.path.splitext(args.input)
	if args.output is None or args.output.strip() == '':
		args.output = root + "_deskewed" + ext
	if '.' not in args.output:
		args.output = f"{args.output}{ext}"
	main(args.input, args.output)
