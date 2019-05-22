import cv2
import numpy as np
import os
import argparse
from scipy.ndimage.filters import maximum_filter
from PIL import Image

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', help="path to input image")
	parser.add_argument('-o', help="path to output image", default="denoised.jpg")
	args = parser.parse_args()

	inp_path = args.i
	out_path = os.path.join("out", args.o)

	img = cv2.imread(inp_path, cv2.IMREAD_GRAYSCALE)
	deshadowed = remove_shadow_grayscale(img)
	binarized = cv2.adaptiveThreshold(deshadowed,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,blockSize=27,C=20.0)
	cv2.imwrite(out_path, binarized)
	print("[INFO] Denoised image written to {}".format(out_path))

def remove_shadow_grayscale(img):

    bg_img = maximum_filter(img, size =(10,10)) # Max Filter

    bg_img = cv2.medianBlur(bg_img, 17) # Median Filter

    diff_img = 255 - cv2.absdiff(img, bg_img) # Extract foreground

    norm_img = np.empty(diff_img.shape)
    norm_img = cv2.normalize(diff_img, dst=norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1) # Normalize pixels
    
    #cv2.imwrite(output_path, norm_img)
    return norm_img


if __name__=="__main__":
	main()