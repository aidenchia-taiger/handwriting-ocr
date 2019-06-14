import cv2
import numpy as np
import os
import argparse
from scipy.ndimage.filters import maximum_filter
from PIL import Image
import matplotlib.pyplot as plt

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--i', help="path to input image")
	parser.add_argument('--t', help="threshold type: global, adaptive, Otsu", default='otsu')
	args = parser.parse_args()

	img = cv2.imread(args.i, cv2.IMREAD_GRAYSCALE)
	if args.t == 'global':
		_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

	elif args.t == 'adaptive':
		_, img = cv2.threshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 11, 2)

	elif args.t == 'otsu':
		_, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

	elif args.t == 'canny':
		img = cv2.Canny(img, 100, 200)

	print(img)
	plt.imshow(img, 'gray')
	plt.show()

def plotHist(path):
	img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	plt.hist(img.ravel(), 256, [0, 256])
	plt.show()



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
	#plotHist('../new_data/b201.png')