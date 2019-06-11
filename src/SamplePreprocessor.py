from __future__ import division
from __future__ import print_function
from DataAugmentator import DataAugmentator
from WordSegmentator import prepareImg, wordSegmentation

import random
import numpy as np
import cv2
import pdb
import matplotlib.pyplot as plt

def custom_preprocess(img, imgSize, dataAugmentation=False):
	if dataAugmentation:
		da = DataAugmentator()
		img = da.augment(img)

	img = cv2.resize(img, (imgSize[0], imgSize[1]))
	img = cv2.transpose(img)
	img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
	return img


def preprocess(img, imgSize, dataAugmentation=False):
	"put img into target img of size imgSize, transpose for TF and normalize gray-values"

	# there are damaged files in IAM dataset - just use black image instead
	if img is None:
		img = np.zeros([imgSize[1], imgSize[0]])

	# increase dataset size by applying random stretches to the images
	if dataAugmentation:
		da = DataAugmentator()
		img = da.augment(img)

	### if not doing data augmentation, must be inferring, so perform word segmentation to get word tightly cropped
	#else:
	#	img = prepareImg(img, 50)
	#	try:
	#		res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100)[0]
	#		img = res[1] # res[0] is the bounding box coordinates, res[1] is the tightly cropped img itself
	#	except IndexError: # sometimes word segmentation doesn't always work
	#		print('Skipping word segmentation')
	###

	# create target image and copy sample image into it
	(wt, ht) = imgSize
	(h, w) = img.shape
	fx = w / wt
	fy = h / ht
	f = max(fx, fy)
	newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1)) # scale according to f (result at least 1 and at most wt or ht)
	img = cv2.resize(img, newSize)
	target = np.ones([ht, wt]) * 255
	target[0:newSize[1], 0:newSize[0]] = img

	# transpose for TF
	img = cv2.transpose(target)

	# normalize
	(m, s) = cv2.meanStdDev(img)
	m = m[0][0]
	s = s[0][0]
	img = img - m
	img = img / s if s>0 else img
	return img

if __name__ == ('__main__'):
	imgSize = (128, 32)
	#img = preprocess(cv2.imread('sample_imgs/iam-test2.png', cv2.IMREAD_GRAYSCALE), imgSize, dataAugmentation=True)
	img = custom_preprocess(cv2.imread('sample_imgs/iam-test2.png', cv2.IMREAD_GRAYSCALE), imgSize, dataAugmentation=True)
	plt.imshow(img, cmap='gray')
	plt.show()