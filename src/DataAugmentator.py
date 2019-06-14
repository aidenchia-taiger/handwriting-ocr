import os
from skimage.util import random_noise
import numpy as np 
import cv2
import random
import imgaug as ia
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
from WordSegmentator import prepareImg, wordSegmentation
import argparse

class DataAugmentator:
	def __init__(self):
		#print('[INFO] Applying data augmentation...')
		sometimes = lambda aug: iaa.Sometimes(0.5, aug)
		rare = lambda aug: iaa.Sometimes(0.25, aug)
		self.seq = iaa.Sequential([sometimes(iaa.Affine(
								   rotate=(-5,5), 
								   translate_percent={'x': (-0.05, 0.05), 'y':(-0.05,0.05)}, 
								   shear=(-10,10))),
							  	   rare(iaa.Pepper(0.05))])

	def augment(self, img):
		img = np.float32(img)
		img = self.seq.augment_image(img)
		return img

	def read_image(self, imgPath):
		img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
		return img

	def show_image(self, img):
		plt.imshow(img, cmap='gray')
		plt.show()

	def save_image(file, img):
		cv2.imwrite(file,img)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--i', help="path to input image")
	args = parser.parse_args()

	da = DataAugmentator()
	img = da.read_image(args.i)
	img = da.augment(img)
	da.show_image(img)

if __name__ == "__main__":
	main()



