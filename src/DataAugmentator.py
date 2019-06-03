import os
from skimage.util import random_noise
import numpy as np 
import cv2
import random
import imgaug as ia
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
from WordSegmentator import prepareImg, wordSegmentation

class DataAugmentator:
	def __init__(self):
		sometimes = lambda aug: iaa.Sometimes(0.5, aug)
		rare = lambda aug: iaa.Sometimes(0.25, aug)
		self.seq = iaa.Sequential([sometimes(iaa.Affine(rotate=(-5,5), translate_percent={'x': (-0.05, 0.05), 'y':(-0.05,0.05)}, shear=(-10,10))),
							  #iaa.ElasticTransformation(alpha=(0.01,0.09), sigma=0.05),
							  rare(iaa.Pepper(0.05))])

	def augment(self, img):
		img = np.float32(img)
		img = self.seq.augment_image(img)
		return img

	def read_image(self, imgPath):
		img = cv2.imread(imgPath, cv2.COLOR_BGR2RGB)
		return img

	def show_image(self, img):
		plt.imshow(img, cmap='gray')
		plt.show()

	def save_image(file, img):
		cv2.imwrite(file,img)


def main(file='sample_imgs/iam-test4.png'):
	da = DataAugmentator()
	img = da.read_image(file)
	img = da.augment(img)
	da.show_image(img)
	#img = read_image(file)
	#img = augment(img)	
	#show_image(img)
	#for i in range(1,11):
	#	img = read_image(file)
	#	img = augment(img)
	#	filename = '../new_data/ic5_{}.png'.format(i)
	#	save_image(file=filename, img=img)

if __name__ == "__main__":
	main('sample_imgs/iam-test4.png')



