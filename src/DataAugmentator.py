import os
from skimage.util import random_noise
import numpy as np 
import cv2
import random
import imgaug as ia
import imgaug.augmenters as iaa
import Augmentor
from WordSegmentator import prepareImg, wordSegmentation

def augment(img):
	seq = iaa.Sequential([
			iaa.Affine(rotate=(random.uniform(-8,0),random.uniform(0,8))),
			iaa.AdditiveGaussianNoise(scale=(30,90)),
			iaa.CoarseDropout(size_percent=0.2, per_channel=True),
			iaa.Multiply(random.uniform(1,2)),
			iaa.PiecewiseAffine(scale=(0.01, 0.1)),
			iaa.ElasticTransformation(alpha=(0.01,1.0), sigma=0.2)
			],
			random_order=True)

	img = seq.augment_image(img)
	return img

def read_image(imgPath):
	img = cv2.imread(imgPath, cv2.COLOR_BGR2RGB)
	return img

def show_image(img):
	cv2.imshow('augmented image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def save_image(file, img):
	cv2.imwrite(file,img)


def main(file='sample_imgs/iam-test4.png'):
	#show_image(img)
	for i in range(1,11):
		img = read_image(file)
		img = augment(img)
		filename = '../new_data/ic2_{}.png'.format(i)
		save_image(file=filename, img=img)

if __name__ == "__main__":
	main('../new_data/ic2.jpeg')



