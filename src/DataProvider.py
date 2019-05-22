import os
import numpy as np
import cv2
import random

class DataProvider():
	"this class creates machine-written text for a word list. TODO: change getNext() to return your samples."

	def __init__(self):
		self.BASE_DIR = '/Users/taiger/Documents/aiden/characters/Hnd/Img'
		self.groundTruths = self.getGroundTruths()
		self.idx = 0
		#print('[INFO] Ground Truths: {}'.format(self.groundTruths))

	def getGroundTruths(self):
		gt_list = []
		f = open('groundTruth.txt','r')
		for line in f.readlines():
			gt_list.append(line.rstrip())
			random.shuffle(gt_list)
		return gt_list


	def hasNext(self):
		"are there still samples to process?"
		return self.idx < len(self.groundTruths)

	def getNext(self):
		"TODO: return a sample from your data as a tuple containing the text and the image"
		line = self.groundTruths[self.idx].split(' ')
		nextPath = line[0]
		word = line[1]
		img = cv2.imread(nextPath)
		self.idx += 1
		return (word, img)


def createIAMCompatibleDataset(dataProvider):
	"this function converts the passed dataset to an IAM compatible dataset"

	# create files and directories
	f = open('words.txt', 'w+')
	if not os.path.exists('sub'):
		os.makedirs('sub')
	if not os.path.exists('sub/sub-sub'):
		os.makedirs('sub/sub-sub')

	# go through data and convert it to IAM format
	ctr = 0
	while dataProvider.hasNext():
		sample = dataProvider.getNext()
		
		# write img
		cv2.imwrite('sub/sub-sub/sub-sub-%d.png'%ctr, sample[1])
		
		# write filename, dummy-values and text
		line = 'sub-sub-%d'%ctr + ' X X X X X X X ' + sample[0] + '\n'
		f.write(line)
		
		ctr += 1
		
		
if __name__ == '__main__':
	#words = ['some', 'words', 'for', 'which', 'we', 'create', 'text-images']
	dataProvider = DataProvider()
	createIAMCompatibleDataset(dataProvider)