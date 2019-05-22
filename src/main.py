from __future__ import division
from __future__ import print_function

import sys
import argparse
import cv2
import editdistance
from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess
import os
import random
import tensorflow as tf
from WordSegmentator import wordSegmentation, prepareImg

class FilePaths:
	"filenames and paths to data"
	fnCharList = '../model/charList.txt'
	fnAccuracy = '../model/accuracy.txt'
	fnTrain = '../data/'
	fnInfer = 'sample_imgs/iam-test4.png'
	fnCorpus = '../data/corpus.txt'
	fnTest = '../test_data'
	fnClfReport = 'out/clf_wrongly_report.txt'

class Params:
	TEST_SAMPLE_SIZE = 50


def train(model, loader):
	"train NN"
	epoch = 0 # number of training epochs since start
	bestCharErrorRate = float('inf') # best valdiation character error rate
	noImprovementSince = 0 # number of epochs no improvement of character error rate occured
	earlyStopping = 5 # stop training after this number of epochs without improvement
	while True:
		epoch += 1
		print('Epoch:', epoch)

		# train
		print('Train NN')
		loader.trainSet()
		while loader.hasNext():
			iterInfo = loader.getIteratorInfo()
			batch = loader.getNext()
			loss = model.trainBatch(batch)
			print('Batch:', iterInfo[0],'/', iterInfo[1], 'Loss:', loss)

		# validate
		charErrorRate = validate(model, loader)
		
		# if best validation accuracy so far, save model parameters
		if charErrorRate < bestCharErrorRate:
			print('Character error rate improved, save model')
			bestCharErrorRate = charErrorRate
			noImprovementSince = 0
			model.save()
			open(FilePaths.fnAccuracy, 'w').write('Validation character error rate of saved model: %f%%' % (charErrorRate*100.0))
		else:
			print('Character error rate not improved')
			noImprovementSince += 1

		# stop training if no more improvement in the last x epochs
		if noImprovementSince >= earlyStopping:
			print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
			break


def validate(model, loader):
	"validate NN"
	print('Validate NN')
	loader.validationSet()
	numCharErr = 0
	numCharTotal = 0
	numWordOK = 0
	numWordTotal = 0
	while loader.hasNext():
		iterInfo = loader.getIteratorInfo()
		print('Batch:', iterInfo[0],'/', iterInfo[1])
		batch = loader.getNext()
		(recognized, _) = model.inferBatch(batch)
		
		print('Ground truth -> Recognized')	
		for i in range(len(recognized)):
			numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
			numWordTotal += 1
			dist = editdistance.eval(recognized[i], batch.gtTexts[i])
			numCharErr += dist
			numCharTotal += len(batch.gtTexts[i])
			print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')
	
	# print validation result
	charErrorRate = numCharErr / numCharTotal
	wordAccuracy = numWordOK / numWordTotal
	print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))
	return charErrorRate


def infer(model, fnImg):
	"recognize text in image provided by file path"
	img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)

	batch = Batch(None, [img])
	(recognized, probability) = model.inferBatch(batch, True)

	clf = recognized[0]
	prob = probability[0]
	print('\n[INFO] Recognized:', '"' + clf + '"')
	print('[INFO] Probability:', prob)

	return clf, prob


def main():
	"main function"

	# Ignore tensorflow deprecation warnings
	#import tensorflow.python.util.deprecation as deprecation
	#deprecation._PRINT_DEPRECATION_WARNINGS = False

	# optional command line args
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', help='train the NN', action='store_true')
	parser.add_argument('--validate', help='validate the NN', action='store_true')
	parser.add_argument('--beamsearch', help='use beam search instead of best path decoding', action='store_true')
	parser.add_argument('--wordbeamsearch', help='use word beam search instead of best path decoding', action='store_true')
	parser.add_argument('--test', help='test the NN with a batch of images', action='store_true')
	parser.add_argument('--infer', help="infer a single image", nargs='?',const=FilePaths.fnInfer)
	parser.add_argument('--model', help='choose the NN model to use, e.g. snapshot-12', nargs='?', const="")
	args = parser.parse_args()

	decoderType = DecoderType.BestPath
	if args.beamsearch:
		decoderType = DecoderType.BeamSearch
	elif args.wordbeamsearch:
		decoderType = DecoderType.WordBeamSearch

	# train or validate on IAM dataset	
	if args.train or args.validate:
		# load training data, create TF model
		loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)

		# save characters of model for inference mode
		open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))
		
		# save words contained in dataset into file
		open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))

		# execute training or validation
		if args.train:
			model = Model(loader.charList, decoderType, mustRestore=True)
			train(model, loader)
		elif args.validate:
			model = Model(loader.charList, decoderType, mustRestore=True)
			validate(model, loader)


	### aiden
	elif args.test:
		print(open(FilePaths.fnAccuracy).read())
		model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True)

		gt_dic = read_gt_values(FilePaths.fnTest + "/words.txt")
		img_paths = get_img_paths(FilePaths.fnTest)

		img_paths = random.sample(img_paths, Params.TEST_SAMPLE_SIZE)
		acc = 0
		wrong_clf_dic = {}

		for img_path in img_paths:
			key = img_path.split(r"/")[-1][:-4]
			gt_value = gt_dic[key]
			print("[INFO] Input: {} | Ground Truth: {}".format(key, gt_value))
			clf, prob = infer(model, img_path)

			if clf == gt_value:
				acc += 1
			else:
				wrong_clf_dic[key] = [clf, gt_value]
			
		print("[INFO] Accuracy = {} / {}".format(acc, Params.TEST_SAMPLE_SIZE))

		write_dic_to_txt(wrong_clf_dic, FilePaths.fnClfReport)
	###

	# infer text on test image
	else:
		print('[INFO] Loading Model')
		print(open(FilePaths.fnAccuracy).read())
		if args.model == "":
			model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True)
		else:
			f = open('../model/checkpoint', 'w+')
			f.write("model_checkpoint_path: '{}'\n".format(args.model))
			f.write("all_model_checkpoint_paths: '{}'".format(args.model))
			f.close()
			model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True)

		word = args.infer
		infer(model, word)

### aiden
def read_gt_values(path):
	f = open(path)
	gt_dic = {}
	for line in f.readlines():
		line = line.split(" ")
		gt_dic[line[0]] = line[-1].strip() # remove EOF
	#print(gt_dic)
	return gt_dic 

def get_img_paths(path):
	paths = []
	for root, _, pathnames in os.walk(path):
		for path in pathnames:
			if ".png" in path:
				paths.append(os.path.abspath(os.path.join(root, path)))

	#print(paths)
	return paths

def write_dic_to_txt(dic, txtfilename):
	f = open(txtfilename, "w+")
	for k, v in dic.items():
		f.write("{} -> {} \n".format(k, v)) 

	print("[INFO] Wrongly classified report written to {}".format(txtfilename))
	f.close()
###

if __name__ == '__main__':
	main()
	#read_gt_values(FilePaths.fnTest + "/words.txt")
	#get_img_paths(FilePaths.fnTest)

	#img = prepareImg(cv2.imread("goodnotes-test4.jpeg"), 50)
	#res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100)
	#cv2.imwrite('new_img.jpeg', )


