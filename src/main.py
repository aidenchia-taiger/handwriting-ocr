from __future__ import division
from __future__ import print_function

import sys
import argparse
import cv2
import editdistance
from DataLoader import DataLoader, Batch, CustomLoader
from Model import Model, DecoderType
from SamplePreprocessor import preprocess, custom_preprocess
from Denoiser import remove_shadow_grayscale
import os
import random
import tensorflow as tf
from WordSegmentator import wordSegmentation, prepareImg

class FilePaths:
	"filenames and paths to data"
	fnModel = '../model/'
	fnCharList = '../model/charList.txt'
	fnAccuracy = '../model/accuracy.txt'
	fnCheckpoint = '../model/checkpoint'
	fnTrain = '../new_data/'
	fnInfer = '../data/test.png'
	fnCorpus = '../data/corpus.txt'
	fnTest = '../test/'
	fnReport = '../model/report.txt'

def train(model, loader):
	"train NN"
	epoch = 0 # number of training epochs since start
	bestCharErrorRate = float('inf') # best valdiation character error rate
	noImprovementSince = 0 # number of epochs no improvement of character error rate occured
	earlyStopping = 500 # stop training after this number of epochs without improvement, default: 5
	maxEpochs = 30
	while True:
		epoch += 1
		print('[INFO] Epoch:', epoch)

		# train
		print('Train NN')
		loader.trainSet()
		while loader.hasNext():
			iterInfo = loader.getIteratorInfo()
			batch = loader.getNext()
			loss = model.trainBatch(batch)
			print('[INFO] Batch:', iterInfo[0],'/', iterInfo[1], 'Loss:', loss)

		# validate
		charErrorRate = validate(model, loader)
		
		# if best validation accuracy so far, save model parameters
		if charErrorRate < bestCharErrorRate:
			print('[INFO] Character error rate improved, save model')
			bestCharErrorRate = charErrorRate
			noImprovementSince = 0
			model.save()
			open(FilePaths.fnAccuracy, 'w').write('Validation character error rate of saved model: %f%%' % (charErrorRate*100.0))
		else:
			print('[INFO] Character error rate not improved')
			noImprovementSince += 1

		# stop training if no more improvement in the last x epochs
		if noImprovementSince >= earlyStopping:
			print('[INFO] No more improvement since %d epochs. Training stopped.' % earlyStopping)
			break

		elif epoch >= maxEpochs:
			print('[INFO] Reached max. no. of epochs')
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
	model.plotMetricsOnTB(charErrorRate, wordAccuracy)
	return charErrorRate


def infer(model, fnImg):
	"recognize text in image provided by file path"
	img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize) # default
	#img = custom_preprocess(cv2.imread(fnImg, cv2.IMREAD_COLOR), Model.imgSize)
	batch = Batch(None, [img])
	(recognized, probability) = model.inferBatch(batch, True)

	clf = recognized[0]
	prob = probability[0]
	print('\n[INFO] Recognized:', '"' + clf + '"')
	print('[INFO] Probability:', prob)
	print('')

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
	parser.add_argument('--beamsearch', help='use beam search instead of greedy decoding', action='store_true')
	parser.add_argument('--wordbeamsearch', help='use word beam search instead of greedy decoding', action='store_true')
	parser.add_argument('--test', help='test the NN with a batch of images', action='store_true')
	parser.add_argument('--infer', help="infer a single image")
	parser.add_argument('--model', help='select the model to use')
	parser.add_argument('--demo', help='turn off writing to report.txt', action='store_true')
	args = parser.parse_args()

	decoderType = DecoderType.BestPath
	if args.beamsearch:
		decoderType = DecoderType.BeamSearch
	elif args.wordbeamsearch:
		decoderType = DecoderType.WordBeamSearch

	if args.model == None: 
		modelName = tf.train.latest_checkpoint(FilePaths.fnModel) # by default, use latest checkpoint
	elif args.model == 'new':
		modelName = 'new' # start from a new model
	else:
		modelName = FilePaths.fnModel + args.model  # specify the exact model to use

	# train or validate on IAM dataset	
	if args.train or args.validate:
		# load training data, create TF model
		#loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)
		loader = CustomLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)
		# save characters of model for inference mode
		open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))
		
		# save words contained in dataset into file
		open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))

		# execute training or validation
		if args.train:
			model = Model(loader.charList, decoderType, modelName=modelName,log=True)
			train(model, loader)
		elif args.validate:
			model = Model(loader.charList, decoderType, modelName=modelName, mustRestore=True)
			validate(model, loader)


	### aiden
	elif args.test:
		if not args.demo:
			report = open(FilePaths.fnReport, 'a')
			report.write('\n[MODEL] {}'.format(open(FilePaths.fnCheckpoint, 'r').readline().split(' ')[-1]))

		model = Model(open(FilePaths.fnCharList).read(), decoderType, modelName=modelName,mustRestore=True)
		gtTexts = read_gt_values(FilePaths.fnTest + 'ground_truth.txt')

		numWordOK = 0
		numWordTotal = 0
		numCharErr = 0
		numCharTotal = 0
		for fullPath, gt in gtTexts.items():
			recognized, _ = infer(model, fullPath)
			numWordOK += 1 if gt == recognized else 0
			numWordTotal += 1
			dist = editdistance.eval(recognized, gt)
			numCharErr += dist	
			numCharTotal += len(gt)
			if dist == 0:
				statement = '[OK] {} -> {}'.format(gt, recognized)
			else:
				statement = '[ERR{}] {} -> {}'.format(dist,gt,recognized)
				
			print(statement)
			if not args.demo:
				report.write(statement + '\n')

		charErrorRate = numCharErr / numCharTotal
		wordAccuracy = numWordOK / numWordTotal
		result = '[INFO] Character Error Rate: {:.{}f}% | Word Accuracy: {:.{}f}%'.format(charErrorRate*100,2,wordAccuracy*100,2)
		print(result)
		if not args.demo:
			report.write(result)
			print('[INFO] Report written to ../model/report.txt')
			report.close()
	###

	# infer text on test image
	else:
		model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True, modelName=modelName)

		word = args.infer
		if word != None:
			infer(model, word)

		else:
			infer(model, FilePaths.fnInfer)

### aiden
def read_gt_values(path):
	f = open(path)
	gt_dic = {}
	for line in f.readlines():
		line = line.split()
		gt_dic[line[0]] = line[-1].strip() # remove EOF
	print(gt_dic)
	return gt_dic 

def get_img_paths(path):
	paths = []
	for root, _, pathnames in os.walk(path):
		for path in pathnames:
			if ".png" in path:
				paths.append(os.path.abspath(os.path.join(root, path)))

	print(paths)
	return paths

def write_report(dic, txtfilename, message=''):
	f = open(txtfilename, "w+")
	f.write()
	for k, v in dic.items():
		f.write("{} -> {} \n".format(k, v)) 

	print("[INFO] Report written to {}".format(txtfilename))
	f.close()
###

if __name__ == '__main__':
	main()
	#read_gt_values('../test/' + 'ground_truth.txt')
