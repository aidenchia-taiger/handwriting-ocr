from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import datetime

class DecoderType:
	BestPath = 0
	BeamSearch = 1
	WordBeamSearch = 2

class Model: 
	"minimalistic TF model for HTR"

	# model constants
	batchSize = 50 # default: 50
	imgSize = (224, 224, 3) # default: (128, 32)
	maxTextLen = 32
	MODULE_URL = 'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/3'
	LOG_PATH = '../model/logs/' + datetime.datetime.now().strftime("Time_%H%M_Date_%d-%m")

	def __init__(self, charList, decoderType=DecoderType.BestPath, mustRestore=False, modelName=None, log=False):
		"init model: add CNN, RNN and CTC and initialize TF"
		self.charList = charList # FilePaths.
		self.decoderType = decoderType
		self.mustRestore = mustRestore
		self.snapID = 0
		self.charErrorRate = 0.0
		self.wordAccuracy = 0.0
		self.modelName = modelName
		# Whether to use normalization over a batch or a population
		self.is_train = tf.placeholder(tf.bool, name='is_train')

		# input image batch
		#self.inputImgs = tf.placeholder(tf.float32, shape=(None, Model.imgSize[0], Model.imgSize[1])) : default
		self.inputImgs = tf.placeholder(tf.float32, shape=(None, Model.imgSize[0], Model.imgSize[1], Model.imgSize[2]))

		# setup CNN, RNN and CTC: default methods
		#self.setupCNN()
		#self.setupRNN()
		#self.setupCTC()

		# setup deep CNN, RNN and CTC
		self.setupDeepCNN()
		self.setupRNN2()
		self.setupCTC()

		# setup optimizer to train NN
		self.batchesTrained = 0
		self.learningRate = tf.placeholder(tf.float32, shape=[])
		self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
		with tf.control_dependencies(self.update_ops):
			self.optimizer = tf.train.AdamOptimizer(self.learningRate).minimize(self.loss)

		# initialize TF
		(self.sess, self.saver) = self.setupTF()
		if log:
			self.writer = tf.summary.FileWriter(Model.LOG_PATH, self.sess.graph)

	def setupDeepCNN(self):
		'Input: (? , 224, 224, 3) => Output: (?, 1, 2048)'
		module = hub.Module(Model.MODULE_URL, trainable=True)
		cnnOut = module(self.inputImgs)

		self.dropout_rate = tf.placeholder_with_default(0.0, shape=())
		prob = tf.math.subtract(1.0, self.dropout_rate)
		dropout = tf.nn.dropout(cnnOut, keep_prob=prob)
		self.cnnOut3d = tf.reshape(tensor=dropout, shape=[-1, 32, 64])

	def setupRNN2(self):
		rnnIn = self.cnnOut3d

		# basic cells which is used to build RNN
		numHidden = 256
		cells = [tf.contrib.rnn.LSTMCell(num_units=numHidden, state_is_tuple=True) for _ in range(2)] # 2 layers

		# stack basic cells
		stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

		# bidirectional RNN
		((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnnIn, dtype=rnnIn.dtype)
		
		# BxTxH + BxTxH -> BxTx2H -> BxTx1X2H i.e. expand dim needed for atrous convolution
		concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)

		# project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
		kernel = tf.Variable(tf.truncated_normal([1, 1, numHidden * 2, len(self.charList) + 1], stddev=0.1))
		self.rnnOut3d = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])

	def setupCNN(self):
		"create CNN layers and return output of these layers"
		"Input: (?, 128, 32) => Output: (?, 32, 1, 256)"  
		"Formula: Output shape = (W - F + 2P) / S + 1, where W is input, F kernel size, S stride, P amount of padding"
		cnnIn4d = tf.expand_dims(input=self.inputImgs, axis=3) # (?, 128, 32) => (?, 128, 32, 1)

		# list of parameters for the layers
		kernelVals = [5, 5, 3, 3, 3]
		featureVals = [1, 32, 64, 128, 128, 256]
		strideVals = poolVals = [(2,2), (2,2), (1,2), (1,2), (1,2)]
		numLayers = len(strideVals)

		# create layers
		pool = cnnIn4d # input to first CNN layer
		for i in range(numLayers):
			kernel = tf.Variable(tf.truncated_normal([kernelVals[i], kernelVals[i], featureVals[i], featureVals[i + 1]], stddev=0.1))
			conv = tf.nn.conv2d(pool, kernel, padding='SAME',  strides=(1,1,1,1))
			conv_norm = tf.layers.batch_normalization(conv, training=self.is_train)
			relu = tf.nn.relu(conv_norm)
			kernel2 = tf.Variable(tf.truncated_normal([kernelVals[i], kernelVals[i], featureVals[i + 1], featureVals[i + 1]], stddev=0.1))
			conv2 = tf.nn.conv2d(relu, kernel2, padding='SAME',  strides=(1,1,1,1))
			conv_norm2 = tf.layers.batch_normalization(conv2, training=self.is_train)
			relu2 = tf.nn.relu(conv_norm2)
			pool = tf.nn.max_pool(relu2, (1, poolVals[i][0], poolVals[i][1], 1), (1, strideVals[i][0], strideVals[i][1], 1), 'VALID')

		self.dropout_rate = tf.placeholder_with_default(0.0, shape=())
		prob = tf.math.subtract(1.0, self.dropout_rate)
		dropout = tf.nn.dropout(pool, keep_prob=prob)
		self.cnnOut4d = dropout

	def setupRNN(self):
		"create RNN layers and return output of these layers"
		rnnIn3d = tf.squeeze(self.cnnOut4d, axis=[2])

		# basic cells which is used to build RNN
		numHidden = 256
		cells = [tf.contrib.rnn.LSTMCell(num_units=numHidden, state_is_tuple=True) for _ in range(2)] # 2 layers

		# stack basic cells
		stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

		# bidirectional RNN
		# BxTxF -> BxTx2H
		((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnnIn3d, dtype=rnnIn3d.dtype)
									
		# BxTxH + BxTxH -> BxTx2H -> BxTx1X2H i.e. expand dim needed for atrous convolution
		concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)
									
		# project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
		kernel = tf.Variable(tf.truncated_normal([1, 1, numHidden * 2, len(self.charList) + 1], stddev=0.1))
		self.rnnOut3d = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])
		


	def setupCTC(self):
		"create CTC loss and decoder and return them"
		# BxTxC -> TxBxC i.e. Time * Batch * Classes, transpose needed for tf ctc loss
		self.ctcIn3dTBC = tf.transpose(self.rnnOut3d, [1, 0, 2]) # (?, 32, 80) => (32, ?, 80)
		# ground truth text as sparse tensor, required by tf ctc loss
		self.gtTexts = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]) , tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int64, [2]))

		# calc loss for batch
		self.seqLen = tf.placeholder(tf.int32, [None])
		self.loss = tf.reduce_mean(tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, ctc_merge_repeated=True, ignore_longer_outputs_than_inputs=True))

		# calc loss for each element to compute label probability
		self.savedCtcInput = tf.placeholder(tf.float32, shape=[Model.maxTextLen, None, len(self.charList) + 1])
		self.lossPerElement = tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.savedCtcInput, sequence_length=self.seqLen, ctc_merge_repeated=True)

		# decoder: either best path decoding or beam search decoding
		if self.decoderType == DecoderType.BestPath:
			self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen)
		elif self.decoderType == DecoderType.BeamSearch:
			self.decoder = tf.nn.ctc_beam_search_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, beam_width=50, merge_repeated=False)
		elif self.decoderType == DecoderType.WordBeamSearch:
			# import compiled word beam search operation (see https://github.com/githubharald/CTCWordBeamSearch)
			word_beam_search_module = tf.load_op_library('/home/aidenchia/Documents/handwriting-ocr/src/TFWordBeamSearch.so')

			# prepare information about language (dictionary, characters in dataset, characters forming words) 
			chars = str().join(self.charList)
			wordChars = open('../model/wordCharList.txt').read().splitlines()[0]
			corpus = open('../data/corpus.txt').read()

			# decode using the "Words" mode of word beam search
			self.decoder = word_beam_search_module.word_beam_search(tf.nn.softmax(self.ctcIn3dTBC, dim=2), 50, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf8'), wordChars.encode('utf8'))


	def setupTF(self):
		"initialize TF"
		print('Python: '+sys.version)
		print('Tensorflow: '+tf.__version__)

		sess=tf.Session() # TF session

		saver = tf.train.Saver(max_to_keep=1) # saver saves model to file

		self.batch_loss_summary = tf.summary.scalar(name="Batch Loss", tensor= self.loss) # log to tensorboard

		# if model must be restored (for inference), there must be a snapshot
		if self.mustRestore and not self.modelName:
			raise Exception('Could not find or open {}'.format(self.modelName))

		# load saved model
		if self.modelName != 'new':
			print('Init with stored values from ' + self.modelName)
			saver.restore(sess, self.modelName)
		else:
			print('Init with new values')
			sess.run(tf.global_variables_initializer())

		return (sess,saver)


	def toSparse(self, texts):
		"put ground truth texts into sparse tensor for ctc_loss"
		indices = []
		values = []
		shape = [len(texts), 0] # last entry must be max(labelList[i])

		# go over all texts
		for (batchElement, text) in enumerate(texts):
			# convert to string of label (i.e. class-ids)
			labelStr = [self.charList.index(c) for c in text]
			# sparse tensor must have size of max. label-string
			if len(labelStr) > shape[1]:
				shape[1] = len(labelStr)
			# put each label into sparse tensor
			for (i, label) in enumerate(labelStr):
				indices.append([batchElement, i])
				values.append(label)

		return (indices, values, shape)


	def decoderOutputToText(self, ctcOutput, batchSize):
		"extract texts from output of CTC decoder"
		
		# contains string of labels for each batch element
		encodedLabelStrs = [[] for i in range(batchSize)]

		# word beam search: label strings terminated by blank
		if self.decoderType == DecoderType.WordBeamSearch:
			blank=len(self.charList)
			for b in range(batchSize):
				for label in ctcOutput[b]:
					if label==blank:
						break
					encodedLabelStrs[b].append(label)

		# TF decoders: label strings are contained in sparse tensor
		else:
			# ctc returns tuple, first element is SparseTensor 
			decoded=ctcOutput[0][0] 

			# go over all indices and save mapping: batch -> values
			idxDict = { b : [] for b in range(batchSize) }
			for (idx, idx2d) in enumerate(decoded.indices):
				label = decoded.values[idx]
				batchElement = idx2d[0] # index according to [b,t]
				encodedLabelStrs[batchElement].append(label)

		# map labels to chars for all batch elements
		return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]


	def trainBatch(self, batch):
		"feed a batch into the NN to train it"
		numBatchElements = len(batch.imgs)
		sparse = self.toSparse(batch.gtTexts)
		rate = 0.01 if self.batchesTrained < 10 else (0.001 if self.batchesTrained < 10000 else 0.0001) # decay learning rate
		#rate = 0.01
		evalList = [self.batch_loss_summary,self.optimizer, self.loss]
		feedDict = {self.inputImgs : batch.imgs, self.gtTexts : sparse , self.seqLen : [Model.maxTextLen] * numBatchElements, self.learningRate : rate, self.is_train: True, self.dropout_rate: 0.6}
		(summary,_, lossVal) = self.sess.run(evalList, feedDict)
		self.writer.add_summary(summary, self.batchesTrained)
		self.batchesTrained += 1
		return lossVal


	def inferBatch(self, batch, calcProbability=False, probabilityOfGT=False):
		"feed a batch into the NN to recognize the texts"
		
		# decode, optionally save RNN output
		numBatchElements = len(batch.imgs)
		evalList = [self.decoder] + ([self.ctcIn3dTBC] if calcProbability else [])
		feedDict = {self.inputImgs : batch.imgs, self.seqLen : [Model.maxTextLen] * numBatchElements, self.is_train: False}
		evalRes = self.sess.run([self.decoder, self.ctcIn3dTBC], feedDict)
		decoded = evalRes[0]
		texts = self.decoderOutputToText(decoded, numBatchElements)
		
		# feed RNN output and recognized text into CTC loss to compute labeling probability
		probs = None
		if calcProbability:
			sparse = self.toSparse(batch.gtTexts) if probabilityOfGT else self.toSparse(texts)
			ctcInput = evalRes[1]
			evalList = self.lossPerElement
			feedDict = {self.savedCtcInput : ctcInput, self.gtTexts : sparse, self.seqLen : [Model.maxTextLen] * numBatchElements, self.is_train: False}
			lossVals = self.sess.run(evalList, feedDict)
			probs = np.exp(-lossVals)
		return (texts, probs)

	def plotMetricsOnTB(self, charErrorRate, wordAccuracy):
		CER_summary = tf.Summary(value=[tf.Summary.Value(tag='Character Error Rate', simple_value=charErrorRate)])
		self.writer.add_summary(CER_summary, self.batchesTrained)
		return	



	def save(self):
		"save model to file"
		self.snapID += 1
		self.saver.save(self.sess, '../model/snapshot', global_step=self.snapID)
		print('[INFO] Saved model to ../model/snapshot')


class ResNet():
	def __init__(self):
		filters = [64, 64, 128, 256, 512]
        kernels = [7, 3, 3, 3, 3]
        strides = [2, 0, 2, 2, 2]

        # conv1
        x = self._conv(self.inputImgs, kernels[0], filters[0], strides[0])
        x = self._bn(x)
        x = self._relu(x)
        x = tf.nn.max_pool(value=x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        # conv2_x
        x = self._residual_block(x, name='conv2_1')
        x = self._residual_block(x, name='conv2_2')

        # conv3_x
        x = self._residual_block_first(x, filters[2], strides[2], name='conv3_1')
        x = self._residual_block(x, name='conv3_2')

        # conv4_x
        x = self._residual_block_first(x, filters[3], strides[3], name='conv4_1')
        x = self._residual_block(x, name='conv4_2')

        # conv5_x
        x = self._residual_block_first(x, filters[4], strides[4], name='conv5_1')
        x = self._residual_block(x, name='conv5_2')




    def conv(self, x, filter_size, out_channel, strides, pad='SAME', name='conv'):
    	in_shape = x.get_shape()
    	with tf.variable_scope(name):
    		kernel = tf.get_variable(name='kernel', 
    			shape=[filter_size, filter_size, in_shape[3], out_channel],
    			dtype=tf.float32, 
    			initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/filter_size/filter_size/out_channel)))

    	conv = tf.nn.conv2d(input=x, filter=kernel, strides=[1, strides, strides, 1], padding=pad)
    	return conv

    def _bn(self, x):
    	return tf.nn.batch_normalization(x, trainable=self.is_train)

    def _relu(self, x):
    	return tf.nn.relu(x, name='relu')

    def _residual_block(self, x, name="unit"):
    	num_channel = x.get_shape().as_list()[-1]
    	with tf.variable_scope(name) as scope:
    		shortcut = x
    		x = self._conv(x, 3, num_channel, 1, input_q=input_q, output_q=output_q, name='conv_1')
            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            x = self._conv(x, 3, num_channel, 1, input_q=output_q, output_q=output_q, name='conv_2')
            x = self._bn(x, name='bn_2')

            x = x + shortcut
            x = self._relu(x, name='relu_2')
        return x

    def _residual_block_first(self, x, out_channel, strides, name="unit"):
        in_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            # Shortcut connection
            if in_channel == out_channel:
                if strides == 1:
                    shortcut = tf.identity(x)
                else:
                    shortcut = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
            else:
                shortcut = self._conv(x, 1, out_channel, strides, name='shortcut')
            # Residual
            x = self._conv(x, 3, out_channel, strides, name='conv_1')
            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            x = self._conv(x, 3, out_channel, 1, name='conv_2')
            x = self._bn(x, name='bn_2')
            # Merge
            x = x + shortcut
            x = self._relu(x, name='relu_2')
        return x


