from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
import datetime
import pdb # for debugging
from SamplePreprocessor import preprocess, custom_preprocess

class DecoderType:
    BestPath = 0
    BeamSearch = 1
    WordBeamSearch = 2

class Model: 
    "minimalistic TF model for HTR"

    # model constants
    batchSize = 50 # default: 50
    #imgSize = (128, 32) # default: (128, 32)
    imgSize = (256, 64)
    maxTextLen = 32
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
        self.inputImgs = tf.placeholder(tf.float32, shape=(None, Model.imgSize[0], Model.imgSize[1])) # : default
        #self.inputImgs = tf.placeholder(tf.float32, shape=(None, Model.imgSize[0], Model.imgSize[1], Model.imgSize[2]))

        # setup CNN, RNN and CTC
        #self.setupCNN() # default
        self.setupWideCNN()
        self.setupRNN()
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

    def setupCNN(self):
        "create CNN layers and return output of these layers"
        "Input: (?, 128, 32) => Output: (?, 32, 1, 256)"
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
            conv = tf.nn.conv2d(pool, kernel, padding='SAME',  strides=(1,1,1,1)) # (?, 128, 32, 1) => (?, 128, 32, 32)
            conv_norm = tf.layers.batch_normalization(conv, training=self.is_train)
            relu = tf.nn.relu(conv_norm)

            kernel2 = tf.Variable(tf.truncated_normal([kernelVals[i], kernelVals[i], featureVals[i + 1], featureVals[i + 1]], stddev=0.1))
            conv2 = tf.nn.conv2d(relu, kernel2, padding='SAME',  strides=(1,1,1,1)) 
            conv_norm2 = tf.layers.batch_normalization(conv2, training=self.is_train) 
            relu2 = tf.nn.relu(conv_norm2)

            conv3 = tf.nn.conv2d(relu2, kernel2, padding='SAME', strides=(1,1,1,1))
            conv_norm3 = tf.layers.batch_normalization(conv3, training=self.is_train)
            relu3 = tf.nn.relu(conv_norm3)

            pool = tf.nn.max_pool(relu3, (1, poolVals[i][0], poolVals[i][1], 1), 
                                    (1, strideVals[i][0], strideVals[i][1], 1), 'VALID')
        self.dropout_rate = tf.placeholder_with_default(0.0, shape=())
        prob = tf.math.subtract(1.0, self.dropout_rate)
        dropout = tf.nn.dropout(pool, keep_prob=prob)
        self.cnnOut4d = dropout

    def setupWideCNN(self):
        "Input: (?, 256, 64) => Output: (?, 32, 1, 256)"
        cnnIn4d = tf.expand_dims(input=self.inputImgs, axis=3) # (?, 256, 64) => (?, 256, 64, 1)
        # list of parameters for the layers
        kernelVals = [5, 5, 3, 3, 3]
        featureVals = [1, 32, 64, 128, 128, 256]
        strideVals = poolVals = [(2,2), (2,2), (2,2), (1,2), (1,2)]
        numLayers = len(strideVals)

        # create layers
        pool = cnnIn4d # input to first CNN layer
        for i in range(numLayers):
            kernel = tf.Variable(tf.truncated_normal([kernelVals[i], kernelVals[i], featureVals[i], featureVals[i + 1]], stddev=0.1)) # , name='kernel{}_1'.format(i)
            conv = tf.nn.conv2d(pool, kernel, padding='SAME',  strides=(1,1,1,1)) # (?, 256, 64, 1) => (?, 256, 64, 32)
            conv_norm = tf.layers.batch_normalization(conv, training=self.is_train) 
            relu = tf.nn.relu(conv_norm) 

            kernel2 = tf.Variable(tf.truncated_normal([kernelVals[i], kernelVals[i], featureVals[i + 1], featureVals[i + 1]], stddev=0.1)) #, name='kernel{}_2'.format(i)
            conv2 = tf.nn.conv2d(relu, kernel2, padding='SAME',  strides=(1,1,1,1)) 
            conv_norm2 = tf.layers.batch_normalization(conv2, training=self.is_train) 
            relu2 = tf.nn.relu(conv_norm2)

            conv3 = tf.nn.conv2d(relu2, kernel2, padding='SAME', strides=(1,1,1,1)) 
            conv_norm3 = tf.layers.batch_normalization(conv3, training=self.is_train)
            relu3 = tf.nn.relu(conv_norm3)

            pool = tf.nn.max_pool(relu3, (1, poolVals[i][0], poolVals[i][1], 1), 
                                    (1, strideVals[i][0], strideVals[i][1], 1), 'VALID') # (?, 32, 2, 256)

        kernel3 = tf.Variable(tf.truncated_normal([1, 1, 256, 256], stddev=0.1)) # 1 x 1 convolution
        conv4 = tf.nn.conv2d(pool, kernel3, padding='SAME', strides=(1,1,2,1))
        #pdb.set_trace()
        self.dropout_rate = tf.placeholder_with_default(0.0, shape=())
        prob = tf.math.subtract(1.0, self.dropout_rate)
        dropout = tf.nn.dropout(conv4, keep_prob=prob)
        self.cnnOut4d = dropout

    #def visualizeCNN(self, image):


    def setupRNN(self):
        "create RNN layers and return output of these layers"
        rnnIn3d = tf.squeeze(self.cnnOut4d, axis=[2]) # (?, 32, 1, 256) => (?, 32, 256)

        # basic cells which is used to build RNN
        numHidden = 256
        cells = [tf.contrib.rnn.LSTMCell(num_units=numHidden, state_is_tuple=True) for i in range(2)] # default: 2 layers

        # stack basic cells
        stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        # bidirectional RNN
        ((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnnIn3d, dtype=rnnIn3d.dtype)
                       
        # BxTxH + BxTxH -> BxTx2H -> BxTx1X2H i.e. expand dim needed for atrous convolution
        concat = tf.expand_dims(tf.concat([fw, bw], 2), 2) # (?, 32, 256) => (?, 32, 512) => (?, 32, 1, 512)
                        
        # project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
        kernel = tf.Variable(tf.truncated_normal([1, 1, numHidden * 2, len(self.charList) + 1], stddev=0.1))
        self.rnnOut3d = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2]) # (?, 32, 1, 512) => (?, 32, 1, 80) => (?, 32, 80)

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

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess=tf.Session(config=config) # TF session

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
            # print(ctcOutput)

            # go over all indices and save mapping: batch -> values
            idxDict = { b : [] for b in range(batchSize) }
            for (idx, idx2d) in enumerate(decoded.indices):
                label = decoded.values[idx] # e.g. decoded.values = [67, 64, 57]
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
        feedDict = {self.inputImgs : batch.imgs, self.gtTexts : sparse , self.seqLen : [Model.maxTextLen] * numBatchElements, self.learningRate : rate, self.is_train: True, self.dropout_rate: 0.6} # 
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

        # get the indexes of the characters 
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

    def save(self):
        "save model to file"
        self.snapID += 1
        self.saver.save(self.sess, '../model/snapshot', global_step=self.snapID)
        print('[INFO] Saved model to ../model/snapshot')

def main():
    from tensorflow.python.tools import inspect_checkpoint as chkp 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='name of model')
    parser.add_argument('--i', help='rel or abs path to input image to pass through model')
    args = parser.parse_args()

    # See weights
    chkp.print_tensors_in_checkpoint_file('../model/' + args.model, tensor_name='', all_tensors=True)

    model = Model(open('../model/charList.txt').read(), decoderType, mustRestore=True, modelName=modelName)
    img = preprocess(cv2.imread(args.i, cv2.IMREAD_GRAYSCALE), model.imgSize)
    img = tf.expand_dims(input=img, axis=3)
    kernel1_1 = tf.get_collection(tf.GraphKeys.VARIABLES, 'conv1_1')
    conv1_1 = tf.nn.conv2d(img, kernel1_1, padding='SAME',  strides=(1,1,1,1))
    with model.writer.as_default():
        tf.summary.image('conv1_1', conv1_1, step=0)

if __name__=='__main__':
    main()