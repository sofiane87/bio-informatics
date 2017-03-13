#################### LOADING PYTHON PACKAGES ##########################
import numpy as np
from time import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os.path as path
import os
import tensorflow as tf
import sys
import platform
from Bio import SeqIO
from sklearn.linear_model import Ridge as ridge



####################### Auxiliary function ###########################
def printResults(results,index):
	if (platform.system().lower() == 'windows'):
		os.system('cls')
	else:
		os.system('clear')

	for i in range(index+1):
		print('------------- EPOCH : ' + str(i) + ' ------------' )
		print('Test-loss : %.4f'  %(results[i][0]) + ' - Test-Accuracy %.4f'  %(results[i][1]) )
		print('Train-loss : %.4f'  %(results[i][2]) + ' - Train-Accuracy %.4f'  %(results[i][3]) )

####################### EXTRACTING ARGUMENTS ##########################
args = sys.argv[1:] ## getting the arguments


if not('-t' in args): # -t : Option to train or retrain the model
	retrain = False
else : 
	retrain = True

if not('-ct' in args): # -t : Option to train or retrain the model
	completeTraining = False
else : 
	completeTraining = True


plot = False

if ('-plot' in args): ## -plot : This option displays write the learning curves into .png files
	plot = True 


prediction = True
if ('-np' in args): ## -np : option to avoid predictions
	prediction = False


learningRate = 0.001

if ('-lr' in args): ## -lr : option to specify the learning rate 
	learningRate = float(args[args.index('-lr') + 1])

latex = False
if('-latex' in args):
	latex = True

bagging = True
if ('-bag' in args):
	baggin = True

rnn = False
if ('-rnn' in args):
	bagging = False
	rnn = True

rnn_cell = 'GRU'
lstm_bool = False
if ('-lstm' in args):
	rnn_cell = 'LSTM'
	lstm_bool = True

hiddenSize = 128
if ('-units' in args):
	hiddenSize = int(args[args.index('-units') +1])

stacked_text = '' 
stacked = False
if ('-stacked' in args): ## -stacked : use a layer stacked model 
	stacked = True
	stacked_text = 'STACKED-'


if not('-layers' in args): ## -layers : This option specifies the number 
	if stacked:
		numberOfLayers = 3 
	else:
		numberOfLayers = 1
else:
	numberOfLayers = int(args[args.index('-layers') + 1])


if not('-epoch' in args): ## -batches : Number of batches tu run during training
	numberOfEpochs = 5
else : 
	numberOfEpochs = int(args[args.index('-batches') + 1])



if not('-bs' in args): ## -bs :  Batch size 
	batchSize = 300
else : 
	batchSize = int(args[args.index('-bs') + 1])

####################### DISPLAYING PARAMETERS ##########################

print('CURRENT PARAMETERS : ')
print('TRAINING THE MODEL : ' + str(retrain)) 

####################### BUILDING PATHS ################################
scriptPath = os.path.realpath(__file__)
rootPath = os.path.sep.join(scriptPath.split(os.path.sep)[:-1])
dataPath = rootPath + os.path.sep + 'data' +  os.path.sep
modelPath = rootPath + os.path.sep + 'models' +  os.path.sep
if rnn:
	if not(os.path.exists(modelPath + rnn_cell + '_%.3f' % learningRate + os.path.sep )):
		os.makedirs(modelPath + rnn_cell + '_%.3f' % learningRate + os.path.sep)
	saveFileName  = modelPath + rnn_cell + '_%.3f' % learningRate + os.path.sep + 'model.ckpt'
	learningCurvePath = modelPath + rnn_cell + '_%.3f' % learningRate + os.path.sep + 'learningCurve'
####################### Loading the Data ################################

dataNames = ['cyto', 'mito', 'nucleus', 'secreted']

tags = []
names = []
sequences = []

listOfValues = []
maxSeqLen = 0
lengthArray = []

index = 0
for name in dataNames:
	fasta_sequences =  SeqIO.parse(open(dataPath + name + '.fasta'),'fasta') # The function I miss
	for fasta in fasta_sequences:
			seqName, sequence = fasta.id, str(fasta.seq)
			### 
			names.append(seqName)
			sequences.append(sequence)
			tagArray = np.zeros(len(dataNames))
			tagArray[index] = 1
			tags.append(tagArray)
			lengthArray.append(len(sequence))
			###
			listOfValues = list(set(listOfValues).union(set(sequence)))
			maxSeqLen = max(maxSeqLen, len(sequence))
	index += 1

print('Number of Amino Acids possible : ', len(listOfValues))
print('Number of Samples : ', len(sequences))
print('Maximum Sequence Length : ', maxSeqLen)
print('Number Of Different lengths : ', len(set(lengthArray)))
print('Shuffling the data : ')


seqLength =len(sequences)
randPerm = np.random.permutation(range(0,seqLength))
tags = np.array(tags)[randPerm]
names = np.array(names)[randPerm]
sequences = np.array(sequences)[randPerm]
lengthArray = np.array(lengthArray)[randPerm]
sequenceList = lengthArray[np.argsort(-lengthArray)]



if bagging:
	print('RIDGE MODEL CHOSEN')
	print('COMPUTING BAGS')
	full_bag = np.array([[sequences[j].count(listOfValues[i]) for i in range(len(listOfValues)) ] for j in range(len(sequences))])
	train_set = full_bag[:-222]
	train_tags = tags[:-222]

	dev_set = full_bag[-222:]
	dev_tags = tags[-222:]

	print('FITTING MODEL')
	ridge_model = ridge(alpha = 0.01)
	ridge_model.fit(train_set,train_tags)

	print('Predicting')
	predicted_tags = ridge_model.predict(dev_set)
	print('SCORE : ', ridge_model.score(dev_set,dev_tags))
	accuracy = np.mean(np.array(predicted_tags.argmax(axis=1) == dev_tags.argmax(axis=1)).astype(int))
	print('ACCURACY : ', accuracy )



if rnn:
	############################ Preprocessing The Data ####################################


	print('RUNNING RNN MODEL : ', rnn_cell)

	numpyDataPath = dataPath + 'rnn' + os.path.sep + 'Preprocessing' + os.path.sep

	print('PRE-PROCESSING : ')

	if not(os.path.exists(numpyDataPath + 'dev_set.npy')):
		tooLong_set = sequences[np.argsort(-lengthArray)[:2]]
		tooLong_length = lengthArray[np.argsort(-lengthArray)[:2]]
		tooLong_tags = tags[np.argsort(-lengthArray)[:2]]

		## Shortened
		sequences =  sequences[np.argsort(-lengthArray)[2:]]
		tags = tags[np.argsort(-lengthArray)[2:]]
		lengthArray = lengthArray[np.argsort(-lengthArray)[2:]]

		## Train
		Sequence_train = np.full([seqLength,max(lengthArray)],-1)
		print('Sequence_train size : ', Sequence_train.shape)
		for i in range(len(sequences)):
			if i % 1000 == 0:
				print('Sample : ', i)
			Sequence_train[i,:len(sequences[i])] = [listOfValues.index(sequences[i][e]) for e in range(len(sequences[i]))]
		train_set = Sequence_train[:-220]
		train_tags = tags[:-220]
		train_length =lengthArray[:-220]

		## Dev Set
		dev_set = Sequence_train[-220:]
		dev_tags = tags[-220:]
		dev_length = lengthArray[-220:]

		if not(os.path.exists(numpyDataPath)):
			os.makedirs(numpyDataPath)

		np.save(numpyDataPath + 'train_set', train_set)
		np.save(numpyDataPath + 'train_tags', train_tags)
		np.save(numpyDataPath + 'train_length', train_length)

		np.save(numpyDataPath + 'dev_set', dev_set)
		np.save(numpyDataPath + 'dev_tags', dev_tags)
		np.save(numpyDataPath + 'dev_length', dev_length)

		np.save(numpyDataPath + 'tooLong_set', tooLong_set)
		np.save(numpyDataPath + 'tooLong_tags', tooLong_tags)
		np.save(numpyDataPath + 'tooLong_length', tooLong_length)
	else:
		train_set = np.load('train_set.npy')
		train_tags = np.load('train_tags.npy')
		train_length = np.load('train_length.npy')

		dev_set = np.load('dev_set.npy')
		dev_tags = np.load('dev_tags.npy')
		dev_length = np.load('dev_length.npy')

		tooLong_set = np.load('tooLong_set.npy')
		tooLong_tags = np.load('tooLong_tags.npy')
		tooLong_length = np.load('tooLong_length.npy')		
 	########################### BUILDING THE MODEL ###############################################

	print('BUILDING THE MODEL :' + stacked_text + rnn_cell)

	## Place Holders
	class_size = 4
	x = tf.placeholder(tf.float32, [None, None])
	y_ = tf.placeholder(tf.float32, [None, class_size])
	lengths = tf.placeholder(tf.int32 , [None])
	seq_length = tf.shape(x)[1]


	print('RESHAPING THE INPUT : ADDING ONE DIMENSON')

	new_x = tf.reshape(x,[-1,seq_length,1])


	######################## RECCURENT LAYER ################################

	print('ADDING ' + stacked_text + rnn_cell + ' CELL')

	if lstm_bool:
		cell = tf.nn.rnn_cell.LSTMCell(hiddenSize, state_is_tuple=True)
	else:
		cell = tf.nn.rnn_cell.GRUCell(hiddenSize)

	if stacked:
		cell = tf.nn.rnn_cell.MultiRNNCell([cell] * numberOfLayers)

	full_lstm_out, _ = tf.nn.dynamic_rnn(cell, new_x, dtype=tf.float32, sequence_length = lengths)  

	lstm_out = full_lstm_out[:,-1]

	######################## OUTPUT LAYER ##################################
	print('ADDING AFFINE LAYERS')

	## Affine + relu

	W2 = tf.Variable(tf.truncated_normal([hiddenSize, 100], stddev=0.1))
	b2 = tf.Variable(tf.zeros([100]))
	l2 = tf.nn.relu(tf.matmul(lstm_out, W2) + b2)

	## Affine + softmax

	W3 = tf.Variable(tf.truncated_normal([100, class_size], stddev=0.1))
	b3 = tf.Variable(tf.zeros([class_size]))
	y = tf.matmul(l2,W3) + b3
	y2 = tf.nn.softmax(tf.matmul(l2, W3) + b3)

	####################### CROSS ENTROPY FUNCTION #########################

	print('ADDING CROSS ENTROPY LAYER')

	##Cross entropy layer
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))

	## Defining the evaluation function
	correct_prediction = tf.equal(tf.argmax(y2,1), tf.argmax(y_,1))

	## Defining the accuracy
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	####################### TRAINING STEP  ###################################

	## Adam Optimizer
	print('TRAINING USING : AdamOptimizer')

	train_step = tf.train.AdamOptimizer(learningRate).minimize(cross_entropy)


	####################### DEFINING THE SAVER ##################################

	saver = tf.train.Saver()


	print('INITIALIZING MODEL ')

	## Initiliazing the variables
	init = tf.global_variables_initializer()


	## Start the session 
	sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

	sess.run(init)

	test_errorRate = 1;

	if (retrain or not(path.exists(saveFileName)) ) : 
		
		results = np.zeros([numberOfEpochs,4])
		if completeTraining:

			print('CONTINUING TRAINING OF THE MODEL')
			saver.restore(sess, saveFileName)
			print('MODEL RELOADED ')

			try: ## parsing the previous File to extract the last epoch and the previous accuracy
				temp_results = np.load(learningCurvePath + '.npy')
				initialEpoch = np.array(range(temp_results.shape[0]))[np.absolute(temp_results).sum(axis = 1) == 0][0]
				results[:initialEpoch] = temp_results
			except:
				initialEpoch = 0
		else :
			print('TRAINING THE MODEL')
			initialEpoch = 0


		for epoch in range(initialEpoch,numberOfEpochs):
			for i in range(int(np.ceil(train_set.shape[0]/batchSize))):
				batch_xs = np.array(train_set[i*batchSize:max((i+1)*batchSize,train_set.shape[0])])
				batch_ys = np.array(train_tags[i*batchSize:max((i+1)*batchSize,train_set.shape[0])])
				batch_lengths = np.array(train_length[i*batchSize:max((i+1)*batchSize,train_set.shape[0])])
				sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys, lengths:batch_lengths})
			test_cross_entropy, test_accuracy = sess.run([cross_entropy, accuracy], feed_dict={x: dev_set, y_: dev_tags, lengths:dev_length})
			train_cross_entropy, train_accuracy = sess.run([cross_entropy, accuracy], feed_dict={x: train_set, y_: train_tags, lengths: train_length})
			results[epoch] = [test_cross_entropy, test_accuracy,train_cross_entropy, train_accuracy]
			printResults(results, epoch)
			if (test_errorRate >= 1- test_accuracy):
				test_errorRate = 1- test_accuracy
				saver.save(sess,saveFileName)
				print('SAVING DONE')
	else:
		print('LOADING THE MODEL')
		saver.restore(sess, saveFileName)


	test_cross_entropy, test_accuracy = sess.run([cross_entropy, accuracy], feed_dict={x: dev_set, y_: dev_tags})
	train_cross_entropy, train_accuracy = sess.run([cross_entropy, accuracy], feed_dict={x: train_set, y_: train_tags})
	print('------------- EPOCH : ' + str(i) + ' ------------' )
	print('Test-loss : %.4f'  %(test_cross_entropy) + ' - Test-Accuracy %.4f'  %(test_accuracy) )
	print('Train-loss : %.4f'  %(train_cross_entropy) + ' - Train-Accuracy %.4f'  %(train_accuracy) )

