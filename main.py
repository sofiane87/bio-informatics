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
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.linear_model import Ridge as ridge
from sklearn import svm
from sklearn.model_selection import train_test_split
from  sklearn.ensemble import RandomForestClassifier as RFC
import re
###################### Useful Paths ##################################
scriptPath = os.path.realpath(__file__)
rootPath = os.path.sep.join(scriptPath.split(os.path.sep)[:-1])
dataPath = rootPath + os.path.sep + 'data' +  os.path.sep


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


#################### LOADING FOLDERS #############################

def loadFastaFiles(fileList):
	index = 0

	## Initializing array
	tags = []
	names = []
	sequences = []
	lengthArray = []
	setOfAminoAcids = []
	## For each file
	for name in fileList:
		print('LOADING ELEMENTS FROM FILE : ' + name)
		secondIndex = 0
		## LOAD file
		fasta_sequences =  SeqIO.parse(open(dataPath + name + '.fasta'),'fasta') # The function I miss
		for fasta in fasta_sequences:
			seqName, sequence = fasta.id, str(fasta.seq)

			sequence = re.sub('X|U', '', sequence)
			sequence = re.sub('B', ['N', 'D'][np.random.randint(0,2)], sequence)

			setOfAminoAcids = list(set(setOfAminoAcids).union(set(sequence)))

			# if secondIndex < 20 : 
			secondIndex += 1
			# 	print(len(sequence))


			## Adding sequqnce name to array
			names.append(seqName)
			## addind sequence to sequence array
			sequences.append(sequence)
			## Adding the tag
			tagArray = np.zeros(len(fileList))
			tagArray[index] = 1
			tags.append(tagArray)

			## append the sequence length the length array
			lengthArray.append(len(sequence))
			## append to the list of different amino-acids
		print('Number of Samples for '+ name + ' : ' ,secondIndex)
		index += 1

	return names, sequences, tags, lengthArray, setOfAminoAcids 

def featurize(sequences,setOfAminoAcids,lengthArray):
	unambiguousAminoAcids = [letter for letter in 'ACDEFGHIKLMNPQRSTVWY']
	featurized_sequence = np.zeros([len(sequences), 7+3*len(setOfAminoAcids)])
	featurized_sequence[:,1:len(setOfAminoAcids)+1] = np.array([[sequences[j].count(setOfAminoAcids[i])/lengthArray[j] for i in range(len(setOfAminoAcids))] for j in range(len(sequences))]) 
	featurized_sequence[:,len(setOfAminoAcids)+1:2*len(setOfAminoAcids)+1] =  np.array([[sequences[j][:50].count(setOfAminoAcids[i]) for i in range(len(setOfAminoAcids))] for j in range(len(sequences))]) 
	featurized_sequence[:,2*len(setOfAminoAcids)+1:3*len(setOfAminoAcids)+1] =  np.array([[sequences[j][-50:].count(setOfAminoAcids[i]) for i in range(len(setOfAminoAcids))] for j in range(len(sequences))]) 
	featurized_sequence[:,3*len(setOfAminoAcids)+1] = np.array([ProteinAnalysis(sequences[j]).isoelectric_point() for j in range(len(sequences))])
	featurized_sequence[:,3*len(setOfAminoAcids)+2] = np.array([ProteinAnalysis(sequences[j]).molecular_weight() for j in range(len(sequences))])
	featurized_sequence[:,3*len(setOfAminoAcids)+3] = np.array([ProteinAnalysis(sequences[j]).aromaticity() for j in range(len(sequences))])
	featurized_sequence[:,3*len(setOfAminoAcids)+4:3*len(setOfAminoAcids)+7] = np.array([ProteinAnalysis(sequences[j]).secondary_structure_fraction() for j in range(len(sequences))])

	return featurized_sequence


def findCommonSubstrig(sequences, sequenceLengths):
	initialSequenceIndex = np.argmin(sequenceLengths)
	initialSequence = sequences[initialSequenceIndex]

	longestSequence = initialSequence
	for i in range(len(sequences)):
		if i != initialSequenceIndex:
			if not(longestSequence in sequences[i]):
				print('blue')

####################### EXTRACTING ARGUMENTS ##########################
args = sys.argv[1:] ## getting the arguments

ridge_option = True
svm_option = False
rf_option = False
if ('-ridge' in args):
	ridge_option = True
elif('-svm' in args):
	ridge_option = False
	svm_option = True
elif('-rf' in args):
	ridge_option = False
	rf_option = True

nFolds = 1 
foldString = '-nFold'
if (foldString in args):
	nFolds = int(args[args.index(foldString)+1])

alphaValues = [10**i for i in range(-6,1)]
####################### Loading the Data ################################

print('LOADING THE DATA SET')

dataNames = ['cyto', 'mito', 'nucleus', 'secreted']
names, sequences, tags, lengthArray, setOfAminoAcids  = loadFastaFiles(dataNames)
maxSeqLen = np.max(np.array(lengthArray))

print('Number of Amino Acids possible : ', len(setOfAminoAcids))
print('Number of Samples : ', len(sequences))
print('Maximum Sequence Length : ', maxSeqLen)
print('Number Of Different lengths : ', len(set(lengthArray)))
print('Shuffling the data : ')

print('FEATURIZATION')

featurized_input = featurize(sequences,setOfAminoAcids,lengthArray)

accuracies = np.zeros(nFolds)
scores = np.zeros(nFolds)

train_accuracies = np.zeros(nFolds)
train_scores = np.zeros(nFolds)


tags = np.array(tags)
names = np.array(names)
lengthArray = np.array(lengthArray)


print('Number Of Folds : ', nFolds)

if not(ridge_option) and not(svm_option):
	alphaValues = [0]


for alpha in alphaValues:

	if ridge_option:
		print('alpha = ', alpha)
	elif svm_option:
		print('gamma = ', alpha)
	for i in range(nFolds):

		train_set, dev_set, train_tags, dev_tags = train_test_split(featurized_input, tags, test_size=0.25)


		if ridge_option:
			#print('Ridge Regression - Fold : ', i)

			ridge_model = ridge(alpha = alpha)
			ridge_model.fit(train_set,train_tags)

			predicted_tags = ridge_model.predict(dev_set)
			train_predicted_tags = ridge_model.predict(train_set)

			scores[i] = ridge_model.score(dev_set,dev_tags)
			train_scores[i] = ridge_model.score(train_set,train_tags)
			
			accuracies[i] = np.mean(np.array(predicted_tags.argmax(axis=1) == dev_tags.argmax(axis=1)).astype(int))
			train_accuracies[i] = np.mean(np.array(train_predicted_tags.argmax(axis=1) == train_tags.argmax(axis=1)).astype(int))

			
		elif svm_option:
			#print('SVM - Fold : ', i)

			clf = svm.SVC(probability = True, C = 2, gamma = alpha)
			clf.fit(train_set, train_tags.argmax(axis=1))
			
			predicted_tags = clf.predict(dev_set)
			train_predicted_tags = clf.predict(train_set)

			scores[i] = clf.score(dev_set,dev_tags.argmax(axis=1))
			train_scores[i] = clf.score(dev_set,dev_tags.argmax(axis=1))
			
			accuracies[i] = np.mean(np.array(predicted_tags == dev_tags.argmax(axis=1)).astype(int))
			train_accuracies[i] = np.mean(np.array(train_predicted_tags == train_tags.argmax(axis=1)).astype(int))

		elif rf_option:
			#print('SVM - Fold : ', i)

			clf = RFC(n_estimators = 500)
			clf.fit(train_set, train_tags.argmax(axis=1))
			
			predicted_tags = clf.predict(dev_set)
			train_predicted_tags = clf.predict(train_set)

			scores[i] = clf.score(dev_set,dev_tags.argmax(axis=1))
			train_scores[i] = clf.score(train_set,train_tags.argmax(axis=1))
			
			accuracies[i] = np.mean(np.array(predicted_tags == dev_tags.argmax(axis=1)).astype(int))
			train_accuracies[i] = np.mean(np.array(train_predicted_tags == train_tags.argmax(axis=1)).astype(int))

		if len(alphaValues) == 1:
			print('Fold : ', i )
			print('Accuracy : ', accuracies[i])
	accuracy = np.mean(accuracies)
	score = np.mean(scores)

	train_accuracy = np.mean(train_accuracies)
	train_score = np.mean(train_scores)

	print('ACCURACY : ', accuracy )
	print('SCORE : ', score)
	print('TRAIN ACCURACY : ', train_accuracy )
	print('TRAIN SCORE : ', train_score)
