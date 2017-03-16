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
from sklearn.linear_model import RidgeClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from  sklearn.ensemble import RandomForestClassifier as RFC
import re
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, make_scorer, confusion_matrix
import matplotlib.pyplot as plt
import itertools

###################### Useful Paths ##################################
scriptPath = os.path.realpath(__file__)
rootPath = os.path.sep.join(scriptPath.split(os.path.sep)[:-1])
dataPath = rootPath + os.path.sep + 'data' +  os.path.sep
seed  = 42

####################### Auxiliary function ###########################
def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):


	if normalize:
		cm = np.floor((cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])*100).astype(int)/100
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')


def sklearn_Grid_Search(model, parameters, X, Y, n_folds = 4):
    # grid search cross validation
    GSCV = GridSearchCV(model, parameters, cv=KFold(n_splits=n_folds, shuffle=True, random_state=42), verbose = 3)
    print(X.shape)
    print(Y.shape)
    GSCV.fit(X, Y)

    print("Best parameters set found on development set:")
    print(GSCV.best_params_)

    print("Grid scores on development set:")
    means = GSCV.cv_results_['mean_test_score']
    stds = GSCV.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, GSCV.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    return GSCV


def grid_search(model, parameters, X, Y, seed=seed, n_folds = 10):
    # grid search cross validation
    grid_CV = GridSearchCV(model, parameters, cv=KFold(n_splits=n_folds, shuffle=True, random_state=seed), verbose = 3)
    grid_CV.fit(X, Y)

    print("Best parameters set found on development set:")
    print(grid_CV.best_params_)

    print("Grid scores on development set:")
    means = grid_CV.cv_results_['mean_test_score']
    stds = grid_CV.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_CV.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    return grid_CV


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

			secondIndex += 1

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
	full_seq = np.array([ProteinAnalysis(sequences[j]) for j in range(len(sequences))])
	initial_seq = np.array([ProteinAnalysis(sequences[j][:50]) for j in range(len(sequences))])
	final_seq = np.array([ProteinAnalysis(sequences[j][-50:]) for j in range(len(sequences))])

	featurized_sequence = np.zeros([len(sequences), 17+3*len(setOfAminoAcids)])
	featurized_sequence[:,0] = np.array([len(sequences[j]) for j in range(len(sequences))]) 
	featurized_sequence[:,1:len(setOfAminoAcids)+1] = np.array([[full_seq[j].get_amino_acids_percent()[i] for i in setOfAminoAcids] for j in range(len(sequences))]) 
	featurized_sequence[:,len(setOfAminoAcids)+1:2*len(setOfAminoAcids)+1] =  np.array([[initial_seq[j].get_amino_acids_percent()[i] for i in setOfAminoAcids]  for j in range(len(sequences))]) 
	featurized_sequence[:,2*len(setOfAminoAcids)+1:3*len(setOfAminoAcids)+1] =  np.array([[final_seq[j].get_amino_acids_percent()[i] for i in setOfAminoAcids]  for j in range(len(sequences))]) 
	featurized_sequence[:,3*len(setOfAminoAcids)+1] = np.array([full_seq[j].isoelectric_point() for j in range(len(sequences))])
	featurized_sequence[:,3*len(setOfAminoAcids)+2] = np.array([full_seq[j].gravy() for j in range(len(sequences))])
	featurized_sequence[:,3*len(setOfAminoAcids)+3] = np.array([full_seq[j].molecular_weight() for j in range(len(sequences))])
	featurized_sequence[:,3*len(setOfAminoAcids)+4] = np.array([full_seq[j].aromaticity() for j in range(len(sequences))])
	featurized_sequence[:,3*len(setOfAminoAcids)+5] = np.array([initial_seq[j].isoelectric_point() for j in range(len(sequences))])
	featurized_sequence[:,3*len(setOfAminoAcids)+7] = np.array([initial_seq[j].gravy() for j in range(len(sequences))])
	featurized_sequence[:,3*len(setOfAminoAcids)+8] = np.array([initial_seq[j].molecular_weight() for j in range(len(sequences))])
	featurized_sequence[:,3*len(setOfAminoAcids)+9] = np.array([initial_seq[j].aromaticity() for j in range(len(sequences))])	
	featurized_sequence[:,3*len(setOfAminoAcids)+10] = np.array([final_seq[j].isoelectric_point() for j in range(len(sequences))])
	featurized_sequence[:,3*len(setOfAminoAcids)+11] = np.array([final_seq[j].gravy() for j in range(len(sequences))])
	featurized_sequence[:,3*len(setOfAminoAcids)+12] = np.array([final_seq[j].molecular_weight() for j in range(len(sequences))])
	featurized_sequence[:,3*len(setOfAminoAcids)+13] = np.array([final_seq[j].aromaticity() for j in range(len(sequences))])	
	featurized_sequence[:,3*len(setOfAminoAcids)+14:3*len(setOfAminoAcids)+17] = np.array([full_seq[j].secondary_structure_fraction() for j in range(len(sequences))])
	#featurized_sequence[:,3*len(setOfAminoAcids)+17:3*len(setOfAminoAcids)+19] = np.array([[len(sequences[j][:50]), len(sequences[j][-50:])] for j in range(len(sequences))]) 

	#featurized_sequence[:,3*len(setOfAminoAcids)+8:3*len(setOfAminoAcids)+11] = np.array([[len(re.findall('RL.{5}?HL', sequences[j][:50])) ,len(re.findall('RL.{5}?HL', sequences[j][-50:])), len(re.findall('RL.{5}?HL', sequences[j]))] for j in range(len(sequences))])
	#featurized_sequence[:,] = np.array([[sequences[j][-50:].count('KDEL'),sequences[j][-50:].count('SKL')] for j in range(len(sequences))])

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

X_train_all = featurize(sequences,setOfAminoAcids,lengthArray)

Y_train_all = np.array(tags).argmax(axis = 1)
names = np.array(names)


if ridge_option : 
	classifier = RidgeClassifier() 
	alpha_range = np.logspace(-6, 5, num=11, endpoint=False)
	model_params = {'alpha': alpha_range}

elif svm_option : 
	classifier = svm.SVC()
	model_params = {'C': [1, 2, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}
	 
elif rf_option : 
	classifier = RFC()
	estimators_range = np.int_(np.linspace(250, 1000, num=4, endpoint=True))
	min_samples_leaf = np.array(list(range(1, 21)))
	max_features = np.linspace(0.1, 1.0, num=9, endpoint=True)
	model_params = {'n_jobs' : [-1], 'n_estimators': estimators_range, 'min_samples_leaf': min_samples_leaf, 'max_features' : max_features}


classifier = grid_search(classifier, model_params, X_train_all, Y_train_all, seed)
print(classifier.best_params_, classifier.best_score_)




	
# train_accuracy = np.mean()
# train_score = np.mean()

# print('ACCURACY : ', accuracy )
# print('SCORE : ', score)
# print('TRAIN ACCURACY : ', train_accuracy )
# print('TRAIN SCORE : ', train_score)


# cm = confusion_matrix(y_true = dev_tags.argmax(axis=1), y_pred = predicted_tags)
# plt.figure()	
# plot_confusion_matrix(cm, dataNames)
# plt.show()