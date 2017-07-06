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


from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import mean_squared_error, make_scorer, confusion_matrix


import re
import itertools

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.normalization import BatchNormalization



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

	#print(cm)

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
    print('BIGINING GRID SEACH')
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
			sequence = re.sub('X', ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V" ][np.random.randint(0,20)], sequence)
			sequence = re.sub('U', '', sequence)
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

def featurize(sequences,setOfAminoAcids,lengthArray, isForSVM = False):
	specific_sequences = ['WW','MWW', 'CWW', 'WWM', 'WMW', 'WCW', 'CWM', 'WCM', 'MCW', 'WWH', 'FWW', 'WCWE', 'WMYH', 'WMYI', 'WMYL', 'WMYM', 'WMYQ', 'WMYV', 'WMYW', 'WMYY', 'QPWR']
	initial_sequences = ['WWF', 'WMM', 'WQW', 'HWC', 'FWC', 'HWD', 'CMW', 'VMW', 'MM', 'MHV']
	final_sequences = ['MW', 'CMM', 'WCM', 'WHF', 'PMW', 'AWW', 'CM', 'MMC', 'WFY', 'YWC']

	unambiguousAminoAcids = [letter for letter in 'ACDEFGHIKLMNPQRSTVWY']
	full_seq = np.array([ProteinAnalysis(sequences[j]) for j in range(len(sequences))])
	initial_seq = np.array([ProteinAnalysis(sequences[j][:50]) for j in range(len(sequences))])
	final_seq = np.array([ProteinAnalysis(sequences[j][-50:]) for j in range(len(sequences))])
	initial_seq_100 = np.array([ProteinAnalysis(sequences[j][:100]) for j in range(len(sequences))])
	final_seq_100 = np.array([ProteinAnalysis(sequences[j][-100:]) for j in range(len(sequences))])



	print('BUILDING DICTIONNARY ')
	featureDictionnary = dict()

	print('LENGTH FEATURE')
	featureDictionnary['length'] = np.array([[len(sequences[j])] for j in range(len(sequences))])
	
	print('FREQUENCY FEATURE')
	featureDictionnary['frequency'] = np.array([[full_seq[j].get_amino_acids_percent()[i] for i in setOfAminoAcids] for j in range(len(sequences))]) 
	featureDictionnary['initialFrequency'] = np.array([[initial_seq[j].get_amino_acids_percent()[i] for i in setOfAminoAcids]  for j in range(len(sequences))]) 
	featureDictionnary['finalFrequency'] = np.array([[final_seq[j].get_amino_acids_percent()[i] for i in setOfAminoAcids]  for j in range(len(sequences))]) 
	featureDictionnary['initialFrequency100'] = np.array([[initial_seq_100[j].get_amino_acids_percent()[i] for i in setOfAminoAcids]  for j in range(len(sequences))]) 
	featureDictionnary['finalFrequency100'] = np.array([[final_seq_100[j].get_amino_acids_percent()[i] for i in setOfAminoAcids]  for j in range(len(sequences))]) 
	
	print('COUNT FEATURE')
	featureDictionnary['counts'] = np.array([[full_seq[j].count_amino_acids()[i] for i in setOfAminoAcids] for j in range(len(sequences))]) 
	featureDictionnary['initialcounts'] = np.array([[initial_seq[j].count_amino_acids()[i] for i in setOfAminoAcids]  for j in range(len(sequences))]) 
	featureDictionnary['finalcounts'] = np.array([[final_seq[j].count_amino_acids()[i] for i in setOfAminoAcids]  for j in range(len(sequences))]) 
	featureDictionnary['initialcounts100'] = np.array([[initial_seq_100[j].count_amino_acids()[i] for i in setOfAminoAcids]  for j in range(len(sequences))]) 
	featureDictionnary['finalcounts100'] = np.array([[final_seq_100[j].count_amino_acids()[i] for i in setOfAminoAcids]  for j in range(len(sequences))]) 

	print('ISO-ELECTRIC POINT FEATURE')
	featureDictionnary['iso'] =  np.array([[full_seq[j].isoelectric_point()] for j in range(len(sequences))])
	featureDictionnary['initialIso'] =  np.array([[initial_seq[j].isoelectric_point()] for j in range(len(sequences))])
	featureDictionnary['finalIso'] =  np.array([[final_seq[j].isoelectric_point()] for j in range(len(sequences))])
	featureDictionnary['initialIso100'] =  np.array([[initial_seq_100[j].isoelectric_point()] for j in range(len(sequences))])
	featureDictionnary['finalIso100'] =  np.array([[final_seq_100[j].isoelectric_point()] for j in range(len(sequences))])
	
	print('GRAVY FEATURE')
	featureDictionnary['gravy']  = np.array([[full_seq[j].gravy()] for j in range(len(sequences))])
	featureDictionnary['initialGravy']  = np.array([[initial_seq[j].gravy()] for j in range(len(sequences))])
	featureDictionnary['finalGravy']  = np.array([[final_seq[j].gravy()] for j in range(len(sequences))])
	featureDictionnary['initialGravy100']  = np.array([[initial_seq_100[j].gravy()] for j in range(len(sequences))])
	featureDictionnary['finalGravy100']  = np.array([[final_seq_100[j].gravy()] for j in range(len(sequences))])
	
	print('MOLECULAR WEIGHT FEATURE')

	if not(isForSVM):
	
		featureDictionnary['weight']  = np.array([[full_seq[j].molecular_weight()] for j in range(len(sequences))])
		featureDictionnary['initialweight']  = np.array([[initial_seq[j].molecular_weight()] for j in range(len(sequences))])
		featureDictionnary['finalweight']  = np.array([[final_seq[j].molecular_weight()] for j in range(len(sequences))])
		featureDictionnary['initialweight100']  = np.array([[initial_seq_100[j].molecular_weight()] for j in range(len(sequences))])
		featureDictionnary['finalweight100']  = np.array([[final_seq_100[j].molecular_weight()] for j in range(len(sequences))])

	print('AROMATICITY FEATURE')
	featureDictionnary['aromaticity'] = np.array([[full_seq[j].aromaticity()] for j in range(len(sequences))])
	featureDictionnary['initialAromaticity'] = np.array([[initial_seq[j].aromaticity()] for j in range(len(sequences))])	
	featureDictionnary['finalAromaticity'] = np.array([[final_seq[j].aromaticity()] for j in range(len(sequences))])	

	# print('INSTABILITY INDEX FEATURE')
	# featureDictionnary['instability'] = np.array([[full_seq[j].instability_index()] for j in range(len(sequences))])
	# featureDictionnary['initialinstability'] = np.array([[initial_seq[j].instability_index()] for j in range(len(sequences))])	
	# featureDictionnary['finalinstability'] = np.array([[final_seq[j].instability_index()] for j in range(len(sequences))])	


	# print('SECONDARY-STRUCTURE FEATURE')
	# featureDictionnary['secondary'] = np.array([full_seq[j].secondary_structure_fraction() for j in range(len(sequences))])

	# print('SPECIFIC-SEQUENCES FEATURE')
	# featureDictionnary['specific'] = np.array([[sequence.count(specific_sequence) for specific_sequence in specific_sequences] for sequence in sequences])
	# featureDictionnary['initialSpecific'] = np.array([[sequence[:50].count(specific_sequence) for specific_sequence in initial_sequences] for sequence in sequences])
	# featureDictionnary['finalSpecific'] = np.array([[sequence[-50:].count(specific_sequence) for specific_sequence in final_sequences] for sequence in sequences])

	print('SPECIFIC-SEQUENCES FEATURE')
	featureDictionnary['specific'] = np.array([[ int(specific_sequence in sequence) for specific_sequence in specific_sequences] for sequence in sequences])
	featureDictionnary['initialSpecific'] = np.array([[ int(specific_sequence in sequence[:50]) for specific_sequence in initial_sequences] for sequence in sequences])
	featureDictionnary['finalSpecific'] = np.array([[ int(specific_sequence in sequence[-50:]) for specific_sequence in final_sequences] for sequence in sequences])

	print('NES COUNT FEATURE')

	featureDictionnary['NES'] = np.array([[len(re.findall('[L|I|V|F|M][A-Z][A-Z][A-Z][L|I|V|F|M][A-Z][A-Z][L|I|V|F|M][A-Z][L|I|V|F|M]', sequence))] for sequence in sequences])
	featureDictionnary['InitialNES'] = np.array([[len(re.findall('[L|I|V|F|M][A-Z][A-Z][A-Z][L|I|V|F|M][A-Z][A-Z][L|I|V|F|M][A-Z][L|I|V|F|M]', sequence[:50]))] for sequence in sequences])
	featureDictionnary['finalNES'] = np.array([[len(re.findall('[L|I|V|F|M][A-Z][A-Z][A-Z][L|I|V|F|M][A-Z][A-Z][L|I|V|F|M][A-Z][L|I|V|F|M]', sequence[-50:]))] for sequence in sequences])

	print('POS LIST FEATURE')

	tempPOS = [re.findall('[K|R|H]*', sequence) for sequence in sequences]
	featureDictionnary['5POS'] = np.array([[ len([subseq for subseq in tempPOS[i] if len(subseq) == 5])] for i in range(len(sequences))])
	featureDictionnary['4POS'] = np.array([[ len([subseq for subseq in tempPOS[i] if len(subseq) > 4])] for i in range(len(sequences))])
	tempInitialPOS = [re.findall('[K|R|H]*', sequence[:50]) for sequence in sequences]
	featureDictionnary['5InitialPOS'] = np.array([[ len([subseq for subseq in tempInitialPOS[i] if len(subseq) == 5])] for i in range(len(sequences))])
	featureDictionnary['4InitialPOS'] = np.array([[ len([subseq for subseq in tempInitialPOS[i] if len(subseq) > 4])] for i in range(len(sequences))])
	tempFinalPOS = [re.findall('[K|R|H]*', sequence[-50:]) for sequence in sequences]
	featureDictionnary['5FinalPOS'] = np.array([[ len([subseq for subseq in tempFinalPOS[i] if len(subseq) == 5])] for i in range(len(sequences))])
	featureDictionnary['4FinalPOS'] = np.array([[ len([subseq for subseq in tempFinalPOS[i] if len(subseq) > 4])] for i in range(len(sequences))])


	# print('N-TERMINUS SEQUENCE')
	# tempNTerminus = [re.findall('[V|I|L|M|F|A|C]*', sequence[:50]) for sequence in sequences]
	# featureDictionnary['NTerminus'] = np.array([[ len([subseq for subseq in tempNTerminus[i] if len(subseq) >= 5])] for i in range(len(sequences))])

	print('TRANSFORMING INTO ARRAY')

	featureSize = 0
	for key in featureDictionnary:
		featureSize += featureDictionnary[key].shape[1]

	print('NUMBER OF FEATURES : ', featureSize)

	featurized_sequence = np.zeros([len(sequences), featureSize])
	currentIndex = 0
	for key in featureDictionnary:
		featurized_sequence[:,currentIndex:currentIndex+featureDictionnary[key].shape[1]] = featureDictionnary[key]
		currentIndex += featureDictionnary[key].shape[1]

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
nn_option = False
lr_option = False
ensemble_option = False
if ('-ridge' in args):
	ridge_option = True
elif('-svm' in args):
	ridge_option = False
	svm_option = True
elif('-rf' in args):
	ridge_option = False
	rf_option = True
elif('-nn' in args):
	ridge_option = False
	nn_option = True
elif('-lr' in args):
	ridge_option = False
	lr_option = True
elif('-ensemble' in args):
	ensemble_option = True
	ridge_option = False
	maxMode = False
	meanMode = True
	if ('-max' in args):
		meanMode = False
		maxMode = True

train = False
if('-train' in args) or ('-t' in args):
	train = True

nFolds = 2
foldString = '-nFold'
if (foldString in args):
	nFolds = int(args[args.index(foldString)+1])

testMode = False
if ('-test' in args):
	testMode = True


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

X_train_all = featurize(sequences,setOfAminoAcids,lengthArray, isForSVM= svm_option)

if not(nn_option):
	Y_train_all = np.array(tags).argmax(axis = 1)
else:
	Y_train_all = np.array(tags)
names = np.array(names)


if train and not(nn_option): 
	if ridge_option : 
		classifier = RidgeClassifier() 
		alpha_range = np.logspace(-6, 5, num=11, endpoint=False)
		model_params = {'alpha': alpha_range}

	elif svm_option : 
		classifier = svm.SVC()
		model_params = {'C': [1, 2, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}
		 
	elif rf_option : 
		classifier = RFC()
		estimators_range = np.int_(np.linspace(200, 1000, num=9, endpoint=True))
		min_samples_leaf = np.int_(np.linspace(1, 40, num=20, endpoint=True))
		max_features = np.linspace(0.1, 0.8, num=8, endpoint=True)
		model_params = {'n_jobs' : [-1], 'n_estimators': estimators_range, 'min_samples_leaf': min_samples_leaf, 'max_features' : max_features}
	elif lr_option:
		c_range = np.int_(np.linspace(1, 1000, num=200, endpoint=True))
		model_params = {'n_jobs' : [-1], 'C': c_range}
		classifier = LogisticRegression(C=1)

	classifier = grid_search(classifier, model_params, X_train_all, Y_train_all, seed)
	print('BEST PARAMS')
	print(classifier.best_params_)
	print('BEST SCORE')
	print(classifier.best_score_)

else:
	if ridge_option : 
		print('RIDGE CLASSIFIER')

		classifier = RidgeClassifier(alpha = 1.0) 
	elif svm_option : 
		print('SVM CLASSIFIER')
	
		classifier = svm.SVC(C = 10, gamma = 0.0001)
	elif rf_option : 
		print('RANDOM FOREST')

		classifier = RFC(max_features =  0.4, n_estimators =  750, min_samples_leaf =  1, n_jobs =  -1)
	elif lr_option:
		print('LOGISTIC REGRESSION')
		classifier = LogisticRegression(C=1)
	elif nn_option:

		x_train, x_test, y_train, y_test = train_test_split(X_train_all, Y_train_all, test_size=0.1, random_state=0)
		
		print(x_train[0])
		print(y_train[0])
		model = Sequential()
		model.add(Dense(units=128, input_dim=X_train_all.shape[1]))
		model.add(Activation('relu'))
		model.add(BatchNormalization())
		# model.add(Dense(units=64, input_dim=X_train_all.shape[1]))
		# model.add(Activation('relu'))
		# model.add(BatchNormalization())		
		model.add(Dense(units=4))
		model.add(Activation('softmax'))
		model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
		model.fit(x_train, y_train, epochs=50, batch_size=10, validation_data=(x_test, y_test))


if ensemble_option:
	X_svm_all = featurize(sequences,setOfAminoAcids,lengthArray, isForSVM =  True)


train_ratio = 0.9
confusion_matrices = np.zeros([nFolds, 4,4])
y_proba = np.zeros([nFolds, Y_train_all.shape[0] - int(np.floor(Y_train_all.shape[0]*train_ratio)), 4])
y_mistake = np.zeros([nFolds])
y_mistake_perClass = np.zeros([nFolds, 4])

if not(nn_option) and not(testMode):
	print('CROSS VALIDATING ...')
	for i in range(nFolds):
		if ensemble_option: 
			random_permutation = np.random.permutation(Y_train_all.shape[0])
			train_index = random_permutation[:int(np.floor(Y_train_all.shape[0]*train_ratio))]
			test_index = random_permutation[int(np.floor(Y_train_all.shape[0]*train_ratio)):]
			
			x_svm_train = X_svm_all[train_index]
			x_train = X_train_all[train_index]

			x_svm_test = X_svm_all[test_index]
			x_test = X_train_all[test_index]

			y_train = Y_train_all[train_index]
			y_test = Y_train_all[test_index]

			# x_train, x_test, y_train, y_test = train_test_split(X_train_all, Y_train_all, test_size=0.1, random_state=0)

			listOfClassiferNames = ["SVMs", "Random Forest" ,"Ridge Classifier", "Logistic Regression"]
			listOfClassifer = [svm.SVC(C = 10, gamma = 0.0001, probability = True), RFC(max_features =  0.4, n_estimators =  750, min_samples_leaf =  1, n_jobs =  -1), RidgeClassifier(alpha = 1.0) ,   LogisticRegression(C=1)]	
			listOfFeatures_train = [x_svm_train] + [x_train] * (len(listOfClassifer)-1)


			for j in range(len(listOfClassifer)):
				print("Model Being Trained : ", listOfClassiferNames[j])
				listOfClassifer[j].fit(listOfFeatures_train[j],y_train)	

			listOfFeatures_test = [x_svm_test] + [x_test] * (len(listOfClassifer) - 1)

			probabilities = np.zeros([len(listOfClassifer), x_test.shape[0], 4])
			for j in range(len(listOfClassifer)):
				print("Model Predicting : ", listOfClassiferNames[j])
				try:
					probabilities[j] = listOfClassifer[j].predict_proba(listOfFeatures_test[j])
				except:
					scores = np.exp(listOfClassifer[j].decision_function(listOfFeatures_test[j]))
					probabilities[j] = scores/(1+scores)

			coefList = [0.25,0.25,0.25,0.25]
			if meanMode :
				predictions = np.average(probabilities, axis = 0, weights = coefList).argmax(axis = 1)
			else:
				predictions = np.max(probabilities, axis = 0).argmax(axis = 1)

			dev_accuracy = sum((predictions == y_test).astype(int))/predictions.shape[0]
		else:

			temp_classifier = classifier
			x_train, x_test, y_train, y_test = train_test_split(X_train_all, Y_train_all, test_size=0.1)
			temp_classifier.fit(x_train,y_train)
			dev_accuracy =  temp_classifier.score(x_test, y_test)
			predictions = temp_classifier.predict(x_test)
			# if ridge_option:
			# 	print(temp_classifier.decision_function(x_test))
			try:
				y_proba[i] = classifier.predict_proba(x_test)
			except:
				scores = classifier.decision_function(x_test)
				y_proba[i] = scores/(1+scores)
			y_mistake[i] = np.mean(y_proba[i][y_proba[i].argmax(axis = 1) != y_test].max(axis = 1))
			testArray = np.array([np.mean(y_proba[i][y_test == j][y_proba[i][y_test == j].argmax(axis = 1) != j].max(axis = 1)) for j in range(4)])
			y_mistake_perClass[i] = testArray

		confusion_matrices[i] = confusion_matrix(y_test, predictions)
		print('Fold N°',str(i))
		print('SCORE : ', dev_accuracy)

	if not(ensemble_option): 

		y_full_mistake = np.mean(y_mistake)
		y_mistake_perClass = np.mean(y_mistake_perClass, axis = 0)

		print('Average False Confidence: ', y_full_mistake)
		print('Average False Confidence Per Class: ', y_mistake_perClass)

	final_confusion_matrix = np.mean(confusion_matrices,axis = 0)
	text = 'Ridge Model'
	textFile = 'Ridge_cc'

	if svm_option:
		textFile = 'SVM_cc'
		text = 'SVM Model'
	elif rf_option: 
		textFile = 'RF_cc'
		text = 'Random Forest Model'
	elif lr_option:
		textFile = 'LR_cc'
		text = 'Logistic Regression Model'
	elif ensemble_option:
		textFile = 'ENSEMBLE_cc'
		if maxMode:
			text = 'Ensemble Model - Max'
		else:
			text = 'Ensemble Model - Mixture'

	text += ' Confusion Matrix'
	textFile += '.png'
	fig = plt.figure()


	plot_confusion_matrix(final_confusion_matrix,dataNames, title= text)
	fig.savefig(textFile)
elif testMode:
	x_train = X_train_all
	y_train = Y_train_all
	print('TEST FEATURIZATION')
	test_names, test_sequences, test_tags, test_lengthArray, test_setOfAminoAcids  = loadFastaFiles(['blind'])
	x_test = featurize(test_sequences,setOfAminoAcids,test_lengthArray, isForSVM= svm_option)
	print('FITTING')
	classifier.fit(x_train,y_train)
	print('PREDICTION')
	try:
		y_pred = classifier.predict_proba(x_test)
	except:
		scores = classifier.decision_function(x_test)
		y_pred = scores/(1+scores)

	for i in range(y_pred.shape[0]):
		print(test_names[i] + ' ' + dataNames[y_pred[i].argmax()] + ' Confidence ' +  str(int(y_pred[i].max() * 100)) + '%')