import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
import sys
import re
import os
from Bio import SeqIO
import kmr
######### PATH ##################
scriptPath = os.path.realpath(__file__)
rootPath = os.path.sep.join(scriptPath.split(os.path.sep)[:-1])
dataPath = rootPath + os.path.sep + 'data' +  os.path.sep
####################### EXTRACTING ARGUMENTS ##########################
args = sys.argv[1:] ## getting the arguments

N_gram = 3
if ('-ngram' in args):
	N_gram = int(args[args.index('-ngram') + 1])

sequenceType = 'all'
if ('-type' in args):
	sequenceType = args[args.index('-type')+1]

numberToDisplay = 10
if ('-display' in args):
	numberToDisplay = int(args[args.index('-display') + 1])

length = 0
if ('-length' in args):
	length = int(args[args.index('-length') + 1])

selection = False
if('-select' in args):
	selection = True
	coef = 10
	if('-coef' in args):
		coef = float(args[args.index('-coef') + 1])
####################### Loading the Data ################################


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

def computeTFIDF(sequenceList,N,numberToDisplay,length=0):
	print('building the N-GRAM sequence')
	if length == 0:
		corpus = [" ".join(["".join(sequence[i:i+N]) for i in range(len(sequence)-N)]) for sequence in sequenceList]
	else:
		if length >0:
			corpus = [" ".join(["".join(sequence[:length][i:i+N]) for i in range(len(sequence)-N)]) for sequence in sequenceList]
		else:
			corpus = [" ".join(["".join(sequence[length:][i:i+N]) for i in range(len(sequence)-N)]) for sequence in sequenceList]
	print('COMPUTING TF-IDF')
	vectorizer = TfidfVectorizer(min_df=1)
	X = vectorizer.fit_transform(corpus)
	idf = vectorizer.idf_
	sortedIndex = np.array(np.argsort(-idf))
	sequence_names = np.array(vectorizer.get_feature_names())[sortedIndex]
	idf = np.array(idf)[sortedIndex]
	#print(dict(zip(sequence_names[:numberToDisplay], idf[:numberToDisplay])))
	toReturn = [word.upper() for word in sequence_names[:numberToDisplay]]
	return toReturn

def select(listOfClasses,listOfSequences,tags, allSequenceList, coef=10):
	ClassDict = dict()
	fullList = []
	for i in range(len(listOfClasses)):
		print('CLASS : ', listOfClasses[i])
		ClassDict[listOfClasses[i]] = []
		tempList = []
		candidateSequences = listOfSequences[i]
		targetedSequences = allSequenceList[tags.argmax(axis = 1) == i]
		ReamainingSequences = allSequenceList[tags.argmax(axis = 1) != i]
		for seq in candidateSequences:
			alphaSum = np.sum([int(seq in target) for target in targetedSequences])
			alphaValue = alphaSum/targetedSequences.shape[0]
			betaSum = np.sum([int(seq in target) for target in ReamainingSequences])
			betaValue = betaSum/ReamainingSequences.shape[0]
			Bool = alphaValue>=betaValue*coef
			if Bool:
				ClassDict[listOfClasses[i]].append(seq)
				fullList.append(seq)

	return ClassDict, fullList
print('LOADING THE DATA SET')

dataNames = ['cyto', 'mito', 'nucleus', 'secreted']
names, sequences, tags, lengthArray, setOfAminoAcids  = loadFastaFiles(dataNames)


if selection :
	offeredSequences = []
	listOfSequences = []
	nameList = []
	for i in range(len(dataNames)):
		dataIndex = i
		name = dataNames[i]
		print('COMPUTING TF_IDF FOR : ', name)
		sequenceList = np.array(sequences)[np.array(tags).argmax(axis = 1) == dataIndex]
		offeredSequences.append(computeTFIDF(sequenceList,N_gram,numberToDisplay,length))
		listOfSequences.append(sequenceList)
		nameList.append(name)
	selectedPerClass, FullSelectedList = select(nameList,offeredSequences,np.array(tags), np.array(sequences), coef )
	print(selectedPerClass)
	print(FullSelectedList)
elif sequenceType == 'all':
	computeTFIDF(sequences,N_gram,numberToDisplay,length)
else:
	print('COMPUTING TF-IDF for : ', sequenceType)
	dataIndex = dataNames.index(sequenceType)
	sequenceList = np.array(sequences)[np.array(tags).argmax(axis = 1) == dataIndex]
	computeTFIDF(sequenceList,N_gram,numberToDisplay, length)
