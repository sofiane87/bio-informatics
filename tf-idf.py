import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
import sys
import re
import os
from Bio import SeqIO
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

def computeTFIDF(sequenceList,N,numberToDisplay,length):
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
	print(dict(zip(sequence_names[:numberToDisplay], idf[:numberToDisplay])))
	print([word.upper() for word in sequence_names[:numberToDisplay]])
	return idf, X


print('LOADING THE DATA SET')

dataNames = ['cyto', 'mito', 'nucleus', 'secreted']
names, sequences, tags, lengthArray, setOfAminoAcids  = loadFastaFiles(dataNames)


print('COMPUTING TF-IDF for : ', sequenceType)
if sequenceType == 'all':
	computeTFIDF(sequences,N_gram,numberToDisplay,length)
else:
	dataIndex = dataNames.index(sequenceType)
	sequenceList = np.array(sequences)[np.array(tags).argmax(axis = 1) == dataIndex]
	computeTFIDF(sequenceList,N_gram,numberToDisplay)
