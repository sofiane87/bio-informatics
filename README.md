# Bio Informatics Assignment :

## What is this repo ? 
This repository presents the work realised for a *Bio-Informatics Assignment*. 

## The assignment : 

The goal of this assignment is to provide an automated system that is able to classify proteins (Amino Acid sequences) into four classes each being a subcellular locations : *[Cytosolic, Secreted, Nuclear, Mitochondrial]*. The subcellular location is an important information as it provides more knowledge about the protein function, it is therefore, very helpful to be able to automatically extract that information for any given protein.

*Results :* Using a **Random Forest Classifer** we manage to reach a *67\% cross-validation accuracy*.

*Improvements:* In order to improve the results of the classifier, deepening the feature extraction method seems to be the way to go. Another method would be to use neural network techniques.


## The Code : 

The code contains three files :

*	**main.py : ** This file contains most of the work and all attempted machine learning techniques aside from RNNs
*	**temp_rnn.py : ** This file contains all attempts related to RNNs [both LSTM and GRU cells have been attempted]
*	**tf_idf.py : ** This file contains a preprocessing step where various interesting key words were identified using metrics such as **tf-idf** score