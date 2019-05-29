#!/usr/bin/python
#Usage: python RunModel.py modelPath inputPath"
from __future__ import print_function
import nltk
from util.preprocessing import addCharInformation, createMatrices, addCasingInformation
from neuralnets.BiLSTM_withAttention import BiLSTM
from util.preprocessing import prepareDataset, loadDatasetPickle
import sys
import numpy as np
import pdb
import cPickle as pickle



if len(sys.argv) < 2:
    print("Usage: python RunModel.py modelPath")
    exit()

modelPath = sys.argv[1]
datasetName = 'swbd_diff_lm'

# :: Load the model ::
lstmModel = BiLSTM()
lstmModel.loadModel(modelPath)
pdb.set_trace()

# :: Load the data
# :: Train / Dev / Test-Files ::
dataColumns = {0:'tokens', 1:'POS', 2:'feat1', 3:'feat2', 4:'feat3', 5:'feat4',
               6:'feat5', 7:'feat6', 8:'feat7', 9:'feat8', 10:'feat9',
               11:'feat10', 12:'feat11', 13:'feat12', 14:'feat13', 15:'feat14',
               16:'feat15', 17:'feat16', 18:'feat17', 19:'feat18', 20:'TAG'}
labelKey = 'TAG'

embeddingsPath = 'levy_deps.words' #Word embeddings by Levy et al: https://levyomer.wordpress.com/20\14/04/25/dependency-based-word-embeddings/

datasetFiles = [
    (datasetName, dataColumns),
    ]

    
# :: Prepares the dataset to be used with the LSTM-network. Creates and stores cPickle files in the pkl/ folder ::
pickleFile = prepareDataset(embeddingsPath, datasetFiles)

#Load the embeddings and the dataset
embeddings, word2Idx, datasets = loadDatasetPickle(pickleFile)
data = datasets[datasetName]

if 'with_s' in modelPath and 'full' in modelPath:
    labels = 'full_with_s'
elif 'with_s' in modelPath:
    labels = 'with_s'
elif 'full' in modelPath:
    labels = 'full'
else:
    labels = 'base'
    
lstmModel.setTrainDataset(data, labelKey, extend_labels=labels)
lstmModel.predictLabels(lstmModel.dataset['devMatrix'])
