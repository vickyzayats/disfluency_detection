#!/g/tial/sw/pkgs/as70-amd64/python2.7.9/bin/python
from __future__ import print_function
import os, re
import logging
import sys
import pdb
import socket
import traceback

config_file = sys.argv[1]
experiment_num = 0
if len(sys.argv) > 2 and sys.argv[2].isdigit():
    experiment_num = int(sys.argv[2])

param_tuning = False
if len(sys.argv) > 3 and not sys.argv[2].isdigit():
    param_tuning = True
    param_to_tune = sys.argv[2]
    param_num = sys.argv[3]

if len(sys.argv) > 4:
    load_model_path = sys.argv[4]

with open(config_file, 'r') as inf:
    exec(inf.read())

if param_tuning and param_to_tune in jobconfig and isinstance(jobconfig[param_to_tune], list):
    jobconfig[param_to_tune] = jobconfig[param_to_tune][int(param_num)]

n = experiment_num
if 'gpu' in jobconfig:    
    gpu = (experiment_num / int(jobconfig['num_concurrent_jobs'])) % 2
if len(sys.argv) > 3:
    gpu = int(sys.argv[3])
import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu%d' % gpu)

from neuralnets.BiLSTM_withAttention import BiLSTM
from util.preprocessing import prepareDataset, loadDatasetPickle

# :: Change into the working dir of the script ::
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

datasetName = jobconfig['datasetName']
basedirName = jobconfig['basedirName']
modelName = jobconfig['modelName']
expdate = jobconfig['expdate']
labels = jobconfig['labels']
outlayer = jobconfig['outlayer']
train_embeddings = jobconfig['train_embeddings']

if modelName is None:
    modelName = os.path.basename(config_file.replace('.py',''))

exppathName = datasetName + '/' + expdate
if len(jobconfig['expPath']) > 0:
    exppathName += '/' + jobconfig['expPath']

dataColumns = {0:'tokens', 1:'POS', 2:'feat1', 3:'feat2', 4:'feat3', 5:'feat4',
              6:'feat5', 7:'feat6', 8:'feat7', 9:'feat8', 10:'feat9',
              11:'feat10', 12:'feat11', 13:'feat12', 14:'feat13', 15:'feat14',
              16:'feat15', 17:'feat16', 18:'feat17', 19:'feat18', 20:'TAG'}
labelKey = 'TAG'

embeddingsPath = 'levy_deps.words' #Word embeddings by Levy et al: https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/
#embeddingsPath = 'lm_vocab_100_jjjj.txt'

#Parameters of the network
params = {'dropout': [0.25, 0.25], 'classifier': outlayer, 'LSTM-Size': [150],
          'optimizer': 'adam', 'charEmbeddings': None, 'miniBatchSize': 32,
          'addFeatureDimensions': 1}
params.update(jobconfig.copy())

frequencyThresholdUnknownTokens = 50 #If a token that is not in the pre-trained embeddings file appears at least 50 times in the train.txt, then a new embedding is generated for this word

print('Running job %d' % n)
baseModelName = modelName
if param_tuning:
    modelName = '%s/%s_%s_%d' % (modelName, modelName, param_to_tune + param_num, n)
else:
    modelName = '%s/%s_%d' % (modelName, modelName, n)
logfile = "%s/logs/%s/%s.log" % (basedirName, exppathName, modelName)
if not os.path.exists("%s/logs/%s" % (basedirName, exppathName)):
    os.makedirs("%s/logs/%s" % (basedirName, exppathName))
if not os.path.exists("%s/logs/%s/%s" % (basedirName, exppathName, baseModelName)):
    os.makedirs("%s/logs/%s/%s" % (basedirName, exppathName, baseModelName))
    
hostfile = "%s/%s.host" % ('../' + exppathName, modelName)
if not os.path.exists('../' + exppathName + '/' + baseModelName):
    os.makedirs('../' + exppathName + '/' + baseModelName)
with open(hostfile, 'w') as outf:
    outf.write(socket.gethostname())

# :: Logging level ::
loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)

ch = logging.FileHandler(logfile)
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# Data preprocessing
datasetFiles = [
    (datasetName, dataColumns),
]
pickleFile = prepareDataset(embeddingsPath, datasetFiles)

#Load the embeddings and the dataset
embeddings, word2Idx, datasets = loadDatasetPickle(pickleFile)
data = datasets[datasetName]

print('Getting predictions')
model = BiLSTM(params)
model.setMappings(embeddings, data['mappings'])
model.additionalFeatures.extend(jobconfig['feats_to_include'])
model.setTrainDataset(data, labelKey, extend_labels=labels)
model.verboseBuild = True
model.train_embeddings = train_embeddings
load_model_path = '../trained_models/best_models/02_22_18#attention_multi_sim_conv#attention_multi_sim_conv_num_filters0_16/0.8795_0.8780_4_500.h5'
model.loadWeights(load_model_path)
output_file = 'output'
for m in ['train','dev','test']:
    pdb.set_trace()
    labels = model.predictLabels(model.dataset[m + 'Matrix'])
    with open(output_file + '_' + m + '.dat', 'w') as outf:
        for lab, feat in zip(labels, model.dataset[m + 'Matrix']):
            tokens = feat['raw_tokens']
            tags = feat['TAG_full_with_s']
            for i,l in enumerate(lab):
                out.write(tokens[i] + ' ' + tags[i] + ' ' + l + '\n')
            outf.write('\n')
pdb.set_trace()
