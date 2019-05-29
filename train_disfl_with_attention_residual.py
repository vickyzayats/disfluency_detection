from __future__ import print_function
import os, re
import logging
import sys
import pdb
import socket
import traceback

config_file = sys.argv[1]
experiment_num = 0


forward_lm = None
backward_lm = None
experiment_num = 0
if len(sys.argv) > 2 and sys.argv[2].isdigit():
    experiment_num = int(sys.argv[2])
    
param_tuning = False
if len(sys.argv) > 3 and not sys.argv[2].isdigit():
    param_tuning = True
    param_to_tune = sys.argv[2]
    param_num = sys.argv[3]
    
if len(sys.argv) > 4:
    experiment_num = int(sys.argv[4])
                                
with open(config_file, 'r') as inf:
    exec(inf.read())

if param_tuning and param_to_tune in jobconfig and isinstance(jobconfig[param_to_tune], list):
    jobconfig[param_to_tune] = jobconfig[param_to_tune][int(param_num)]

n = experiment_num

if 'model' in jobconfig and jobconfig['model'] != None:
    model_to_import = 'from neuralnets.'+jobconfig['model']+' import BiLSTM'
    exec(model_to_import)
elif 'phone_feats' in jobconfig:
    from neuralnets.BiLSTM_withAttention_residual_top_attention import BiLSTM
else:
    from neuralnets.BiLSTM_withAttention_residual import BiLSTM
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

if 'text_only_preds' in jobconfig and jobconfig['text_only_preds']:
    datasetName += '_with_text_preds'
exppathName = datasetName + '/' + expdate
if len(jobconfig['expPath']) > 0:
    exppathName += '/' + jobconfig['expPath']

'''
dataColumns = {0:'tokens', 1:'POS', 2:'feat1', 3:'feat2', 4:'feat3', 5:'feat4',
              6:'feat5', 7:'feat6', 8:'feat7', 9:'feat8', 10:'feat9',
              11:'feat10', 12:'feat11', 13:'feat12', 14:'feat13', 15:'feat14',
              16:'feat15', 17:'feat16', 18:'feat17', 19:'feat18', 25:'TAG',
              20:'tokenname', 21:'filename'}
'''
    
dataColumns = {0:'tokens', 1:'POS', 2:'feat1', 3:'feat2', 4:'feat3', 5:'feat4',
              6:'feat5', 7:'feat6', 8:'feat7', 9:'feat8', 10:'feat9',
              11:'feat10', 12:'feat11', 13:'feat12', 14:'feat13', 15:'feat14',
              16:'feat15', 19:'tokenname', 20:'filename', 17:'TAG'}#, 23:'TAG'}
if 'text_only_preds' in jobconfig and jobconfig['text_only_preds']:
    dataColumns[24] = 'text_only_TAG'
if 'tree' in jobconfig['datasetName']:
    dataColumns = {0:'tokens', 1:'POS', 2:'feat1', 3:'feat2',
                   4:'feat3', 5:'feat4',
                   6:'feat5', 7:'feat6', 8:'feat7', 9:'feat8', 10:'feat9',
                   11:'feat10', 12:'feat11', 13:'feat12', 14:'feat13',
                   15:'feat14',
                   16:'feat15', 17:'tokenname', 18:'filename', 20:'TAG'}
    if 'text_only_preds' in jobconfig and jobconfig['text_only_preds']:
        dataColumns[21] = 'text_only_TAG'
    
labelKey = 'TAG'

embeddingsPath = '../levy_deps.words' #Word embeddings by Levy et al: https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/

#Parameters of the network
params = {'dropout': [0.5, 0.5], 'classifier': outlayer, 'LSTM-Size': [150],
          'optimizer': 'adam', 'charEmbeddings': None, 'miniBatchSize': 32,
          'addFeatureDimensions': 1, 'forward_lm': forward_lm, 'backward_lm': backward_lm}
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
pickleFile = prepareDataset(embeddingsPath, datasetFiles,
                            prosody=params['prosody'],
                            prosody_feats=params['prosody_feats'])

#Load the embeddings and the dataset
embeddings, word2Idx, datasets = loadDatasetPickle(pickleFile)
data = datasets[datasetName]

print('Training, experiment %d' % n)
model = BiLSTM(params)
model.setMappings(embeddings, data['mappings'])
if params['lm']:
    model.setLMMappings(lm_f_embeddings, lm_f_word2Idx)
    model.setLMMappings(embeddings, lm_b_word2Idx, forward=False)
if params['prosody']:
    model.prosody = True
    params['feats_to_include']#.extend(params['prosody_feats'])
model.additionalFeatures.extend(params['feats_to_include'])
if 'phone_feats' in params:
    model.additionalDenseFeatures.extend(params['phone_feats'])
model.additionalDenseFeatures.extend(params['prosody_feats'])
model.setTrainDataset(data, labelKey, extend_labels=labels)
model.verboseBuild = True
model.train_embeddings = train_embeddings
model.modelSavePath = "%s/models/%s/%s/[DevScore]_[TestScore]_[Epoch]_[Batch].h5" % (basedirName, exppathName, modelName) #Enable this line to save the model to the disk
if params['pretrain_text']:
    model.loadWeights(params['pretrained_text_only_model'], by_name=True)
max_dev, max_test = model.evaluate(50)

with open(os.path.join('../' + exppathName, modelName + '.results'), 'a') as outf:
    outf.write('%.6f, %.6f\n' % (max_dev, max_test))

#remove all except the best file
outpath = '%s/models/%s/%s' % (basedirName, exppathName, modelName)
all_results_files = os.listdir(outpath)
for f in all_results_files[:-1]:
        os.remove(os.path.join(outpath, f))
