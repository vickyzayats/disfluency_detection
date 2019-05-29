#!/g/tial/sw/pkgs/as70-amd64/python2.7.9/bin/python
from __future__ import print_function
import os, re
import logging
import sys
import pdb
import socket
import traceback
import cPickle

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

exppathName = datasetName + '/' + expdate
if len(jobconfig['expPath']) > 0:
    exppathName += '/' + jobconfig['expPath']

dataColumns = {0:'tokens', 1:'POS', 2:'feat1', 3:'feat2', 4:'feat3', 5:'feat4',
              6:'feat5', 7:'feat6', 8:'feat7', 9:'feat8', 10:'feat9',
              11:'feat10', 12:'feat11', 13:'feat12', 14:'feat13', 15:'feat14',
              16:'feat15', 17:'feat16', 18:'feat17', 19:'feat18', 20:'TAG'}
labelKey = 'TAG'

embeddingsPath = 'levy_deps.words' #Word embeddings by Levy et al: https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/

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
if 'phone_feats' in params:
        model.additionalDenseFeatures.extend(params['phone_feats'])
if params['prosody']:
    model.prosody = True
    model.additionalDenseFeatures.extend(params['prosody_feats'])
model.additionalFeatures.extend(jobconfig['feats_to_include'])
model.setTrainDataset(data, labelKey, extend_labels=labels)
model.verboseBuild = True
model.train_embeddings = train_embeddings
model.loadWeights(load_model_path)
output_file = 'output'
nname = datasets.keys()
mappings = datasets[nname[0]]['mappings']
mappings_pos_inv = {mappings['POS'][k]:k for k in mappings['POS']}
mappings_tag_inv = {mappings['TAG'][k]:k for k in mappings['TAG']}
for m in ['test']:
    sentences = []
    len_ii = 0
    splited = []
    sentences =  model.dataset[m + 'Matrix']
    labels = model.predictLabels(sentences, mode='prosody_late_fusion')
    true = [x['TAG_full'] for x in sentences]
    pdb.set_trace()
    outname = 'output.pickle'
    outname = re.sub('/0\..*','outputs_' + m + '.pickle', outname.replace('models','outputs'))
    outname = outname.replace('ms_word_level_total','treebank_full')
    with open(outname, 'wb') as outf:
        cPickle.dump([labels, true], outf)
    output_file = outname.replace('.pickle','.dat')
    with open(output_file + '_' + m + '.dat', 'w') as outf:
        for lab, feat in zip(labels, model.dataset[m + 'Matrix']):
            tokens = feat['raw_tokens']
	    tokenname = feat['tokenname']
	    filename = feat['filename']
            tags = ['O' if x in (0,2) else 'IE' for x in feat['TAG_full']]
            labs = ['O' if x in (0,2) else 'IE' for x in lab]
	    for i,l in enumerate(lab):
                outf.write(' '.join([tokens[i],tokenname[i], filename[i], str(l), tags[i], labs[i]]) + '\n')
            outf.write('\n')
