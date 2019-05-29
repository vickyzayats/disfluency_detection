"""
A bidirectional LSTM with optional CRF and character-based presentation for NLP sequence tagging.

Author: Nils Reimers
License: CC BY-SA 3.0
"""

from __future__ import print_function
import keras
from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import *
import keras.backend as K

import os
import sys
sys.setrecursionlimit(10000)
import random
import time
import math
import numpy as np
import logging
import pdb

from .keraslayers.ChainCRF import ChainCRF
from .keraslayers.InnerAttention_keras import SingleMultiModalAttention, FlattenCNN, SelfAttention
import util.BIOF1Validation as BIOF1Validation


import sys
if (sys.version_info > (3, 0)):
    import pickle as pkl
else: #Python 2.7 imports
    import cPickle as pkl

class BiLSTM:
    additionalFeatures = []
    additionalDenseFeatures = []
    learning_rate_updates = {'sgd': {1: 0.1, 3:0.05, 5:0.01} } 
    verboseBuild = True
    prosody = False

    model = None 
    epoch = 0 
    skipOneTokenSentences=True
    
    dataset = None
    embeddings = None
    labelKey = None
    writeOutput = False    
    devAndTestEqual = False
    resultsOut = None
    modelSavePath = None
    maxCharLen = None
    train_embeddings = False
    
    params = {'miniBatchSize': 32, 'dropout': [0.25, 0.25], 'classifier': 'Softmax', 'LSTM-Size': [100], 'optimizer': 'nadam', 'earlyStopping': 5, 'addFeatureDimensions': 10, 'posDimensions': 10, 
                'charEmbeddings': None, 'charEmbeddingsSize':30, 'charFilterSize': 30,
                'charFilterLength':3, 'charLSTMSize': 25, 'clipvalue': 0, 'clipnorm': 1,
                'attention': None , 'attention_norm': 'uniform', 'num_filters': 12,
                'window_size': 10, 'text_prosody_loss': 'mean_squared_error', 'num_heads': 8}
        #Default params
   

    def __init__(self,   params=None):        
        if params != None:
            self.params.update(params)
        
        logging.info("BiLSTM model initialized with parameters: %s" % str(self.params))
        
    def setMappings(self, embeddings, mappings):
        self.mappings = mappings
        self.embeddings = embeddings
        self.idx2Word = {v: k for k, v in self.mappings['tokens'].items()}

    def setLMMappings(self, embeddings, mappings, forward=True):
        lm2Tokens = {}
        for k, v in mappings.items():
            if k in self.mappings['tokens']:
                lm2Tokens[v] = self.mappings['tokens'][k]
        tokens2Lm = {v: k for k, v in lm2Tokens.items()}
        if forward:
            self.lm_f_mappings = mappings
            self.lm_f_embeddings = embeddings
            self.lm_f_lm2Tokens = lm2Tokens
            self.lm_f_tokens2Lm = tokens2Lm
        else:
            self.lm_b_mappings = mappings
            self.lm_b_embeddings = embeddings
            self.lm_b_lm2Tokens = lm2Tokens
            self.lm_b_tokens2Lm = tokens2Lm

    def removeNonProsodyFiles(self, data, mappings, remove_ac=False):
        i = 0
        while i < len(data):
            filename = data[i]['filename'][0]
            stat = False
            if remove_ac:
                #stat = mappings['prosody_files'][filename] == 0 or len(data[i]['pause_before']) == 0
                stat = len(data[i]['pause_before']) == 0
            if stat:
                data.pop(i)
            else:
                i += 1
                
        
    def setTrainDataset(self, dataset, labelKey, extend_labels='base'):
        # filter data that does not contain prosody features
        remove_ac = False
        if 'valid_prosody_only' in self.params and self.params['valid_prosody_only']:
            print('Removing instances without prosodic information')
            remove_ac = True
        if self.prosody:
            self.removeNonProsodyFiles(dataset['trainMatrix'], dataset['mappings'], remove_ac=True)
        elif remove_ac:
            self.removeNonProsodyFiles(dataset['trainMatrix'], dataset['mappings'], remove_ac=remove_ac)
        if remove_ac:
            self.removeNonProsodyFiles(dataset['devMatrix'], dataset['mappings'], remove_ac=remove_ac)
            self.removeNonProsodyFiles(dataset['testMatrix'], dataset['mappings'], remove_ac=remove_ac)

        self.dataset = dataset
        self.label2Idx = self.mappings[labelKey]
        self.labelKey = '_'.join([labelKey, extend_labels])
        self.idx2Label = {v: k for k, v in self.label2Idx.items()}

        with_s, full = False, False
        if 'with_s' in extend_labels:
            with_s = True
        if 'full' in extend_labels:
            full = True

        labels_map = self.idx2Label.copy()
        new_tags = []
        for key in sorted(labels_map.keys()):
            new_tag = labels_map[key]
            if not with_s:
                new_tag = new_tag.replace('_s','')
            if not full:
                if new_tag in ('C', 'C_s'):
                    new_tag = 'O'
                else:
                    new_tag = new_tag.replace('C_s_','').replace('C_','')
            if not new_tag in new_tags:
                new_tags.append(new_tag)
            labels_map[key] = new_tags.index(new_tag)

        # iterate through the data to updated the labels
        for data in [self.dataset['trainMatrix'], self.dataset['devMatrix'], self.dataset['testMatrix']]:       
            # change old to updated labels according to labels_map 
            for sentenceIdx in range(len(data)):
                data[sentenceIdx][self.labelKey] = []
                for tagIdx in range(len(data[sentenceIdx][labelKey])):
                    tag = data[sentenceIdx][labelKey][tagIdx]
                    data[sentenceIdx][self.labelKey].append(labels_map[tag])

        # update mappings
        self.mappings[self.labelKey] = {k: v for v, k in enumerate(new_tags)}
        self.mappings['label'] = self.mappings[self.labelKey]
        self.label2Idx = self.mappings[self.labelKey]
        self.idx2Label = {k: v for k, v in enumerate(new_tags)}
        
    def padCharacters(self, field):
        """ Pads the character representations of the words to the longest word in the dataset """
        #Find the longest word in the dataset
        maxCharLen = 0
        for data in [self.dataset['trainMatrix'], self.dataset['devMatrix'], self.dataset['testMatrix']]:            
            for sentence in data:
                for token in sentence[field]:
                    maxCharLen = max(maxCharLen, len(token))
                    
        for data in [self.dataset['trainMatrix'], self.dataset['devMatrix'], self.dataset['testMatrix']]:       
            #Pad each other word with zeros
            for sentenceIdx in range(len(data)):
                for tokenIdx in range(len(data[sentenceIdx][field])):
                    if field == 'conv_phone_feats':
                        pdb.set_trace()
                    token = data[sentenceIdx][field][tokenIdx]
                    data[sentenceIdx][field][tokenIdx] = np.pad(token, (0,maxCharLen-len(token)), 'constant')

        if field == 'characters':
            self.maxCharLen = maxCharLen
        elif field == 'phones':
            self.maxPhoneLen = maxCharLen
        
    def trainModel(self, iterator=None, max_num_batches=None, epoch=None):
        if self.model == None:
            self.buildModel()        
        if iterator is None:
            if self.params['optimizer'] in self.learning_rate_updates and self.epoch in self.learning_rate_updates[self.params['optimizer']]:
                K.set_value(self.model.optimizer.lr, self.learning_rate_updates[self.params['optimizer']][self.epoch])          
                logging.info("Update Learning Rate to %f" % (K.get_value(self.model.optimizer.lr)))

            iterator = self.online_iterate_dataset(trainMatrix, self.labelKey) if self.params['miniBatchSize'] == 1 else self.batch_iterate_dataset(trainMatrix, self.labelKey)

        l = len(self.additionalDenseFeatures)
        i = 0
        for batch in iterator:
            disfl_labels = batch[0]
            labels = [disfl_labels]*len(self.model.outputs)
            prosody_labels = []
            mask = 1 - np.sum(disfl_labels, axis=1).flatten().astype(bool)
            mask_idx = [ii for ii in range(len(mask)) if mask[ii] > 0]
            
            nnInput = batch[1:]
            for feat_num in range(l):
                if 'pause' in self.additionalDenseFeatures[feat_num][0]:
                    nnInput[-l+feat_num][:,:,0] = np.log(np.minimum(nnInput[-l+feat_num][:,:,0] + 1, np.exp(1)))
                    
                feat_log = nnInput[-l+feat_num]
                prosody_labels.append(feat_log)

            if epoch > -1:
                try:
                    err = self.model.train_on_batch(nnInput, labels)
                    if np.isnan(err).sum():
                        logging.info('The error is Nan')
                        pdb.set_trace()
                except:
                    logging.info('Cannot calculate error')
                    pdb.set_trace()
            if len(self.params['phone_feats']) + len(self.params['prosody_feats']) > 0 and \
              (not 'no_diff_feats' in self.params or ('no_diff_feats' in self.params and \
                                                  not self.params['no_diff_feats'])) and \
                                                  self.params['prosody'] and sum(mask) > 0:
                prosodyInput = [x[mask_idx] for x in nnInput[:-len(self.additionalDenseFeatures)]]
                prosody_labels = [x[mask_idx] for x in prosody_labels]
                prosody_err = self.prosody_model.train_on_batch(prosodyInput, prosody_labels)
                if len(prosody_err) > 1:
                    prosody_err = prosody_err[0]
                if i % 100 == 0:
                    logging.info('Cost for i=%d, prosody: %.4f' % (i, prosody_err))
            i += 1
            if max_num_batches and i == max_num_batches:
                return False
        return True
            
    def predictLabels(self, sentences, mode=''):
        if self.model == None:
            self.buildModel()

        predLabels = [None]*len(sentences)

        if 'text_only_preds' in self.params and self.params['text_only_preds'] \
          and 'text_only_TAG' in self.mappings:
            text_only_id2tag = {v: k for k, v in self.mappings['text_only_TAG'].items()}
        sentenceLengths = self.getSentenceLengths(sentences)
        
        for senLength, indices in sentenceLengths.items():        
            
            if self.skipOneTokenSentences and senLength == 1:
                if 'O' in self.label2Idx:
                    dummyLabel = self.label2Idx['O']
                else:
                    dummyLabel = 0
                predictions = [[dummyLabel]] * len(indices) #Tag with dummy label
            else:
                features = ['tokens', 'casing']
                features += self.additionalFeatures
                if self.prosody:
                    if 'phone_feats' in self.params and len(self.params['phone_feats']) > 0 and \
                      (not 'no_diff_feats' in self.params or ('no_diff_feats' in self.params and \
                                    not self.params['no_diff_feats'])):
                        features += ['phones','stress']
                    features += [x[0] for x in self.additionalDenseFeatures]
                if self.params['lm']:
                    features += ['lm_f_tokens','lm_f_casing','lm_b_tokens','lm_b_casing']

                inputData = {name: [] for name in features}              
                def convert_vocab(tokens, forward=True):
                    if forward:
                        tokens2Lm = self.lm_f_tokens2Lm
                    else:
                        tokens2Lm = self.lm_b_tokens2Lm
                    out = []
                    for t in tokens:
                        if t in tokens2Lm:
                            out.append(tokens2Lm[t])
                        else:
                            out.append(tokens2Lm[self.mappings['tokens']['UNKNOWN_TOKEN']])
                    return out

                for i,idx in enumerate(indices):
                    if 'debug' in self.params and self.params['debug']:
                        k = True
                    else:
                        k = self.prosody and len(sentences[idx]['pause_before']) == 0
                    if k:
                        if 'text_only_preds' in self.params and self.params['text_only_preds'] \
                          and 'text_only_TAG' in self.mappings:
                            text_only_tags = [text_only_id2tag[x] for x in sentences[idx]['text_only_TAG']]
                            predLabels[idx] = [self.label2Idx[x] for x in text_only_tags]
                        continue
                    for name in features:
                        if name in ('lm_f_casing','lm_b_casing'):
                            inputData[name].append(sentences[idx]['casing'])
                        elif name == 'lm_f_tokens':
                            inputData[name].append(convert_vocab(sentences[idx]['tokens']))
                        elif name == 'lm_b_tokens':
                            inputData[name].append(convert_vocab(sentences[idx]['tokens'], False))
                        elif name == 'phone_durations':
                            inputData[name].append([np.expand_dims(x, -1) for x in sentences[idx][name]])
                        elif name in ['total_phone_durations','pause_before','pause_after', 'conv_phone_feats']:
                            feat_val = sentences[idx][name]
                            splited = [ii+1 for ii in range(len(sentences[idx]['tokenname'])-1) if \
                                       sentences[idx]['tokenname'][ii].endswith('_a') and \
                                       sentences[idx]['tokenname'][ii+1].endswith('_b')]
                            pro_mask = np.ones_like(sentences[idx][name])
                            pro_mask[splited] = 0
                            inputData[name].append(np.concatenate([np.asarray(sentences[idx][name]),
                                                                   pro_mask], axis=-1))
                        else:
                            inputData[name].append(sentences[idx][name])
                            
                for name in features:
                    inputData[name] = np.asarray(inputData[name])
                                 
                if len(inputData['tokens']) > 0:
                    if mode and mode=='debug':
                        predictions = self.debug_model.predict([inputData[name] \
                                    for name in features], verbose=False)
                    elif mode and mode=='mse':
                        predictions = self.prosody_model.predict([inputData[name] \
                                    for name in features[:-len(self.additionalDenseFeatures)]],
                                    verbose=False)
                        if len(self.additionalDenseFeatures) > 1:
                            predictions = predictions[1]
                    else:
                        predictions = self.model.predict([inputData[name] for name in features], verbose=False)
                        if isinstance(predictions, list):
                            if mode == 'prosody_late_fusion':
                                predictions = predictions[-1]
                            else:
                                predictions = predictions[0]
                        predictions = predictions.argmax(axis=-1) #Predict classes
                        
                     #f = K.function([self.attention_layers[0].input],[self.attention_layers[0].output])

            predIdx = 0
            for idx in indices:
                if predLabels[idx] is None:
                    try:
                        predLabels[idx] = predictions[predIdx]
                    except:
                        pdb.set_trace()
                    predIdx += 1   

        return predLabels

    def predictProbs(self, sentences):
        if self.model == None:
            self.buildModel()
            
        predProbs = [None]*len(sentences)
        
        sentenceLengths = self.getSentenceLengths(sentences)
        
        for senLength, indices in sentenceLengths.items():        
            
            if self.skipOneTokenSentences and senLength == 1:
                if 'O' in self.label2Idx:
                    dummyLabel = self.label2Idx['O']
                else:
                    dummyLabel = 0
                predictions = [[dummyLabel]] * len(indices) #Tag with dummy label
            else:          
                
                features = ['tokens', 'casing']
                features += self.additionalFeatures
                if self.prosody:
                    if 'phone_feats' in self.params and len(self.params['phone_feats']) > 0 and \
                      (not 'no_diff_feats' in self.params or ('no_diff_feats' in self.params and \
                                    not self.params['no_diff_feats'])):
                        features += ['phones','stress']
                    features += [x[0] for x in self.additionalDenseFeatures]
                if self.params['lm']:
                    features += ['lm_f_tokens','lm_f_casing','lm_b_tokens','lm_b_casing']
                inputData = {name: [] for name in features}              
                def convert_vocab(tokens, forward=True):
                    if forward:
                        tokens2Lm = self.lm_f_tokens2Lm
                    else:
                        tokens2Lm = self.lm_b_tokens2Lm
                    out = []
                    for t in tokens:
                        if t in tokens2Lm:
                            out.append(tokens2Lm[t])
                        else:
                            out.append(tokens2Lm[self.mappings['tokens']['UNKNOWN_TOKEN']])
                    return out

                for idx in indices:                    
                    if self.prosody and len(sentences[idx]['pause_before']) == 0:
                        continue
                    for name in features:
                        if name in ('lm_f_casing','lm_b_casing'):
                            inputData[name].append(sentences[idx]['casing'])
                        elif name == 'lm_f_tokens':
                            inputData[name].append(convert_vocab(sentences[idx]['tokens']))
                        elif name == 'lm_b_tokens':
                            inputData[name].append(convert_vocab(sentences[idx]['tokens'], False))
                        elif name in ['total_phone_durations','pause_before','pause_after','conv_phone_feats']:
                            feat_val = sentences[idx][name]
                            splited = [ii+1 for ii in range(len(sentences[idx]['tokenname'])-1) if \
                                       sentences[idx]['tokenname'][ii].endswith('_a') and \
                                       sentences[idx]['tokenname'][ii+1].endswith('_b')]
                            pro_mask = np.ones_like(sentences[idx][name])
                            pro_mask[splited] = 0
                            inputData[name].append(np.concatenate([np.asarray(sentences[idx][name]),
                                                                   pro_mask], axis=-1))
                        else:
                            inputData[name].append(sentences[idx][name])                 
                                                    
                for name in features:
                    inputData[name] = np.asarray(inputData[name])
                    
                    
                predictions = self.model.predict([inputData[name] for name in features], verbose=False)[-1]
                
            
            predIdx = 0
            for idx in indices:
                predProbs[idx] = predictions[predIdx]    
                predIdx += 1   
        
        return predProbs

    
    # ------------ Some help functions to train on sentences -----------
    def online_iterate_dataset(self, dataset, labelKey): 
        idxRange = list(range(0, len(dataset)))
        random.shuffle(idxRange)
        
        for idx in idxRange:
                labels = []                
                features = ['tokens', 'casing']
                features += self.additionalFeatures
                if self.prosody:
                    if 'phone_feats' in self.params and len(self.params['phone_feats']) > 0:
                        features += ['phones','stress']
                    features += [x[0] for x in self.additionalDenseFeatures]
                if self.params['lm']:
                    features += ['lm_f_tokens','lm_f_casing','lm_b_tokens','lm_b_casing']
                def convert_vocab(tokens, forward=True):
                    if forward:
                        tokens2Lm = self.lm_f_tokens2Lm
                    else:
                        tokens2Lm = self.lm_b_tokens2Lm
                    out = []
                    for t in tokens:
                        if t in tokens2Lm:
                            out.append(tokens2Lm[t])
                        else:
                            out.append(tokens2Lm[self.mappings['tokens']['UNKNOWN_TOKEN']])
                    return out

                labels = dataset[idx][labelKey]
                labels = [labels]
                labels = np.expand_dims(labels, -1)  
                    
                inputData = {}              
                for name in features:
                    if self.prosody and len(dataset[idx]['pause_before']) == 0:
                        continue
                    if name in ('lm_f_casing','lm_b_casing'):
                        inputData[name] = np.asarray(dataset[idx]['casing'])
                    elif name == 'lm_f_tokens':
                        inputData[name] = np.asarray(convert_vocab(dataset[idx]['tokens']))
                    elif name == 'lm_b_tokens':
                        inputData[name] = np.asarray(convert_vocab(dataset[idx]['tokens'], False))
                    elif name in ['total_phone_durations','pause_before','pause_after','conv_phone_feats']:
                        feat_val = dataset[idx][name]
                        splited = [ii+1 for ii in range(len(dataset[idx]['tokenname'])-1) if \
                                   dataset[idx]['tokenname'][ii].endswith('_a') and \
                                   dataset[idx]['tokenname'][ii+1].endswith('_b')]
                        pro_mask = np.ones_like(dataset[idx][name])
                        pro_mask[splited] = 0
                        inputData[name] = np.concatenate([np.asarray(dataset[idx][name]),
                                                               pro_mask], axis=-1)
                    else:
                        inputData[name] = np.asarray([dataset[idx][name]])                 
                                    
                 
                yield [labels] + [inputData[name] for name in features] 
            
            
            
    def getSentenceLengths(self, sentences):
        sentenceLengths = {}
        for idx in range(len(sentences)):
            sentence = sentences[idx]['tokens']
            if len(sentence) not in sentenceLengths:
                sentenceLengths[len(sentence)] = []
            sentenceLengths[len(sentence)].append(idx)
        
        return sentenceLengths
            
    
    trainSentenceLengths = None
    trainSentenceLengthsKeys = None        
    def batch_iterate_dataset(self, dataset, labelKey):
        if self.trainSentenceLengths == None:
            self.trainSentenceLengths = self.getSentenceLengths(dataset)
            self.trainSentenceLengthsKeys = list(self.trainSentenceLengths.keys())
            
        trainSentenceLengths = self.trainSentenceLengths
        trainSentenceLengthsKeys = self.trainSentenceLengthsKeys
        random.shuffle(trainSentenceLengthsKeys)
        for senLength in trainSentenceLengthsKeys:
            if self.skipOneTokenSentences and senLength == 1: #Skip 1 token sentences
                continue
            sentenceIndices = trainSentenceLengths[senLength]
            random.shuffle(sentenceIndices)
            sentenceCount = len(sentenceIndices)
            
            
            bins = int(math.ceil(sentenceCount/float(self.params['miniBatchSize'])))
            binSize = int(math.ceil(sentenceCount / float(bins)))
           
            numTrainExamples = 0
            for binNr in range(bins):
                tmpIndices = sentenceIndices[binNr*binSize:(binNr+1)*binSize]
                numTrainExamples += len(tmpIndices)
                
                
                labels = []                
                features = ['tokens', 'casing']
                features += self.additionalFeatures
                if self.prosody:
                    if 'phone_feats' in self.params and len(self.params['phone_feats']) > 0 and \
                        (not 'no_diff_feats' in self.params or ('no_diff_feats' in self.params and \
                                    not self.params['no_diff_feats'])):
                        features += ['phones', 'stress']
                    features += [x[0] for x in self.additionalDenseFeatures]
                if self.params['lm']:
                    features += ['lm_f_tokens','lm_f_casing','lm_b_tokens','lm_b_casing']
                inputData = {name: [] for name in features}

                def convert_vocab(tokens, forward=True):
                    if forward:
                        tokens2Lm = self.lm_f_tokens2Lm
                    else:
                        tokens2Lm = self.lm_b_tokens2Lm
                    out = []
                    for t in tokens:
                        if t in tokens2Lm:
                            out.append(tokens2Lm[t])
                        else:
                            out.append(tokens2Lm[self.mappings['tokens']['UNKNOWN_TOKEN']])
                    return out
                
                for idx in tmpIndices:
                    labels.append(dataset[idx][labelKey])
                    if self.prosody and len(dataset[idx]['pause_before']) == 0:
                        pdb.set_trace()
                        continue
                    for name in features:
                        if name in ('lm_f_casing','lm_b_casing'):
                            inputData[name].append(dataset[idx]['casing'])
                        elif name == 'lm_f_tokens':
                            inputData[name].append(convert_vocab(dataset[idx]['tokens']))
                        elif name == 'lm_b_tokens':
                            inputData[name].append(convert_vocab(dataset[idx]['tokens'], False))
                        elif name == 'phone_durations':
                            inputData[name].append([np.expand_dims(x, -1) for x in dataset[idx][name]])
                        elif name in ['total_phone_durations','pause_before','pause_after','conv_phone_feats']:
                            feat_val = dataset[idx][name]
                            splited = [ii+1 for ii in range(len(dataset[idx]['tokenname'])-1) if \
                                       dataset[idx]['tokenname'][ii].endswith('_a') and \
                                       dataset[idx]['tokenname'][ii+1].endswith('_b')]
                            pro_mask = np.ones_like(dataset[idx][name])
                            pro_mask[splited] = 0
                            inputData[name].append(np.concatenate([np.asarray(dataset[idx][name]),
                                                                   pro_mask], axis=-1))
                        else:
                            inputData[name].append(dataset[idx][name])                 
                            
                                    
                labels = np.asarray(labels)
                labels = np.expand_dims(labels, -1)

                for name in features:
                    inputData[name] = np.asarray(inputData[name])
                    
                yield [labels] + [inputData[name] for name in features]   
                
            assert(numTrainExamples == sentenceCount) #Check that no sentence was missed 
            
          
        
    
    def buildModel(self):        
        params = self.params  
        
        if self.params['charEmbeddings'] not in [None, "None", "none", False, "False", "false"]:
            self.padCharacters('characters')

        if 'phone_feats' in self.params and len(self.params['phone_feats']) > 0:
            self.padCharacters('phones')
            self.padCharacters('stress')
            
        embeddings = self.embeddings
        casing2Idx = self.dataset['mappings']['casing']
        
        caseMatrix = np.identity(len(casing2Idx), dtype='float32')

        inputs = []

        tokens = Input(shape=(None,), name='input_tokens')
        inputs.append(tokens)
        tokens = Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1],  weights=[embeddings], trainable=self.train_embeddings, name='token_emd')(tokens)
        
        casing = Input(shape=(None,), name='input_casing')
        inputs.append(casing)
        casing = Embedding(input_dim=caseMatrix.shape[0], output_dim=caseMatrix.shape[1], weights=[caseMatrix], trainable=False, name='casing_emd')(casing)
        
        mergeLayers = [tokens, casing]
            
        if self.additionalFeatures != None:
            for addFeature in self.additionalFeatures:
                try:
                    maxAddFeatureValue = max([max(sentence[addFeature]) for sentence in self.dataset['trainMatrix']+self.dataset['devMatrix']+self.dataset['testMatrix']])
                except:
                    pdb.set_trace()
                addFeatureEmd = Input(shape=(None,), name='input_'+addFeature)
                inputs.append(addFeatureEmd)
                if addFeature.lower() in ('pos',): #'pause_before', 'pause_after'):
                    addFeatureEmd = Embedding(input_dim=maxAddFeatureValue+1, output_dim=self.params['posDimensions'], trainable=True, name=addFeature+'_emd')(addFeatureEmd)
                else:
                    addFeatureEmd = Embedding(input_dim=maxAddFeatureValue+1, output_dim=self.params['addFeatureDimensions'], trainable=True, name=addFeature+'_emd')(addFeatureEmd)
                    
                mergeLayers.append(addFeatureEmd)
                

        text_model = concatenate(mergeLayers) 

        # Add LSTMs
        cnt = 1
        for size in params['LSTM-Size']:
            if isinstance(params['dropout'], (list, tuple)):
                text_model = Bidirectional(LSTM(size, return_sequences=True, dropout_W=params['dropout'][0], dropout_U=params['dropout'][1]), name="varLSTM_"+str(cnt))(text_model)
            
            else:
                """ Naive dropout """
                text_model = Bidirectional(LSTM(size, return_sequences=True), name="LSTM_"+str(cnt))(text_model) 
                
                if params['dropout'] > 0.0:
                    text_model = TimeDistributed(Dropout(params['dropout']), name="dropout_"+str(cnt))(text_model)
            
            cnt += 1

        outputs = []
        prosody_outputs = []
        losses = []
        prosody_losses = []

        prosodyLayers = []#[text_model]
        # :: Phone Embeddings ::
        if 'phone_feats' in self.params and 'no_diff_feats' in self.params and self.params['no_diff_feats']:
            for addFeature, feat_dim in self.params['phone_feats']:
                input_shape = (None,feat_dim*2)
                addFeatureDense = Input(shape=input_shape, name='input_' + addFeature)
                inputs.append(addFeatureDense)
                prosodyLayers.append(addFeatureDense)
        
        elif 'phone_feats' in self.params and len(self.params['phone_feats']) > 0 and params['prosody']:
            phoneset = self.dataset['mappings']['phones']
            phoneEmbeddingsSize = params['phoneEmbeddingsSize']
            stressEmbeddingsSize = params['stressEmbeddingsSize']
            maxPhoneLen = self.maxPhoneLen
            phoneEmbeddings= []

            for _ in phoneset:
                limit = math.sqrt(3.0/phoneEmbeddingsSize)
                vector = np.random.uniform(-limit, limit, phoneEmbeddingsSize) 
                phoneEmbeddings.append(vector)
                
            phoneEmbeddings[0] = np.zeros(phoneEmbeddingsSize) #Zero padding
            phoneEmbeddings = np.asarray(phoneEmbeddings)

            stressEmbeddings= []

            for _ in phoneset:
                limit = math.sqrt(3.0/stressEmbeddingsSize)
                vector = np.random.uniform(-limit, limit, stressEmbeddingsSize) 
                stressEmbeddings.append(vector)
                
            stressEmbeddings[0] = np.zeros(stressEmbeddingsSize) #Zero padding
            stressEmbeddings = np.asarray(stressEmbeddings)

            phones = Input(shape=(None,maxPhoneLen))
            stress = Input(shape=(None,maxPhoneLen))
            inputs.append(phones)
            inputs.append(stress)
            phones = TimeDistributed(Embedding(input_dim=phoneEmbeddings.shape[0], output_dim=phoneEmbeddings.shape[1],  weights=[phoneEmbeddings], trainable=True, mask_zero=True), input_shape=(None, maxPhoneLen), name='phone_emd')(phones)
            stress = TimeDistributed(Embedding(input_dim=stressEmbeddings.shape[0], output_dim=stressEmbeddings.shape[1],  weights=[stressEmbeddings], trainable=True, mask_zero=True), input_shape=(None, maxPhoneLen), name='stress_emd')(stress)
            phones = concatenate([phones, stress])
            context = Lambda(lambda x: K.expand_dims(x), output_shape=(None,size*2,1))(text_model)
            context = Lambda(lambda x: K.repeat_elements(x,maxPhoneLen,-1), output_shape=(None,size*2,maxPhoneLen))(context)
            context = Permute((1,3,2))(context)
            phones = concatenate([phones, context])
            phoneLSTMSize = params['phoneLSTMSize']
            num_heads = params['num_heads']
            if 'phone_att' in params and params['phone_att']:
                phones = TimeDistributed(Bidirectional(LSTM(phoneLSTMSize, return_sequences=True)), name="phone_lstm")(phones)
                att = SelfAttention(num_heads)(phones)
            else:
                phones = TimeDistributed(Bidirectional(LSTM(phoneLSTMSize, return_sequences=False)), name="phone_lstm")(phones)

            def z_score(x):
                num_feat = x.shape[-1] / 3
                if x.ndim == 3:
                    miu = x[:,:,:num_feat]
                    sigma = x[:,:,num_feat:2*num_feat]
                    pred = x[:,:,2*num_feat:]
                elif x.ndim == 4:
                    miu = x[:,:,:,:num_feat]
                    sigma = x[:,:,:,num_feat:2*num_feat]
                    pred = x[:,:,:,2*num_feat:]
                cost = (pred - miu)/K.sqrt(sigma)
                return cost

            def z_score_mask(x):
                num_feat = x.shape[-1] / 4
                if K.ndim(x) == 3:
                    miu = x[:,:,:num_feat]
                    sigma = x[:,:,num_feat:2*num_feat]
                    pred = x[:,:,2*num_feat:3*num_feat]
                    mask = x[:,:,3*num_feat:]
                else:
                    pdb.set_trace()
                masked_miu = miu * (1 - mask)
		miu = K.concatenate([miu[:,:-1,:] + masked_miu[:,1:,:], 
							K.expand_dims(miu[:,-1,:],1)], axis=1) 
                masked_sigma = sigma * (1 - mask)
		sigma = K.concatenate([sigma[:,:-1,:] + masked_sigma[:,1:,:],
                                                        K.expand_dims(sigma[:,-1,:],1)], axis=1)
                cost = (pred - miu)/K.sqrt(sigma)
                cost = cost * mask
                masked_cost = cost[:,:-1,:] * (1 - mask[:,1:,:])
		cost = K.concatenate([K.expand_dims(cost[:,0,:],1), 
							cost[:,1:,:] + masked_cost], axis=1)
                return cost

            def log_likelihood_normal_cost(y_true, y_pred):
                feat_dim = y_pred.shape[-1]/2
                miu = y_pred[:,:,:feat_dim]
                sigma = y_pred[:,:,feat_dim:]
                true = y_true[:,:,:feat_dim]
                mask = y_true[:,:,feat_dim:] # mask is 0 for all tokenname_b that have first part
                # masked_miu
                masked_miu = miu * (1 - mask)
                miu = K.concatenate([miu[:,:-1,:] + masked_miu[:,1:,:],
                                                        K.expand_dims(miu[:,-1,:],1)], axis=1)
                masked_sigma = sigma * (1 - mask)
		sigma = K.concatenate([sigma[:,:-1,:] + masked_sigma[:,1:,:],
                                                        K.expand_dims(sigma[:,-1,:],1)], axis=1)
                cost = K.log(2*math.pi*sigma + K.epsilon())/2 + K.square(true - miu)/(2 * sigma + K.epsilon())
                return K.sum(cost * mask, axis=-1)

            def get_last(x):
                forward = x[:,:,-1,:phoneLSTMSize]
                backward = x[:,:,0,phoneLSTMSize:]
                return K.concatenate([forward, backward])
            
            def apply_attention(inp):
                x, att = inp
                x_ext = K.repeat_elements(K.expand_dims(x, dim=-2), att.shape[-1], axis=-2)
                att_ext = K.repeat_elements(K.expand_dims(att, dim=-1), x.shape[-1], axis=-1)
                return K.sum(x_ext * att_ext, axis=2) # size(batch, seq_l, num_heads, feat_dim)

                
            for addFeature, feat_dim in self.params['phone_feats']:
                if 'total' in addFeature and 'phone_att' in params and params['phone_att']:
                    #hid_phones = Lambda(lambda x: K.mean(x, axis=-2),
                    #                output_shape=(None,phoneLSTMSize*2))(phones)
                    hid_phones = Lambda(lambda x: get_last(x),
                                    output_shape=(None,phoneLSTMSize*2))(phones)
                    input_shape = (None,feat_dim*2)
                    output_shape = (None,feat_dim)
                elif 'phone_att' in params and params['phone_att']:
                    hid_phones = phones
                    input_shape = (None,maxPhoneLen,feat_dim*2)
                    output_shape = (None,maxPhoneLen,feat_dim)
                else:
                    hid_phones = phones
                    input_shape = (None,feat_dim*2)
                    output_shape = (None,feat_dim)
                if feat_dim == 1:
                    activation = 'softplus'
                else:
                    activation = 'softplus'#'tanh'
                miu = TimeDistributed(Dense(feat_dim, activation=activation), name=addFeature+ '_miu_layer')(hid_phones)
                sigma = TimeDistributed(Dense(feat_dim, activation='softplus'), name=addFeature+ '_sigma_layer')(hid_phones)
                miu_sigma = concatenate([miu, sigma])
                prosody_outputs.append(miu_sigma)

            
                prosody_losses.append(log_likelihood_normal_cost)
                addFeatureDense = Input(shape=input_shape, name='input_' + addFeature)
                    
                inputs.append(addFeatureDense)
                minus_feat = concatenate([miu_sigma, addFeatureDense])
                addDiffFeatureDense = Lambda(z_score_mask,
                                            output_shape=output_shape)(minus_feat)
                self.temp_param = [minus_feat, addDiffFeatureDense]
         
                if not 'total' in addFeature and 'phone_att' in params and params['phone_att']:
                        addDiffFeatureDense = Lambda(lambda x: apply_attention(x),
                                output_shape=(None,feat_dim,num_heads))([addDiffFeatureDense, att])
                        addDiffFeatureDense = FlattenCNN()(addDiffFeatureDense)
                addDiffFeatureDense = normalization.BatchNormalization()(addDiffFeatureDense)
                #prosodyLayers.append(addFeatureDense)
                prosodyLayers.append(addDiffFeatureDense)
                
        # Predict prosody losses
        if 'phone_feats' in self.params and 'no_diff_feats' in self.params and self.params['no_diff_feats']:
            for addFeature, feat_dim in self.params['prosody_feats']:
                input_shape = (None,feat_dim*2)
                addFeatureDense = Input(shape=input_shape, name='input_' + addFeature)
                inputs.append(addFeatureDense)
                prosodyLayers.append(addFeatureDense)
        else:
            for addFeature, feat_dim in self.params['prosody_feats']:
                input_shape = (None,feat_dim*2)
                output_shape = (None,feat_dim)

                miu = TimeDistributed(Dense(feat_dim, activation='softplus'),
                                  name=addFeature+ '_miu_layer')(text_model)
                sigma = TimeDistributed(Dense(feat_dim, activation='softplus'),
                                    name=addFeature+ '_sigma_layer')(text_model)
                miu_sigma = concatenate([miu, sigma])
                prosody_outputs.append(miu_sigma)
                prosody_losses.append(log_likelihood_normal_cost)
                
                addFeatureDense = Input(shape=input_shape, name='input_' + addFeature)
                inputs.append(addFeatureDense)
                minus_feat = concatenate([miu_sigma, addFeatureDense])
                addDiffFeatureDense = Lambda(z_score_mask,
                                         output_shape=output_shape)(minus_feat)
                addDiffFeatureDense = normalization.BatchNormalization()(addDiffFeatureDense)
                #prosodyLayers.append(addFeatureDense)
                prosodyLayers.append(addDiffFeatureDense)
        
        #mergeLayers.extend(prosodyLayers)

        if params['attention'] == 'single_multimodal':
            assert params['attention_type'] in ('sim',)
            forward_att = SingleMultiModalAttention(params['window_size'], params['num_feat'],
                            params['attention_activation'],
                            params['attention_type'],
                            params['attention_norm'])
            backward_att = SingleMultiModalAttention(params['window_size'], params['num_feat'],
                            params['attention_activation'],
                            params['attention_type'],
                            params['attention_norm'], backwards=True)

            num_prosody_feats = len(self.additionalDenseFeatures)
            filters = [(3,3),(5,3),(3,1),(5,1),(1,1)]
            if 'filter_shapes' in params and params['filter_shapes'] != None:
                filters = params['filter_shapes']

            fmodel = concatenate(mergeLayers[:3])
            fmodel = forward_att(fmodel)
            self.fmodel = fmodel
            
            convs = []
            for sh in filters:
                conv = Conv2D(params['num_filters'], (sh[0], sh[1]), padding='same', 
							data_format='channels_last')(fmodel)
                if 'conv_nonlin' in self.params and self.params['conv_nonlin']:
                    conv = advanced_activations.LeakyReLU()(conv)
                pool = MaxPooling2D(pool_size=(1,3))(conv)
                convs.append(pool)

            fmodel = concatenate(convs)
            fmodel = FlattenCNN()(fmodel)

            bmodel = concatenate(mergeLayers[:3])
            bmodel = backward_att(bmodel)
            
            convs = []
            for sh in filters:
                conv = Conv2D(params['num_filters'], sh[0], sh[1], border_mode='same',
                              dim_ordering='tf')(bmodel)
                if 'conv_nonlin' in self.params and self.params['conv_nonlin']:
                    conv = advanced_activations.LeakyReLU()(conv)
                pool = MaxPooling2D(pool_size=(1,3))(conv)
                convs.append(pool)

            bmodel = concatenate(convs)
            bmodel = FlattenCNN()(bmodel)

            mergeLayers.append(fmodel)
            mergeLayers.append(bmodel)

            self.attention_layers = [fmodel, bmodel]

        model_reg = Sequential()
        model_reg = concatenate(mergeLayers)
        
        # Add second LSTMs
        cnt = 1
        for size in params['LSTM-Size']:
            if isinstance(params['dropout'], (list, tuple)):
                model_reg_rnn = Bidirectional(LSTM(size, return_sequences=True,
                                        dropout_W=params['dropout'][0],
                                        dropout_U=params['dropout'][1]),
                                        name="varLSTM_top_"+str(cnt))(model_reg)
            
            else:
                """ Naive dropout """
                model_reg_rnn = Bidirectional(LSTM(size, return_sequences=True),
                                              name="LSTM_top_"+str(cnt))(model_reg)     
                
                if params['dropout'] > 0.0:
                    model_reg_rnn = TimeDistributed(Dropout(params['dropout']),
                                                    name="dropout_top_"+str(cnt))(model_reg)
            
            cnt += 1

        # Softmax Decoder - disfluency detection based on text
        if params['classifier'].lower() == 'softmax':    
            model_reg = TimeDistributed(Dense(len(self.dataset['mappings'][self.labelKey]),
                                                  activation='softmax'),
                                                  name='top_softmax_output')(model_reg_rnn)
            lossFct = 'sparse_categorical_crossentropy'
        elif params['classifier'].lower() == 'crf':
            model_reg_out = TimeDistributed(Dense(len(self.dataset['mappings'][self.labelKey]),
                                                  activation=None),
                                                  name='top_hidden_layer')(model_reg_rnn)
            crf = ChainCRF()
            model_reg = crf(model_reg_out)            
            lossFct = crf.sparse_loss 
        elif params['classifier'].lower() == 'tanh-crf':
            model_reg_out = TimeDistributed(Dense(len(self.dataset['mappings'][self.labelKey]),
                                                  activation='tanh'),
                                                  name='top_hidden_layer')(model_reg_rnn)
            crf = ChainCRF()
            model_reg = crf(model_reg_out)
            lossFct = crf.sparse_loss 
        else:
            print("Please specify a valid classifier")
            assert(False) #Wrong classifier
        outputs.append(model_reg)
        losses.append(lossFct)

        # Add second prosody only LSTMs
        if len(params['phone_feats']) + len(params['prosody_feats']) > 0 and params['prosody']:
            model_prosody = concatenate(prosodyLayers) # size(num_batch, seq_len, feat_dim)

            cnt = 1
            for size in params['LSTM-Size']:
                if isinstance(params['dropout'], (list, tuple)):
                    model_prosody_rnn = Bidirectional(LSTM(size, return_sequences=True,
                                        dropout_W=params['dropout'][0],
                                        dropout_U=params['dropout'][1]),
                                        name="varLSTM_prosody_top_"+str(cnt))(model_prosody)
            
                else:
                    """ Naive dropout """
                    model_prosody_rnn = Bidirectional(LSTM(size, return_sequences=True),
                                              name="LSTM_prosody_top_"+str(cnt))(model_prosody)     
                
                    if params['dropout'] > 0.0:
                        model_prosody_rnn = TimeDistributed(Dropout(params['dropout']),
                                                    name="dropout_prosody_top_"+str(cnt))(model_prosody)
            
                cnt += 1

            # Softmax Decoder - disfluency detection based on prosody
            if params['classifier'].lower() == 'softmax':    
                model_prosody = TimeDistributed(Dense(len(self.dataset['mappings'][self.labelKey]),
                                                  activation='softmax'),
                                                  name='top_prosody_softmax_output')(model_prosody_rnn)
                lossFct = 'sparse_categorical_crossentropy'
            elif params['classifier'].lower() == 'crf':
                model_prosody_out = TimeDistributed(Dense(len(self.dataset['mappings'][self.labelKey]),
                                                  activation=None),
                                                  name='top_prosody_hidden_layer')(model_prosody_rnn)
                crf = ChainCRF()
                model_prosody = crf(model_prosody_out)            
                lossFct = crf.sparse_loss 
            elif params['classifier'].lower() == 'tanh-crf':
                model_prosody_out = TimeDistributed(Dense(len(self.dataset['mappings'][self.labelKey]),
                                                  activation='tanh'),
                                                  name='top_prosody_hidden_layer')(model_prosody_rnn)
                crf = ChainCRF()
                model_prosody = crf(model_prosody_out)
                lossFct = crf.sparse_loss 
            else:
                print("Please specify a valid classifier")
                assert(False) #Wrong classifier
            outputs.append(model_prosody)
            losses.append(lossFct)


        # merge rnn layers of text and prosody
        '''
        model_shared_rnn = concatenate([model_reg_rnn, model_prosody_rnn])
        model_shared = TimeDistributed(Dense(len(self.dataset['mappings'][self.labelKey]),
                                                  activation=None),
                                                  name='top_shared_hidden_layer')(model_shared_rnn)
        '''
        if len(params['phone_feats']) + len(params['prosody_feats']) > 0 and params['prosody']:
            model_prosody_out = Lambda(lambda x,aa: x * aa,
                                   arguments={'aa': params['alpha']})(model_prosody_out)
            model_reg_out = Lambda(lambda x,aa: x * (1 - aa),
                                   arguments={'aa': params['alpha']})(model_reg_out)
 
            model_shared = add([model_prosody_out, model_reg_out])
            crf = ChainCRF()
            model_shared = crf(model_shared)
            outputs.append(model_shared)
            losses.append(crf.sparse_loss)


        optimizerParams = {}
        if 'clipnorm' in self.params and self.params['clipnorm'] != None and  self.params['clipnorm'] > 0:
            optimizerParams['clipnorm'] = self.params['clipnorm']
        
        if 'clipvalue' in self.params and self.params['clipvalue'] != None and  self.params['clipvalue'] > 0:
            optimizerParams['clipvalue'] = self.params['clipvalue']
        
        if params['optimizer'].lower() == 'adam':
            #opt = Adam(**optimizerParams)
            opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        elif params['optimizer'].lower() == 'nadam':
            opt = Nadam(**optimizerParams)
        elif params['optimizer'].lower() == 'rmsprop': 
            opt = RMSprop(**optimizerParams)
        elif params['optimizer'].lower() == 'adadelta':
            opt = Adadelta(**optimizerParams)
        elif params['optimizer'].lower() == 'adagrad':
            opt = Adagrad(**optimizerParams)
        elif params['optimizer'].lower() == 'sgd':
            opt = SGD(lr=0.1, **optimizerParams)

        model = Model(input=inputs, output=outputs)

        main_weights = params['weights'][:1] + params['weights'][len(self.additionalDenseFeatures)+1:]
        prosody_weights = params['weights'][1:len(self.additionalDenseFeatures)+1]
        model.compile(loss=losses, optimizer=opt)#, loss_weights=main_weights)
        self.model = model
        if len(params['phone_feats']) + len(params['prosody_feats']) > 0 and params['prosody'] and\
          (not 'no_diff_feats' in self.params or ('no_diff_feats' in self.params and \
                                                  not self.params['no_diff_feats'])):
            prosody_model = Model(input=inputs[:-len(self.additionalDenseFeatures)],
                              output=prosody_outputs)
            prosody_model.compile(loss=prosody_losses, optimizer=opt, loss_weights=prosody_weights)
            self.prosody_model = prosody_model        
            logging.info('Prosody model')
            logging.info(prosody_model.summary())
            self.prosody_model.summary()
            
        
        logging.info('Disfluency prediction model')
        logging.info(model.summary())
        if self.verboseBuild: 
            self.model.summary()
            logging.debug(self.model.get_config())            
            logging.debug("Optimizer: %s, %s" % (str(type(opt)), str(opt.get_config())))
            
    def storeResults(self, resultsFilepath):
        if resultsFilepath != None:
            directory = os.path.dirname(resultsFilepath)
            if not os.path.exists(directory):
                os.makedirs(directory)
                
            self.resultsOut = open(resultsFilepath, 'w')
        else:
            self.resultsOut = None 

    
    def evaluate(self, epochs):
        logging.info("%d train sentences" % len(self.dataset['trainMatrix']))     
        logging.info("%d dev sentences" % len(self.dataset['devMatrix']))   
        logging.info("%d test sentences" % len(self.dataset['testMatrix']))   
        
        trainMatrix = self.dataset['trainMatrix']
        devMatrix = self.dataset['devMatrix']
        testMatrix = self.dataset['testMatrix']
   
        total_train_time = 0
        max_dev_score = 0
        max_test_score = 0
        no_improvement_since = 0
        
        for epoch in range(epochs):      
            sys.stdout.flush()           
            logging.info("--------- Epoch %d -----------" % (epoch+1))

            if self.params['optimizer'] in self.learning_rate_updates and self.epoch in self.learning_rate_updates[self.params['optimizer']]:
                K.set_value(self.model.optimizer.lr, self.learning_rate_updates[self.params['optimizer']][self.epoch])          
                logging.info("Update Learning Rate to %f" % (K.get_value(self.model.optimizer.lr)))

            iterator = self.online_iterate_dataset(trainMatrix, self.labelKey) if self.params['miniBatchSize'] == 1 else self.batch_iterate_dataset(trainMatrix, self.labelKey)
            batch_num = 0
            eval_every = 500
            start_time = time.time()
            while True:
                batch_num += 1
                done_iterating = self.trainModel(iterator, eval_every, epoch)
                if eval_every:
                    logging.info("-- Batch %d --" % (batch_num * eval_every))
                
                time_diff = time.time() - start_time
                total_train_time += time_diff
                logging.info("%.2f sec for training (%.2f total)" % (time_diff, total_train_time))
            
            
                start_time = time.time()
                if not 'no_diff_feats' in self.params or ('no_diff_feats' in self.params and \
                                                  not self.params['no_diff_feats']):
                    dev_score, test_score = self.computeScores(devMatrix, testMatrix)
                else:
                    dev_score, test_score = self.computeScores(devMatrix, testMatrix, 'F1')
                print(epoch, batch_num, dev_score)
                dev_score_prosody, test_score_prosody = self.computeScores(devMatrix, testMatrix, score='prosody_late_fusion')
                print(dev_score_prosody, test_score_prosody)
                if dev_score > max_dev_score:
                    no_improvement_since = epoch
                    max_dev_score = dev_score 
                    max_test_score = test_score
                
                    if self.modelSavePath != None:
                        if eval_every:
                            savePath = self.modelSavePath.replace("[DevScore]", "%.4f" % dev_score).replace("[TestScore]", "%.4f" % test_score).replace("[Epoch]", str(epoch)).replace("[Batch]", str(batch_num * eval_every))
                        else:
                            savePath = self.modelSavePath.replace("[DevScore]", "%.4f" % dev_score).replace("[TestScore]", "%.4f" % test_score).replace("[Epoch]", str(epoch)).replace("[Batch]", '')
                        directory = os.path.dirname(savePath)
                        if not os.path.exists(directory):
                            os.makedirs(directory)
                        
                        if not os.path.isfile(savePath):
                            self.model.save(savePath, False)
                        
                        
                            #self.save_dict_to_hdf5(self.mappings, savePath, 'mappings')
                        
                            import json
                            import h5py
                            mappingsJson = json.dumps(self.mappings)
                            with h5py.File(savePath, 'a') as h5file:
                                h5file.attrs['mappings'] = mappingsJson
                                h5file.attrs['additionalFeatures'] = json.dumps(self.additionalFeatures)
                                if self.prosody:
                                    h5file.attrs['additionalDenseFeatures'] = json.dumps(self.additionalDenseFeatures)
                                h5file.attrs['maxCharLen'] = str(self.maxCharLen)
                            
                            #mappingsOut = open(savePath+'.mappings', 'wb')                        
                            #pkl.dump(self.dataset['mappings'], mappingsOut)
                            #mappingsOut.close()
                        else:
                            logging.info("Model", savePath, "already exists")
                
                
                if self.resultsOut != None:
                    self.resultsOut.write("\t".join(map(str, [epoch+1, batch_num, dev_score, test_score, max_dev_score, max_test_score])))
                    self.resultsOut.write("\n")
                    self.resultsOut.flush()
                
                logging.info("Max: %.4f on dev; %.4f on test" % (max_dev_score, max_test_score))
                logging.info("%.2f sec for evaluation" % (time.time() - start_time))

                # break while loop when done circulating through the batch
                if done_iterating:
                    break
            
            if self.params['earlyStopping'] > 0 and (epoch - no_improvement_since) >= self.params['earlyStopping']:
                logging.info("!!! Early stopping, no improvement after "+str(no_improvement_since)+" epochs !!!")
                break
        return max_dev_score, max_test_score
            
            
    def computeScores(self, devMatrix, testMatrix, score='F1_mse'):
        #if self.labelKey.endswith('_BIO') or self.labelKey.endswith('_IOB') or self.labelKey.endswith('_IOBES') or self.labelKey.endswith('TAG'):
        if 'mse' in score and self.params['prosody']:
            mse_score = self.computeMSEScores(devMatrix, testMatrix)
            if score == 'mse':
                return mse_score
        if score == 'prosody_late_fusion':
            return self.computeF1Scores(devMatrix, testMatrix, 'prosody_late_fusion')
        return self.computeF1Scores(devMatrix, testMatrix)
        #else:
        #    return self.computeAccScores(devMatrix, testMatrix)
            
    def computeMSEScores(self, devMatrix, testMatrix):
        dev_mse, dev_log_prob = self.computeMSE(devMatrix, 'dev')
        if dev_log_prob:
            logging.info("Dev-Data: MSE: %.4f, Avg-log-likelihood %.4f" % (dev_mse, dev_log_prob))
        else:
            logging.info("Dev-Data: MSE: %.4f" % dev_mse)
            
            
        test_mse, test_log_prob = self.computeMSE(testMatrix, 'test')
        if test_log_prob:
            logging.info("Test-Data: MSE: %.4f, Avg-log-likelihood %.4f" % (test_mse, test_log_prob))
        else:
            logging.info("Test-Data: MSE: %.4f" % test_mse)
            
        return dev_mse, test_mse
        
    def computeF1Scores(self, devMatrix, testMatrix, score=''):       
        #dev_pre, dev_rec, dev_f1 = self.computeF1(devMatrix, 'dev', score)
        dev_pre, dev_rec, dev_f1 = self.computeF1(devMatrix, score)
        if len(score) > 0:
            logging.info("Prosody_only preds:")
        logging.info("Dev-Data: Prec: %.3f, Rec: %.3f, F1: %.4f" % (dev_pre, dev_rec, dev_f1))
        
        if self.devAndTestEqual:
            test_pre, test_rec, test_f1 = dev_pre, dev_rec, dev_f1 
        else:        
            #test_pre, test_rec, test_f1 = self.computeF1(testMatrix, 'test', score)
            test_pre, test_rec, test_f1 = self.computeF1(testMatrix, score)
        logging.info("Test-Data: Prec: %.3f, Rec: %.3f, F1: %.4f" % (test_pre, test_rec, test_f1))
        
        return dev_f1, test_f1
        
    def computeAccScores(self, devMatrix, testMatrix):
        dev_acc = self.computeAcc(devMatrix)
        test_acc = self.computeAcc(testMatrix)
        
        logging.info("Dev-Data: Accuracy: %.4f" % (dev_acc))
        logging.info("Test-Data: Accuracy: %.4f" % (test_acc))
        
        return dev_acc, test_acc
          

    def tagSentences(self, sentences):
        
        #Pad characters
        if 'characters' in self.additionalFeatures:       
            maxCharLen = self.maxCharLen
            for sentenceIdx in range(len(sentences)):
                for tokenIdx in range(len(sentences[sentenceIdx]['characters'])):
                    token = sentences[sentenceIdx]['characters'][tokenIdx]
                    sentences[sentenceIdx]['characters'][tokenIdx] = np.pad(token, (0, maxCharLen-len(token)), 'constant')
        
    
        paddedPredLabels = self.predictLabels(sentences)        
        predLabels = []
        for idx in range(len(sentences)):           
            unpaddedPredLabels = []
            for tokenIdx in range(len(sentences[idx]['tokens'])):
                if sentences[idx]['tokens'][tokenIdx] != 0: #Skip padding tokens                     
                    unpaddedPredLabels.append(paddedPredLabels[idx][tokenIdx])
            
            predLabels.append(unpaddedPredLabels)
            
            
        idx2Label = {v: k for k, v in self.mappings['label'].items()}
        labels = [[idx2Label[tag] for tag in tagSentence] for tagSentence in predLabels]
        
        return labels

    def computeMSE(self, sentences, name=''):
        correct = []
        pred = []
        paddedPredLabels = self.predictLabels(sentences, mode='mse')
        import numpy as np
        from sklearn.metrics import mean_squared_error
        preds = [x for x in paddedPredLabels if len(x) > 1]
        means = np.concatenate([x[:,0] for x in preds])
        true = np.concatenate([x['total_phone_durations'] for x in sentences \
                               if len(x['total_phone_durations']) > 1]).flatten()
        if preds[0].shape[1] > 1:
            sigma = np.concatenate([x[:,1] for x in preds])
            cost = np.log(2*math.pi*sigma)/2 + np.square(true - means)/(2 * sigma)
            return mean_squared_error(true, means), np.average(cost)
        return mean_squared_error(true, means), None
        
    def computeF1(self, sentences, name=''):
        correctLabels = []
        predLabels = []
        paddedPredLabels = self.predictLabels(sentences, name)        
        
        for idx in range(len(sentences)):
            unpaddedCorrectLabels = []
            unpaddedPredLabels = []
            for tokenIdx in range(len(sentences[idx]['tokens'])):
                if sentences[idx]['tokens'][tokenIdx] != 0: #Skip padding tokens 
                    unpaddedCorrectLabels.append(sentences[idx][self.labelKey][tokenIdx])
                    unpaddedPredLabels.append(paddedPredLabels[idx][tokenIdx])
                    # calculate you_know as two separate tokens
                    if sentences[idx]['raw_tokens'][tokenIdx] == 'you_know':
                        unpaddedCorrectLabels.append(sentences[idx][self.labelKey][tokenIdx])
                        unpaddedPredLabels.append(paddedPredLabels[idx][tokenIdx])
                    
            correctLabels.append(unpaddedCorrectLabels)
            predLabels.append(unpaddedPredLabels)
            
        
        #encodingScheme = self.labelKey[self.labelKey.index('_')+1:]
        pre, rec, f1 = BIOF1Validation.compute_f1(predLabels, correctLabels, self.idx2Label, 'O')  
        #pre_b, rec_b, f1_b = BIOF1Validation.compute_f1(predLabels, correctLabels, self.idx2Label, 'B', encodingScheme)
        
        
        #if f1_b > f1:
        #    logging.debug("Setting incorrect tags to B yields improvement from %.4f to %.4f" % (f1, f1_b))
        #    pre, rec, f1 = pre_b, rec_b, f1_b 
        
    
        if self.writeOutput:
            self.writeOutputToFile(sentences, predLabels, '%.4f_%s' % (f1, name))
        return pre, rec, f1

    def getProbs(self, sentences, name=''):
        correctLabels = []
        predProbs = []
        
        if 'with_s' in self.labelKey and 'full' in self.labelKey:
            index2labels = ['O','BE','IE','IP','BE_IP','C','BE_s','IE_s','IP_s',
                            'BE_IP_s','C_s','C_IE','C_IP','C_IE_s','C_IP_s',
                            'C_s_IE','C_s_IP']
        elif 'full' in self.labelKey:
            index2labels = ['O','BE','IE','IP','BE_IP','C','C_IE','C_IP']
        else:
            index2labels = ['O','BE','IE','IP','BE_IP']

        newKeys = []
        for key in index2labels:
            newKeys.append(self.mappings[self.labelKey][key])
        
            
        paddedPredProbs = self.predictProbs(sentences)        
        for idx in range(len(sentences)):
            unpaddedCorrectLabels = []
            unpaddedPredProbs = []
            for tokenIdx in range(len(sentences[idx]['tokens'])):
                if sentences[idx]['tokens'][tokenIdx] != 0: #Skip padding tokens 
                    unpaddedCorrectLabels.append(sentences[idx][self.labelKey][tokenIdx])
                    unpaddedPredProbs.append(paddedPredProbs[idx][tokenIdx])
                    # calculate you_know as two separate tokens
                    if sentences[idx]['raw_tokens'][tokenIdx] == 'you_know':
                        unpaddedCorrectLabels.append(sentences[idx][self.labelKey][tokenIdx])
                        unpaddedPredProbs.append(paddedPredProbs[idx][tokenIdx])

            unpaddedCorrectLabels = [newKeys.index(x) for x in unpaddedCorrectLabels]
            correctLabels.append(unpaddedCorrectLabels)
            predProbs.append(np.asarray(unpaddedPredProbs)[:,newKeys])
        return predProbs, correctLabels
    
    def writeOutputToFile(self, sentences, predLabels, name):
            outputName = 'tmp/'+name
            fOut = open(outputName, 'w')
            
            for sentenceIdx in range(len(sentences)):
                for tokenIdx in range(len(sentences[sentenceIdx]['tokens'])):
                    token = self.idx2Word[sentences[sentenceIdx]['tokens'][tokenIdx]]
                    label = self.idx2Label[sentences[sentenceIdx][self.labelKey][tokenIdx]]
                    predLabel = self.idx2Label[predLabels[sentenceIdx][tokenIdx]]
                    
                    fOut.write("\t".join([token, label, predLabel]))
                    fOut.write("\n")
                
                fOut.write("\n")
            
            fOut.close()
            
        
    
    def computeAcc(self, sentences):
        correctLabels = [sentences[idx][self.labelKey] for idx in range(len(sentences))]
        predLabels = self.predictLabels(sentences) 
        
        numLabels = 0
        numCorrLabels = 0
        for sentenceId in range(len(correctLabels)):
            for tokenId in range(len(correctLabels[sentenceId])):
                numLabels += 1
                if correctLabels[sentenceId][tokenId] == predLabels[sentenceId][tokenId]:
                    numCorrLabels += 1

  
        return numCorrLabels/float(numLabels)
    
    def loadWeights(self, modelPath, by_name=False):
        if self.model == None:
            self.buildModel()
       
        if by_name:
            self.model.load_weights(modelPath, by_name=True)
        else:
            self.model.load_weights(modelPath)

    def loadModel(self, modelPath):
        import h5py
        import json
        from neuralnets.keraslayers.ChainCRF import create_custom_objects
        from neuralnets.keraslayers.InnerAttention import attention_custom_object

        custom_objects = create_custom_objects()
        att_custom_objects =  attention_custom_object()
        custom_objects.update(att_custom_objects)
            
        model = keras.models.load_model(modelPath, custom_objects=custom_objects)

        with h5py.File(modelPath, 'r') as f:
            mappings = json.loads(f.attrs['mappings'])
            if 'additionalFeatures' in f.attrs:
                self.additionalFeatures = json.loads(f.attrs['additionalFeatures'])

            if 'maxCharLen' in f.attrs and not f.attrs['maxCharLen'] == 'None':
                self.maxCharLen = int(f.attrs['maxCharLen'])
            
        self.model = model        
        self.setMappings(None, mappings)
        
