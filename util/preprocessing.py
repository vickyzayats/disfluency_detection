from __future__ import print_function
import numpy as np
import gzip
import os.path
import nltk
import logging
import re
import code,pdb
import math
import copy
from nltk import FreqDist

from .WordEmbeddings import wordNormalize
from .CoNLL import readCoNLL

import sys
if (sys.version_info > (3, 0)):
    import pickle as pkl
else: #Python 2.7 imports
    import cPickle as pkl
    from io import open

def prepareDataset(embeddingsPath, datasetFiles, frequencyThresholdUnknownTokens=50, reducePretrainedEmbeddings=False, commentSymbol=None, prosody=False, prosody_feats=None, prosody_files=None):
    """
    Reads in the pre-trained embeddings (in text format) from embeddingsPath and prepares those to be used with the LSTM network.
    Unknown words in the trainDataPath-file are added, if they appear at least frequencyThresholdUnknownTokens times
    
    # Arguments:
        datasetName: The name of the dataset. This function creates a pkl/datasetName.pkl file
        embeddingsPath: Full path to the pre-trained embeddings file. File must be in text format.
        datasetFiles: Full path to the [train,dev,test]-file
        tokenIndex: Column index for the token 
        frequencyThresholdUnknownTokens: Unknown words are added, if they occure more than frequencyThresholdUnknownTokens times in the train set
        reducePretrainedEmbeddings: Set to true, then only the embeddings needed for training will be loaded
        commentSymbol: If not None, lines starting with this symbol will be skipped
    """
    embeddingsName = os.path.splitext(embeddingsPath)[0]
    datasetName = "_".join(sorted([datasetFile[0] for datasetFile in datasetFiles])+[os.path.basename(embeddingsName)])
    outputPath = './pkl/'+datasetName+'.pkl'

    if os.path.isfile(outputPath):
        logging.info("Using existent pickle file: %s" % outputPath)
        return outputPath
    if outputPath.endswith('2'):
        outputPath = outputPath[:-1]
    #Check that the embeddings file exists
    if not os.path.isfile(embeddingsPath):
        if embeddingsPath == 'levy_deps.words':
            getLevyDependencyEmbeddings()
        elif embeddingsPath == '2014_tudarmstadt_german_50mincount.vocab.gz':
            getReimersEmbeddings()
        else:
            print("The embeddings file %s was not found" % embeddingsPath)
            exit()

    tree = False
    if 'tree' in datasetName:
        tree = True
    if prosody:
        prosody_files = copyProsodyFiles(tree)
    else:
        prosody_files = None
        
    logging.info("Generate new embeddings files for a dataset: %s" % outputPath)
    
    neededVocab = {}    
    if reducePretrainedEmbeddings:
        logging.info("Compute which tokens are required for the experiment")
        def createDict(filename, tokenPos, vocab):    
            for line in open(filename):                
                if line.startswith('#'):
                    continue                
                splits = line.strip().split() 
                if len(splits) > 1:  
                    word = splits[tokenPos]     
                    wordLower = word.lower() 
                    wordNormalized = wordNormalize(wordLower)
                    
                    vocab[word] = True
                    vocab[wordLower] = True
                    vocab[wordNormalized] = True        
                
                
        for datasetFile in datasetFiles:
            dataColumnsIdx = {y:x for x,y in datasetFile[1].items()}
            tokenIdx = dataColumnsIdx['tokens']
            datasetPath = '../data/%s/' % datasetName
            
            for dataset in ['train.txt', 'dev.txt', 'test.txt']:  
                createDict(datasetPath+dataset, tokenIdx, neededVocab)

        
    
    # :: Read in word embeddings ::   
    logging.info("Read file: %s" % embeddingsPath) 
    word2Idx = {}
    embeddings = []
    
    embeddingsIn = gzip.open(embeddingsPath, "rt") if embeddingsPath.endswith('.gz') else open(embeddingsPath, encoding="utf8")
    
    embeddingsDimension = None
    
    for line in embeddingsIn:
        split = line.rstrip().split(" ")
        word = split[0]
        
        if embeddingsDimension == None:
            embeddingsDimension = len(split)-1
            
        if (len(split)-1) != embeddingsDimension:  #Assure that all lines in the embeddings file are of the same length
            print("ERROR: A line in the embeddings file had more or less  dimensions than expected. Skip token.")
            continue
        
        if len(word2Idx) == 0: #Add padding+unknown
            word2Idx["PADDING_TOKEN"] = len(word2Idx)
            vector = np.zeros(embeddingsDimension) 
            embeddings.append(vector)
            
            word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
            vector = np.random.uniform(-0.25, 0.25, embeddingsDimension) #Alternativ -sqrt(3/dim) ... sqrt(3/dim)
            embeddings.append(vector)
    
        
        vector = np.array([float(num) for num in split[1:]])
        
        
        if len(neededVocab) == 0 or word in neededVocab:
            if word not in word2Idx:                     
                embeddings.append(vector)
                word2Idx[word] = len(word2Idx)
    
    
    
    # Extend embeddings file with new tokens 
    def createFD(filename, tokenIndex, fd, word2Idx):
        for line in open(filename):
            
            if line.startswith('#'):
                continue
            
            splits = line.strip().split()      
            
            if len(splits) > 1:  
                word = splits[tokenIndex]     
                wordLower = word.lower() 
                wordNormalized = wordNormalize(wordLower)
                
                if word not in word2Idx and wordLower not in word2Idx and wordNormalized not in word2Idx: 
                    fd[wordNormalized] += 1
            
    if frequencyThresholdUnknownTokens != None and frequencyThresholdUnknownTokens >= 0:
        fd = nltk.FreqDist()
        for datasetFile in datasetFiles:
            dataColumnsIdx = {y:x for x,y in datasetFile[1].items()}
            tokenIdx = dataColumnsIdx['tokens']
            datasetPath = '../data/%s/' % datasetFile[0]            
            createFD(datasetPath+'train.txt', tokenIdx, fd, word2Idx)        
        
        addedWords = 0
        for word, freq in fd.most_common(10000):
            if freq < frequencyThresholdUnknownTokens:
                break
            
            addedWords += 1        
            word2Idx[word] = len(word2Idx)
            vector = np.random.uniform(-0.25, 0.25, len(split)-1)  #Alternativ -sqrt(3/dim) ... sqrt(3/dim)
            embeddings.append(vector)
            
            assert(len(word2Idx) == len(embeddings))
        
        
        logging.info("Added words: %d" % addedWords)
    embeddings = np.array(embeddings)

    embeddings, word2Idx, ds = loadDatasetPickle('../pkl/swbd_diff_lm_levy_deps.pkl')
    pklObjects = {'embeddings': embeddings, 'word2Idx': word2Idx, 'datasets': {}}
    casing2Idx = getCasingVocab()
    for datasetName, datasetColumns in datasetFiles:
        trainData = '../data/%s/train.txt' % datasetName 
        devData = '../data/%s/dev.txt' % datasetName 
        testData = '../data/%s/test.txt' % datasetName 
        paths = [trainData, devData, testData]

        pklObjects['datasets'][datasetName] = createPklFiles(paths, word2Idx, casing2Idx, datasetColumns, commentSymbol, padOneTokenSentence=True, prosody=prosody, prosody_feats=prosody_feats, prosody_files=prosody_files, loaded_mappings=ds['swbd_diff_lm']['mappings'])
    
    f = open(outputPath, 'wb')
    pkl.dump(pklObjects, f, -1)
    f.close()
    
    logging.info("DONE - Embeddings file saved: %s" % outputPath)
    
    return outputPath


def addCharInformation(sentences):
    """Breaks every token into the characters"""
    for sentenceIdx in range(len(sentences)):
        sentences[sentenceIdx]['characters'] = []
        for tokenIdx in range(len(sentences[sentenceIdx]['tokens'])):
            token = sentences[sentenceIdx]['tokens'][tokenIdx]
            chars = [c for c in token]
            sentences[sentenceIdx]['characters'].append(chars)

def addCasingInformation(sentences):
    """Adds information of the casing of words"""
    for sentenceIdx in range(len(sentences)):
        sentences[sentenceIdx]['casing'] = []
        for tokenIdx in range(len(sentences[sentenceIdx]['tokens'])):
            token = sentences[sentenceIdx]['tokens'][tokenIdx]
            sentences[sentenceIdx]['casing'].append(getCasing(token))



def addPauseInformation(sentences, prosody_files, tree=False):
    """Adds information of the pause duration between the words"""
    nxt_dir = '/s0/vzayats/acoustic_features/'
    #conv_feat_dir = '/s0/vzayats/acoustic_features/normalized_features_ms_word_boundaries/'
    conv_feat_dir = '/s0/vzayats/acoustic_features/normalized_3sigma/'
    if tree:
        durations_dir = nxt_dir + 'boundaries/'
        nxt_dir += 'tree_aligned'
        durations_dir += 'tree_aligned'
    else:
        durations_dir = nxt_dir + 'ms_word_boundaries/'
        nxt_dir += 'ms_aligned'
        durations_dir += 'ms_aligned'
    
    word_level = True if 'word_boundaries' in durations_dir else False
    vowels = ['aa','ae','ah','ao','aw','ax','axr','ay','eh','er','ey','ih','iy','ow','oy','uh','uw']
    stress_file = '/g/ssli/data/switchboard/dicts/resources/BerklingSyllableDict'
    #stress_file = '/homes/vzayats/project/disfluency/cmudict-0.7b'
    with open(stress_file, 'r') as inf:
        inf = inf.read().strip().split('\n')
        stress_dict = {}
        for line in inf:
            key, value = line.strip().split('   ')
            value = re.sub('[%.+]', '', value)
            value = re.sub(' +', ' ', value)
            syl = value.strip().split(' ] [ ')
            
            syl[0] = syl[0][2:]
            syl[-1] = syl[-1][:-2]
            stressed = [i for i,x in enumerate(syl) if "'" in x]
            syl = [x.replace("' ",'').split(' ') for x in syl]
            syl_flat = []
            i = 0
            for x in syl:
                for y in x:
                    syl_flat.append((i,y))
                i += 1
            v = [[y for y in x if y in vowels] for x in syl]
            if len(stressed) == 0:
                stress_dict[key.lower()] = [[], syl_flat]
            else:
                stress_dict[key.lower()] = [zip(stressed, [v[x][0] for x in stressed]), syl_flat]
            
    prev_nxt_filename = ''
    features = [{},{}]
    durations = [{},{}]
    conv_features = [{},{}]
    cat = ''
    if 'nxt' in nxt_dir:
        cat = '_cat'
        
    def pause2cat(p):
        if np.isnan(p):
            cat = 6
        elif p == 0.0:
            cat = 0
        elif p <= 0.05:
            cat = 1
        elif p <= 0.1:
            cat = 2
        elif p <= 0.2:
            cat = 3
        elif p <= 1.0:
            cat = 4
        else:
            cat = 5
        return cat

    incorrect_tokenization = []
    sentenceIdx = -1
    while sentenceIdx < len(sentences) - 1:
        sentenceIdx += 1
        filename = sentences[sentenceIdx]['filename']
        tokennames = sentences[sentenceIdx]['tokenname']
        sentences[sentenceIdx]['pause_before'] = []
        sentences[sentenceIdx]['pause_before_cat'] = []
        sentences[sentenceIdx]['phones'] = []
        sentences[sentenceIdx]['stress'] = []
        sentences[sentenceIdx]['total_phone_durations'] = []
        sentences[sentenceIdx]['conv_phone_feats'] = []
        sentences[sentenceIdx]['phone_feats_len'] = []
        if not word_level:
            sentences[sentenceIdx]['pause_after'] = []
            sentences[sentenceIdx]['pause_after_cat'] = []
            sentences[sentenceIdx]['word_norm'] = []
            sentences[sentenceIdx]['rhyme_norm'] = []
            sentences[sentenceIdx]['phone_durations'] = []
        nxt_filename = os.path.join(nxt_dir, filename[0])
        duration_filename = os.path.join(durations_dir, 'word_times_' + filename[0])
        conv_filename = os.path.join(conv_feat_dir, filename[0])
        if nxt_filename != prev_nxt_filename:
            if filename[0] in prosody_files:
                print('Loading files %s' % (nxt_filename + '{A,B}.features'))
                if not 'word_boundaries' in duration_filename:
                    with open(nxt_filename + 'A.features', 'rb') as inf:
                        features[0] = pkl.load(inf)
                    with open(nxt_filename + 'B.features', 'rb') as inf:
                        features[1] = pkl.load(inf)
                with open(duration_filename + 'A.pickle', 'rb') as inf:
                    durations[0] = pkl.load(inf)
                with open(duration_filename + 'B.pickle', 'rb') as inf:
                    durations[1] = pkl.load(inf)
                with open(conv_filename + '-A.pickle', 'rb') as inf:
                    conv_features[0] = pkl.load(inf)
                with open(conv_filename + '-B.pickle', 'rb') as inf:
                    conv_features[1] = pkl.load(inf)
            else:
                pdb.set_trace()
                continue
            prev_nxt_filename = nxt_filename
            
        tokenIdx = 0
        prev_tokid = ''
        while tokenIdx < len(sentences[sentenceIdx]['tokens']):
            tokenname = sentences[sentenceIdx]['tokenname'][tokenIdx]
            if 'A' in tokenname or 'B' in tokenname:
                nxt_filename = os.path.join(nxt_dir, filename[0])
            else:
                pdb.set_trace()
                # temporal fix for tokennames that are not in the feature file
                sentences[sentenceIdx]['pause_before'].append(6)
                sentences[sentenceIdx]['pause_before_cat'].append(6)
                if not word_level:
                    sentences[sentenceIdx]['pause_after'].append(6)
                    sentences[sentenceIdx]['pause_after_cat'].append(6)
                    sentences[sentenceIdx]['word_norm'].append(6)
                    sentences[sentenceIdx]['rhyme_norm'].append(6)
                tokenIdx += 1
                continue
            tokid = tokenname.replace('_a','').replace('_b','')
            multitokid = None
            if '@' in tokenname:
                multitokid = tokid.split('@')
                if len(multitokid) > 2:
                    print('long combined')
                    pdb.set_trace()
                tokid = multitokid[0]
            if tokid in durations[0]:
                sp = 0
            elif tokid in durations[1]:
                sp = 1
            else:
                #print('Token not found: %s' % sentences[sentenceIdx]['tokens'][tokenIdx])
                tokenIdx += 1
                continue
            try:
                if word_level:
                    if prev_tokid == '':
                        pause_before = [0]
                    else:
                        if tokid == prev_tokid:
                            pause_before = [0]
                        else:
                            pause_before = [durations[sp][tokid]['start_time'] - durations[sp][prev_tokid]['end_time']]
                        if pause_before[0] < 0:
                            pdb.set_trace()
                    if '_a' in tokenname or '_b' in tokenname:
                        pause_before.append(0)
                else:
                    pause_before = features[sp][tokid]['pause_before' + cat]
                    if math.isnan(pause_before[0]):
                        pause_before[0] = 0
                    pause_after = features[sp][tokid]['pause_after' + cat]
                    if multitokid:
                        pause_after = features[sp][multitokid[-1]]['pause_after' + cat]
                    if math.isnan(pause_after[-1]):
                        pause_after[-1] = 0
                        
                    word_norm = features[sp][tokid]['word_norm']
                    rhyme_norm = features[sp][tokid]['rhyme_norm']
                prev_tokid = tokid
                phones = durations[sp][tokid]['phones'][:]
                phone_durations = []
                if not word_level:
                    phone_durations = [x-y for x,y in zip(durations[sp][tokid]['phone_end_times'],
                                                      durations[sp][tokid]['phone_start_times'])]
                text = durations[sp][tokid]['text']
                if multitokid:
                    phones.extend(durations[sp][multitokid[-1]]['phones'])
                    if not word_level:
                        phone_durations.extend([x-y for x,y in zip(durations[sp][multitokid[-1]]['phone_end_times'],
                                                                   durations[sp][multitokid[-1]]['phone_start_times'])])
                if not word_level:
                    cnn_feats = features[sp][tokid]['cnn_feats'][0]
                    feats = []
                    start_time = 0
                    for dur in phone_durations:
                        end_time = start_time + dur
                        feat = cnn_feats[:,int(np.floor(start_time/0.015)):int(np.ceil(end_time/0.015))]
                        feats.append(np.average(feat, axis=-1))
                else:
                    feats = conv_features[sp][tokid]
                    tot_time = durations[sp][tokid]['end_time'] - durations[sp][tokid]['start_time']
                    if multitokid:
                        tot_time += durations[sp][multitokid[-1]]['end_time'] - durations[sp][multitokid[-1]]['start_time']
                        feats = np.concatenate([conv_features[sp][tokid],
                                                conv_features[sp][multitokid[-1]]], axis=0)
                    mean_feats = np.mean(feats, axis=0)
                    #this part takes only 3 first mfccs and polynomic coefficients
                    '''
                    xx = range(feats.shape[0])
                    poly_coef = []
                    for ff in (0,6,7,8):
                        if feats.shape[0] == 0:
                            poly_coef.append(np.zeros((3)))
                        elif feats.shape[0] > 2:
                            poly_coef.append(np.polyfit(xx, feats[:,ff],2))
                        else:
                            poly_coef.append(np.polyfit(xx, feats[:,ff],feats.shape[0]-1))
                            poly_coef.append(np.zeros((3-feats.shape[0])))
                    feats = np.concatenate(poly_coef)
                    '''
                    feats = mean_feats
                    feats_len = feats.shape[0]
                if '[lau' in text:
                    text = text.replace('[laughter-','').replace(']','')
                text = text.replace('_1','')
                phones_syl = []
                k = 0
                for p in phones:
                    phones_syl.append((k, p))
                    if p in vowels:
                        k += 1
                phones_stress = []
                if text in stress_dict or text.replace('-','') in stress_dict or  text.replace('-','_') in stress_dict:
                    try:
                        stress = stress_dict[text]
                    except:
                        try:
                            stress = stress_dict[text.replace('-','')]
                        except:
                            stress = stress_dict[text.replace('-','_')]
                    # you_know
                    if multitokid:
                        last_token = durations[sp][multitokid[-1]]['text']
                        last_token = last_token.replace('[laughter-','').replace(']','')
                        last_token = re.sub('\[\w+/','',last_token)
                        if last_token != 'know':
                            pdb.set_trace()
                        stress = copy.deepcopy(stress)
                        for p1,p2 in stress_dict[last_token][0]:
                            stress[0] += [(p1+1, p2)]
                        for p1,p2 in stress_dict[last_token][1]:
                            stress[1] += [(p1+1, p2)]
                    stressed_syl = [x[0] for x in stress[0]]
                    # 1 - stressed vowel, 2 - consonant, 3 - non-stressed vowel
                    for p in phones_syl:
                        if p[0] in stressed_syl and p[1] in vowels:
                            phones_stress.append(1)
                        elif not p[1] in vowels:
                            phones_stress.append(2)
                        else:
                            phones_stress.append(3)
                    if sum([x for x in phones_stress if x == 1]) != len(stress[0]):
                        phone_vowels = [x for x in phones if x in vowels]
                        if len(phone_vowels) > 0:
                            #hard-coded:
                            if (stress[0][0][0]-1, stress[0][0][1]) in phones_syl:
                                stress_index = phones_syl.index((stress[0][0][0]-1, stress[0][0][1]))
                                phones_stress[stress_index] = 1
                            else:    
                                pass
                else:
                    for p in phones_syl[::-1]:
                        if not p[1] in vowels:
                            phones_stress.append(2)
                        #heuristics
                        elif 1 in phones_stress or '[' in text:
                            phones_stress.append(3)
                        else:
                            phones_stress.append(1)
                    phones_stress = phones_stress[::-1]
                if tokenname.endswith('_b'):
                    tid = 1
                if len(pause_before) == 1 and (tokenname.endswith('_a') or tokenname.endswith('_b')):
                    if tokid + '_a' in sentences[sentenceIdx]['tokenname'] and \
                      tokid + '_b' in sentences[sentenceIdx]['tokenname']:
                        incorrect_tokenization.append([filename[0], tokenname,
                                                       sentences[sentenceIdx]['tokens'][tokenIdx]])
                        print('Incorrect tokenization: %s' % sentences[sentenceIdx]['tokens'][tokenIdx])
                    else:
                        print('Just _a or _b present')
                    tid = 0
                s_idx = 0
                e_idx = split_idx = len(phones)
                tid = 0
                inconsistent = False
                if tokenname.endswith('_a') or tokenname.endswith('_b'):
                    if phones[-1] in ('s','d','z','r','ax','m','v','l','er'):
                        split_idx = -1
                    elif phones[-1] in ('t'):
                        split_idx = -2
                    else:
                        inconsistent = True
                        print(' '.join(phones))
                    if phones[-1] != 'z' and len(phones) > -split_idx + 1 and phones[split_idx - 1] == 'ax':
                        split_idx -= 1
                if tokenname.endswith('_a'):
                    e_idx = split_idx
                elif tokenname.endswith('_b'):
                    s_idx = split_idx
                    if inconsistent:
                        s_idx = -1
                phones = phones[s_idx:e_idx]
                phone_durations = phone_durations[s_idx:e_idx]
                phones_stress = phones_stress[s_idx:e_idx]
                sentences[sentenceIdx]['pause_before'].append(np.array(pause_before[tid]).reshape(1,))
                sentences[sentenceIdx]['pause_before_cat'].append(pause2cat(pause_before[tid]))
                sentences[sentenceIdx]['phones'].append(phones)
                sentences[sentenceIdx]['stress'].append(phones_stress)
                sentences[sentenceIdx]['conv_phone_feats'].append(feats)
                sentences[sentenceIdx]['phone_feats_len'].append(feats_len)
                if not word_level:
                    sentences[sentenceIdx]['pause_after'].append(np.array(pause_after[tid]).reshape(1,))
                    sentences[sentenceIdx]['pause_after_cat'].append(pause2cat(pause_after[tid]))
                    sentences[sentenceIdx]['word_norm'].append(np.array(word_norm[tid]).reshape(1,))
                    sentences[sentenceIdx]['rhyme_norm'].append(np.array(rhyme_norm[tid]).reshape(1,))
                    sentences[sentenceIdx]['total_phone_durations'].append(np.array(sum(phone_durations)).reshape(1,))
                    sentences[sentenceIdx]['phone_durations'].append(np.array(phone_durations))
                else:
                    sentences[sentenceIdx]['total_phone_durations'].append(np.array(tot_time).reshape(1,))
                #sentences[sentenceIdx]['last_syl_durations'].append(sum(phone_durations))
            except:
                print('stopped here 4')
                pdb.set_trace()
            
            tokenIdx += 1
            
        #pdb.set_trace()
        if len(sentences[sentenceIdx]['pause_before']) != len(sentences[sentenceIdx]['feat1']):
            sentences[sentenceIdx]['pause_before'] = []
            sentences[sentenceIdx]['pause_before_cat'] = []
            sentences[sentenceIdx]['phones'] = []
            sentences[sentenceIdx]['stress'] = []
            sentences[sentenceIdx]['total_phone_durations'] = []
            sentences[sentenceIdx]['conv_phone_feats'] = []
            sentences[sentenceIdx]['phone_feats_len'] = []
            if not word_level:
                sentences[sentenceIdx]['pause_after'] = []
                sentences[sentenceIdx]['pause_after_cat'] = []
                sentences[sentenceIdx]['word_norm'] = []
                sentences[sentenceIdx]['rhyme_norm'] = []
                sentences[sentenceIdx]['phone_durations'] = []
    '''
    import pandas
    db = pandas.DataFrame(incorrect_tokenization)
    a = 'train'
    pdb.set_trace()
    db.to_csv('incorrect_tokenization' + a + '.csv', sep='\t')
    '''
    
def getCasing(word):   
    """Returns the casing for a word"""
    casing = 'other'
    
    numDigits = 0
    for char in word:
        if char.isdigit():
            numDigits += 1
            
    digitFraction = numDigits / float(len(word))
    
    if word.isdigit(): #Is a digit
        casing = 'numeric'
    elif digitFraction > 0.5:
        casing = 'mainly_numeric'
#    elif word.islower(): #All lower case
#        casing = 'allLower'
#    elif word.isupper(): #All upper case
#        casing = 'allUpper'
#    elif word[0].isupper(): #is a title, initial char upper, then all lower
#        casing = 'initialUpper'
    elif numDigits > 0:
        casing = 'contains_digit'
    
    return casing

def getCasingVocab():
    # entries = ['PADDING', 'other', 'numeric', 'mainly_numeric', 'allLower', 'allUpper', 'initialUpper', 'contains_digit']
    entries = ['PADDING', 'other', 'numeric', 'mainly_numeric', 'contains_digit']
    return {entries[idx]:idx for idx in range(len(entries))}

def getPOSVocab():
    return None

def createMatrices(sentences, mappings, padOneTokenSentence=False):
    data = []
    numTokens = 0
    numUnknownTokens = 0    
    missingTokens = FreqDist()
    paddedSentences = 0

    for sentence in sentences:
        row = {name: [] for name in list(mappings.keys())+['raw_tokens']}
        
        for mapping, str2Idx in mappings.items():    
            if mapping not in sentence:
                continue
            for entry in sentence[mapping]:                
                if mapping.lower() == 'tokens':
                    numTokens += 1
                    idx = str2Idx['UNKNOWN_TOKEN']
                    
                    if entry in str2Idx:
                        idx = str2Idx[entry]
                    elif entry.lower() in str2Idx:
                        idx = str2Idx[entry.lower()]
                    elif wordNormalize(entry) in str2Idx:
                        idx = str2Idx[wordNormalize(entry)]
                    else:
                        numUnknownTokens += 1    
                        missingTokens[wordNormalize(entry)] += 1
                        
                    row['raw_tokens'].append(entry)
                elif mapping.lower() in ('characters','phones','stress'):  
                    idx = []
                    for c in entry:
                        if c in str2Idx:
                            idx.append(str2Idx[c])
                        else:
                            idx.append(str2Idx['UNKNOWN'])
                elif mapping.lower() in ('tokenname', 'filename', 'pause_before', 'pause_after',
                                         'word_norm', 'rhyme_norm', 'phone_durations',
                                         'total_phone_durations', 'conv_phone_feats',
                                         'phone_feats_len'):
                    idx = entry
                else:
                    idx = str2Idx[entry]
                        
                row[mapping].append(idx)
                
        if len(row['tokens']) == 1 and padOneTokenSentence:
            paddedSentences += 1
            for mapping, str2Idx in mappings.items():
                if mapping.lower() == 'tokens':
                    row['tokens'].append(mappings['tokens']['PADDING_TOKEN'])
                    row['raw_tokens'].append('PADDING_TOKEN')
                elif mapping.lower() in ('characters', 'phones', 'stress'):
                    row[mapping.lower()].append([0])
                else:
                    row[mapping].append(0)
            
        data.append(row)
    
    if numTokens > 0:           
        logging.info("Unknown-Tokens: %.2f%%" % (numUnknownTokens/float(numTokens)*100))
        
    return data
    
def file_contains_prosody(sentences, prosody_list=[]):
    files = {}
    for s in sentences:
        f = s['filename'][0]
        if not f in files:
            if f in prosody_list:
                files[f] = 1
            else:
                files[f] = 0
    return files             
  
def get_phones_mappings(sentences):
    phoneset = {"PADDING":0, "UNKNOWN":1}
    for s in sentences:
        for w in s['phones']:
            for ph in w:
                if not ph in phoneset:
                    phoneset[ph] = len(phoneset)
    return phoneset
  
def createPklFiles(datasetFiles, word2Idx, casing2Idx, cols, commentSymbol=None, valTransformation=None, padOneTokenSentence=False, prosody=False, prosody_feats=None, prosody_files=[], loaded_mappings=[]):
    
              
    trainSentences = readCoNLL(datasetFiles[0], cols, commentSymbol, valTransformation)
    devSentences = readCoNLL(datasetFiles[1], cols, commentSymbol, valTransformation)
    testSentences = readCoNLL(datasetFiles[2], cols, commentSymbol, valTransformation)    
    mappings = createMappings(trainSentences+devSentences+testSentences)
    mappings['tokens'] = word2Idx
    mappings['casing'] = casing2Idx
    if prosody:
        mappings['prosody_files'] = file_contains_prosody(trainSentences + devSentences + testSentences, prosody_files)
    if 'tokenname' in cols.values():
        mappings['tokenname'] = {}
    if 'filename' in cols.values():
        mappings['filename'] = {}                
    
    charset = {"PADDING":0, "UNKNOWN":1}
    for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
        charset[c] = len(charset)
    mappings['characters'] = charset

    tree = False
    if 'tree' in datasetFiles[0]:
        tree = True
    
    addCharInformation(trainSentences)
    addCasingInformation(trainSentences)
    if prosody:
        addPauseInformation(trainSentences, prosody_files, tree)
        maxpb = max([max(x['pause_before_cat']) for x in trainSentences if len(x['pause_before_cat']) > 0])
        mappings['pause_before_cat'] = dict(zip(range(maxpb+1),range(maxpb+1)))
        mappings['pause_before'] = {}
        mappings['phones'] = {}
        mappings['stress'] = {'PADDING':0, 1:1, 2:2, 3:3}
        mappings['total_phone_durations'] = {}
        mappings['conv_phone_feats'] = {}
        mappings['phone_feats_len'] = {}
        if 'pause_after' in trainSentences[0]:
            maxpa = max([max(x['pause_after_cat']) for x in trainSentences if len(x['pause_after_cat']) > 0])
            mappings['pause_after_cat'] = dict(zip(range(maxpa+1),range(maxpa+1)))
            mappings['pause_after'] = {}
            mappings['word_norm'] = {}
            mappings['rhyme_norm'] = {}
            mappings['phone_durations'] = {}

    mappings['phones'] = get_phones_mappings(trainSentences)
    for c in cols:
        if c in loaded_mappings.keys():
            mappings[c] = loaded_mappings[c]
    
    addCharInformation(devSentences)
    addCasingInformation(devSentences)
    if prosody:
        addPauseInformation(devSentences, prosody_files, tree)
    
    addCharInformation(testSentences)   
    addCasingInformation(testSentences)   
    if prosody:
        addPauseInformation(testSentences, prosody_files, tree)
    
    trainMatrix = createMatrices(trainSentences, mappings)
    devMatrix = createMatrices(devSentences, mappings)
    testMatrix = createMatrices(testSentences, mappings)       
    data = { 'mappings': mappings,
                'trainMatrix': trainMatrix,
                'devMatrix': devMatrix,
                'testMatrix': testMatrix
            }        
       
    
    return data

def createMappings(sentences):
    sentenceKeys = list(sentences[0].keys())
    sentenceKeys.remove('tokens')
    if 'tokenname' in sentenceKeys:
        sentenceKeys.remove('tokenname')
    if 'filename' in sentenceKeys:
        sentenceKeys.remove('filename')
        
    
    vocabs = {name:{'O':0} for name in sentenceKeys} #Use 'O' also for padding
    #vocabs = {name:{} for name in sentenceKeys}
    for sentence in sentences:
        for name in sentenceKeys:
            for item in sentence[name]:              
                if item not in vocabs[name]:
                    vocabs[name][item] = len(vocabs[name]) 
                    
    
    return vocabs  


    
def loadDatasetPickle(embeddingsPickle):
    """ Loads the cPickle file, that contains the word embeddings and the datasets """
    f = open(embeddingsPickle, 'rb')
    pklObjects = pkl.load(f)
    f.close()
    

        
        
    return pklObjects['embeddings'], pklObjects['word2Idx'], pklObjects['datasets']



def getLevyDependencyEmbeddings():
    """
    Downloads from https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/
    the dependency based word embeddings and unzips them    
    """ 
    if not os.path.isfile("levy_deps.words.bz2"):
        print("Start downloading word embeddings from Levy et al. ...")
        os.system("wget -O levy_deps.words.bz2 http://u.cs.biu.ac.il/~yogo/data/syntemb/deps.words.bz2")
    
    print("Start unzip word embeddings ...")
    os.system("bzip2 -d levy_deps.words.bz2")

def getReimersEmbeddings():
    """
    Downloads from https://www.ukp.tu-darmstadt.de/research/ukp-in-challenges/germeval-2014/
    embeddings for German
    """
    if not os.path.isfile("2014_tudarmstadt_german_50mincount.vocab.gz"):
        print("Start downloading word embeddings from Reimers et al. ...")
        os.system("wget https://public.ukp.informatik.tu-darmstadt.de/reimers/2014_german_embeddings/2014_tudarmstadt_german_50mincount.vocab.gz")
    
   
def copyProsodyFiles(tree=False):

    #nxt_directory = '/s0/vzayats/experiments/disfl/ntx_features'
    nxt_directory = '/s0/vzayats/acoustic_features/'
    if tree:
        nxt_directory += 'tree_aligned'
    else:
        nxt_directory += 'ms_aligned'
    copy = False
    if not os.path.exists(nxt_directory):
        os.makedirs(nxt_directory)
        copy = True

    from shutil import copyfile
    import glob

    files = []
    #for f in glob.glob('/g/ssli/projects/disfluencies/julia/projects/prosody/nxt_features/sw*.features'):
    for f in glob.glob(nxt_directory + '/sw*.features'):
        filename = os.path.join(nxt_directory, os.path.basename(f))
        if copy:
            copyfile(f, filename)
        if f.endswith('A.features'):
            files.append(os.path.basename(f).replace('A.features',''))

    return files
