from __future__ import print_function
import logging
import code
from sklearn import metrics
"""
Computes the F1 score on BIO tagged data

@author: Nils Reimers
"""



def compute_f1_token_basis(predictions, correct, O_Label): 
       
    prec = compute_precision_token_basis(predictions, correct, O_Label)
    rec = compute_precision_token_basis(correct, predictions, O_Label)
    
    f1 = 0
    if (rec+prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec);
        
    return prec, rec, f1

def compute_precision_token_basis(guessed_sentences, correct_sentences, O_Label):
    assert(len(guessed_sentences) == len(correct_sentences))
    correctCount = 0
    count = 0
    
    
    for sentenceIdx in range(len(guessed_sentences)):
        guessed = guessed_sentences[sentenceIdx]
        correct = correct_sentences[sentenceIdx]
        assert(len(guessed) == len(correct))
        for idx in range(len(guessed)):
            
            if guessed[idx] != O_Label:
                count += 1
               
                if guessed[idx] == correct[idx]:
                    correctCount += 1
    
    precision = 0
    if count > 0:    
        precision = float(correctCount) / count
        
    return precision


def compute_f1(predictions, correct, idx2Label, correctBIOErrors='No', encodingScheme='BIO'): 
    label_pred = []
    for sentence in predictions:
        label_pred.append([idx2Label[element] for element in sentence])
        
    label_correct = []    
    for sentence in correct:
        label_correct.append([idx2Label[element] for element in sentence])
            
          
    #checkBIOEncoding(label_pred, correctBIOErrors)

    prec, rec = compute_disfl_scores(label_pred, label_correct)
    
    f1 = 0
    if (rec+prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec);
        
    return prec, rec, f1


def convertIOBtoBIO(dataset):
    """ Convert inplace IOB encoding to BIO encoding """
    for sentence in dataset:
        prevVal = 'O'
        for pos in range(len(sentence)):
            firstChar = sentence[pos][0]
            if firstChar == 'I':
                if prevVal == 'O' or prevVal[1:] != sentence[pos][1:]:
                    sentence[pos] = 'B'+sentence[pos][1:] #Change to begin tag

            prevVal = sentence[pos]

def convertIOBEStoBIO(dataset):
    """ Convert inplace IOBES encoding to BIO encoding """    
    for sentence in dataset:
        for pos in range(len(sentence)):
            firstChar = sentence[pos][0]
            if firstChar == 'S':
                sentence[pos] = 'B'+sentence[pos][1:]
            elif firstChar == 'E':
                sentence[pos] = 'I'+sentence[pos][1:]
                
def testEncodings():
    """ Tests BIO, IOB and IOBES encoding """
    
    goldBIO   = [['O', 'B-PER', 'I-PER', 'O', 'B-PER', 'B-PER', 'I-PER'], ['O', 'B-PER', 'B-LOC', 'I-LOC', 'O', 'B-PER', 'I-PER', 'I-PER'], ['B-LOC', 'I-LOC', 'I-LOC', 'B-PER', 'B-PER', 'I-PER', 'I-PER', 'O', 'B-LOC', 'B-PER']]
    
    
    print("--Test IOBES--")
    goldIOBES = [['O', 'B-PER', 'E-PER', 'O', 'S-PER', 'B-PER', 'E-PER'], ['O', 'S-PER', 'B-LOC', 'E-LOC', 'O', 'B-PER', 'I-PER', 'E-PER'], ['B-LOC', 'I-LOC', 'E-LOC', 'S-PER', 'B-PER', 'I-PER', 'E-PER', 'O', 'S-LOC', 'S-PER']]
    convertIOBEStoBIO(goldIOBES)
    
    for sentenceIdx in range(len(goldBIO)):
        for tokenIdx in range(len(goldBIO[sentenceIdx])):
            assert(goldBIO[sentenceIdx][tokenIdx] == goldIOBES[sentenceIdx][tokenIdx])
            
    print("--Test IOB--")        
    goldIOB   = [['O', 'I-PER', 'I-PER', 'O', 'I-PER', 'B-PER', 'I-PER'], ['O', 'I-PER', 'I-LOC', 'I-LOC', 'O', 'I-PER', 'I-PER', 'I-PER'], ['I-LOC', 'I-LOC', 'I-LOC', 'I-PER', 'B-PER', 'I-PER', 'I-PER', 'O', 'I-LOC', 'I-PER']]
    convertIOBtoBIO(goldIOB)
    
    for sentenceIdx in range(len(goldBIO)):
        for tokenIdx in range(len(goldBIO[sentenceIdx])):
            assert(goldBIO[sentenceIdx][tokenIdx] == goldIOB[sentenceIdx][tokenIdx])
            
    print("test encodings completed")
    

def compute_disfl_scores(guessed_sentences, correct_sentences):

    def label2bin(label):
        label = label.replace('_s','').replace('C_','')
        if label in ('BE','IP','IE','BE_IP'):
            return 1
        else:
            return 0
           
    
    correct = [label2bin(item) for sent in correct_sentences for item in sent]
    guessed = [label2bin(item) for sent in guessed_sentences for item in sent]
    return metrics.precision_score(correct, guessed), metrics.recall_score(correct, guessed)
    

def compute_precision(guessed_sentences, correct_sentences):
    assert(len(guessed_sentences) == len(correct_sentences))
    correctCount = 0
    count = 0
    
    
    for sentenceIdx in range(len(guessed_sentences)):
        guessed = guessed_sentences[sentenceIdx]
        correct = correct_sentences[sentenceIdx]
        code.interact(local=locals())
        
        assert(len(guessed) == len(correct))
        idx = 0
        while idx < len(guessed):
            if guessed[idx][0] == 'B': #A new chunk starts
                count += 1
                
                if guessed[idx] == correct[idx]:
                    idx += 1
                    correctlyFound = True
                    
                    while idx < len(guessed) and guessed[idx][0] == 'I': #Scan until it no longer starts with I
                        if guessed[idx] != correct[idx]:
                            correctlyFound = False
                        
                        idx += 1
                    
                    if idx < len(guessed):
                        if correct[idx][0] == 'I': #The chunk in correct was longer
                            correctlyFound = False
                        
                    
                    if correctlyFound:
                        correctCount += 1
                else:
                    idx += 1
            else:  
                idx += 1
    
    precision = 0
    if count > 0:    
        precision = float(correctCount) / count
        
    return precision

def checkBIOEncoding(predictions, correctBIOErrors):
    errors = 0
    labels = 0
    
    for sentenceIdx in range(len(predictions)):
        labelStarted = False
        labelClass = None
        

        for labelIdx in range(len(predictions[sentenceIdx])):
            label = predictions[sentenceIdx][labelIdx]
            if 'C_' in label:
                label.replace('C_','')
            elif not label in ('BE', 'IE', 'IP', 'BE_IP'):
                label = 'O'
            if label.startswith('B-'):
                labels += 1
                labelStarted = True
                labelClass = label[2:]
            
            elif label == 'O':
                labelStarted = False
                labelClass = None
            elif label.startswith('I-'):
                if not labelStarted or label[2:] != labelClass:
                    errors += 1        
                    
                    if correctBIOErrors.upper() == 'B':
                        predictions[sentenceIdx][labelIdx] = 'B-'+label[2:]
                        labelStarted = True
                        labelClass = label[2:]
                    elif correctBIOErrors.upper() == 'O':
                        predictions[sentenceIdx][labelIdx] = 'O'
                        labelStarted = False
                        labelClass = None
            else:
                code.interact(local=locals())
                assert(False) #Should never be reached
           
    
    if errors > 0:
        labels += errors
        logging.info("Wrong BIO-Encoding %d/%d labels, %.2f%%" % (errors, labels, errors/float(labels)*100),)


def compute_f1_argument(predictions, correct, idx2Label):     
    prec = compute_argument_chunk_precision(predictions, correct)
    rec = compute_argument_chunk_precision(correct, predictions)
    
    f1 = 0
    if (rec+prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec);
        
    return prec, rec, f1

def compute_f1_argument_token_basis(predictions, correct, idx2Label):     
    prec = compute_argument_token_precision(predictions, correct)
    rec = compute_argument_token_precision(correct, predictions)
    
    f1 = 0
    if (rec+prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec);
        
    return prec, rec, f1

def compute_argument_token_precision(predictions, correct):
    count = 0
    correctCount = 0
    
    for sentenceIdx in range(len(predictions)):
        for tokenIdx in range(len(predictions[sentenceIdx])):
            for argIdx in range(len(predictions[sentenceIdx][tokenIdx])):
                pred = predictions[sentenceIdx][tokenIdx][argIdx]
                corr = correct[sentenceIdx][tokenIdx][argIdx]
                
                if pred:
                    count += 1
                    
                    if pred == corr:
                        correctCount += 1
    
    if count == 0:
        return 0
    
    return correctCount / float(count)

def compute_argument_chunk_precision(guessed_sentences, correct_sentences):
    assert(len(guessed_sentences) == len(correct_sentences))
    correctCount = 0
    count = 0
    
    
    for sentenceIdx in range(len(guessed_sentences)):
        assert(len(guessed_sentences[sentenceIdx]) == len(correct_sentences[sentenceIdx]))
        
        for argIdx in range(len(guessed_sentences[sentenceIdx][0])):
            idx = 0
            guessed = guessed_sentences[sentenceIdx]
            correct = correct_sentences[sentenceIdx]
            while idx < len(guessed):
                if guessed[idx][argIdx]: #A new chunk starts
                    count += 1
                    
                    if guessed[idx][argIdx] == correct[idx][argIdx]:
                        idx += 1
                        correctlyFound = True
                        
                        while idx < len(guessed) and guessed[idx][argIdx]: #Scan until it no longer starts with I
                            if guessed[idx][argIdx] != correct[idx][argIdx]:
                                correctlyFound = False
                            
                            idx += 1
                        
                        if idx < len(guessed):
                            if correct[idx][argIdx]: #The chunk in correct was longer
                                correctlyFound = False
                            
                        
                        if correctlyFound:
                            correctCount += 1
                    else:
                        idx += 1
                else:  
                    idx += 1
    
    precision = 0
    if count > 0:    
        precision = float(correctCount) / count
        
    return precision


if __name__ == "__main__":
    testEncodings()
