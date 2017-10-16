from collections import defaultdict
import sys
import matplotlib.pyplot as plt
import dill as pickle
from collections import OrderedDict
import numpy as np
import time

#HYPERPARAMETERS
    #Phrase translation parameters
GLOBAL_lamb_fe = 1
GLOBAL_lamb_ef = 1
GLOBAL_lamb_lex_fe = 1
GLOBAL_lamb_lex_ef = 1
GLOBAL_language_constant = 1

    #language model parameters
GLOBAL_language_model_window = 3
GLOBAL_backoff_param = 1 #normally 0.4

    #reordering weights
GLOBAL_lamb_lr = [1] * 3 #left to right weights as [monotone, swap, discontinuous ]
GLOBAL_lamb_rl = [1] * 3 #right to left weights as [monotone, swap, discontinuous ]

    #all weights
GLOBAL_phrase_transl_weight = 1
GLOBAL_language_model_weight = 1
GLOBAL_phrase_weight = 1
GLOBAL_word_weight = 1
GLOBAL_reordering_weight = 1


#Small fix for python 3 code
python3Code = False;
if (sys.version_info > (3, 0)):
    python3Code = True

#Load/Save directory
DATA_DIR = 'data/'

GLOBAL_f_en = open(DATA_DIR +'test.en', 'r')
GLOBAL_f_de = open(DATA_DIR+'test.de', 'r')
GLOBAL_phrase_table = open(DATA_DIR+'phrase-table', 'r')
GLOBAL_test_results = open(DATA_DIR+'testresults.trans.txt.trace', 'r')
GLOBAL_language_model = open(DATA_DIR+'file.en.lm', 'r')
GLOBAL_reordering = open(DATA_DIR+'dm.fe.0.75', 'r')


#TODO: Transition cost function:
#For all of the following, consider log_10() of prob so that they can be added?
#       -Phrase translation: htm(state) = p(e|f) + p(f|e), use lexical weight
#       -LM continuation: sum for all english word of new state p(english word |previous words) with previous words depending on the sliding window size
#       -Phrase penalty: apply -1 at each transition call
#       -Word penalty: normally length of target sentence
#       -Reordering: use hierarchical ordering to determine event? Or just use the probability as a cost, use weights here too

#phrase_pair: Tuple of string (German,English) to be translated
#translation_dict: dictionnary [(German phrase,english phrase)] = (p(f|e) lex(f|e) p(e|f) lex(e|f))
#Returns the translation cost as ln(prob)
def phrase_translation_cost(phrase_pair, translation_dict):
    cost = 0
    if phrase_pair in translation_dict:
        val = translation_dict[phrase_pair]
        #add GLOBAL_lamb_ef * log(p(e|f)) + GLOBAL_lamb_fe * log(p(f|e)) +GLOBAL_lamb_lex_ef * log(lex(e|f)) + GLOBAL_lamb_lex_fe * log(lex(f|e))
        cost += GLOBAL_lamb_fe * val[0] + GLOBAL_lamb_lex_fe * val[1] + \
                GLOBAL_lamb_ef * val[2] + GLOBAL_lamb_lex_ef * val[3]
    else:#This happens if key not found or german word align to itself (UNK handling)
        cost+= np.log(GLOBAL_language_constant)
    return cost

#target_phrase: string composed of words separated by spaces
#language_model_dict: dictionnary [english phrase] = (log10(prob),log10(back of prob) or None)
#Returns the language model continuation cost as ln(prob)
def LM_cost(target_phrase,language_model_dict):
     #Logarithm base switch formula: ln(x) = log_10(x) / log_10(e)
     #TODO:loop over all the english word and sum p(wn|wn-1,wn-2...wn-(GLOBAL_language_model_window)+1)= X (short name for explanation)
     #TODO: to get X : X= p(wn,wn-1,...wn-(GLOBAL_language_model_window)+1) / p(wn-1,...wn-(GLOBAL_language_model_window)+1)
     #TODO: if either of those probs are not found in language_model, use the backof prob * GLOBAL_backoff_param
     #TODO: return np.log of this
    return 0

def phrase_penalty_cost():
    #phrase penalty is 2.718 = exp(1) hence the log of it
    return -1

#target_phrase: string composed of words separated by spaces
#Returns the word penalty
def word_penalty_cost(target_phrase):
    #Do not use log as the log penalty is the length of the sentence
    return len(target_phrase.split())

#phrase_pair: Tuple of string (German,English) to be translated
#event_type : Tuple of integer -1:None, 0=monotonic, 1=swap, 2=discontinuous, first one is for right to left, 2nd left to right
#reordering_dict: dictionnary[(German,English)] = [pr->l(mono|f,e),...,pl->r(disc|f,e)]
def reordering_cost(phrase_pair,event_type,reordering_dict):
    prob_rl=1
    prob_lr=1
    o1 = event_type[0]
    o2 = event_type[1]
    if phrase_pair in reordering_dict:
        val = reordering_dict[phrase_pair]
    else:
        val = reordering_dict[("UNK","UNK")]
    if o1 != -1:
        prob_rl = np.power(val[o1],GLOBAL_lamb_lr[o1])
    if o2 != -1:
        prob_lr = np.power(val[3+o2],GLOBAL_lamb_lr[o2])
    return np.log(prob_rl* prob_lr)

#prevAlign: previous source_target_align or None
#currentAlign: current source_target_align or None
#nextAlign: next source_target_align or None
def computeReorderingEvent(prevAlign,currentAlign,nextAlign):
    res = (-1,-1) #-1 means no reordering event, the res is in the format (r->l reordering event, l->r reordering event)
    if prevAlign != None :
        step = currentAlign[0] - prevAlign[1]
        if step == 1:
            res[0]=0 #monotonic
        elif step==-1:
            res[0] = 1 #swap
        else:
            res[0]=2 #discontinuous
    if nextAlign != None :
        step = nextAlign[0] - currentAlign[1]
        if step == 1:
            res[0] = 0  # monotonic
        elif step == -1:
            res[0] = 1  # swap
        else:
            res[0] = 2  # discontinuous
    return res
#source_sentence: List of source words
#target_phrases: List of target phrases (from testresults.trans.txt.trace)
#source_target_align: List of tuple of int matching with target_phrases to retrieve the corresponding source_phrase
#translation_dict: dictionnary [(German phrase,english phrase)] = (p(f|e) lex(f|e) p(e|f) lex(e|f))
#language_model_dict: dictionnary [english phrase] = (log10(prob),log10(back of prob) or None)
#reordering_dict: dictionnary[(German,English)] = [pr->l(mono|f,e),...,pl->r(disc|f,e)]
#return the cost for the source_sentence
def sentence_cost(source_sentence,target_phrases,source_target_align,translation_dict,language_model_dict,reordering_dict):
    total_cost = 0
    #Compute phrase pair
    for i in range(len(source_target_align)):
        source_phrase = ""
        for j in range(source_target_align[i][0],source_target_align[i][0]+1):
            if j != source_target_align[i][0]:
                source_phrase+=" "
            source_phrase+= source_sentence[j]
        phrase_pair =(source_phrase,target_phrases[i])

        #Compute reordering event
        prevAlign = None
        nextAlign = None
        if i>0:
            prevAlign=source_target_align[i-1]
        if i< (len(source_target_align)-1):
            nextAlign = source_target_align[i+1]
        event_type = computeReorderingEvent(prevAlign,source_target_align[i],nextAlign)

        total_cost = GLOBAL_phrase_transl_weight * phrase_translation_cost (phrase_pair,translation_dict) +\
           GLOBAL_language_model_weight * LM_cost(phrase_pair[1],language_model_dict) + \
           GLOBAL_phrase_weight * phrase_penalty_cost() +\
           GLOBAL_word_weight * word_penalty_cost(phrase_pair[1]) + \
           GLOBAL_reordering_weight * reordering_cost(phrase_pair,event_type,reordering_dict)

    return total_cost


def main():
    #TODO: create translation_dict by reading from GLOBAL_phrase_table

    #TODO: create language_model_dict by reading from GLOBAL_language_model

    #TODO: create reordering_dict by reading from GLOBAL_reordering

    #TODO: store all source sentence in a list (1 index for 1 sentence) by reading from GLOBAL_f_de

    #TODO: for each line in GLOBAL_test_results

        #TODO: extract target phrase and source->target alignement

        #TODO: compute the sentence cost with:
        #TODO: sentence_cost(source_sentence,target_phrases,source_target_align,translation_dict,language_model_dict,reordering_dict):

        #TODO: write GERMAN ||| ENGLISH ||| cost in a file
    print('')