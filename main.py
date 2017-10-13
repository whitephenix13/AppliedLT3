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
def phrase_translation_cost():
    # TODO:Phrase translation: htm(state) = GLOBAL_lamb_ef * log(p(e|f)) + GLOBAL_lamb_fe * log(p(f|e)) +\
    # TODO:  GLOBAL_lamb_lex_ef * log(lex(e|f)) + GLOBAL_lamb_lex_fe * log(lex(f|e))
    #TODO: however we also have to use lexical weight... how?
    # TODO: how to handle missing phrases?
    return 0

#target_phrase: string composed of words separated by spaces
#language_model: dictionnary [english phrase] = (log10(prob),log10(back of prob) or None)
#Returns the language model continuation cost as ln(prob)
def LM_cost(target_phrase,prev_target_phrase,language_model):
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

def reordering_cost():
    #TODO: look for the phrase pair in the reordering dictionnary. use the .trace to determine which event it is
    #TODO: if we call "o1" the r->l event and "o2" the l->r event, the final prob is then
    #TODO: # use P = p r->l (o1) ^ {lamb_rl(o1)} *  p l->r (o2) ^ {lamb_lr(o2)}
    #TODO: the cost is then the np.log of this probability
    # TODO: how to handle missing phrases?
    return 0
def transition_cost():
    return GLOBAL_phrase_transl_weight * phrase_translation_cost () + GLOBAL_language_model_weight * LM_cost() + \
           GLOBAL_phrase_weight * phrase_penalty_cost() + GLOBAL_word_weight * word_penalty_cost() + \
           GLOBAL_reordering_weight * reordering_cost()

