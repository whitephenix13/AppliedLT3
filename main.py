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

GLOBAL_e = sys.float_info.epsilon

#Small fix for python 3 code
python3Code = False;
if (sys.version_info > (3, 0)):
    python3Code = True

#Load/Save directory
DATA_DIR = './data/'

GLOBAL_f_en = open(DATA_DIR +'file.test.en', 'r')
GLOBAL_f_de = open(DATA_DIR+'file.test.de', 'r')
GLOBAL_phrase_table = open(DATA_DIR+'phrase-table', 'r')
GLOBAL_test_results = open(DATA_DIR+'testresults.trans.txt.trace', 'r')
GLOBAL_language_model = open(DATA_DIR+'file.en.lm', 'r')
GLOBAL_reordering = open(DATA_DIR+'dm_fe_0.75', 'r')


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

    # assumption : the probability given is already in log space and it's already an ngram
    target_words = target_phrase.split()
    total_prob = 0.0
    for i, word in enumerate(target_words):
        ngram = ""
        ### create the ngrams according to the window ###
        if(i+1 > GLOBAL_language_model_window):
            for j in range(i+1-GLOBAL_language_model_window,i+1):
                ngram += target_words[j] + " "
            ngram = ngram[:-1]
        else:
            for j in range(i+1):
                ngram += target_words[j] + " "
            ngram = ngram[:-1]
        ### checking ngram and calculate probability ###
        if ngram in language_model_dict:
            total_prob += language_model_dict[word][0]
        else:
            total_prob += calculate_back_off(ngram, language_model_dict)
    #return total_prob / np.log(GLOBAL_e)
    return total_prob

# TODO : For Alex please read this
# Basically my calculate_back_off method + LM_Cost method computes prob in a recursive way and handle 3 of these cases below.
# Case 1 (assume in log space)
# Say we have phrase "the way of life", but p(life|the way of) doesn't exists, then in my implementation I will back off to : p(life|way of) + back off prob p(way|the).
# Is this method correct?
# Case 2
# Say that in the Case 1 example, the back off prob p(way|the) doesn't exists (None), then my implementation will penalize this by : p(life|way of) + SOME_BIG_NUMBERS. I dont know if this is the right way to go
# Case 3
# Say that in the Case 1 example, p(life|way of) doesn't exists but back off prob p(way|the) does exist, then my implementation will recurse to p(life|of) + back off prob p(of|way) ignoring completely back off prob p(way|the).
# In this case should we consider also back off prob p(way|the)?
#
# If my understanding is completely wrong, please do tell me.

def calculate_back_off(ngram, language_model_dict):
    # base case
    if len(ngram.split()) == 1:
        if ngram in language_model_dict:
            return language_model_dict[ngram]
        else:
            # a very large number? im not sure about this
            return -20
    # if not base case
    else:
        result = 0
        words = ngram.split()
        new_ngram = " ".join(words[1:])
        backoff_ngram = " ".join(words[:2])
        if(new_ngram in language_model_dict):
            result += language_model_dict[new_ngram][0]
            # penalize with big numbers
            if(backoff_ngram not in language_model_dict):
                result += -20
             # penalize with big numbers also
            elif(language_model_dict[backoff_ngram][1] == None):
                result += -20
            else:
                result += language_model_dict[backoff_ngram][1]
        else:
            result += calculate_back_off(new_ngram,language_model_dict)
    return result

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
    language_model_dict = defaultdict(tuple)
    for line in GLOBAL_language_model:
        elems = line.replace("\n","").split("\t")
        if len(elems) == 3:
            language_model_dict[elems[1]] = (float(elems[0]), float(elems[2]))
        # case for empty backoff prob
        elif len(elems) == 2:
            language_model_dict[elems[1]] = (float(elems[0]), None)
    #print(LM_cost("the way of life", language_model_dict))
    print(LM_cost("life of shit man", language_model_dict))
    print(LM_cost("the the the the the the", language_model_dict))
    #TODO: create reordering_dict by reading from GLOBAL_reordering

    #TODO: store all source sentence in a list (1 index for 1 sentence) by reading from GLOBAL_f_de

    #TODO: for each line in GLOBAL_test_results

        #TODO: extract target phrase and source->target alignement

        #TODO: compute the sentence cost with:
        #TODO: sentence_cost(source_sentence,target_phrases,source_target_align,translation_dict,language_model_dict,reordering_dict):

        #TODO: write GERMAN ||| ENGLISH ||| cost in a file
    print('')

main()