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
GLOBAL_translation_constant = 1

    #language model parameters
GLOBAL_language_model_window = 5
GLOBAL_backoff_weight = 0 #P(wN| w{N-1},w1) = P(wN|w{N-1},w2)*backoff-weight(w{N-1}|w{N-2},w1 ), if not found backoff-weight=1 (0 in log space)
GLOBAL_language_constant = -np.log10(np.exp(1)) #so that the division gives -1

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
DATA_DIR = './data/'

GLOBAL_f_en = open(DATA_DIR +'file.test.en', 'r')
GLOBAL_f_de = open(DATA_DIR+'file.test.de', 'r')
GLOBAL_phrase_table = open(DATA_DIR+'phrase-table', 'r')
GLOBAL_test_results = open(DATA_DIR+'testresults.trans.txt.trace', 'r')
GLOBAL_language_model = open(DATA_DIR+'file.en.lm', 'r')
GLOBAL_reordering = open(DATA_DIR+'dm_fe_0.75', 'r')


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
        cost+= np.log(GLOBAL_translation_constant)
    return cost

#target_phrase: string composed of words separated by spaces
#language_model_dict: dictionnary [english phrase] = (log10(prob),log10(back of prob) or None)
#Returns the language model continuation cost as ln(prob)
def LM_cost(target_phrase,language_model_dict):
    # assumption : the probability given is already in log space and it's already an ngram
    target_words = target_phrase.split()
    total_prob = 0.0
    for i, word in enumerate(target_words):
        ngram = ""
        ### create the ngrams according to the window ###
        #Note: the n-gram consists of this word with a history of the n-1 previous words
        enough_words = (i - (GLOBAL_language_model_window -1)) >= 0 #<=> can we have n-1 words in the history
        min_index_history = (i - (GLOBAL_language_model_window -1)) if enough_words else 0
        max_index = i
        for j in range(min_index_history,max_index+1):
            ngram += target_words[j] + " "
        ngram = ngram[:-1]
        ###calculate probability ###
        if ngram in language_model_dict:
            total_prob += language_model_dict[ngram][0]
        else:
            total_prob += calculate_back_off(ngram, language_model_dict)
    return total_prob / np.log10(np.exp(1)) #convert log10 to log prob using log_a(x) = log_b(x) / log_b(a)

#ngram : string of words seperated by spaces
#language_model_dict: dictionnary [english phrase] = (log10(prob),log10(back of prob) or None)
#This function is recursive and it terminates since the number of words in the ngrams strictly decrease at each call
#This function is based on the following link: https://cmusphinx.github.io/wiki/arpaformat/
def calculate_back_off(ngram, language_model_dict):
    # end case
    if len(ngram.split()) == 1:
        if ngram in language_model_dict:
            return language_model_dict[ngram][0]
        else:
            return GLOBAL_language_constant
    # general case
    else:
        result = 0
        words = ngram.split() #w1 ... wn
        new_ngram = " ".join(words[1:]) #w2 ... wn
        backoff_ngram = " ".join(words[:-1]) #w1 ... w-1
        if new_ngram in language_model_dict:
            result += language_model_dict[new_ngram][0]
            #Also use the backoff prob
            if(backoff_ngram not in language_model_dict):
                result += GLOBAL_backoff_weight
            elif(language_model_dict[backoff_ngram][1] == None):
                result += GLOBAL_backoff_weight
            else:
                result += language_model_dict[backoff_ngram][1]
        else:
            #Recursive call with the shorter ngram
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
    #get all the reordering probabilite related to the phrase pair : p(reordering event |phrase pair)
    if phrase_pair in reordering_dict:
        val = reordering_dict[phrase_pair]
    else: #phras pair not found, use (UNK,UNK) instead
        val = reordering_dict[("UNK","UNK")]
    if o1 != -1: #handle case wehre there are no right phrase
        prob_rl = np.power(val[o1],GLOBAL_lamb_rl[o1])
    if o2 != -1: #handle case wehre there are no left phrase
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
            res=(0,-1) #monotonic
        elif step==-1:
            res = (1,-1) #swap
        else:
            res= (1,-1) #discontinuous
    if nextAlign != None :
        step = nextAlign[0] - currentAlign[1]
        if step == 1:
            res = (res[0],0)  # monotonic
        elif step == -1:
            res = (res[0],1)   # swap
        else:
            res = (res[0],2)   # discontinuous
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
        for j in range(source_target_align[i][0],source_target_align[i][1]+1):
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

        total_cost += GLOBAL_phrase_transl_weight * phrase_translation_cost (phrase_pair,translation_dict) +\
           GLOBAL_language_model_weight * LM_cost(phrase_pair[1],language_model_dict) + \
           GLOBAL_phrase_weight * phrase_penalty_cost() +\
           GLOBAL_word_weight * word_penalty_cost(phrase_pair[1]) + \
           GLOBAL_reordering_weight * reordering_cost(phrase_pair,event_type,reordering_dict)

    return total_cost

########################### DICTIONNARY CREATION FUNCTIONS   #############################################################

#Returns dictionnary [(German phrase,english phrase)] = (p(f|e) lex(f|e) p(e|f) lex(e|f))
def createTranslationDict():
    translation_dict = defaultdict(tuple)
    for line in GLOBAL_phrase_table:
        elems_bloc = line.replace("\n", "").replace("\t", "").split(" ||| ") #[source, target,probs,alignments,counts]
        if python3Code:
            translation_dict[(elems_bloc[0], elems_bloc[1])] = list(map(float, elems_bloc[2].split()))
        else:
            translation_dict[(elems_bloc[0], elems_bloc[1])] = map(float, elems_bloc[2].split())
    return translation_dict

#Returns dictionnary [english phrase] = (log10(prob),log10(back of prob) or None)
def createLMDict():
    language_model_dict = defaultdict(tuple)
    for line in GLOBAL_language_model:
        elems = line.replace("\n", "").replace("\t", " ").split(" ")
        valid = False
        try:
            float(elems[0])
            valid = True
        except ValueError:
            valid = False

        if valid:
            size = len(elems)
            last_elem_is_num = False
            try:
                float(elems[size - 1])
                if float(elems[size - 1]) < 0:  # handle the 6.66 and 6:66 cases (all log probs are <0)
                    last_elem_is_num = True
            except ValueError:
                last_elem_is_num = False
            target_phrase = " ".join(elems[1:-1]) if last_elem_is_num else " ".join(elems[1:])

            if last_elem_is_num:
                language_model_dict[target_phrase] = (float(elems[0]), float(elems[size - 1]))
            # case for empty backoff prob
            else:
                language_model_dict[target_phrase] = (float(elems[0]), None)
    return language_model_dict

#Returns dictionnary[(German,English)] = [pr->l(mono|f,e),...,pl->r(disc|f,e)]
def createReorderingDict():
    reordering_dict = defaultdict(tuple)
    for line in GLOBAL_reordering:
        elems = line.replace("\n", "").replace("\t", " ").split(" ||| ")
        if python3Code:
            reordering_dict[(elems[0],elems[1])] = list(map(float, elems[2].split()))
        else:
            reordering_dict[(elems[0],elems[1])] = map(float, elems[2].split())
    return reordering_dict
###########################             MAIN                      #############################################################
def main():
    #File to write results
    result_f = open('results.txt', 'w')

    print('Creation of translation dict')
    #create translation_dict by reading from GLOBAL_phrase_table
    translation_dict= createTranslationDict()

    print('Creation of language model dict')
    #create language_model_dict by reading from GLOBAL_language_model
    language_model_dict= createLMDict()

    print('Creation of reordering dict')
    #create reordering_dict by reading from GLOBAL_reordering
    reordering_dict=createReorderingDict()

    print('Storing all the source sentences')
    #store all source sentence in a list (1 index for 1 sentence) by reading from GLOBAL_f_de
    source_sentences = []
    for line in GLOBAL_f_de:
        source_sentences.append(line.replace("\n", "").replace("\t", " "))

    index = 0
    for line in GLOBAL_test_results:
        if index>0 and index%10==0:
            print('Step:'+str(index)+ '/500')
        #extract target phrase and source->target alignement
        target_phrases=[]
        source_target_align=[]
        data = line.replace("\n","").split(" ||| ")# list of type "ind1-ind2:target_phrase"
        for d in data:
            data2 = d.split(":")# ["ind1-ind2","target_phrase"]
            if len(data2)>1 : #get rid of empty words
                #handle sentences where they have ":" in it
                target_ph =data2[1]
                for k in range(2,len(data2)):
                    target_ph+=":"+data2[k]
                target_phrases.append(target_ph)

                align = data2[0].split("-")
                source_target_align.append((int(align[0]),int(align[1])))

        source_sentence = source_sentences[index].split(" ")

        #compute the sentence cost with:
        sentence_c = sentence_cost(source_sentence,target_phrases,source_target_align,translation_dict,language_model_dict,reordering_dict)

        #write GERMAN ||| ENGLISH ||| cost in a file
        target_sentence = " ".join(target_phrases)
        result_line = source_sentences[index]+" ||| "+target_sentence +" ||| "+str(sentence_c)+"\n"
        result_f.write(result_line)
        index = index + 1

    #close all used files
    result_f.close()
    GLOBAL_f_en.close()
    GLOBAL_f_de.close()
    GLOBAL_phrase_table.close()
    GLOBAL_test_results.close()
    GLOBAL_language_model.close()
    GLOBAL_reordering.close()
    print('done')
main()