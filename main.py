from collections import defaultdict
import sys
import matplotlib.pyplot as plt
import dill as pickle
from collections import OrderedDict
import numpy as np
import time

#HYPERPARAMETERS


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
#       -Word penalty: normally length of target sentence, wait for M.fadaee answer to have more information
#       -Reordering: use hierarchical ordering to determine event? Or just use the probability as a cost, use weights here too 