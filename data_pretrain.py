import os
from os import listdir
from os.path import isfile, join
import numpy as np
from tensorflow.contrib import learn
from collections import Counter

# This Data_Loader file is copied online
class Data_Loader:
    def __init__(self, options):

        positive_data_file = options['dir_name']
        positive_examples = list(open(positive_data_file, "r").readlines())
        positive_examples = [s for s in positive_examples]

        max_document_length = max([len(x.split(",")) for x in positive_examples])
        # max_document_length = max([len(x.split()) for x in positive_examples])  #split by space, one or many, not sensitive
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)#, min_frequency=10
        self.item = np.array(list(vocab_processor.fit_transform(positive_examples)))
        self.item_dict = vocab_processor.vocabulary_._mapping

# if __name__ == '__main__':
#     data = Data_Loader({'dir_name':'Data/Session/data_pretrain_small.csv'})
#     print len(data.item[0])
#     print len(data.item_dict)