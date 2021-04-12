from nltk.util import ngrams
from nltk import word_tokenize
from nltk.corpus import stopwords 
from nltk.tag import pos_tag
from collections import Counter


class CollocationsByFrequency:

    def tag_collocations(self, collocations):
        
        ''' part of speech tagging for all the collocations passed'''
        
        freq_collocation_tag = []
        for collocation in collocations:
            frequency_of_collocation = collocation[1]
            tag_of_collocation = pos_tag([collocation[0][0]])[0][1], pos_tag([collocation[0][1]])[0][1]
            freq_collocation_tag_tuple = frequency_of_collocation, collocation[0], tag_of_collocation
            freq_collocation_tag.append(freq_collocation_tag_tuple)
            
        return freq_collocation_tag

    def pos_filter(self, tagged_collocations, filter_tags_list):
        
        ''' Takes a set of tuples  containing the tags to be filtered and returns
            filtered version of the collocations
            e.g. filter_list = set([('NN', 'NN'), ('AN', 'NN')])
        '''
        filtered_collocations = []
        for collocation in tagged_collocations:
            tag = collocation[2]
            if tag not in filter_tags_list:
                filtered_collocations.append(collocation)
        return filtered_collocations

    def get_collocations(self, donem_text):
        
        """Returns bigrams(collocations) list given a donem_text """

        collocations = list(ngrams(donem_text, 2)) # extracting bigrams
        collocations_freqs = Counter(collocations)
        collocations_freqs = sorted(collocations_freqs.items(), key=lambda kv: kv[1], reverse=True)[:10]
        tagged_collocations = self.tag_collocations(collocations_freqs)
        pos_filter_list = set([('NA', 'NN'), ('NA', 'NN')])
        filtered_collocations = self.pos_filter(tagged_collocations, pos_filter_list)
        return filtered_collocations