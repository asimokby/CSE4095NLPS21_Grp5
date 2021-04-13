from collections import Counter
from nltk.util import ngrams
import math

class MutualInformation():

    def get_bigrams_with_freqs(self, donem_text):
        collocations = list(ngrams(donem_text, 2)) # extracting bigrams
        collocations_freqs = Counter(collocations)
        collocations_freqs = sorted(collocations_freqs.items(), key=lambda kv: kv[1], reverse=True)[:100]
        return dict(collocations_freqs)

    def calc_mi(self, bigram):
        w1_freq = self.word_freq[bigram[0]]
        w2_freq = self.word_freq[bigram[1]]
        bigram_frequncy = self.bigram_freqs[bigram]
        normlize = self.words_count
        mutual_information = ((bigram_frequncy/normlize)/((w1_freq/normlize)*(w2_freq/normlize)))
        mi_value = math.log2(mutual_information)
        return (mi_value, w1_freq, w2_freq, bigram_frequncy, bigram)
        
    def get_collocations(self, donem_text):
        collocations = []
        self.words_count = len(donem_text)
        self.word_freq = Counter(donem_text)
        self.bigram_freqs = self.get_bigrams_with_freqs(donem_text)
        collocations = sorted([self.calc_mi(bigram) for bigram in self.bigram_freqs], reverse=True)
        return collocations