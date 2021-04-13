from collections import Counter
import math

class MutualInformation():

    def calc_mi(self, bigram, bigrams_with_freqs):
        w1_freq = self.word_freq[bigram[0]]
        w2_freq = self.word_freq[bigram[1]]
        bigram_frequncy = bigrams_with_freqs[bigram]
        normlize = self.words_count
        mutual_information = ((bigram_frequncy/normlize)/((w1_freq/normlize)*(w2_freq/normlize)))
        mi_value = math.log2(mutual_information)
        return (mi_value, w1_freq, w2_freq, bigram_frequncy, bigram)
        
    def get_collocations(self, donem_text, bigrams_with_freqs):
        collocations = []
        self.words_count = len(donem_text)
        self.word_freq = Counter(donem_text)
        collocations = sorted([self.calc_mi(bigram, bigrams_with_freqs) for bigram in bigrams_with_freqs], reverse=True)
        return collocations