from collections import Counter
import math

class Ttest():
    
    def calc_t_value(self, bigram, bigrams_with_freqs):
        w1_freq = self.word_freq[bigram[0]]
        w2_freq = self.word_freq[bigram[1]]
        bigram_frequncy = bigrams_with_freqs[bigram]
        normlize = self.words_count
        expected_mean = (w1_freq/normlize)*(w2_freq/normlize)
        observed_mean = bigram_frequncy/normlize
        variance = observed_mean
        t_value = (observed_mean - expected_mean)/math.sqrt(observed_mean/normlize)
        return (t_value, w1_freq, w2_freq, bigram_frequncy, bigram)

    def get_collocations(self, donem_text, bigrams_with_freqs):
        collocations = []
        self.words_count = len(donem_text)
        self.word_freq = Counter(donem_text)
        collocations = sorted([self.calc_t_value(bigram, bigrams_with_freqs) for bigram in bigrams_with_freqs], reverse=True)
        return collocations