import os
import re
from collocations_by_frequency import CollocationsByFrequency
from collocations_by_mutual_information import MutualInformation
from Ttest import Ttest
from collections import Counter
from nltk.util import ngrams
import csv

def clean_text(text):
    """This function will process the t1ext and clean it before extracting the collocations
    """
    words=re.sub("[IVX]+\\.","", text) #roman numbers
    words = re.split(r'\W+', words)  #punctionation
    string_words = ' '.join((item for item in words if not item.isdigit())) #numbers
    tokens = [token for token in string_words.split(" ") if (token != "" and len(token)>1)] 
    return tokens

def load_donem_data(donem_number):
        
    """Returns all the text found in a donem directory"""
    
    donem_text = ''
    donem_dir_path = os.path.join(os.getcwd(), f'corpus/donem{donem_number}') 
    walks = os.walk(donem_dir_path)
    for walk in walks:
        path, dirs, files = walk
        for file in files:
            file_path = os.path.join(path, file)
            if not file_path.endswith('txt'): continue  # avoid reading .DS_Store files (for mac users)
            with open(file_path, 'r') as f:
                donem_text += f.read()
    return clean_text(donem_text)


def get_bigrams_with_freqs(donem_text):

    collocations = list(ngrams(donem_text, 2)) # extracting bigrams
    collocations_freqs = Counter(collocations)
    collocations_freqs = sorted(collocations_freqs.items(), key=lambda kv: kv[1], reverse=True)[:100]
    
    return dict(collocations_freqs)


def save_as_csv(data, file_name, donem_num, header):

    path = os.path.join(os.getcwd(), f'collocations/results/donem_{donem_num}/')
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+f'{file_name}.csv','w') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(header)
        for row in data:
            csv_out.writerow(row)



def main():

    collocations_by_frquency = CollocationsByFrequency() 
    mutual_information = MutualInformation()
    t_test = Ttest()

    donem_nums = range(20, 28) 
    for donem_num in donem_nums:
        donem_text = load_donem_data(donem_num)
        bigrams_with_freqs = get_bigrams_with_freqs(donem_text)

        # Method 1: Frequency 
        collocations_frequency = collocations_by_frquency.get_collocations(bigrams_with_freqs)
        table_header = ['C(w1;w2)', 'collocation', 'tag pattern']
        save_as_csv(collocations_frequency, 'freq', donem_num, table_header)

        # Method 2: MutualInformation
        collocations_mi = mutual_information.get_collocations(donem_text, bigrams_with_freqs)
        table_header = ['I(w1;w2)', 'C(w1)', 'C(w2)', 'C(w1;w2)', 'collocation']
        save_as_csv(collocations_mi, 'MI', donem_num, table_header)
        
        # Method 3: T-test
        collocations_Ttest = t_test.get_collocations(donem_text, bigrams_with_freqs)
        table_header = ['t', 'C(w1)', 'C(w2)', 'C(w1;w2)', 'collocation']
        save_as_csv(collocations_mi, 't_test', donem_num, table_header)

main()
