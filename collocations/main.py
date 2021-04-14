import os
import re
from collocations_by_frequency import CollocationsByFrequency
from collocations_by_mutual_information import MutualInformation
from Ttest import Ttest
from collections import Counter
from nltk.util import ngrams
import pickle
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
            with open(file_path, 'r', encoding="UTF-8") as f:
                donem_text += f.read()
    return clean_text(donem_text)


def get_bigrams_with_freqs(donem_text):

    collocations = list(ngrams(donem_text, 2)) # extracting bigrams
    collocations_freqs = Counter(collocations)
    collocations_freqs = sorted(collocations_freqs.items(), key=lambda kv: kv[1], reverse=True)[0:1000]
    
    return dict(collocations_freqs)


def save_as_csv(data, file_name, donem_num, header, all_donems=False):

    if all_donems:
        path = os.path.join(os.getcwd(), f'collocations/results/whole_corpus/')
    else: 
        path = os.path.join(os.getcwd(), f'collocations/results/donem_{donem_num}/')

    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+f'{file_name}.csv','w', encoding="UTF-8") as out:
        csv_out=csv.writer(out)
        csv_out.writerow(header)
        for row in data:
            csv_out.writerow(row)



def main():

    out_base = os.path.join(os.getcwd(), "collocations","out")

    collocations_by_frquency = CollocationsByFrequency() 
    mutual_information = MutualInformation()
    t_test = Ttest()

    donem_nums = range(20, 28) 
    all_donem_text = []
    table_header_freq = ['C(w1;w2)', 'collocation', 'tag pattern']
    table_header_mi = ['I(w1;w2)', 'C(w1)', 'C(w2)', 'C(w1;w2)', 'collocation']
    table_header_ttest = ['t', 'C(w1)', 'C(w2)', 'C(w1;w2)', 'collocation']
    
    for donem_num in donem_nums:

        print(f"Donem {str(donem_num)} processing..")
        donem_text = load_donem_data(donem_num)

        all_donem_text.extend(donem_text)

        """print(f"Donem {str(donem_num)} bigrams extracting processing..")
        bigrams_with_freqs = get_bigrams_with_freqs(donem_text)"""

        if donem_num != 27:
            continue
        print("i am in asem")
        pk_file = open(out_base+os.sep+"donem_"+str(donem_num)+".pk", "rb")
        bigrams_with_freqs = pickle.load(pk_file)
        
        print(f"Donem {str(donem_num)} frequency processing..")
        # Method 1: Frequency 
        collocations_frequency = collocations_by_frquency.get_collocations(bigrams_with_freqs)
        save_as_csv(collocations_frequency, 'freq', donem_num, table_header_freq)

        print(f"Donem {str(donem_num)} MutualInformation processing..")
        #Method 2: MutualInformation
        collocations_mi = mutual_information.get_collocations(donem_text, bigrams_with_freqs)
        save_as_csv(collocations_mi, 'MI', donem_num, table_header_mi)
        
        print(f"Donem {str(donem_num)} T-test processing..")
        # Method 3: T-test
        collocations_Ttest = t_test.get_collocations(donem_text, bigrams_with_freqs)
        save_as_csv(collocations_Ttest, 't_test', donem_num, table_header_ttest)

    # All donems together
    """ print("Collocations will start..")
    all_donems_bigrams_with_freqs = get_bigrams_with_freqs(all_donem_text)
    
    print("Saving all bigrams in pickle..")
    pk_file = open(out_base+os.sep+"donem_all.pk", "wb")
    pickle.dump(bigrams_with_freqs,file=pk_file)"""
    
    pk_file = open(out_base+os.sep+"donem_all.pk", "rb")
    all_donems_bigrams_with_freqs = pickle.load(pk_file)

    print("Collocations process is finished..")
    collocations_frequency = collocations_by_frquency.get_collocations(all_donems_bigrams_with_freqs)
    save_as_csv(collocations_frequency, 'freq', 0, table_header_freq, all_donems=True)
    print("Frequencies extraction finished..")
    collocations_mi = mutual_information.get_collocations(all_donem_text, all_donems_bigrams_with_freqs)
    save_as_csv(collocations_mi, 'MI', 0, table_header_mi, all_donems=True)
    print("Mutual Information extraction finished..")
    collocations_Ttest = t_test.get_collocations(all_donem_text, all_donems_bigrams_with_freqs)
    save_as_csv(collocations_Ttest, 't_test', 0, table_header_ttest, all_donems=True)
    print("T-test extraction finished..")

main()
