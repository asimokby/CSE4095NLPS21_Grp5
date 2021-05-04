from nltk import ngrams
from os.path import join
import os
import re
import collections
import plotly.express as px
# to save image also kaleido should be installed
# pip install -U kaleido
import pandas as pd
from concurrent.futures import ThreadPoolExecutor 

stopwords = []

# parse stopwords from
def parse_stopwords():
    print("Parsing stopwords..")
    global stopwords
    stopwords_src_path = join(os.getcwd(), "stopwords.txt")
    f = open(stopwords_src_path, encoding="UTF-8")
    stopwords = [ word[:-1] for word in f.readlines()]

# preprocess a text to clean space characters, roman numbers and stopwords (optional)
def preprocess_text(text, with_stopwords):
    global stopwords

    words=re.sub("[IVX]+\\.","", text) #roman numbers
    words = re.split(r'\W+', words)  #punctionation
    string_words = ' '.join((item for item in words if not item.isdigit())) #numbers
    
    tokens = []

    for token in string_words.split(" "):
        token = token.lower().strip()
        if (token != "") and (len(token) > 1):
            # dont check the token is a stopword or not
            if with_stopwords:
                tokens.append(token)
            else:
                # if it is a stopword don't append to list
                if token not in stopwords:
                    tokens.append(token)
                
    return tokens

# get all words as preprocessed in a donem with or without stopwords
def get_donem_words(donem_number, with_stopwords):
    
    print(f"Processing donem {donem_number}")

    all_words_in_specified_donem = []

     # base src path ./src
    base_src_path = join(os.getcwd(), "src")
    # donem src path ./src/donem20
    donem_src_path = join(base_src_path, "donem"+str(donem_number))
    donem = os.listdir(donem_src_path)
    count = 0
    for yil in donem:
        count += 1
        print(f"Processing yıl {str(count)}")
        # yıl src path ./src/donem20/yıl1
        yil_src_path = join(donem_src_path, yil)
        text_files = os.listdir(yil_src_path)
        
        for file_name in text_files:
            # txt src path ./src/donem20/yıl1/32.txt
            text_file_src_path = join(yil_src_path, file_name)
            file_handler = open(text_file_src_path, encoding='UTF-8')
            will_processed_text = file_handler.read()
            file_handler.close()
            all_words_in_file = preprocess_text(will_processed_text, with_stopwords)
            all_words_in_specified_donem += all_words_in_file
    
    return all_words_in_specified_donem

def create_n_grams(words, donem_spec):
    
    figure_plotting_keys = ["unigrams", "bigrams", "trigrams"]
    
    V = ...

    # create all n-grams in the loop ( unigram -> 1, bigram -> 2, trigram -> 3 )
    # plot them and svae the plotted figures in ./out path
    for i in range(1,4):
        current_n_gram_type = figure_plotting_keys[i-1]
        print("Creating " + str(current_n_gram_type))
        # with stopwords

        ngram_in_specified_type_with_stopwords = list(ngrams(words,i))
        l_total_n_gram_instances = len(ngram_in_specified_type_with_stopwords)
        # count frequencies N -> l_total
        ngram_freq = collections.Counter(ngram_in_specified_type_with_stopwords) # unigram
        
        if i==1:
            V=len(ngram_freq)

        ngram_freq_sorted = sorted(ngram_freq.items(), key=lambda kv: kv[1], reverse=True)
        
        df_n_gram_freq = dict()

        for item in ngram_freq_sorted:
            df_n_gram_freq[str(item[0])] = item[1]
        
        ngram_freq_sorted_10 = ngram_freq_sorted[0:10] 
        # plot them
        print(str(ngram_freq_sorted_10))
        
        df_dict = dict()
        for item in ngram_freq_sorted_10:
            df_dict[str(item[0])] = item[1]

        df = pd.Series(df_dict).to_frame()
        fig = px.bar(df, 
                    labels={'x':'words', 'y':'frequencies of '+ current_n_gram_type}, 
                    title= "Top 10 frequent " + current_n_gram_type + " of Donem " + donem_spec,
                    width=960,
                    height=540)
        
        # set out path for saving the figure of the donem
        out_path_graphs_base = join(os.getcwd(), "out","graphs", "donem_"+donem_spec )

        if not os.path.exists(out_path_graphs_base):
            os.mkdir(out_path_graphs_base)

        out_path_current_donem = join(out_path_graphs_base, current_n_gram_type+".png")
        fig.write_image(out_path_current_donem)

        save_as_csv(df_dict, "donem_"+donem_spec, current_n_gram_type)

        # c() / N
        calculate_MLE(df_n_gram_freq, l_total_n_gram_instances, V, "donem_"+donem_spec, current_n_gram_type)
        calculate_LAP(df_dict, l_total_n_gram_instances, V, i ,"donem_"+donem_spec, current_n_gram_type)
        calculate_LDSTN(df_dict, l_total_n_gram_instances, V, i ,"donem_"+donem_spec, current_n_gram_type)
        calculate_JP(df_dict, l_total_n_gram_instances, V, i ,"donem_"+donem_spec, current_n_gram_type)

        #calculate_GT(ngram_freq, l_total_n_gram_instances, V , i, "donem_"+donem_spec, current_n_gram_type )

def calculate_GT(freq_dict, N, V, n, donem, file_name):

    Nr_ngrams = {n:[k for k in freq_dict.keys() if freq_dict[k] == n] for n in set(freq_dict.values())}
    r_adjustedr_pgt_nr = dict()
    
    # unseen prob
    prob_for_all_unseen = len(Nr_ngrams[1])/N #TODO use this in the presentation
    r_adjustedr_pgt_nr[0] = (0, prob_for_all_unseen, prob_for_all_unseen/((V**n) - N)) #TODO replace 2 by n-gram's value of n
    
    for r in Nr_ngrams:
        k = r + 1
        if k not in Nr_ngrams: k = r # case when r does not exist or r is the highest value. 
        adjusted_r = (r+1) * (len(Nr_ngrams[k])/N)
        prob_gt = adjusted_r/N
        r_adjustedr_pgt_nr[r] = (r, adjusted_r, prob_gt, len(Nr_ngrams[r]))

    save_as_csv(r_adjustedr_pgt_nr, donem, file_name=file_name+"_gt")

# Maximum Likelihood Estimation
def calculate_MLE(freq_dict, total_ngram_instances, V, donem, file_name):

    r_d = dict()
    mle = dict()

    summ_all_p = 0

    count = 0

    for key in freq_dict.keys():
        
        freq = float(freq_dict[key])
        mle_p = float(freq / total_ngram_instances)

        summ_all_p += mle_p

        if freq in r_d:
            r_d[freq] += mle_p
        else:
            r_d[freq] = mle_p

        if count < 10:
            mle[key] = f"{str(freq)}, {str(mle_p)}"
        
        count+=1

    rd_last_10 = list(r_d)[-10:]
    rd_will_save = dict()
    
    for item in rd_last_10:
        rd_will_save[item]=r_d[item]

    save_as_csv(rd_will_save, donem, file_name=file_name+"_MLE_r")
    save_as_csv(mle, donem, file_name=file_name+"_MLE")

# Laplace Law
def calculate_LAP(freq_dict, total_ngram_instances, V, n, donem, file_name ):

    B = V ** n

    lap = dict()
    r_d = dict()
    
    

    # lap_est = (c + 1) / N + B 
    for key in freq_dict:
        val = float(freq_dict[key])
        lap[key] = f"{str(val)},{str(float( (val + 1) / (total_ngram_instances + B)))}"

    save_as_csv(lap, donem, file_name=file_name+"_LAP")

# Lidstone's Law
def calculate_LDSTN(freq_dict, total_ngram_instances, V, n, donem, file_name ):

    lambda_ = 0.25

    B = V ** n

    lap = dict()

    for key in freq_dict:
        val = float(freq_dict[key])
        lap[key] = f"{str(val)},{str(float( (val + lambda_) / (total_ngram_instances + (B*lambda_))))}" 

    save_as_csv(lap, donem, file_name=file_name+"_LDSTN")

# Jeffreys Perks Law
def calculate_JP(freq_dict, total_ngram_instances, V, n, donem, file_name ):

    lambda_ = 0.5
    B = V ** n

    lap = dict()

    for key in freq_dict:
        val = float(freq_dict[key])
        lap[key] = f"{str(val)},{str(float( (val + lambda_) / (total_ngram_instances + (B*lambda_))))}"

    save_as_csv(lap, donem, file_name=file_name+"_JP")

def save_as_csv(est_dict, donem, file_name):

    out_path_base = join(os.getcwd(), "out","est_stats", donem )

    if not os.path.exists(out_path_base):
        os.mkdir(out_path_base)

    out_path = join(out_path_base, file_name+".csv")

    with open(out_path, "w", encoding="UTF-8") as f:
        for key in est_dict.keys():
            f.write("%s,%s\n"%(key,est_dict[key]))
        f.flush()
        f.close()

def main():
    parse_stopwords()

    all_words_in_corpus_with_stopwords = []

    all_words_in_corpus_without_stopwords = []

    for index in range(0,8):
        # initialize the donem number combining with 2 (ex: 20, 21, 22)
        donem_number = "2" + str(index) 

        # with stopwords
        all_donem_words_with_stopwords = get_donem_words(donem_number=donem_number, with_stopwords=True)
        # add donem words to general with stopwords corpus list (with stopwords)
        all_words_in_corpus_with_stopwords += all_donem_words_with_stopwords

        # without stopwords
        all_donem_words_without_stopwords = get_donem_words(donem_number=donem_number, with_stopwords=False)
        # add donem words to general wtihout stopwords corpus list (without stopwords)
        all_words_in_corpus_without_stopwords += all_donem_words_without_stopwords
        
        create_n_grams(all_donem_words_with_stopwords, donem_number)
        create_n_grams(all_donem_words_without_stopwords, donem_number+"_without_stopwords")

    create_n_grams(all_words_in_corpus_with_stopwords, "all_donems")
    create_n_grams(all_words_in_corpus_without_stopwords, "all_donems_without_stopwords")
    
    print("all process is done !")
    #print(str(all_words_in_corpus_without_stopwords))

main()
