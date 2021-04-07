from nltk import ngrams
from os.path import join
import os
import re
import collections
import plotly.express as px
# to save image also kaleido should be installed
# pip install -U kaleido
import pandas as pd

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

    # create all n-grams in the loop ( unigram -> 1, bigram -> 2, trigram -> 3 )
    # plot them and svae the plotted figures in ./out path
    for i in range(1,4):
        current_n_gram_type = figure_plotting_keys[i-1]
        print("Creating " + str(current_n_gram_type))
        # with stopwords
        ngram_in_specified_type_with_stopwords = list(ngrams(words,i))
        # count frequencies
        ngram_freq = collections.Counter(ngram_in_specified_type_with_stopwords)
        ngram_freq = sorted(ngram_freq.items(), key=lambda kv: kv[1], reverse=True)[0:10]
        # plot them
        print(str(ngram_freq))
        df_dict = dict()
        for item in ngram_freq:
            df_dict[str(item[0])] = item[1]

        df = pd.Series(df_dict).to_frame()
        fig = px.bar(df, 
                    labels={'x':'words', 'y':'frequencies of '+ current_n_gram_type}, 
                    title= "Top 10 frequent " + current_n_gram_type + " of Donem " + donem_spec,
                    width=960,
                    height=540)
        
        # set out path for saving the figure of the donem
        out_path_current_donem = join(os.getcwd(), "out", "donem_"+donem_spec.replace(" ","_")+"_"+current_n_gram_type+".png")
        fig.write_image(out_path_current_donem)

def main():
    parse_stopwords()

    all_words_in_corpus_with_stopwords = []

    all_words_in_corpus_without_stopwords = []

    for index in range(0,2):
        print(str(index))
        # initialize the donem number combining with 2 (ex: 20, 21, 22)
        donem_number = "2" + str(index) 
        print(donem_number)

        # with stopwords
        all_donem_words_with_stopwords = get_donem_words(donem_number=donem_number, with_stopwords=True)
        # add donem words to general with stopwords corpus list (with stopwords)
        all_words_in_corpus_with_stopwords += all_donem_words_with_stopwords

        # without stopwords
        all_donem_words_without_stopwords = get_donem_words(donem_number=donem_number, with_stopwords=False)
        # add donem words to general wtihout stopwords corpus list (without stopwords)
        all_words_in_corpus_without_stopwords += all_donem_words_without_stopwords

        create_n_grams(all_donem_words_with_stopwords, donem_number)
        create_n_grams(all_donem_words_without_stopwords, donem_number+" without stopwords")

    create_n_grams(all_words_in_corpus_with_stopwords, "all donems")
    create_n_grams(all_words_in_corpus_without_stopwords, "all donems without stopwords")
    
    print("all process is done !")
    #print(str(all_words_in_corpus_without_stopwords))

main()
