import os
import re
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
import pickle
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

print(str(tf.__version__))

def clean_text(text):
    """This function will process the t1ext and clean it before extracting the collocations
    """
    words=re.sub("[IVX]+\\.","", text) #roman numbers
    words = re.split(r'\W+', words)  #punctionation
    string_words = ' '.join((item for item in words if not item.isdigit())) #numbers
    # tokens = [token for token in string_words.split(" ") if (token != "" and len(token)>1)] 
    return string_words

def load_donem_data(donem_number):
        
    """Returns all the text found in a donem directory"""
    
    donem_sentences = {}
    yil_sentences = []
    donem_dir_path = os.path.join(os.getcwd(), f'corpus/donem{donem_number}') 
    walks = os.walk(donem_dir_path)
    year = 1
    for walk in walks:
        path, dirs, files = walk
        for file_ in files:
            file_path = os.path.join(path, file_)
            if not file_path.endswith('txt'): continue  # avoid reading .DS_Store files (for mac users)
            with open(file_path, 'r', encoding="UTF-8") as f:
                for line in f:
                    cleaned_line = clean_text(line.strip())
                    if len(cleaned_line) > 30:
                        yil_sentences.append(cleaned_line)
        
        if len(yil_sentences) < 1: continue
        donem_sentences[year] = yil_sentences
        year+=1   
        yil_sentences = [] 

    return donem_sentences

def test_model():
    
    max_tokens = 52
    model = keras.models.load_model('SA_lstm_with_w2v_embedd.h5')
    
    with open('turkish_tokenizer.pickle', 'rb') as f:
        
        neg_results_file = open("negative_results_w2v_donem27.results", "w")
        pos_results_file = open("positive_results_w2v_donem27.results", "w")
        
        turkish_tokenizer = pickle.load(f)
        donem_number = 27
        donem_sentences = load_donem_data(donem_number)

        neg_results = {}
        pos_results = {}
        
        donem_results = {}
        
        for year, sentences in donem_sentences.items():
            
            print(f'Processing year: {year}')
            tokens = turkish_tokenizer.texts_to_sequences(sentences)
        
            if len(tokens) == 0: continue
        
            tokens_pad = pad_sequences(tokens, maxlen=max_tokens)
            res = model.predict(tokens_pad)
        
            pos = 0
            neg = 0
        
            neg_results[year] = {}
            pos_results[year] = {}

            for idx, prob in enumerate(res):

                if prob[0] > 0.5:
                    pos+=1
                    pos_results[year][idx] = sentences[idx]
                    pos_results_file.write(f"{year},{sentences[idx]}\n")
                    pos_results_file.flush()
                else: 
                    neg+=1
                    neg_results[year][idx] = sentences[idx]
                    neg_results_file.write(f"{year},{sentences[idx]}\n")
                    neg_results_file.flush()
            
            pos_neg_count = {'pos':pos, 'neg':neg}
            donem_results[year] = pos_neg_count
        
        neg_results_file.close()
        pos_results_file.close()

    with open(f'donem_{donem_number}_neg_results.pickle', 'wb') as f:
        pickle.dump(neg_results, f)

test_model()

def load_results(pk_file_name):

    tokenizer_file = open('turkish_tokenizer.pickle', "rb")
    tokenizer = pickle.load(tokenizer_file)

    pk_file = open(pk_file_name, "rb")
    res = pickle.load(pk_file)
    pk_file.close()
    with open(str(pk_file_name)[0:(str(pk_file_name).rindex(".")-1)]+".results", "w") as results_file:
        results_file.write(str(res))
        results_file.flush()
        results_file.close()

"""load_results("donem_27_results_keras.pickle")
load_results("donem_27_results_w2v.pickle")"""




def test_with_sentence(sentence):
    
    max_tokens = 52
    model = keras.models.load_model('SA_lstm_with_keras_embedd.h5')
    
    tokenizer_file = open('turkish_tokenizer.pickle', "rb")
    tokenizer = pickle.load(tokenizer_file)

    tokens = tokenizer.texts_to_sequences(sentence)

    tokens_pad = pad_sequences(tokens, maxlen=max_tokens)
    res = model.predict(tokens_pad)
    pos = 0
    neg = 0
    for prob in res:
        if prob[0] > 0.5: return "positive"
        else: return "negative"

