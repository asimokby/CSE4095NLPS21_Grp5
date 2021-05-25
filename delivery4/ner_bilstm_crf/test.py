import bilstm
import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models.keyedvectors import KeyedVectors
import os
import pickle5 as  pk

def read_data(line_count):
    file_path = os.getcwd() + os.sep + 'TWNERTC_TC_Coarse Grained NER_DomainIndependent_NoiseReduction.DUMP'
    will_parsed_row_of_corpus = line_count
    f = open(file_path, "r")

    line_count = 0

    data = dict()
    data["sentence_id"] = list()
    data["tag"] = list()
    data["word"] = list()

    sentence_id_arr = data["sentence_id"]
    tag_arr = data["tag"]
    word_arr = data["word"]


    for line in f.readlines():

        if line_count > will_parsed_row_of_corpus:
            break

        line_count += 1
        # each line seperated by ht (horizontal tabs)
        splitted = line.split("\t")
        
        if len(splitted) ==3:
            tag_split = splitted[1].split(" ")
            word_split = splitted[2].split(" ")
        
        for tag, word in zip(tag_split, word_split):
            word = word.strip()
            if word[len(word)-1] == "\n":
                word = word[:-1]
            
            sentence_id_arr.append(line_count)
            tag_arr.append(tag)
            word_arr.append(word)

    df = pd.DataFrame(data, columns=["sentence_id", "tag", "word"])
    return df

class SentenceGetter(object):
    
    def __init__(self, dataset):
        self.n_sent = 1
        self.dataset = dataset
        self.empty = False
        agg_func = lambda s: [(w, t) for w,t in zip(s["word"].values.tolist(),
                                                        s["tag"].values.tolist())]
        self.grouped = self.dataset.groupby("sentence_id").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

def get_data(line_count):
    df = read_data(line_count)
    getter = SentenceGetter(df)

    x_train = []
    y_train = []
    for sentence in getter.sentences:
        words = []
        tags = []
        for word, tag in sentence:
            words.append(word)
            tags.append(tag)
        x_train.append(words) 
        y_train.append(tags)
    return x_train, y_train


X, y = get_data(50000)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


def train_model():

    embeddings = get_embeddings() 

    print("Pickle word embeddings..")

    model = bilstm.Sequence(embeddings=embeddings.vectors, word_embedding_dim=300, word_lstm_size=20, dropout=0.8)
    model.fit(X_train, y_train, epochs=20)

    res = model.analyze('Corina Casanova , İsviçre Federal Şansölyesidir .')
    print(res)

    score = model.score(X_test, y_test)
    print(f'f1_score: {score}')

    model.save('weights_word_lstm_20_dropout_0.8.h5', 'params.json', 'preprocessor.pickle')
    
    print("Saved trained model..")

def load_model(test_w_our_sentences = False):

    embeddings = get_embeddings()

    print("Loading pre-trained model..")
    # reload model again
    model = bilstm.Sequence(embeddings=embeddings.vectors, word_embedding_dim=300, word_lstm_size=20, dropout=0.8)
    loaded_model = model.load('weights_word_lstm_20_dropout_0.8.h5', 'params.json', 'preprocessor.pickle')
    
    """res  = model.analyze("ankara nasıl bir yer")
    print(str(res))"""

    if test_w_our_sentences:
        X_test, Y_test = prepare_test_data("test_sentences.txt")
        score, y_predict = loaded_model.score(X_test, Y_test)
        write_text_with_tags(y_predict, X_test)
        #print(y_predict)
    else:
        score, y_predict = loaded_model.score(X_train, y_train)
        print(score)
    """print(f"{y_predict}")
    print(f'{score}')"""

def write_text_with_tags(predicitions, X_test):

    results = open("prettified_resuts.txt", "w")

    for sentence_prediction, sentence in zip(predicitions, X_test):
        for idx , (tag, word) in  enumerate(zip(sentence_prediction, sentence)):
            
            if idx != 0:
                results.write(" ")
            results.write(word)
        
            if tag != "O":
                results.write(f" ({tag})")
        
        results.write("\n")
        results.flush()
    
    results.close()
    
def prepare_test_data(file_name):

    test_file = open(file_name, "r")

    X_test = []
    Y_test = []

    for line in test_file.readlines():
        split_line = line.split()
        words = []
        tags = []
        for word in split_line:
            # remove new line character with strip
            words.append(word.strip())
            tags.append("O")

        X_test.append(words)
        Y_test.append(tags)
    
    return X_test, Y_test

def get_embeddings():

    """pickle_path = os.getcwd() + os.sep + "word_embedding_vectors.pkl"""

    """if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as pk:
            return pk.load(pk)"""

    EMBEDDING_PATH = os.getcwd() +os.sep +  "glove_vectors.txt"
    print("Loading pre-trained word embeddings..")
    embeddings = KeyedVectors.load_word2vec_format(EMBEDDING_PATH)

    """with open(pickle_path, "wb") as pk:
        pk.dump(embeddings, pk)"""

    return embeddings

if __name__ == "__main__":
    #train_model()
    load_model(test_w_our_sentences=True)