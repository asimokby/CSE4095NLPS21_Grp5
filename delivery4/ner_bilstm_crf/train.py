import bilstm
import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models.keyedvectors import KeyedVectors
import pickle
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer


def read_data():
    file_path = 'data/TWNERTC_TC_Coarse Grained NER_DomainIndependent_NoiseReduction.DUMP'
    will_parsed_row_of_corpus = 1000
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

def get_data():
    df = read_data()
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
    




##### Getting DATA & Splitting #######
# X, y = get_data()
# X_rest, X_test, y_rest, y_test = train_test_split(X,y, test_size=.2, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_rest,y_rest,test_size = 0.2, random_state=42)

########## pickling embeddings #########
embed = 'with_w2v_'
# EMBEDDING_PATH = 'turkish_embeds/model.bin'
# embeddings = KeyedVectors.load_word2vec_format(EMBEDDING_PATH, binary=True)
# with open('turkish_embeddings_w2v.pickle', 'wb') as f:
#     pickle.dump(embeddings.vectors, f)

############# unpickling embeddings ############
# with open('turkish_embeddings_w2v.pickle', 'rb') as f:
#     embeddings = pickle.load(f)

######## training model ########
# model = bilstm.Sequence(embeddings=embeddings, word_embedding_dim=100,
        # word_lstm_size=100, dropout=.88,)
# model.fit(X_train, y_train, epochs=5, x_valid=X_val, y_valid=y_val)

###### saving ######
# model.save(f'{embed}weights.h5', f'{embed}params.json', f'{embed}preprocessor.pickle')


####### scoring #######
# score = model.score(X_train, y_train)   
# print(f'f1_score-training: {score}')
# score = model.score(X_test, y_test)
# print(f'f1_score-test: {score}')



########## Loading and Testing ############
# model = bilstm.Sequence(embeddings=embeddings, word_embedding_dim=100, word_lstm_size=100, dropout=.88,)
# loaded_model = model.load(f'{embed}weights.h5', f'{embed}params.json', f'{embed}preprocessor.pickle')
# score = loaded_model.score(X_train, y_train)
# print(f'f1_score-training: {score}')
# score = loaded_model.score(X_test, y_test)
# print(f'f1_score-test: {score}')



############# resutls for presentation ###########
# model = bilstm.Sequence(embeddings=embeddings, word_embedding_dim=100, word_lstm_size=100, dropout=.88,)
# loaded_model = model.load(f'{embed}weights.h5', f'{embed}params.json', f'{embed}preprocessor.pickle')
# file_name = 'test_sentences.txt'
# x_test, y_test = prepare_test_data(file_name)
# score, y_pred = loaded_model.score(x_test, y_test)
# write_text_with_tags(y_pred, x_test)

########### Demo##########
trained_model = AutoModelForTokenClassification.from_pretrained("savasy/bert-base-turkish-ner-cased")
tokenizer = AutoTokenizer.from_pretrained("savasy/bert-base-turkish-ner-cased")
ner = pipeline('ner', model=trained_model, tokenizer=tokenizer)
res = ner("Mustafa Kemal Atatürk 19 Mayıs 1919'da Samsun'a ayak bastı.")
print('#'*10 + 'Result' + '#'*10)
for i in res:
    print(i)
print('#'*30)
