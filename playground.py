from nltk import ngrams
from os.path import join
import os
import re

# test

text = "bu burada ve ali bu ali\n"

stop_words_src_path = join(os.getcwd(), "stop_words.txt")
f = open(stop_words_src_path, encoding="UTF-8")
stop_words = [ word[:-1] for word in f.readlines()]

words=re.sub("[IVX]+\\.","", text) #roman numbers
words = re.split(r'\W+', words)  #punctionation
string_words = ' '.join((item for item in words if not item.isdigit())) #numbers
tokens = [token.lower().strip() for token in string_words.split(" ") if (token != "" and len(token)>1 and token not in stop_words)]

unigram = list(ngrams(tokens,1))

print(str(unigram))

# In here output is correct there is no stop words but in the implementation there are still stop words in output
# and I added the same things to implementation with here