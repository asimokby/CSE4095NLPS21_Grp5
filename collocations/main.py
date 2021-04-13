import os
import re
from collocations_by_frequency import CollocationsByFrequency
from MutualInformation import MutualInformation


def clean_text(text):
    """This function will process the text and clean it before extracting the collocations
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
                break #TODO  Remove this before generating the actual results. (it is like this now to test the methods faster)
    return clean_text(donem_text)



def main():

    # Method 1: Frequency 
    collocations_by_frquency = CollocationsByFrequency()

    # Method 2: MutualInformation
    mutual_information = MutualInformation()


    # main loop
    donem_nums = range(20, 21) #TODO make this (20, 28)
    for donem_num in donem_nums:
        donem_text = load_donem_data(donem_num)

        #TODO save these as reslults for the presentation
        # Method 1
        collocations_frequency = collocations_by_frquency.get_collocations(donem_text)

        #Method 2
        collocations_mi = mutual_information.get_collocations(donem_text)


main()
