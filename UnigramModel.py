import os
import nltk
import ssl
#from transformers import BertTokenizer
import math
from nltk.tokenize import word_tokenize
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')

# might be useful?
from collections import Counter


def read_files_in_directory(directory_path):
    # key: tokens value: their frequency in all songs belonging to a genre
    dic_term_frequency = {}
    
    for file in os.listdir(directory_path):
         filepath = directory_path + "/"+file
        #for file in os.listdir(filepath):
         with open(filepath, 'r') as rfile:
            for line in rfile:
                current_line = line.strip()

                tokens = word_tokenize(current_line)
                for word in tokens:
                 dic_term_frequency[word] = dic_term_frequency.get(word, 0) + 1


    return dic_term_frequency


def freq_to_prob(dic_term_frequency):

    total_words = len(dic_term_frequency)
    dic_term_prob = {word: count +1/ total_words+1 for word, count in dic_term_frequency.items()}
    #print(dic_term_prob)
    return dic_term_prob


def calculate_probability(dic_term_prob, input_text):
    prob = 0.0
    tokens = word_tokenize(input_text.lower())
    #log_probability_sum = 0.0

    for token in tokens:
        token = token.capitalize()  # Capitalize the token to match the dictionary keys
        if token in dic_term_prob:
             prob += math.log(dic_term_prob[token])
        else:
            print(f"Token: {token} not found in the dictionary")

    return prob


def main():
  list = ["Pop","Rock","Rap","Metal","Country","Blues"]
  list1 = []
  sentance = """
     How they dance in the courtyard
            Sweet summer sweat
            Some dance to remember
            Some dance to forget
            So I called up the Captain
            "Please bring me my wine"
            He said, "We haven't had that spirit here
            Since 1969"""
  for i in range(0,len(list)):
   list1.append(calculate_probability(freq_to_prob(read_files_in_directory("TM_CA1_Lyrics/" + list[i]) ),sentance))
  
  print(list,"\n",list1)
  max_value = max(list1)
  max_index = list1.index(max_value)
  print("MAx:",list[max_index])
if __name__ == "__main__":
    main()