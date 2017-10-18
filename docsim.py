# Things to do:

# Read in files
# Parse files: split into words
# Find word obscurity and score them
# Find out how many words they have in common

import spacy, textract, os, re
import numpy as np
from collections import Counter

# Extract text from files and return list of strings
def extract_text(path):

    text_list = []

    # Pattern match files in directory
    os.chdir(path)
    contents = os.listdir()
    pattern = r'ID-\d+\.docx*'

    # Pull text from files and make unicode
    [text_list.append(nlp(textract.process(item).decode('utf-8')))
                      for item in contents
                      if re.search(pattern,item)]

    return text_list


# Build similarity matrix
def build_sim_matrix(text,method):

    sim_mat = np.zeros((len(text), len(text)))

    if method == 'built-in':

        # Use Spacy's 'similarity' method
        for row, item1 in enumerate(text):

            for col, item2 in enumerate(text):

                sim_mat[row,col] = item1.similarity(item2)

    elif method == 'TF-IDF':

        # Use TF-DIF
        words = [[token.text for token in row if token.is_stop != True
                            and token.is_punct != True] for row in text]
        word_freq = [Counter(row) for row in words]

        print(word_freq[0])


    return sim_mat


if __name__ == '__main__':

    # do some stuffs!
    nlp = spacy.load('en', parser=False)
    path = '/Users/Carl/Documents/9thCo/Articles/'

    # Extract files
    texts = extract_text(path)

    # Build similarity matrix
    similarity_matrix = build_sim_matrix(texts,'TF-IDF')

    print(similarity_matrix)
