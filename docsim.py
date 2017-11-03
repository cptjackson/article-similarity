# Things to do:

# Read in files
# Parse files: split into words
# Find word obscurity and score them
# Find out how many words they have in common

import spacy, textract, os, re
import numpy as np
from collections import Counter
import math
from textblob import TextBlob as tb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Cluster algorithm
def cluster(tfidf_matrix):

    num_clusters = 10
    km = KMeans(n_clusters=num_clusters)
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()

    print(clusters)

# Textblob functions
def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

# Extract text from files and return list of strings
def extract_text(path,method):

    text_list = []

    # Pattern match files in directory
    os.chdir(path)
    contents = os.listdir()
    pattern = r'ID-\d+\.docx*'


    # Pull text from files and make unicode
    if method == 'spacy':
        [text_list.append(nlp(textract.process(item).decode('utf-8')))
                      for item in contents
                      if re.search(pattern,item)]
    else:
        [text_list.append(textract.process(item).decode('utf-8'))
                      for item in contents
                      if re.search(pattern,item)]

    return text_list


def extract_title(text_list):

    title = []
    text = []
    title_pattern = r'^.+\n'
    for item in text_list:
        title_str = re.search(title_pattern,item).group()
        title.append(title_str[0:-1])
        text.append(item[len(title_str):])

    return text,title


# Build similarity matrix
def build_sim_matrix(text,method):

    #words = [[token.text for token in row if token.is_stop != True
                        #and token.is_punct != True] for row in text]

    #print(words)
    #print(words[0])
    #print(len(text))

    sim_mat = np.zeros((len(text), len(text)))

    if method == 'spacy':

        for row, doc1 in enumerate(text):

            for col, doc2 in enumerate(text):

                # Use Spacy's 'similarity' method
                sim_mat[row,col] = doc1.similarity(doc2)

    elif method == 'TF-IDF':

                # Use TF-DIF
                vect = TfidfVectorizer(min_df=1,stop_words='english')
                tfidf = vect.fit_transform(text)
                sim_mat = (tfidf * tfidf.T).A

    return tfidf


if __name__ == '__main__':

    # do some stuffs!
    nlp = spacy.load('en', parser=False)
    path = '/Users/Carl/Documents/9thCo/Articles/'

    # Custom stop words
    nlp.vocab["\t"].is_stop = True
    nlp.vocab["\n\n"].is_stop = True
    nlp.vocab["\n\n\t"].is_stop = True
    nlp.vocab["n\u2019t"].is_stop = True
    nlp.vocab["\u2019s"].is_stop = True
    nlp.vocab["\u2019ll"].is_stop = True

    # Extract files
    texts = extract_text(path,'TF-IDF')
    texts,titles = extract_title(texts)

    #texts = extract_text(path,'spacy')

    # Build similarity matrix
    tfidf = build_sim_matrix(texts,'TF-IDF')
    #similarity_matrix2 = build_sim_matrix(texts,'spacy')

    #dist_matrix = 1-similarity_matrix

    #print(dist_matrix)

    #print(max(similarity_matrix[0,1:]))
    #print(1-max(similarity_matrix2[0,1:]))

    # Cluster
    cluster(tfidf)
