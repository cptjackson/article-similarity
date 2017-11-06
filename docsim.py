import spacy, textract, os, re, math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def token_lemma(text):

    tokens = [token.text for token in text if token.is_stop != True
                         and token.is_punct != True]

    lemmas = [token.lemma_ for token in text if token.is_stop != True
                         and token.is_punct != True]

    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    #tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    filtered_lemmas = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for ind,token in enumerate(tokens):
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token.lower())
            filtered_lemmas.append(lemmas[ind])

    return [filtered_tokens,filtered_lemmas]

# Cluster algorithm
def cluster(tfidf_matrix,num_clusters):

    km = KMeans(n_clusters=num_clusters)
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()

    return [clusters,km]


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

    terms = vect.get_feature_names()

    return [tfidf,terms,sim_mat]


if __name__ == '__main__':

    # Load nlp and set path
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

    spacy_texts = extract_text(path,'spacy')

    # Build similarity matrix
    [tfidf,terms,sim_mat] = build_sim_matrix(texts,'TF-IDF')
    #similarity_matrix2 = build_sim_matrix(texts,'spacy')

    # Tokens
    #tokenize_only(spacy_texts[0])

    totalvocab_tokenized = []
    totalvocab_lemmatized = []
    for i in spacy_texts:
        #allwords_lemmatized = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
        #totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list

        [allwords_tokenized, allwords_lemmatized] = token_lemma(i)
        totalvocab_tokenized.extend(allwords_tokenized)
        totalvocab_lemmatized.extend(allwords_lemmatized)

    vocab_frame = pd.DataFrame({'words': totalvocab_tokenized},index = totalvocab_lemmatized)
    print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')

    #print(vocab_frame.iloc[4147])
    #dist_matrix = 1-similarity_matrix

    #print(dist_matrix)

    #print(max(similarity_matrix[0,1:]))
    #print(1-max(similarity_matrix2[0,1:]))

    # Cluster analysis
    num_clusters = 10
    [clusters,km] = cluster(tfidf,num_clusters)
    articles = { 'title': titles, 'cluster': clusters }
    frame = pd.DataFrame(articles, index = [clusters] , columns = ['title', 'cluster'])
    print("Number of items per cluster: ")
    print(frame['cluster'].value_counts())

    terms_per_cluster = 6

    print("Top terms per cluster:")
    print()
    #sort cluster centers by proximity to centroid
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    for i in range(num_clusters):
        print("Cluster %d words: " % i, end='')

        for ind in order_centroids[i, :terms_per_cluster]: #replace 6 with n words per cluster
            #print(vocab_frame.ix[terms[ind].split(' ')])
            print(terms[ind], end=' ')
            #print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0], end=',')
        print() #add whitespace
        print() #add whitespace

        print("Cluster %d titles:" % i, end='')
        for title in frame.ix[i]['title'].values.tolist():
            print(' %s,' % title, end='')
        print() #add whitespace
        print() #add whitespace

    print()
    print()
