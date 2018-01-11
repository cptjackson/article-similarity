import spacy, textract, os, re, math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt, mpld3
import matplotlib as mpl
from sklearn.manifold import MDS


#define custom toolbar location
class TopToolbar(mpld3.plugins.PluginBase):
    """Plugin for moving toolbar to top of figure"""

    JAVASCRIPT = """
    mpld3.register_plugin("toptoolbar", TopToolbar);
    TopToolbar.prototype = Object.create(mpld3.Plugin.prototype);
    TopToolbar.prototype.constructor = TopToolbar;
    function TopToolbar(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };

    TopToolbar.prototype.draw = function(){
      // the toolbar svg doesn't exist
      // yet, so first draw it
      this.fig.toolbar.draw();

      // then change the y position to be
      // at the top of the figure
      this.fig.toolbar.toolbar.attr("x", 150);
      this.fig.toolbar.toolbar.attr("y", 400);

      // then remove the draw function,
      // so that it is not called again
      this.fig.toolbar.draw = function() {}
    }
    """
    def __init__(self):
        self.dict_ = {"type": "toptoolbar"}


def plot_web_stuff(xs,ys,clusters,titles,labels):

    #set up colors per clusters using a dict
    cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

    #set up cluster names using a dict
    cluster_names = {}
    for i in range(5):
        cluster_names[i] = labels[i]

    #create data frame that has the result of the MDS plus the cluster numbers and titles
    df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles))

    #group by cluster
    groups = df.groupby('label')

    #define custom css to format the font and to remove the axis labeling
    css = """
    text.mpld3-text, div.mpld3-tooltip {
      font-family:Arial, Helvetica, sans-serif;
    }

    g.mpld3-xaxis, g.mpld3-yaxis {
    display: none; }

    svg.mpld3-figure {
    margin-left: -200px;}
    """

    # Plot
    fig, ax = plt.subplots(figsize=(14,6)) #set plot size
    ax.margins(0.03) # Optional, just adds 5% padding to the autoscaling

    #iterate through groups to layer the plot
    #note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
    for name, group in groups:
        points = ax.plot(group.x, group.y, marker='o', linestyle='', ms=18,
                         label=cluster_names[name], mec='none',
                         color=cluster_colors[name])
        ax.set_aspect('auto')
        labels = [i for i in group.title]

        #set tooltip using points, labels and the already defined 'css'
        tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels,
                                           voffset=10, hoffset=10, css=css)
        #connect tooltip to fig
        mpld3.plugins.connect(fig, tooltip, TopToolbar())

        #set tick marks as blank
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])

        #set axis as blank
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)


    ax.legend(numpoints=1) #show legend with only one dot

    mpld3.show() #show the plot
    #mpld3.display() #use this for jupyter notebook

    #uncomment the below to export to html
    #html = mpld3.fig_to_html(fig)
    #print(html)

def plot_stuff(xs,ys,clusters,titles,labels):

    #set up colors per clusters using a dict
    cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

    #set up cluster names using a dict
    cluster_names = {}
    for i in range(5):
        cluster_names[i] = labels[i]

    #create data frame that has the result of the MDS plus the cluster numbers and titles
    df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles))

    #group by cluster
    groups = df.groupby('label')

    # set up plot
    fig, ax = plt.subplots(figsize=(17, 9)) # set size
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

    #iterate through groups to layer the plot
    #note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
                label=cluster_names[name], color=cluster_colors[name],
                mec='none')
        ax.set_aspect('auto')
        ax.tick_params(\
            axis= 'x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')
        ax.tick_params(\
            axis= 'y',         # changes apply to the y-axis
            which='both',      # both major and minor ticks are affected
            left='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelleft='off')

    ax.legend(numpoints=1)  #show legend with only 1 point

    #add label in x,y position with the label as the film title
    for i in range(len(df)):
        ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)

    plt.show() #show the plot


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

    # Distance matrix
    dist = 1-sim_mat

    #similarity_matrix2 = build_sim_matrix(texts,'spacy')

    # Tokens
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
    num_clusters = 5
    [clusters,km] = cluster(tfidf,num_clusters)
    articles = { 'title': titles, 'cluster': clusters }
    frame = pd.DataFrame(articles, index = [clusters] , columns = ['title', 'cluster'])
    print("Number of items per cluster: ")
    print(frame['cluster'].value_counts())

    terms_per_cluster = 3

    print("Top terms per cluster:")
    print()
    #sort cluster centers by proximity to centroid
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    labels = []

    for i in range(num_clusters):

        term_list = ''

        print("Cluster %d words: " % i, end='')

        for ind in order_centroids[i, :terms_per_cluster]: #replace 6 with n words per cluster
            #print(vocab_frame.ix[terms[ind].split(' ')])
            term_list += terms[ind] + ' '
            print(terms[ind], end=' ')
            #print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0], end=',')

        labels.append(term_list)

        print() #add whitespace
        print() #add whitespace

        print("Cluster %d titles:" % i, end='')
        for title in frame.ix[i]['title'].values.tolist():
            print(' %s,' % title, end='')
        print() #add whitespace
        print() #add whitespace

    print()
    print()

    # Visualise clusters
    MDS()

    # convert two components as we're plotting points in a two-dimensional plane
    # "precomputed" because we provide a distance matrix
    # we will also specify `random_state` so the plot is reproducible.
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

    xs, ys = pos[:, 0], pos[:, 1]
    print()
    print()

    #plot_stuff(xs,ys,clusters,titles,labels)
    plot_web_stuff(xs,ys,clusters,titles,labels)

    # Hierarchy
    from scipy.cluster.hierarchy import ward, dendrogram

    linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

    fig, ax = plt.subplots(figsize=(15, 20)) # set size
    ax = dendrogram(linkage_matrix, orientation="right", labels=titles);

    plt.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')

    plt.tight_layout() #show plot with tight layout

    #uncomment below to save figure
    plt.savefig('ward_clusters.png', dpi=200) #save figure as ward_clusters
