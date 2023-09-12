"""
build
Latent Dirichlet Allocation (LDA) model for auto detecting and interpreting
topics in the tweets

1. Scrape tweets
2. Data pre-processing
3. Generating word cloud
4. Train LDA model
5. Visualizing topics
"""

linux_flag = False
if linux_flag:
    import gevent
    import gevent.monkey
    gevent.monkey.patch_all()

import pandas as pd
import re, pickle, os
import nltk
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
from nltk.util import ngrams
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords, wordnet
from collections import Counter
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
from gensim.corpora import MmCorpus
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models
import logging
import ast
from datetime import datetime

logging.basicConfig(level=logging.INFO, filename="log-analysis.txt",
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(funcName)s - %(levelname)s: %(message)s')
print("log is saving into log-analysis.txt ...")

year = '2020'
src_dir = "./copd_{}/".format(year)
tweet_file = "E:/project/twitter/eng_copd_tweets/total_copd_tweets_0707.csv"
data_path = src_dir + '/data/'
figure_path = src_dir + '/figure/'
pre_tweets_file = data_path + 'pre_tweets.csv'
cleaned_tweets_file = data_path + 'tweets_cleaned_df.pkl'
singwordcount_csv_file = data_path + 'singwordcount.csv'
top30_words_file = data_path + 'top30_words_df.pkl'
wordcount_file = data_path + 'wordcount_df.pkl'
wordcount_csv_file = data_path + 'wordcount.csv'
commond_words_freq_png = figure_path + 'commond_words_freq_1.png'
wordcloud_data_file = data_path + 'wordcloud_l.pkl'
wordcloud_png = figure_path + 'wordcloud_l.png'
MODEL_PATH = src_dir + '/model/'
TOPIC_VIS_FILE = figure_path + 'lda.html'
CORPUS_FILE = MODEL_PATH + 'clean_tweets_corpus.mm'
DICT_FILE = MODEL_PATH + 'clean_tweets.dict'
LDA_MODEL_FILE = MODEL_PATH + 'tweets_lda.model'
LDA_TOPICS_FILE = MODEL_PATH + 'tweets_lda_topics.txt'
copd_dict_file = "CopdDictionary.csv"

# ngrams or multi-word expressions
NUM_GRAMS = 2
# ----------------------
# LDA model parameters
# ----------------------
# Number of topics
NUM_TOPICS = 30
# Number of training passes
NUM_PASSES = 50
# Document-Topic Density. The lower alpha is, the more likely that
# a document may contain mixture of just a few of the topics.
# Default is 1.0/NUM_TOPICS
ALPHA = 0.03
# Word-Topic Density. The lower eta is, the more likely that
# a topic may contain a mixture of just a few of the words
# Default is 1.0/NUM_TOPICS
ETA = 'auto'

if not os.path.exists(src_dir):
    os.mkdir(src_dir)

if not os.path.exists(data_path):
    os.mkdir(data_path)

if not os.path.exists(figure_path):
    os.mkdir(figure_path)


if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)
# ----------------------

def get_wordnet_pos(word):
    """
    Map POS tag to first character lemmatize() accepts
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def load_stop_words(stop_word_file):
    """
    Utility function to load stop words from a file and return as a list of words
    @param stop_word_file Path and file name of a file containing stop words.
    @return list A list of stop words.
    """
    stop_words = []
    for line in open(stop_word_file):
        if line.strip()[0:1] != "#":
            for word in line.split():  # in case more than one per line
                stop_words.append(word)
    return stop_words


def build_stop_word_regex(stop_word_file_path):
    stop_word_list = load_stop_words(stop_word_file_path)
    stop_word_regex_list = []
    for word in stop_word_list:
        word_regex = r'\b' + word + r'(?![\w-])'  # added look ahead for hyphen
        stop_word_regex_list.append(word_regex)
    stop_word_pattern = re.compile('|'.join(stop_word_regex_list), re.IGNORECASE)
    return stop_word_pattern


keyphrase_dict = {}

def add_to_dict(word, word_dict):
    if word in word_dict:
        word_dict[word] += 1
    else:
        word_dict[word] = 1

def generate_keywords(sentence_list, stopword_pattern):
    for s in sentence_list:
        tmp = re.sub(stopword_pattern, '|', s.strip())
        phrases = tmp.split("|")
        for phrase in phrases:
            phrase = phrase.strip().lower()
            if phrase != "":
                lemmatizer = WordNetLemmatizer()
                phrase_new = ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in phrase.split()])
                add_to_dict(phrase_new, keyphrase_dict)

def split_sentences(text):
    """
    Utility function to return a list of sentences.
    @param text The text that must be split in to sentences.
    """
    #去除url信息
    text = re.sub(r'(\w+:\/\/\S+)', ' ', text)
    #去除转移字符& (&amp;)
    test = re.sub(r'&amp;', ' ', text)
    sentence_delimiters = re.compile(u'[.!?,;:\t\\\\"\\(\\)\\\'\u2019\u2013]|\\s\\-\\s')
    sentences = sentence_delimiters.split(text)
    return sentences

test = False
if test:
    text = "Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types of systems and systems of mixed types."
    text = 'RT @paulmuaddib611: #COVIDHoax IMO this 2nd wave is from at home test kits giving false positives https://www.medicalnewstoday.com/articles/coronavirus-testing#accuracy'

    # Split text into sentences
    sentenceList = split_sentences(text)
    #stoppath = "FoxStoplist.txt" #Fox stoplist contains "numbers", so it will not find "natural numbers" like in Table 1.1
    stoppath = "SmartStoplist.txt"  #SMART stoplist misses some of the lower-scoring keywords in Figure 1.5, which means that the top 1/3 cuts off one of the 4.0 score words in Table 1.1
    stopwordpattern = build_stop_word_regex(stoppath)

    # generate candidate keywords
    generate_keywords(sentenceList, stopwordpattern)

stoppath = "SmartStoplist.txt"  # SMART stoplist misses some of the lower-scoring keywords in Figure 1.5, which means that the top 1/3 cuts off one of the 4.0 score words in Table 1.1
stop_word_list = load_stop_words(stoppath)
stops = set(stopwords.words('english')).union(stop_word_list)

def build_match_word_regex(copd_dict_file):
    dict_df = pd.read_csv(copd_dict_file, usecols=['name'], encoding="ISO-8859-1")
    dict_set = set()
    for index, word in dict_df.iterrows():
        word_name = re.sub(r'[^a-z0-9 ]', ' ', word['name'].lower()).strip()
        word_name = re.sub(' +', ' ', word_name.strip())
        dict_set.add(r'\b' + word_name + r'(?![\w-])')
    dict_list = list(dict_set)
    dict_list.sort(key=lambda i: len(i), reverse=True)
    dict_pattern = re.compile('|'.join(dict_list), re.IGNORECASE)
    return dict_pattern

def text_cleanup(text):
    '''
    Text pre-processing
        return tokenized list of cleaned words
    '''
    # Convert to lowercase
    if not isinstance(text, str):
        return ""
    text_clean = text.lower()

    # 去除转移字符& (&amp;)
    text_clean = re.sub(r'(&amp;)|(&gt;)|(&nbsp;)|(&lt;)|(&quot;)|(&apos;)|(&times;)|(&divide;)', ' ', text_clean)
    # Remove non-alphabet and non-digital
    text_clean = re.sub(r'[^a-zA-Z0-9]|(\w+:\/\/\S+)', ' ', text_clean)
    # Remove words in copd dictionary
    copd_dict_pattern = build_match_word_regex(copd_dict_file)
    text_clean = re.sub(copd_dict_pattern, ' ', text_clean)
    text_clean = text_clean.split()
    # Remove short words (length < 3)
    text_clean = [w for w in text_clean if len(w) > 2]
    # Lemmatize text with the appropriate POS tag，词形还原
    lemmatizer = WordNetLemmatizer()
    text_clean = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in text_clean]
    # Filter out stop words in English
    text_clean = [w for w in text_clean if w not in stops]

    return text_clean

from PIL import Image
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator
import numpy as np

def wordcloud(word_count_df):
    '''
    Create word cloud image
    '''
    # Convert DataFrame to Map so that word cloud can be generated from freq
    if os.path.isfile(wordcloud_data_file):
        # Read cleaned tweets from saved file
        wordcloud = read_data_from_pickle(wordcloud_data_file)
        print('Loaded wordcloud from file\n')
    else:
        word_count_dict = {}
        for w, f in word_count_df.values:
            word_count_dict[w] = f
        # Generate word cloud
        background = np.array(Image.open("lung_bk.png"))
        #img_colors = ImageColorGenerator(background)
        wordcloud = WordCloud(max_words=300, width=1400, height=900,
                              random_state=12,
                              contour_width=3,
                              contour_color='lightsalmon',
                              mask=background,
                              background_color = 'white')
        wordcloud.generate_from_frequencies(word_count_dict)
        save_data_to_pickle(wordcloud_data_file, wordcloud)

        plt.figure(figsize=(10, 10), facecolor='k')
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        # Save the word cloud image
        wordcloud.to_file(wordcloud_png)
        print('Word cloud saved\n')
        plt.close('all')

    return wordcloud


def read_data_from_pickle(infile):
    with open(infile, 'rb') as fp:
        return pickle.load(fp)


def save_data_to_pickle(outfile, all_tweets):
    with open(outfile, 'wb') as fp:
        pickle.dump(all_tweets, fp)


def save_print_to_file(outfile, msg):
    with open(outfile, 'w') as fp:
        print(msg, file=fp)

def preprocess_tweets(all_tweets_df):
    '''
    Preprocess tweets
    '''
    if os.path.isfile(cleaned_tweets_file):
        # Read cleaned tweets from saved file
        cleaned_tweets_df = read_data_from_pickle(cleaned_tweets_file)
        print('Loaded cleaned tweets from file\n')
    else:
        print('Start preprocessing tweets ...\n')
        cleaned_tweets_df = all_tweets_df.copy(deep=True)
        # parsing tweets
        cleaned_tweets_df['token'] = [text_cleanup(x) for x in cleaned_tweets_df['text']]
        # Save cleaned tweets to file
        save_data_to_pickle(cleaned_tweets_file, cleaned_tweets_df)
        print('Cleaned tweets saved\n')
    return cleaned_tweets_df

def get_single_word_count(tweets_text):
    if os.path.isfile(singwordcount_csv_file):
        print('single wordcount already exist\n')
    else:
        common_words = Counter(tweets_text).most_common()
        word_count = pd.DataFrame(data=common_words,
                                  columns=['word', 'frequency'])
        word_count.to_csv(singwordcount_csv_file, index=False)
        print('single wordcount file saved\n')

def get_word_count(tweets_text, min_gram, max_gram):
    '''
    Get common word counts
    '''
    if os.path.isfile(top30_words_file):
        # Read cleaned tweets from saved file
        df = read_data_from_pickle(top30_words_file)
        word_count = read_data_from_pickle(wordcount_file)
        print('Loaded wordcount and top30 words from file\n')
    else:
        n_grams = list()
        for n in range(min_gram, max_gram):
            n_grams += list(ngrams(tweets_text, n))

        common_words = Counter(n_grams).most_common()
        word_count = pd.DataFrame(data=common_words,
                                  columns=['word', 'frequency'])
        # Convert list to string
        word_count['word'] = word_count['word'].apply(' '.join)
        save_data_to_pickle(wordcount_file, word_count)
        df = word_count.head(30).sort_values('frequency')
        save_data_to_pickle(top30_words_file, df)
        word_count.to_csv(wordcount_csv_file, index=False)
    # Plot word count graph

    df.plot.barh(
        x='word', y='frequency', title='Word Frequency', figsize=(19, 10))
    plt.savefig(commond_words_freq_png)
    print('Word count saved\n')
    plt.close('all')

    return word_count


def word_grams(words, min=1, max=2):
    '''
    Build ngrams word list
    '''
    word_list = []
    for n in range(min, max):
        for ngram in ngrams(words, n):
            word_list.append(' '.join(str(i) for i in ngram))
    return word_list


def train_lda_model(token_tweets, lda_model_file=LDA_MODEL_FILE, num_topics=NUM_TOPICS):
    print('Start LDA model training ...\n')
    # Build dictionary
    tweets_dict = corpora.Dictionary(token_tweets)
    # Remove words that occur less than 10 documents,
    # or more than 50% of the doc
    tweets_dict.filter_extremes(no_below=10, no_above=0.5)
    # Transform doc to a vectorized form by computing frequency of each word
    bow_corpus = [tweets_dict.doc2bow(doc) for doc in token_tweets]
    # Save corpus and dictionary to file
    MmCorpus.serialize(CORPUS_FILE, bow_corpus)
    tweets_dict.save(DICT_FILE)

    # Create tf-idf model and then apply transformation to the entire corpus
    tfidf = models.TfidfModel(bow_corpus)
    tfidf_corpus = tfidf[bow_corpus]
    if os.path.isfile(lda_model_file):
        lda_model = models.ldamodel.LdaModel.load(lda_model_file)
        print('Loaded LDA model from file\n')
    else:
        # Train LDA model
        """
        lda_model = models.ldamodel.LdaModel(corpus=tfidf_corpus,
                                             num_topics=num_topics,
                                             id2word=tweets_dict,
                                             passes=NUM_PASSES,
                                             alpha=ALPHA,
                                             eta=ETA,
                                             random_state=49)
        """
        lda_model = models.ldamodel.LdaModel(corpus=bow_corpus,
                                             num_topics=num_topics,
                                             id2word=tweets_dict,
                                             passes=NUM_PASSES,
                                             alpha=ALPHA,
                                             eta=ETA,
                                             random_state=49,
                                             per_word_topics=True)
        # Save LDA model to file
        lda_model.save(lda_model_file)
        print('LDA model saved\n')

    PLOT_SWITCH = True
    if PLOT_SWITCH:
        import seaborn as sns
        import matplotlib.colors as mcolors
        from sklearn.manifold import TSNE
        from bokeh.plotting import figure, output_file, show

        topic_weights = []
        for i, row_list in enumerate(lda_model[bow_corpus]):
            topic_weights.append([w for i, w in row_list[0]])

        # Array of topic weights
        arr = pd.DataFrame(topic_weights).fillna(0).values

        # Keep the well separated points (optional)
        arr = arr[np.amax(arr, axis=1) > 0.3]

        # Dominant topic number in each doc
        topic_num = np.argmax(arr, axis=1)
        df = pd.DataFrame(data=topic_num, columns=['topic_num'])
        df['year'] = str(year)
        df.to_csv("topic_num_{}.csv".format(year))

        print("{} total num: {}".format(year, df.shape[0]))
        for i in range(10):
            topic_number = sum(df['topic_num'] == i)
            topic_percent = topic_number / df.shape[0] * 100
            print("topic {}: {} ({:.2f}%)".format(i+1, topic_number, topic_percent))
        # tSNE Dimension Reduction
        tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
        tsne_lda = tsne_model.fit_transform(arr)

        # tSNE Dimension Reduction
        #tsne_model2 = TSNE(n_components=2, verbose=1, random_state=99, angle=.5, init='pca')
        #tsne_lda2 = tsne_model2.fit_transform(arr)


        # Plot the Topic Clusters using Bokeh
        # output_notebook()
        #mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
        colors = np.array(["#ed7d31",  # covid 0
                           "#5b9bd5",  # copd 1
                           "#a6b1cf",  # politics 2
                           "#70ad47",  # healthcare 3
                           "#ffe699",  # economics 4
                           "#ff99ff",  # other 5
                           ])
        mycolors = np.array([colors[0], colors[1], colors[4], colors[0], colors[0],
                             colors[0], colors[1], colors[2], colors[0], colors[0],]) #2020
        mycolors = np.array([colors[1], colors[1], colors[0], colors[5], colors[0],
                             colors[0], colors[4], colors[3], colors[3], colors[0], colors[5], ])  # 2021
        mycolors = np.array([colors[5], colors[3], colors[1], colors[1], colors[3],
                             colors[0], colors[1], colors[1], colors[3], colors[1], ])  # 2022
        mycolors = np.array([colors[1], colors[3], colors[0], colors[3], colors[3],
                             colors[1], colors[3], colors[3], colors[1], colors[1], colors[0], colors[5], ])  # 2023

        output_file("twitter_tsne_{}.html".format(year))
        plot = figure(title="t-SNE clustering of the LDA topics in {}".format(year),
                      plot_width=500, plot_height=500)
        plot.scatter(x=tsne_lda[:, 0], y=tsne_lda[:, 1], color=mycolors[topic_num])
        show(plot)

    # Save all generated topics to a file
    msg = ''
    for idx, topic in lda_model.print_topics(-1):
        msg += 'Topic: {} \nWords: {}\n'.format(idx, topic)
    save_print_to_file(MODEL_PATH + 'tweets_lda_topics_{}.txt'.format(num_topics), msg)

    # Evaluate LDA model performance
    coherence, perplexity = eval_lda(lda_model, bow_corpus, tweets_dict, token_tweets)
    # Visualize topics
    vis_topics(lda_model, bow_corpus, tweets_dict, num_topics)

    return lda_model, coherence, perplexity


def eval_lda(lda_model, corpus, dict, token_text):
    # Compute Perplexity: a measure of how good the model is. lower the better.
    perplexity = lda_model.log_perplexity(corpus)
    print('\nPerplexity: ', perplexity)
    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=token_text,
                                         dictionary=dict, coherence='c_v')
    coherence = coherence_model_lda.get_coherence()
    print('\nCoherence: ', coherence)
    return coherence, perplexity


def vis_topics(lda_model, corpus, dict, num_topics):
    '''
    Plot generated topics on an interactive graph
    '''
    lda_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, dict, sort_topics=False) #, mds='mmds'
    pyLDAvis.display(lda_data)
    pyLDAvis.save_html(lda_data, figure_path + 'lda_topics{}.html'.format(num_topics))
    print('Topic visual saved\n')

def read_tweets(start_date, end_date):
    line_count = 0
    tweet_list = []
    tweet_count = 0
    for line in open(tweet_file, 'r', encoding='utf-8'):
        line_count += 1
        try:
            record = ast.literal_eval(line)  # json.dumps
            if 'data' not in record:
                print("no data in line: {}".format(line_count))
                continue
            create_time = datetime.strptime(record['data']['created_at'][:10], '%Y-%m-%d')
            if create_time < start_date or create_time >= end_date:
                continue
            tweet_count += 1
            content = record['data']['text'].lower()
            content_2 = ''
            content_3 = ''
            if 'includes' in record:
                if 'tweets' in record['includes']:
                    for tweeti in range(len(record['includes']['tweets'])):
                        if record['includes']['tweets'][tweeti]['id'] != record['data']['id']:
                            content_2 += ' ' + record['includes']['tweets'][tweeti]['text'].lower()
                if 'users' in record['includes']:
                    for useri in range(len(record['includes']['users'])):
                        if record['includes']['users'][useri]['id'] != record['data']['author_id']:
                            content_3 += ' ' + record['includes']['users'][useri]['description'].lower()
            content = content + ' ' + content_2 # + ' ' + content_3
            content = content.encode('UTF-8','ignore').decode('UTF-8')
            tweet_list.append([record['data']['author_id'],
                               content,
                               record['data']['created_at'],
                               record['data']['id']])
        except:
            print("error line: {}".format(line_count))

    print("from {} to {}, tweet_count = {}.".format(start_date, end_date, tweet_count))
    result_df = pd.DataFrame(data=tweet_list, columns=['author_id', 'text', 'created_at', 'id'])
    result_df.to_csv(pre_tweets_file, index=False)
    return result_df


if __name__ == '__main__':
    num_topics = 11
    # Get all tweets
    tweets_df = None
    tweets_text = None
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2021, 1, 1)
    tweets_df = read_tweets(start_date, end_date)
    # Preprocess tweets
    cleaned_tweets_df = preprocess_tweets(tweets_df)
    # Convert series to list for word count
    tweets_text = [word for one_tweet in cleaned_tweets_df['token'] for word in one_tweet]
    get_single_word_count(tweets_text)
    # Get common ngrams word count
    word_count_df = get_word_count(tweets_text, 2, 3)
    # Generate word cloud
    tweets_wordcloud = wordcloud(word_count_df)
    # Generate ngram tokens
    cleaned_tweets_df['ngram_token'] = [word_grams(x, 1, 3) for
                                        x in cleaned_tweets_df['token']]
    # Train LDA model and visualize generated topics
    lda_model_file = MODEL_PATH + 'tweets_lda_topic{}.model'.format(num_topics)
    lda_model, coherence, perplexity = train_lda_model(cleaned_tweets_df['ngram_token'], lda_model_file=lda_model_file, num_topics=num_topics)

    coherence_list = []
    for topic_num in range(2, 20):
        print("topic_num:", topic_num)
        lda_model_file = MODEL_PATH + 'tweets_lda_topic{}.model'.format(topic_num)
        lda_model, coherence, perplexity = train_lda_model(cleaned_tweets_df['ngram_token'], lda_model_file, topic_num)
        coherence_list.append([topic_num, coherence, perplexity])
    print(coherence_list)
    coherence_df = pd.DataFrame(data=coherence_list, columns=['Topic number', 'Coherence', 'Perplexity'])
    coherence_df.to_csv(MODEL_PATH + 'topic_coherence.csv', index=False)
    print('DONE!')
