#  Copyright (c) 2021.
#  Chunyan Zhang

import numpy as np
from nltk.tokenize import TweetTokenizer
from wordsegment import segment, load
import pandas as pd
import re, pickle, os
import random
import logging
# K-fold splits
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, filename="log.txt",
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(funcName)s - %(levelname)s: %(message)s')
print("log is saving into log.txt ...")

data_path = "C:/cyzhang/project/twitter_projects/Twitter-Occupation-Prediction/Twi_data/"
user_label_data_csv_file = data_path + 'known_user_label_2.jsonl'
raw_data_pkl_file = './data/crawled_data.pkl'
split_data_file = './data/crawled_data_splits.pkl'

num_splits = 10

from config import dir_name, vaccine_file_list

################################################################################
############################ Create training data ##############################
################################################################################
import ast

def generate_label_tweets():
    if os.path.isfile(raw_data_pkl_file):
        return

    tweet_dict = {}
    for line in open(user_label_data_csv_file, 'r', encoding='utf-8'):
        record = ast.literal_eval(line)
        tweet_dict[record['id']] = [record['label_3class'], record['description'], record['name']]

    with open(raw_data_pkl_file, 'wb') as fp:
        pickle.dump(tweet_dict, fp)


################################################################################
############################## Text Preprocessing ##############################
################################################################################
def text_preprocess(text, tknzr):
    FLAGS = re.MULTILINE | re.DOTALL
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"/", " / ")
    text = re_sub(r"@\w+", "<user>")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
    text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
    text = re_sub(r"<3", "<heart>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")
    text = re_sub(r"#\S+", lambda hashtag: " ".join(segment(hashtag.group()[1:])))  # segment hastags

    tokens = tknzr.tokenize(text.lower())
    return " ".join(tokens)


def concat_data():
    path = os.path.dirname(os.path.abspath(__file__)) + "/data/"
    with open(raw_data_pkl_file, "rb") as f:
        id2entities = pickle.load(f)

    ########## Lookup Tables ##########
    labels = sorted(list(set([entity[0] for entity in id2entities.values()])))
    num_classes = len(labels)

    label_lookup = np.zeros((num_classes, num_classes), int)
    np.fill_diagonal(label_lookup, 1)
    ###################################

    text_data, context_data, label_data = [], [], []
    label_dict = {}
    for i, label in enumerate(labels):
        label_dict[label] = i

    load()
    tknzr = TweetTokenizer(reduce_len=True, preserve_case=False, strip_handles=False)
    print("Preprocessing tweets.....")
    for _id in id2entities:
        if id2entities[_id][0] in label_dict.keys():
            text_data.append(text_preprocess(id2entities[_id][1], tknzr))
            context_data.append(text_preprocess(id2entities[_id][2], tknzr))

            label_data.append(label_lookup[label_dict[id2entities[_id][0]]])

    assert len(text_data) == len(context_data) == len(label_data)

    return text_data, context_data, label_data


################################################################################
############################## K-fold Data Split ###############################
################################################################################
def kfold_splits(text_data, context_data, label_data, k):
    kfold_text, kfold_context, kfold_label = [], [], []
    for i in range(k):
        _text_data = {"train": {}, "valid": {}, "test": {}}
        _context_data = {"train": {}, "valid": {}, "test": {}}
        _label_data = {"train": {}, "valid": {}, "test": {}}
        kfold_text.append(_text_data)
        kfold_context.append(_context_data)
        kfold_label.append(_label_data)

    random_state = np.random.randint(0, 10000)
    from sklearn.model_selection import StratifiedKFold
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    # kf = KFold(n_splits=k, shuffle=True, random_state=0)

    kfold_index = 0
    for rest_index, test_index in kf.split(text_data, np.array(label_data)[:, 0]):
        train_index, valid_index, _, _ = train_test_split(rest_index, np.zeros_like(rest_index), test_size=0.05)

        kfold_text[kfold_index]["train"] = [text_data[index] for index in train_index]
        kfold_text[kfold_index]["test"] = [text_data[index] for index in test_index]
        kfold_text[kfold_index]["valid"] = [text_data[index] for index in valid_index]

        kfold_context[kfold_index]["train"] = [context_data[index] for index in train_index]
        kfold_context[kfold_index]["test"] = [context_data[index] for index in test_index]
        kfold_context[kfold_index]["valid"] = [context_data[index] for index in valid_index]

        kfold_label[kfold_index]["train"] = [label_data[index] for index in train_index]
        kfold_label[kfold_index]["test"] = [label_data[index] for index in test_index]
        kfold_label[kfold_index]["valid"] = [label_data[index] for index in valid_index]

        assert len(kfold_text[kfold_index]["train"]) == len(kfold_context[kfold_index]["train"]) == len(
            kfold_label[kfold_index]["train"])
        assert len(kfold_text[kfold_index]["valid"]) == len(kfold_context[kfold_index]["valid"]) == len(
            kfold_label[kfold_index]["valid"])
        assert len(kfold_text[kfold_index]["test"]) == len(kfold_context[kfold_index]["test"]) == len(
            kfold_label[kfold_index]["test"])

        train_length = len(kfold_text[kfold_index]["train"])
        valid_length = len(kfold_text[kfold_index]["valid"])
        test_length = len(kfold_text[kfold_index]["test"])

        kfold_index += 1

    print("Input Data Splitted: %s (train) / %s (valid) / %s (test)" % (train_length, valid_length, test_length))

    return kfold_text, kfold_context, kfold_label


def generate_label_tweets_ourdata():
    global raw_data_pkl_file, split_data_file
    raw_data_pkl_file = './data/our_data/crawled_data.pkl'
    split_data_file = './data/our_data/crawled_data_splits.pkl'
    if os.path.isfile(raw_data_pkl_file):
        return

    tweets_count = 0
    tweet_dict = {}
    for file in vaccine_file_list:
        print("{}".format(file))
        logging.warning("{}".format(file))
        empty_bio_count = 0
        total_count = 0
        covid_file_dir = os.path.join(dir_name, file)
        if not os.path.exists(covid_file_dir):
            print('{} not exsits.'.format(covid_file_dir))
            logging.warning('{} not exsits.'.format(covid_file_dir))
            continue

        tweet_file = os.path.join(dir_name, file, 'vaccine_tweets.csv')

        if not os.path.exists(tweet_file):
            print('{} not exsits.'.format(tweet_file))
            logging.warning('{} not exsits.'.format(tweet_file))
            continue

        for line in open(tweet_file, 'r', encoding='utf-8'):
            tweets_count += 1
            record = ast.literal_eval(line)
            user_id = record['includes']['users'][0]['id']
            bio = record['includes']['users'][0]['description']
            count = 0
            if user_id in tweet_dict:
                count = tweet_dict[user_id][3]+1
                tweet_dict[user_id][3] += 1
                user_id = '{}_{}'.format(user_id, count)

            if len(bio) == 0:
                bio = record['data']['text']
                empty_bio_count += 1
            tweet_dict[user_id] = ['label', bio, record['includes']['users'][0]['name'], count]
            total_count += 1
        print('empty bio:{}, total:{}, percent:{:.2%}'.format(empty_bio_count, total_count, empty_bio_count / total_count))
    with open(raw_data_pkl_file, 'wb') as fp:
        pickle.dump(tweet_dict, fp)

def generate_label_tweets_ourdata_origin():
    global raw_data_pkl_file, split_data_file
    raw_data_pkl_file = './data/our_data/crawled_data.pkl'
    split_data_file = './data/our_data/crawled_data_splits.pkl'
    if os.path.isfile(raw_data_pkl_file):
        return

    tweets_count = 0
    empty_bio_count = 0
    total_count = 0
    tweet_dict = {}
    tweet_dir = "D:/twitter_data/origin_tweets/"
    tweet_file_list = ["Sampled_Stream_detail_20200715_0720_origin/twitter_sample_origin-20200715141905.csv",
                       "Sampled_Stream_detail_20200811_0815_origin/twitter_sample_origin-20200811233217.csv",
                       "Sampled_Stream_detail_20200914_0917_origin/twitter_sample_origin-20200916100506.csv",
                       "Sampled_Stream_detail_20201105_1110_origin/twitter_sample_origin-20201109062630.csv",
                       "Sampled_Stream_detail_20201210_1214_origin/twitter_sample_origin-20201214190040.csv",
                       "Sampled_Stream_detail_20210410_0416_origin/twitter_sample_origin-20210410123126.csv"
                      ]
    tweet_file = tweet_dir + tweet_file_list[5]

    for line in open(tweet_file, 'r', encoding='utf-8'):
        tweets_count += 1
        record = ast.literal_eval(line)
        user_id = record['includes']['users'][0]['id']
        bio = record['includes']['users'][0]['description']
        count = 0
        if user_id in tweet_dict:
            count = tweet_dict[user_id][3]+1
            tweet_dict[user_id][3] += 1
            user_id = '{}_{}'.format(user_id, count)

        if len(bio) == 0:
            bio = record['data']['text']
            empty_bio_count += 1
        tweet_dict[user_id] = ['label', bio, record['includes']['users'][0]['name'], count]
        total_count += 1

    print('empty bio:{}, total:{}, percent:{:.2%}'.format(empty_bio_count, total_count, empty_bio_count / total_count))
    with open(raw_data_pkl_file, 'wb') as fp:
        pickle.dump(tweet_dict, fp)

def process_training_data():
    generate_label_tweets()
    _text, _ctxt, _label = concat_data()
    print("Splitting data into 10 folds.....")
    _text_split, _ctxt_split, _label_split = kfold_splits(_text, _ctxt, _label, num_splits)

    path = os.path.dirname(os.path.abspath(__file__)) + "/data/"
    if not os.path.exists(path):
        os.makedirs(path)
    with open(split_data_file, "wb") as f:
        print("Creating pickle files for each split to " + split_data_file)
        pickle.dump({"text_data": _text_split, "context_data": _ctxt_split, "label_data": _label_split}, f)

def process_our_data():
    text_fold0, context_fold0, label_fold0 = [], [], []
    text_fold0.append({"train": {}, "valid": {}, "test": {}})
    context_fold0.append({"train": {}, "valid": {}, "test": {}})
    label_fold0.append({"train": {}, "valid": {}, "test": {}})
    # 将标注数据集分为训练集和验证集
    generate_label_tweets()
    _text, _ctxt, _label = concat_data()
    from sklearn.model_selection import ShuffleSplit  # or StratifiedShuffleSplit
    sss = ShuffleSplit(n_splits=1, test_size=0.1)
    train_index, val_index = next(sss.split(_text, _label))
    text_fold0[0]["train"] = [_text[index] for index in train_index]
    context_fold0[0]["train"] = [_ctxt[index] for index in train_index]
    label_fold0[0]["train"] = [_label[index] for index in train_index]
    text_fold0[0]["valid"] = [_text[index] for index in val_index]
    context_fold0[0]["valid"] = [_ctxt[index] for index in val_index]
    label_fold0[0]["valid"] = [_label[index] for index in val_index]

    #处理vaccine tweet用generate_label_tweets_ourdata，处理origin tweet用generate_label_tweets_ourdata_origin
    #generate_label_tweets_ourdata()
    generate_label_tweets_ourdata_origin()
    _text, _ctxt, _label = concat_data()
    text_fold0[0]["test"] = _text
    context_fold0[0]["test"] = _ctxt
    label_fold0[0]["test"] = _label

    with open(split_data_file, "wb") as f:
        print("Creating pickle files for each split to " + split_data_file)
        pickle.dump({"text_data": text_fold0, "context_data": context_fold0, "label_data": label_fold0}, f)

if __name__ == "__main__":
    #f = open('./data/our_data/crawled_data_splits.pkl', "rb")
    #id2entities = pickle.load(f)
    #process_training_data()
    process_our_data()