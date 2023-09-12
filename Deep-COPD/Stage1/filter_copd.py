import os
import ast
import re
import pandas as pd
import logging
from unrar import rarfile #"LookupError: Couldn't find path to unrar library" see https://stackoverflow.com/questions/55574212/how-to-set-path-to-unrar-library-in-python
import shutil

logging.basicConfig(level=logging.INFO, filename="./log/log-tweet_filter_eng_copd_batch_20230321.txt",
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(funcName)s - %(levelname)s: %(message)s')
print("log is saving into log-tweet_filter_eng_copd_batch_20230321.txt ...")
rar_file_path = "/data/twitter_data/twitter_compress_data/"
dst_file_path = "../data/eng_copd_tweets/"
copd_dict_file = "CopdDictionary.csv"
rar_file_list = []

def build_match_word_regex():
    dict_df = pd.DataFrame()
    if os.path.isfile(copd_dict_file):
        dict_df = pd.read_csv(copd_dict_file, usecols=['name'], encoding="ISO-8859-1")
    else:
        print("dictionary file is not exit.")
        exit()
    print('Generate word_pattern ...\n')
    word_set = set()
    for index, word in dict_df.iterrows():
        if type(word['name']) == type('a'):
            word_name = re.sub(r'[^a-z0-9 ]', ' ', word['name'].lower()).strip()
            word_name = re.sub(' +', ' ', word_name.strip())
            word_set.add(r'\b' + word_name + r'(?![\w-])')

    word_list = list(word_set)
    word_list.sort(key=lambda i: len(i), reverse=True)
    word_pattern = re.compile('|'.join(word_list), re.IGNORECASE)
    return word_pattern

copd_pattern = build_match_word_regex()

def get_filtered_tweets(src_file_path, dest_file_path):
    logging.info("begin filtering tweets in " + src_file_path)
    print("begin filtering tweets in " + src_file_path)
    if not os.path.exists(src_file_path):
        print("no file in {}.".format(src_file_path))
        logging.info("no file in {}.".format(src_file_path))
        return None

    count = 0
    total_count = 0
    eng_count = 0
    copd_count = 0
    if not os.path.exists(dest_file_path):
        os.mkdir(dest_file_path)

    COPD_TWEET_FILE = dest_file_path + '/copd_tweets.csv'

    copd_object = open(COPD_TWEET_FILE, 'a', encoding='utf8')

    for root, dirs, files in os.walk(src_file_path):
        for file in files:
            if file.endswith('.csv'):
                print(file)
                logging.warning(file)
                for line in open(os.path.join(src_file_path, file), 'r', encoding='utf-8'):
                    total_count += 1
                    record = ast.literal_eval(line)  # json.dumps
                    if 'data' not in record:
                        continue
                    # filter out non-english tweets
                    if record['data']['lang'] != 'en':
                        continue
                    eng_count += 1
                    if 'text' not in record['data']:
                        continue
                    content = record['data']['text'].lower()
                    content_2 = ''
                    content_3 = ''
                    if 'includes' in record:
                        if 'tweets' in record['includes']:
                            content_2 = record['includes']['tweets'][0]['text'].lower()
                        if 'users' in record['includes']:
                            content_3 = record['includes']['users'][0]['description'].lower()

                    if not (re.search(copd_pattern, content)
                            or re.search(copd_pattern, content_2)
                            or re.search(copd_pattern, content_3)):
                        continue

                    copd_count += 1
                    copd_object.write("{}".format(line))


    print(
        '{} total_count: {}, eng_count:{} ({:.2%}), copd_count: {} ({:.2%})'.format(
        src_file_path, total_count, eng_count, eng_count / total_count, copd_count, copd_count / eng_count))
    logging.info(
         '{} total_count: {}, eng_count:{} ({:.2%}), copd_count: {} ({:.2%})'.format(
        src_file_path, total_count, eng_count, eng_count / total_count, copd_count, copd_count / eng_count))

    # closed the file
    if copd_object != None:
        copd_object.close()
    return src_file_path, total_count, eng_count, copd_count


if __name__ == '__main__':
    stat_file = open("{}{}".format(dst_file_path, 'copd_tweet_stats.txt'), 'a', encoding='utf-8')
    try:
        for file in rar_file_list:
            rar_file_name = rar_file_path + file
            if os.path.exists(rar_file_name):
                src_file_path = rar_file_path + file.split('.rar')[0]
                eng_copd_file_path = dst_file_path + file.split('.rar')[0]
                if os.path.exists(eng_copd_file_path):
                    print("{} exists.".format(eng_copd_file_path))
                    logging.warning("{} exists.".format(eng_copd_file_path))
                    continue
                print("unrar {} ... ".format(rar_file_name))
                logging.info("unrar {} ... ".format(rar_file_name))
                rar = rarfile.RarFile(rar_file_path + file)
                rar.extractall(path=rar_file_path)

                # Get english copd tweets
                a, b, c, d = get_filtered_tweets(src_file_path, eng_copd_file_path)
                stat_file.write("{}, {}, {}, {}\r\n".format(a, b, c, d))

                if os.path.exists(src_file_path):
                    shutil.rmtree(src_file_path)
                    print("delete {}. ".format(src_file_path))
                    logging.info("delete {}.".format(src_file_path))
    except:
        if stat_file != None:
            stat_file.close()
        print('exit with exception')

    if stat_file != None:
        stat_file.close()
    print('DONE!')

