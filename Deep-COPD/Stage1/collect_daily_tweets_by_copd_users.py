import os
import ast
import re
import pandas as pd
import logging
from unrar import rarfile #"LookupError: Couldn't find path to unrar library" see https://stackoverflow.com/questions/55574212/how-to-set-path-to-unrar-library-in-python
import shutil

logging.basicConfig(level=logging.INFO, filename="./log/log-tweet_filter_eng_copd_batch_20230410.txt",
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(funcName)s - %(levelname)s: %(message)s')
print("log is saving into log-tweet_filter_eng_copd_batch_20230410.txt ...")
rar_file_path = "/data/twitter_data/twitter_compress_data/"
dst_file_path = "../data/copd_daily_tweets/"
author_ids_file = "author_ids_copd.csv"
author_ids = pd.read_csv(author_ids_file)['author_id'].values
rar_file_list = []


def get_filtered_tweets(src_file_path, dest_file_path, type='ours'):
    logging.info("begin filtering tweets in " + src_file_path)
    print("begin filtering tweets in " + src_file_path)
    if not os.path.exists(src_file_path):
        print("no file in {}.".format(src_file_path))
        logging.info("no file in {}.".format(src_file_path))
        return None

    total_count = 0
    copd_count = 0
    if not os.path.exists(dest_file_path):
        os.mkdir(dest_file_path)

    COPD_TWEET_FILE = dest_file_path + '/copd_tweets_0410.csv'
    if os.path.exists(COPD_TWEET_FILE):
        print("{} exists.".format(COPD_TWEET_FILE))
        logging.warning("{} exists.".format(COPD_TWEET_FILE))
        return

    copd_object = open(COPD_TWEET_FILE, 'a', encoding='utf8')

    for root, dirs, files in os.walk(src_file_path):
        for file in files:
            if file.endswith('.csv'):
                print(file)
                logging.warning(file)
                for line in open(os.path.join(src_file_path, file), 'r', encoding='utf-8'):
                    total_count += 1
                    record = ast.literal_eval(line)  # json.dumps
                    #if 'data' not in record:
                    #    continue
                    if int(record['data']['author_id']) in author_ids:
                        copd_count += 1
                        copd_object.write("{}".format(line))

    print(
        '{} total_count: {}, copd_count: {}.'.format(src_file_path, total_count, copd_count))
    logging.info(
        '{} total_count: {}, copd_count: {}.'.format(src_file_path, total_count, copd_count))

    # 程序结束前关闭文件指针
    if copd_object != None:
        copd_object.close()
    return src_file_path, total_count, copd_count


if __name__ == '__main__':
    stat_file = open("{}{}".format(dst_file_path, 'copd_tweet_stats_0621.txt'), 'a', encoding='utf-8')
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
                os.system('cd {} && unrar x -password {}'.format(rar_file_path, rar_file_name))

                # Get english copd tweets
                a, b, c = get_filtered_tweets(src_file_path, eng_copd_file_path)
                stat_file.write("{}, {}, {}\r\n".format(a, b, c))

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

