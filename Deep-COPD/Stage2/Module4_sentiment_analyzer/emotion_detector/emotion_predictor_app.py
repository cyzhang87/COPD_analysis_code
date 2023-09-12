# 计算时间长，需要在服务器上执行第一步，在本地执行第二步
# 服务器上使用python3.5

import os
os.environ['KERAS_BACKEND'] = 'theano'
import pandas as pd
from emotion_predictor import EmotionPredictor
import logging
import ast
import re
import threading
#from multiprocessing import Process


# Pandas presentation options
pd.options.display.max_colwidth = 150   # show whole tweet's content
pd.options.display.width = 200          # don't break columns
# pd.options.display.max_columns = 7      # maximal number of columns

logging.basicConfig(level=logging.INFO, filename="log-emotion.txt",
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(funcName)s - %(levelname)s: %(message)s')
print("log is saving into log-emotion.txt ...")

# ##Constants##
dir_name = "/data/twitter_data/copd_daily_tweets/" #or "/data/twitter_data/noncopd_daily_tweets/"
tweet_file_list = []


def demo():
    # Predictor for Ekman's emotions in multiclass setting.
    model = EmotionPredictor(classification='ekman', setting='mc', use_unison_model=True)

    tweets = [
        "Watching the sopranos again from start to finish!",

        "Finding out i have to go to the dentist tomorrow",
        "I want to go outside and chalk but I have no chalk",
        "I HATE PAPERS AH #AH #HATE",
        "My mom wasn't mad",
        "Do people have no Respect for themselves or you know others peoples homes",
    ]

    predictions = model.predict_classes(tweets)
    print(predictions, '\n')

    probabilities = model.predict_probabilities(tweets)
    print(probabilities, '\n')

    embeddings = model.embed(tweets)
    print(embeddings, '\n')


def read_tweets(tweet_file):
    tweets_text_list = []
    with open(tweet_file, "r", encoding='utf-8') as fhIn:
        count = 0
        for line in fhIn:
            if isinstance(line, str):
                line = ast.literal_eval(line)  # to dict
                if 'data' in line:
                    text = re.sub(r'[\r\n]|(\w+:\/\/\S+)', ' ', line['data']['text'])
                    tweets_text_list.append([str(line["data"]["id"]), text, line["data"]['created_at'][:10]])
                else:
                    text = re.sub(r'[\r\n]|(\w+:\/\/\S+)', ' ', line['text'])
                    tweets_text_list.append([str(line["id"]), text, line['created_at'][:10]])
            else:
                print(line + "error")
                logging.error(line)
                return None
            count += 1

            if count % 5000 == 0:
                print(count)

    print("read end")
    return tweets_text_list


def transform_results(selection, dir):
    if selection == 1:
        input_file = "tweets_emotion_scores_ekman.csv"
        output_file = "tweets_emotion_scores_ekman_onehot.csv"
    elif selection == 2:
        input_file = "tweets_emotion_scores_plutchik.csv"
        output_file = "tweets_emotion_scores_plutchik_onehot.csv"
    else:
        input_file = "tweets_emotion_scores_poms.csv"
        output_file = "tweets_emotion_scores_poms_onehot.csv"

    input_results_df = pd.read_csv(os.path.join(dir, input_file))
    label_num = input_results_df.shape[1]
    output_results_list = []
    for index, row in input_results_df.iterrows():
        max_index = row.argmax()
        tmp_list = [0] * label_num
        tmp_list[max_index] = 1
        output_results_list.append(tmp_list.copy())

    output_results_df = pd.DataFrame(columns=input_results_df.keys(), data=output_results_list)
    output_results_df.to_csv(os.path.join(dir, output_file), index=False)

def transform_results_2(dir):
    input_files = ["tweets_emotion_scores_ekman.csv", "tweets_emotion_scores_plutchik.csv", "tweets_emotion_scores_poms.csv"]
    output_file = "tweets_emotion_scores.csv"
    output_file = "tweets_emotion_scores_details.csv"
    if os.path.exists(os.path.join(dir, output_file)):
        print("{} exist.".format(os.path.join(dir, output_file)))
        return
    emotion0_df = pd.read_csv(os.path.join(dir, input_files[0]))
    emotion1_df = pd.read_csv(os.path.join(dir, input_files[1]))
    emotion2_df = pd.read_csv(os.path.join(dir, input_files[2]))
    emotion0_df.columns = [emotion+str('_ekman') for emotion in emotion0_df.keys().values]
    emotion1_df.columns = [emotion + str('_plutchik') for emotion in emotion1_df.keys().values]
    emotion2_df.columns = [emotion + str('_poms') for emotion in emotion2_df.keys().values]

    """
    output_results_list = []
    for index in range(emotion0_df.shape[0]):
        output_results_list.append([emotion0_df.loc[index].idxmax(),
                                    emotion1_df.loc[index].idxmax(),
                                    emotion2_df.loc[index].idxmax()])

    output_results_df = pd.DataFrame(columns=['ekman', 'plutchik', 'poms'], data=output_results_list)
    output_results_df.to_csv(os.path.join(dir, output_file), index=False)
    """
    output_results_df = pd.concat([emotion0_df, emotion1_df, emotion2_df], axis=1)
    output_results_df.to_csv(os.path.join(dir, output_file), index=False)


def calculate_emotions(selection, dir):
    tweet_file = os.path.join(dir, 'copd_tweets_0410.csv')
    tweets_text_list = read_tweets(tweet_file)
    labels = []
    if selection == 1:
        classification_mode = 'ekman'
        labels = ['Anger', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise']
        emotion_file = os.path.join(dir, "tweets_emotion_scores_ekman_content.csv")
        emotion_score_file = os.path.join(dir, "tweets_emotion_scores_ekman.csv")
    elif selection == 2:
        classification_mode = 'plutchik'
        labels = ['Anger', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust', 'Anticipation']
        emotion_file = os.path.join(dir, "tweets_emotion_scores_plutchik_content.csv")
        emotion_score_file = os.path.join(dir, "tweets_emotion_scores_plutchik.csv")
    else:
        classification_mode = 'poms'
        labels = ['Anger', 'Depression', 'Fatigue', 'Vigour', 'Tension', 'Confusion']
        emotion_file = os.path.join(dir, "tweets_emotion_scores_poms_content.csv")
        emotion_score_file = os.path.join(dir, "tweets_emotion_scores_poms.csv")

    tweets_df = pd.DataFrame(columns=['id', 'tweets', 'date'], data=tweets_text_list)
    # Predictor for Ekman's emotions in multiclass setting.
    model = EmotionPredictor(classification=classification_mode, setting='mc', use_unison_model=True)
    probabilities = model.predict_probabilities(tweets_df['tweets'])

    probabilities_scores = probabilities[labels]
    # probabilities_scores['id'] = tweets_df['id']
    probabilities_scores.to_csv(emotion_score_file, index=False)
    probabilities.to_csv(emotion_file, index=False)


def thread_process(tweet_file_dir):
    for selection in range(1, 4):
        print("mode: {}".format(selection))
        # step 1: calculate the emotions, run at amax server
        calculate_emotions(selection, tweet_file_dir)

        # step2: transform results to onehot form, run at local
        # transform_results(selection, tweet_file_dir)


class myThread(threading.Thread):
    def __init__(self, tweet_file_dir):
        threading.Thread.__init__(self)
        self.tweet_file_dir = tweet_file_dir
    def run(self):
        print ("开始线程：" + self.tweet_file_dir)
        thread_process(self.tweet_file_dir)
        print ("退出线程：" + self.tweet_file_dir)

if __name__ == '__main__':
    thread_list = []

    file_count = len(tweet_file_list)
    for i in range(file_count):
        file = tweet_file_list[i]
        print("anlyzing {} ...".format(file))
        tweet_file_dir = os.path.join(dir_name, file)
        if not os.path.exists(tweet_file_dir):
            print('{} not exsits.'.format(tweet_file_dir))
            logging.warning('{} not exsits.'.format(tweet_file_dir))
            continue


        emotion_file_1 = os.path.join(tweet_file_dir, "tweets_emotion_scores_ekman.csv")
        emotion_file_2 = os.path.join(tweet_file_dir, "tweets_emotion_scores_plutchik.csv")
        emotion_file_3 = os.path.join(tweet_file_dir, "tweets_emotion_scores_poms.csv")

        """
        if not (os.path.exists(emotion_file_1) and os.path.exists(emotion_file_2) and os.path.exists(emotion_file_3)):
            print('{} not complete.'.format(tweet_file_list))
            #logging.warning('{} exsits.'.format(emotion_file))
            continue
        """
        #thread_process(tweet_file_dir)


    #将三列结果合并
    for file in tweet_file_list:
        print("anlyzing {} ...".format(file))
        tweet_file_dir = os.path.join(dir_name, file)
        if not os.path.exists(tweet_file_dir):
            print('{} not exsits.'.format(tweet_file_dir))
            logging.warning('{} not exsits.'.format(tweet_file_dir))
            continue

        transform_results_2(tweet_file_dir)

    print("analysis end")