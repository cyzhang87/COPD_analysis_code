import os
import numpy as np
import pandas as pd
import re
from collections import Counter
from datetime import datetime
import ast
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

pre_tweets_file = "pre_tweets_0707.csv"
tweet_file = "E:/project/twitter/eng_copd_tweets/total_copd_tweets_0707.csv"
plt.rcParams['font.sans-serif'] = 'Times New Roman'
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 24}
my_color2 = [(91, 155, 213),  #A6D5E9
             (237, 125, 49),  #F4B798
             (142, 179, 117),  #97BEA7
             (149, 223, 231),  #A0AED6
             (255,230,153),  #90AC76
             (255, 153, 255)
             ]
for i in range(len(my_color2)):
        my_color2[i] = (my_color2[i][0] / 255, my_color2[i][1] / 255, my_color2[i][2] / 255)
my_palette_2 = sns.color_palette(my_color2)
sns.set_palette(my_palette_2)


def read_tweets():
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
            content = content + ' ' + content_2 #+ ' ' + content_3
            content = content.encode('UTF-8','ignore').decode('UTF-8')
            tweet_list.append([record['data']['author_id'],
                               content,
                               record['data']['created_at'],
                               record['data']['id']])
        except:
            print("error line: {}".format(line_count))

    print("tweet_count = {}.".format(tweet_count))
    result_df = pd.DataFrame(data=tweet_list, columns=['author_id', 'text', 'created_at', 'id'])
    result_df.to_csv(pre_tweets_file, index=False)
    return result_df


def get_hashtags():
    result_df = pd.read_csv(pre_tweets_file)
    hashtag_list = [[], [], [], []]
    for index, row in result_df.iterrows():
        hashtags = re.findall(r"(#+[a-zA-Z0-9\-_]{1,})", row['text'])
        create_time = datetime.strptime(row['created_at'][:10], '%Y-%m-%d')
        if (create_time >= datetime(2020, 1, 1)) and (create_time < datetime(2021, 1, 1)):
            hashtag_list[0] += hashtags
        elif (create_time >= datetime(2021, 1, 1)) and (create_time < datetime(2022, 1, 1)):
            hashtag_list[1] += hashtags
        elif (create_time >= datetime(2022, 1, 1)) and (create_time < datetime(2023, 1, 1)):
            hashtag_list[2] += hashtags
        elif (create_time >= datetime(2023, 1, 1)) and (create_time < datetime(2024, 1, 1)):
            hashtag_list[3] += hashtags

    all_hashtag_list = hashtag_list[0] + hashtag_list[1] + hashtag_list[2] + hashtag_list[3]
    common_words = Counter(all_hashtag_list).most_common()
    word_count_df = pd.DataFrame(data=common_words, columns=['word', 'frequency'])
    word_count_df.to_csv("all_hashtag_freq.csv")
    word_count_dict = {}
    for w, f in word_count_df.values:
        word_count_dict[w] = f
    # Generate word cloud
    background = np.array(Image.open("lung_bk.png"))
    wordcloud = WordCloud(max_words=500, width=1400, height=900,
                          random_state=12,
                          contour_width=3,
                          contour_color='lightsalmon',
                          mask=background,
                          background_color='white')
    wordcloud.generate_from_frequencies(word_count_dict)

    plt.figure(figsize=(20, 20), facecolor='k')
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    # Save the word cloud image
    wordcloud.to_file("hashtag_wordcloud.png")
    print('Word cloud saved\n')
    plt.close('all')

    years = ['2020', '2021', '2022', '2023']
    for i in range(len(hashtag_list)):
        common_words = Counter(hashtag_list[i]).most_common()
        word_count = pd.DataFrame(data=common_words, columns=['word', 'frequency'])
        word_count.to_csv("hashtag_freq_{}_0707.csv".format(years[i]))


def bar_plot():
    years = ['2020', '2021', '2022', '2023']
    type = ['copd', 'covid', 'health', 'pol', 'eco', 'other']
    category = ['COPD', 'COVID-19', 'Health', 'Politics', 'Economics', 'Others']
    results = []
    for i in range(len(years)):
        tmp_df = pd.read_csv("hashtag_freq_{}.csv".format(years[i]))
        total_f = 0
        for index, row in tmp_df.iterrows():
            if str(row['type']) != 'nan':
                total_f += row['frequency']

        for t in range(len(type)):
            results.append([years[i], category[t], sum(tmp_df['type'] == type[t]),
                            sum(tmp_df[tmp_df['type'] == type[t]]['frequency']) / total_f * 100])

    result_df = pd.DataFrame(data=results, columns=['year', 'Category', 'num', 'frequency'])

    label_fontsize = 24
    fig, ax = plt.subplots(1, 1, figsize=(25, 7.5))
    ax = sns.barplot(ax=ax, x='year', y='frequency', hue='Category', edgecolor=".2", alpha=0.8, data=result_df)

    ax.set_xlabel('Year', fontsize=label_fontsize)
    ax.set_ylabel('Percentage (%)', fontsize=label_fontsize)
    ax.tick_params(axis="x", labelsize=label_fontsize, labelrotation=0)
    ax.tick_params(axis="y", labelsize=label_fontsize)
    ax.legend(fancybox=True, shadow=False, prop=font2, ncol=6, loc='upper center', bbox_to_anchor=(0.5, 1.15))
    plt.savefig('./hashtag_bar.pdf')


if __name__ == '__main__':
    read_tweets()
    get_hashtags()
    bar_plot()
