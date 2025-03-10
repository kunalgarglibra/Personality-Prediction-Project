# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud 
from collections import Counter
import seaborn as sb


def visualising_target(df):
    # Seeing the number of datapoints in each Label.
    target_col = df.groupby(['type']).count()
    print(target_col)


def swarm_plot(df):

    # Visualising using a Swarm Plot:
    # import seaborn as sb
    # Swarm Plot
    df1 = df.copy()
    #this function counts the average number of words in each post of a user
    def mean_row(row):
        l = []
        for i in row.split('|||'):
            l.append(len(i.split()))
        return np.mean(l)
    def var_row(row):
        l = []
        for i in row.split('|||'):
            l.append(len(i.split()))
        return np.var(l)
    #this function counts the no of words per post out of the total 50 posts in the whole row
    df1['words_per_datapoint'] = df1['posts'].apply(lambda x: len(x.split()))
    df1['avg_words_per_comment'] = df1['posts'].apply(lambda x: len(x.split())/50)
    df1['mean_of_word_counts'] = df1['posts'].apply(lambda x: mean_row(x))
    df1['var_of_word_counts'] = df1['posts'].apply(lambda x: var_row(x))

    plt.figure(figsize=(15,10))
    sb.swarmplot("type", "mean_of_word_counts", data=df1['mean_of_word_counts'])

def word_cloud_most_commonN(df, n):

    # Visulaising the dataset using the WordCloud:
    # from wordcloud import WordCloud 
    # from collections import Counter

    #Plotting WordCloud.
    #Finding the most common words in all posts.
    df1 = df.copy()
    words = list(df1["posts"].apply(lambda x: x.split()))
    words = [x for y in words for x in y]

    most_commonN = Counter(words).most_common(n)
    wc = WordCloud(width=1200, height=500, 
                            collocations=False, background_color="white", 
                            colormap="tab20b").generate(" ".join(words))

    # collocations to False  is set to ensure that the word cloud doesn't appear as if it contains any duplicate words
    plt.figure(figsize=(25,10))
    # generate word cloud, interpolation 
    plt.imshow(wc, interpolation='bilinear')
    _ = plt.axis("off")

    Counter(words).most_common(n)


def sub_plots(df):
    df1 = df.copy()
    fig, ax = plt.subplots(len(df1['type'].unique()), sharex=True, figsize=(15,len(df1['type'].unique())))
    k = 0
    for i in df1['type'].unique():
        df_4 = df[df['type'] == i]
        wordcloud = WordCloud(max_words=1628,relative_scaling=1,normalize_plurals=False).generate(df_4['posts'].to_string())
        plt.subplot(4,4,k+1)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(i)
        ax[k].axis("off")
        k+=1

if __name__ == "__main__":
    df = pd.read_csv("data/raw_data.csv")
    visualising_target(df)
    # swarm_plot(df)
    n = int(input("Enter the number of most common words you want to evaluate, ex 40"))
    word_cloud_most_commonN(df, n)
    sub_plots(df)
    print("Data visulaisation completed.")
