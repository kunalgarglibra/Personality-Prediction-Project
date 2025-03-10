import re
import string
import nltk
import pandas as pd
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer


# Returns train, target
def preprocess_data(df):
    # Remove duplicate entries
    df.drop_duplicates(inplace=True)

    # Expanding Contractions
    contractions_dict = { 
    "ain't": "am not / are not / is not / has not / have not",
    "aren't": "are not / am not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had / he would",
    "he'd've": "he would have",
    "he'll": "he shall / he will",
    "he'll've": "he shall have / he will have",
    "he's": "he has / he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how has / how is / how does",
    "I'd": "I had / I would",
    "I'd've": "I would have",
    "I'll": "I shall / I will",
    "I'll've": "I shall have / I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it had / it would",
    "it'd've": "it would have",
    "it'll": "it shall / it will",
    "it'll've": "it shall have / it will have",
    "it's": "it has / it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had / she would",
    "she'd've": "she would have",
    "she'll": "she shall / she will",
    "she'll've": "she shall have / she will have",
    "she's": "she has / she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as / so is",
    "that'd": "that would / that had",
    "that'd've": "that would have",
    "that's": "that has / that is",
    "there'd": "there had / there would",
    "there'd've": "there would have",
    "there's": "there has / there is",
    "they'd": "they had / they would",
    "they'd've": "they would have",
    "they'll": "they shall / they will",
    "they'll've": "they shall have / they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had / we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what shall / what will",
    "what'll've": "what shall have / what will have",
    "what're": "what are",
    "what's": "what has / what is",
    "what've": "what have",
    "when's": "when has / when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where has / where is",
    "where've": "where have",
    "who'll": "who shall / who will",
    "who'll've": "who shall have / who will have",
    "who's": "who has / who is",
    "who've": "who have",
    "why's": "why has / why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had / you would",
    "you'd've": "you would have",
    "you'll": "you shall / you will",
    "you'll've": "you shall have / you will have",
    "you're": "you are",
    "you've": "you have"
    }

    # Regular expression for finding contractions
    contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys()))

    def expand_contractions(text,contractions_dict=contractions_dict):
        def replace(match):
            return contractions_dict[match.group(0)]
        return contractions_re.sub(replace, text)


    # Expanding Contractions in the reviews
    df['posts'] = df['posts'].apply(lambda x:expand_contractions(x))

    # Making the whole data in lower case
    df['posts'] = df['posts'].str.lower()

    #Define the text from which you want to replace the url with "".
    def remove_URL(text):
        """Remove URLs from a text string"""
        return re.sub(r"http\S+", "", text)

    # Removing URL's in the reviews
    df['posts'] = df['posts'].apply(lambda x:remove_URL(x))

    # removing Punctuations
    punctuation = """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~)"""
    print(punctuation)
    #remove punctuation
    df['posts'] = df['posts'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '' , x))

    #remove stopwords
    # import nltk
    # nltk.download('stopwords')
    # from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    stop_words.add('http')
    def remove_stopwords(text):
        return " ".join([word for word in str(text).split() if word not in stop_words])
    df['posts'] = df['posts'].apply(lambda x: remove_stopwords(x))

    #stemming
    # from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    def stem_words(text):
        return " ".join([stemmer.stem(word) for word in text.split()])
    df['posts'] = df['posts'].apply(lambda x: stem_words(x))

    # nltk.download('wordnet')
    # from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    def lemmatize_words(text):
        return " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    df['posts'] = df['posts'].apply(lambda text: lemmatize_words(text))


    # Removal of Frequent words
    # In the previos preprocessing step, we removed the stopwords based on language information. But say, if we have a domain specific corpus, we might also have some frequent words which are of not so much importance to us.
    # So this step is to remove the frequent words in the given corpus. If we use something like tfidf, this is automatically taken care of.
    # Let us get the most common words adn then remove them in the next step
    # from collections import Counter
    # cnt = Counter()
    # for text in df['posts'].values:
    #     for word in text.split():
    #         cnt[word] += 1
    
    # Getting count of most frequently occuring top 10 words:
    # cnt.most_common(10)
    return df

    


if __name__ == "__main__":
    df = pd.read_csv("data/raw_data.csv")
    df = preprocess_data(df)
    df.to_csv("data/processed_data.csv", index=False)
    print("Data preprocessing completed.")