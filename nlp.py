from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import string
import nltk

def preprocess_text(text):
    """
    Preprocess text by converting to lowercase, removing punctuation, and removing stopwords.
    """
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def train_text_classifier(df, comment_column, target_column):
    """
    Train a text classification model.
    This function first pre-processes the comments, then converts them to numerical data using TF-IDF, 
    and finally trains a Multinomial Naive Bayes model.
    """
    df[comment_column] = df[comment_column].apply(preprocess_text)

    X = df[comment_column]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tfidf = TfidfVectorizer()
    model = MultinomialNB()

    text_clf = make_pipeline(tfidf, model)
    text_clf.fit(X_train, y_train)

    return text_clf
