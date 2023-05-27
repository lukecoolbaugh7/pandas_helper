from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import nltk 

class group_comments: 
    
    def main(): 
        new_comment = preprocess_comment(new_comment)
        new_comment_bow = dictionary.doc2bow(new_comment)
        new_category = lda_model[new_comment_bow][0][0]

    def preprocess_comment(comment):
        """
        Tokenize, remove stop words, and apply stemming and lemmatization to a single comment.
        
        Tokenization is the process of splitting the text into individual words (tokens).
        
        Stop words are common words that do not carry much meaning and are often removed in text processing.
        
        Stemming is the process of reducing a word to its base or root form. It's a somewhat crude method 
        and often includes mistakes.
        
        Lemmatization is similar to stemming, but it considers the grammar of the word and tries to find 
        the root word instead of just getting to the root word by brute force methods. 
        """
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')

        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        def get_wordnet_pos(tag):
            """
            Map POS tag to first character lemmatize() accepts.
            """
            tag = tag[0].upper()
            tag_dict = {"J": wordnet.ADJ,
                        "N": wordnet.NOUN,
                        "V": wordnet.VERB,
                        "R": wordnet.ADV}

            return tag_dict.get(tag, wordnet.NOUN)

        tokenized_comment = word_tokenize(comment.lower())
        tokenized_comment = [lemmatizer.lemmatize(w, get_wordnet_pos(pos_tag([w])[0][1])) for w in tokenized_comment if not w in stop_words]

        return tokenized_comment
    
    def train_lda_model(df, comment_column, num_topics):
        """
        Train an LDA model and add a new column to the DataFrame with the category of each comment.
        
        This function takes as input a DataFrame and the name of the column containing the comments.
        It first preprocesses the comments using the preprocess_comment function, then trains an LDA model 
        with the specified number of topics, and assigns each comment to its most probable topic.
        
        The function returns the DataFrame with a new column containing the category of each comment, 
        as well as the trained LDA model.
        """
        df[comment_column] = df[comment_column].apply(preprocess_comment)

        dictionary = corpora.Dictionary(df[comment_column])
        corpus = [dictionary.doc2bow(comment) for comment in df[comment_column]]

        lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

        df['category'] = df[comment_column].apply(lambda comment: lda_model[dictionary.doc2bow(comment)][0][0])

        return df, lda_model
