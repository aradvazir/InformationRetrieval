from hazm import stopwords_list, Stemmer, Lemmatizer
from sklearn.model_selection import train_test_split
import pandas as pd
import copy
import re


# In case the nltk library got you with error like ..... package not found use download method for your desired package, all packages that are needed were included here
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

PATH = './data2.csv'

class Preprocess:
    def __init__(self) -> None:
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    @staticmethod
    def load_data():
        X_data = pd.read_csv(PATH)
        y_data = X_data[['Author', 'Author_ID']]
        X_data = X_data.drop(columns=['Author', 'Author_ID'])
        return X_data, y_data


    @staticmethod
    def special_characters_remover(text: str):  # Eliminates all the special characters like {, . : ; }
        normalized_word = re.sub(r'[^\w\s]', '', text)
        return normalized_word
    
    @staticmethod
    def englisg_word_remover(text: str):
        # Regular expression to match English words
        # [a-zA-Z]+ matches one or more English alphabetic characters
        cleaned_text = re.sub(r'[a-zA-Z0-9]+', '', text)
        return cleaned_text

    @staticmethod
    def stop_word_remover(text: list):  # Eliminate stop words
        stop_words = stopwords_list()
        new_text = ""
        for word in text:
            if word not in stop_words:
                new_text += word
        return copy.deepcopy(new_text)

    @staticmethod
    def stemmer(text: str):  # Stemming the tokens
        tokens = text.split(' ')
        stemmer_obj = Stemmer()
        stemmed_tokens = [stemmer_obj.stem(token) for token in tokens]
        cleaned_text = " ".join(stemmed_tokens)
        return cleaned_text

    @staticmethod
    def lemmatizer(text: str):  # Lemmatize the tokens
        tokens = text.split(' ')
        lemmatizer_obj = Lemmatizer()
        lemmatized_tokens = [lemmatizer_obj.lemmatize(token) for token in tokens]
        cleaned_text = " ".join(lemmatized_tokens)
        return cleaned_text
    
    @staticmethod
    def preprocess():
        X_data, y_data = Preprocess.load_data()
        temp = []
        for data in X_data['Text']:
            # Special characters remover
            data = Preprocess.special_characters_remover(copy.deepcopy(data))
            # Englisg words remover
            data = Preprocess.englisg_word_remover(copy.deepcopy(data))
            # Stop word remover
            data = Preprocess.stop_word_remover(copy.deepcopy(data))
            # Stemming
            data = Preprocess.stemmer(copy.deepcopy(data))
            # Lemmatization
            data = Preprocess.lemmatizer(copy.deepcopy(data))
            # Append processed data
            temp.append(copy.deepcopy(data))
        X_data = copy.deepcopy(temp)
        X_data = pd.DataFrame(X_data)
        X_data.to_csv(encoding='utf-8-sig')
        
        return X_data, y_data

    
    def split_data(self)
        X_data, y_data = Preprocess.preprocess()
        
        # Split data based on labels
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_data, y_data, test_size=0.2, stratify=y_data['Author_ID'], random_state=42)