
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = stopwords.words("english")


class tokenize_class(object):
   # def __init__(self, text):
   #     self.text = text
        
    #def __call__(self):
    #    self.clean_tokens = self.tokenize(self.text)
    #    return self.clean_tokens
        
    def tokenize(self, text):
         # normalize case and remove punctuation
        #text = self.text
        text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower().strip())
    
        # tokenize text
        tokens = word_tokenize(text)
    
        # lemmatize and remove stop words
        lemmatizer = WordNetLemmatizer()
        clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

        return clean_tokens