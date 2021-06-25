from inspect import CO_VARARGS
import sys
import pandas as pd
import numpy as np
import sqlalchemy as db
from sqlalchemy import create_engine

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = stopwords.words("english")

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV

#from joblib import dump, load
#import pickle
#from sklearn.externals import joblib
import joblib

#from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import OneHotEncoder

from tokenize_class.tokenize_class import tokenize_class
#from tokenize_class.tokenize_class import ColumnSelector

def load_data(database_filepath):
    """
    Load Data from database
    Split into Features X and Target Y (Last 36 columns)
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)

    X = df.iloc[:,1]
    Y = df.iloc[:,4:40]
    category_names = Y.columns
    return X, Y, category_names


#def tokenize(text):
#     # normalize case and remove punctuation
#    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower().strip())
#    
#    # tokenize text
#    tokens = word_tokenize(text)
#    
#    # lemmatize and remove stop words
#    lemmatizer = WordNetLemmatizer()
#    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

#    return clean_tokens


def build_model():
    """Create pipeline with CountVectorizer, TfidfTransformer, MultioutputClassifier and RandomForestClassifier"""
    parameters = {
        #'vect__ngram_range': ((1, 1)),
        #'vect__max_df': (0.9, 1.0),
        'vect__max_features': (None, 950),
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [20, 100],
        'clf__estimator__min_samples_split': [0.1],
    }
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize_class().tokenize)),
        ('tfidf', TfidfTransformer(use_idf = False)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(min_samples_split = 3,random_state=42 )))
    ])
    cv = GridSearchCV(pipeline, parameters, verbose = 3)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred,target_names=category_names))

    return Y_pred


def save_model(model, model_filepath):
    #outfile = open(model_filepath,'wb')
    #pickle.dump(model, outfile)
    #outfile.close()
    joblib.dump(model, model_filepath)



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()