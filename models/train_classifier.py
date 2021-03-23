#importing all required libraries
import sys
import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle

import nltk
nltk.download('wordnet')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix, classification_report


def load_data(database_filepath):
    """
    Function to load data from SQLite database
    
    Input: data_filepath: path to SQLite database
           table_name:name of table inside the database
    Output: X: a dataframe containing features
            y: a dataframe containing target variables
            category_names: list of the category column names
    """
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df.message
    y = df[df.columns[4:]]
    category_names = y.columns
    return X, y, category_names 

def tokenize(text):
    """
    Function to tokenize the text of messages
    
    Input:
    text: message to be tokenized
    Output:
    tokens: List of clean tokens extracted from the provided text
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls_present = re.findall(url_regex, text) #finding if there are any urls in the text
    for url in urls_present:
        text = text.replace(url, "urlplaceholder") #replacing all urls with the placeholder

    raw_token = word_tokenize(text) #tokenization
    lemmatizer = WordNetLemmatizer() #lemmatization

    tokens = []
    for i in raw_token:
        j = lemmatizer.lemmatize(i).lower().strip()
        tokens.append(j)

    return tokens    


def build_model():
    """
    Function to build a model pipeline
    
    Output:
    model: A ML model that processes text messages and applies a classifier
    """   
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    param= {
        'clf__estimator__n_estimators': [10],
        'clf__estimator__min_samples_split': [2],
    
    } #setting parameters for gridsearch
    model = GridSearchCV(pipeline, param_grid=param)
    return model


def evaluate_model(model, X_test, y_test, category_names):
    """
    Function to evaluate the model and print out the model performance
    
    Input:
    model: ML model we created in build model function
    X_test: Test set features
    y_test: Test set target variables
    category_names: label names
    """
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=category_names))

def save_model(model, model_filepath):
    """
    Function to save the trained model as a pickle file
    
    Input:
    model: trained model
    model_filepath: destination path of .pkl file
    
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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