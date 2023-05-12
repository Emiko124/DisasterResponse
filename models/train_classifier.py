import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
import sqlite3
import re
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('wordnet')
nltk.download('words')
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score, recall_score, precision_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV,cross_val_score, cross_validate
from sklearn.metrics import fbeta_score, make_scorer, SCORERS


def load_data(database_filepath):
    """
    load data from the database and split them to X and Y for training and testing
    
    input: database filepath
    
    output: X,Y and category names
    
    """
    # read the data from the sql database
    tablename = 'DisasterResponse'
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(tablename, engine)
    X = df['message'].values
    Y = df[df.columns[4:]]
    #extract category names
    category_names = list(Y.columns)
    
    return X, Y, category_names


def tokenize(text):
    """
    loads the text and convert them to lower case and apply Lemmatizer to group together different forms of the same word
    
    input: text
    
    output:a lemmatized text (lem_text)
    
    """
    #normalize text
    text = text.lower()
    #remove special characterss
    text = re.sub(r'[^a-zA-Z0-9]', ' ',text)
    #tokenize text
    text = word_tokenize(text)
    #remove stopwords
    text = [w for w in text if w not in stopwords.words('english')]
    #lemmatize text
    lemm_txt = [WordNetLemmatizer().lemmatize(w, pos = 'v') for w in text]
    
    return lemm_txt
    

def build_model():
    """
    Uses multiple machine learning model to transform and classify text 
    
    """
    #build the pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    #state the parameters
    parameters = {
        'clf__estimator__oob_score': [True],
        'clf__estimator__min_samples_split': [2, 4]     
    }
    #initiate the Grid search
    cv = GridSearchCV(pipeline, parameters)
    
    return cv
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    """
    evaluate the model
    
    input: the model, X_test, Y_test and category names
    
    output: category names, y_pred and the accuracy score
    
    """
    #predict the y variable using the model
    y_pred = model.predict(X_test)
    
    for i in range(len(category_names)):
        
        print('Category: {} '.format(category_names[i]))
        # print the y_pred
        print(classification_report(Y_test.iloc[:, i].values, y_pred[:, i]))
        # print the accuracy score
        print('Accuracy {}\n\n'.format(accuracy_score(Y_test.iloc[:, i].values, y_pred[:, i])))

    pass


def save_model(model, model_filepath):
    """
    save the model in a pickle file
    """
    pickle.dump(model, open(model_filepath, 'wb'))
    pass


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