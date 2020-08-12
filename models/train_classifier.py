# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import multioutput
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support
import pickle
import re
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

def load_data(database_filepath):
    
    """Load the filepath and return the data"""
    
    engine = create_engine('sqlite:///' + database_filepath)
    #engine = create_engine('sqlite:///%s' % database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df.iloc[:,4:]
    #Y=df.drop(['id', 'message', 'original', 'genre'], axis=1)
    #Y['related'] = Y['related'].map(lambda x: min(x, 1))
    category_names    = list(np.array(Y.columns))
    Y['related'] = Y['related'].map(lambda x: min(x, 1))
    return X, Y, category_names, 

#word processing
def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    #words = [w for w in tokens if w not in stopwords.tokens('english')]    
    cleaned_text = [lemmatizer.lemmatize(w, pos='n').strip() for w in tokens]
    processed_txt = [lemmatizer.lemmatize(w, pos='v').strip() for w in cleaned_text]
    return processed_txt
    

#model instantiation and training
def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier (RandomForestClassifier()))
        ])


    parameters = {'clf__estimator__max_depth': [None,50],
                'clf__estimator__min_samples_leaf':[2, 5, 10]}
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2)
    return cv
    #return model

#model evaluation
def evaluate_model(cv, true, pred):
    #test_pred = cv.predict (X_test)
    results = pd.DataFrame(columns=['Category', 'f_score', 'precision', 'recall'])
    num = 0
    #model scoring
    results = pd.DataFrame(columns=['Category', 'f_score', 'precision', 'recall'])
    num = 0
    for cat in true.columns:
        precision, recall, f_score, support = precision_recall_fscore_support(true[cat], pred[:,num], average='weighted')
        results.set_value(num+1, 'Category', cat)
        results.set_value(num+1, 'f_score', f_score)
        results.set_value(num+1, 'precision', precision)
        results.set_value(num+1, 'recall', recall)
        num += 1
    print('Average f_score:', results['f_score'].mean())
    print('AVerage precision:', results['precision'].mean())
    print('Average recall:', results['recall'].mean())
    #return results
    

    #save tarined model
def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    
    if len(sys.argv) == 3:

        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 10)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        y_pred = model.predict(X_test)
        evaluate_model(model, Y_test, y_pred)

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