#importing all the needed libraries 
import nltk
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')
'''
Training the Logistic Regression model on train.csv
'''
#loading data
def load_data(file_path):
  #file_path = 'train.csv'
  df = pd.read_csv(file_path, encoding='latin1')

  return df
'''
Preprocesses a text by removing special characters, numbers, converting to lowercase,
    tokenizing, removing stopwords, and applying stemming.

    Parameters:
    text (str): The input text to be preprocessed.

    Returns:
    str: The preprocessed text.
'''
#preprocessing text
def remove_spl_char(text):
  #removing numbers and spl char
  #text=re.sub(r'[^a-zA-Z\s]',' ',text)
  text = ''.join([c for c in text if c.isalpha() or c.isspace()])
  #converting to lower case
  text = text.lower()
  #tokenizing
  words = word_tokenize(text)
  #setting stopwords
  stop_words = set(stopwords.words('english'))
  #remove them
  good_words = [w for w in words if w not in stop_words]
  #stemming
  stemmer = PorterStemmer()
  #apply it
  poststem = [stemmer.stem(s) for s in good_words]
  final_txt = '  '.join(poststem)

  return final_txt
  
#splitting data
'''
Splits a DataFrame into training and testing sets for a sentiment analysis task.

    Parameters:
    df (DataFrame): The input DataFrame containing text data and sentiment labels.

    Returns:
    tuple: A tuple containing four elements - X_train, X_test, y_train, y_test.
           X_train (Series): Training data containing cleaned text.
           X_test (Series): Testing data containing cleaned text.
           y_train (Series): Training labels (sentiments).
           y_test (Series): Testing labels (sentiments).
'''
def split_data(df):
  X = df['clean_text']
  y = df['sentiment']

  X_train, X_test, y_train, y_test = train_test_split(X,
                                                      y,
                                                      test_size=0.2,
                                                      random_state=1)
  return X_train, X_test, y_train, y_test


#vectorize tfidf
'''
Vectorize text data using TF-IDF (Term Frequency-Inverse Document Frequency) representation.

    Parameters:
        X_train (list or array-like): Training text data.
        X_test (list or array-like): Test text data.

    Returns:
        tuple: A tuple containing:
            - X_train_vect (scipy.sparse.csr_matrix): TF-IDF vectors for training data.
            - X_test_vect (scipy.sparse.csr_matrix): TF-IDF vectors for test data.
            - vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer.
'''
def vectorize(X_train, X_test):
  vectorizer = TfidfVectorizer(max_features=1000)
  X_train_vect = vectorizer.fit_transform(X_train)
  X_test_vect = vectorizer.transform(X_test)
  return X_train_vect, X_test_vect,vectorizer  

#training the model
'''
Builds and trains a logistic regression model using the provided training data.

    Parameters:
    X_train_vect : Training data vectors (TF-IDF or other vectorized representation).
    y_train : Training labels (sentiments).

    Returns:
    LogisticRegression: Trained logistic regression model.
'''
def build_model(X_train_vect, y_train):
  model = LogisticRegression()
  model.fit(X_train_vect, y_train)
  return model


#prediction
'''
 Evaluates a trained model's performance on the testing data and provides a classification report.

    Parameters:
    model (LogisticRegression or other classifier): Trained classification model.
    X_test_vect : Testing data vectors (TF-IDF or other vectorized representation).
    y_test : Testing labels (sentiments).

    Returns:
    str: Classification report containing F1-score, recall, support, and precision.
'''
def evaluate_model(model,X_test_vect,y_test):
 y_pred = model.predict(X_test_vect)

 print("Length of y_test:", len(y_test))
 print("Length of y_pred:", len(y_pred))

 #evaluation-F1-SCORE,RECALL,SUPPORT,PRECISION
 report = classification_report(y_test, y_pred)
 return report 


#finding model accuracy
'''
Calculates and returns the accuracy of a model's predictions.

    Parameters:
    y_pred1 : Model's predicted labels.
    sentiment : True sentiment labels.

    Returns:
'''
def accuracy_of_model(y_pred1,sentiment):
  correct_pred = 0
  for x, y in zip(y_pred1, sentiment):
    if x == y:
        correct_pred += 1
  total_samples = len(y_pred1)
  accuracy = correct_pred / total_samples
  return accuracy
  
  
#trying hyperparameter tuning
'''
"""
    Perform hyperparameter tuning for a given model using GridSearchCV and evaluate the final model.

    Parameters:
        model (estimator): The machine learning model to tune and evaluate.
        X_train_vect : TF-IDF vectors for training data.
        X_test_vect : TF-IDF vectors for test data.
        y_train : True labels for the training data.
        y_test : True labels for the test data.

    Returns:
        float: Accuracy of the final model on the test data.
    """
'''
def hyperpara_tuning(model,X_train_vect,X_test_vect,y_train,y_test):
  param_grid={'C':[0.001,0.01,0.1,1,10],'penalty':['l2'],'solver': ['lbfgs', 'sag'],  # Trying both solvers
    'max_iter': [1000]}
  grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=5)
  #fit grid_search on training data
  grid_search.fit(X_train_vect, y_train)
  #get the best parameters
  best_params = grid_search.best_params_
  #train model on new best parameters 
  final_model = LogisticRegression(**best_params)
  final_model.fit(X_train_vect, y_train)
  #predict values and accuracy
  y_pred2 = final_model.predict(X_test_vect)
  accuracy = accuracy_of_model(y_pred2,y_test)
  return accuracy



def main():
  file_path = 'train.csv'
  df = load_data(file_path)

  df['text'] = df['text'].astype(str)
  df['clean_text'] = df['text'].apply(remove_spl_char)
  X_train, X_test, y_train, y_test = split_data(df)
  X_train_vect, X_test_vect,vectorizer = vectorize(X_train, X_test)
  model = build_model(X_train_vect, y_train)
  report=evaluate_model(model,X_test_vect,y_test)
  print(report)
  #trying model on trained model and vectorizer
  
  df_test = load_data('testnew.csv')
  dff = load_data('test.csv')

  df_test['text'] = df_test['text'].astype(str)
  df_test['clean_text'] = df_test['text'].apply(remove_spl_char)

  X_test1 = df_test['clean_text']
  X_test1_vect = vectorizer.transform(X_test1)

  y_pred1 = model.predict(X_test1_vect)

 #np.set_printoptions(threshold=10000)
 # print(y_pred1)
  
  #printing accuracy
  accuracy=accuracy_of_model(y_pred1,dff['sentiment'])
  print("accuracy without hyperparameter tuning",accuracy)

  #hyperparameter tuning 
  new_accuracy=hyperpara_tuning(model,X_train_vect,X_test_vect,y_train,y_test)
  print("accuracy with hyperparameter tuning",new_accuracy)
 
  



 
  

if __name__ == "__main__":
    main()





  
