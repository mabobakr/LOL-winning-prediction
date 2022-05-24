import pandas as pd
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import train_test_split
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer

def separate_feature_label(data_frame, output_label):


    X = data_frame.drop( labels=[output_label], axis=1 )
    y = data_frame[output_label]
    return X,y

def test_this_model(model,newCols):
    test_data = pd.read_csv("test/test.csv").dropna()


    X_test,y_test = separate_feature_label(test_data, "blueWins")
    X_test = X_test[newCols]

    print( "Accuracy for tree classifier: "+str( metrics.accuracy_score(y_test, model.predict(X_test)) ) )
import numpy as np

def hello():
    print("hello34 ", np.random.randint(0,1000))

