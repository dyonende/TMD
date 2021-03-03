'''
authors: Dyon van der Ende, Eva den Uijl, Myrthe Buckens
a simple SDG classifier

Based on https://github.com/cltl/ma-ml4nlp-labs/blob/main/code/assignment1/basic_system.ipynb

'''

import argparse
import sys
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
import torch
from transformers import AutoModel, AutoTokenizer, pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

MODEL_NAME = 'xlm-roberta-base'


def read_data(path):
    """
    Function that reads in the data and returns a dataframe. 
    
    :param path: path to data file
    :type path: string
    :return: pandas dataframe 
    """
    with open(path, encoding = 'utf-8') as infile:
        df = pd.read_csv(infile, delimiter=',')
        
    return df 

def check_path(path):
    """
    Function that checks the input format of a path. 
    
    :param path: path to data file or folder
    :type path: string
    :return: correct path to file or folder
    
    """ 
    if os.path.isfile(path) == False and os.path.isdir(path) == False:
        print(f"{path} is not valid")
        sys.exit()
        
    if os.path.isdir(path) and path[:-1] != '/':
        path += '/'
    
    return path
    
    
def extract_features(df, classifier):
    """
    Function to extract features from data.
    
    :param df: a pandas dataframe
    :returns: the selected features and the gold labels in the data
    
    """
    gold = []
    features = []
    MAX_LEN = 20
    num_rows = df.shape[0]
    current_row = 1
    for index, row in df.iterrows():
        feature_dict = dict()
        print(f'feature extraction: \t{str(current_row/num_rows*100)[:4]}%', end='\r')
        title = row['title']
        title_features = classifier(title)
        title_len = len(title_features[0])
        if title_len > MAX_LEN:
            title_len = MAX_LEN
        vector_len = len(title_features[0][0])
        counter = 0
        for i in range(title_len):
            for j in range(vector_len):
                feature_dict[counter] = title_features[0][i][j]
                counter+=1
                
        for i in range(title_len, MAX_LEN):
            for j in range(vector_len):
                feature_dict[counter] = 0.000000001
                counter+=1
                    
        features.append(feature_dict)  
        gold.append(row['SDG_label'])
        current_row+=1
        
    print()
    return features, gold  

def create_classifier(train_features, train_targets):
    """
    Function to create a classifier. Variable 'model' denotes type of classifier.
    
    :param train_features: list with features extracted from training data
    :param train_targets: list with gold labels from training data 
    :type train_features: list
    :type train_targets: list 
    :return: trained model and vectors for features 
    
    """
    #selected model and vectorizer
    model = svm.LinearSVC()
    vec = DictVectorizer()
    
    #vectorizing the selected features
    features_vectorized = vec.fit_transform(train_features)
    
    print("predicting labels")
    
    #fitting the model to the features
    model.fit(features_vectorized, train_targets)

    return model, vec     
         
        
def run_classifier(train_set, test_set, selected_features):
    """
    Function to run the classifier and get the predicted labels. 
    
    :param train_set: training data 
    :param test_set: test data
    :param selected_features: the selected features for training the model
    :type train_set: pandas dataframe
    :type test_set: pandas dataframe
    :type selected_features: list 
    :return: predictions (list with predicted labels)
    
    """
    classifier = pipeline('feature-extraction', model=MODEL_NAME)
    print("train data")
    train_features, train_gold = extract_features(train_set, classifier)
    
    
    print("creating classifier")
    model, vec = create_classifier(train_features, train_gold)
    
    train_features = None
    
    print("test data")
    test_features, goldlabels = extract_features(test_set, classifier)
    
    test_features = vec.transform(test_features)
    predictions = model.predict(test_features)
    
    
    return predictions



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_set',
                        help='file path to training data. Example path: "../data/SDG-training.csv"')
    parser.add_argument('test_set',
                        help='file path to test data. Example path: "../data/SDG-test.csv"')
    parser.add_argument('output_path',
                        help='file path to output folder. Example path: "../data/"')

    args = parser.parse_args()
    
    #checking paths for arguments 
    train_set = check_path(args.train_set)
    test_set = check_path(args.test_set)
    output_path = check_path(args.output_path)
    
    #reading in data 
    train_set = read_data(train_set)
    test_set = read_data(test_set)
    
    #selected features for training
    selected_features = ['title']
    
    #running classifier and generating statistics on performance 
    predictions = run_classifier(train_set, test_set, selected_features)
    
    #writing the predictions to a new file
    test = pd.read_csv(args.test_set, encoding = 'utf-8', sep = ',')
    test['prediction'] = predictions
    filename = args.test_set.replace(".csv", "-predictions_title.csv")
    test.to_csv(filename, sep = ',', index = False)

if __name__ == '__main__':
    main()
