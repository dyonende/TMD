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
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


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
    
    
def extract_features(df, selected_features):
    """
    Function to extract features from data.
    
    :param df: a pandas dataframe
    :returns: the selected features and the gold labels in the data
    
    """
    gold = []
    features = []
    
    #mapping features to matching columns
    feature_to_index = {'retrieval_date': 0, 'id': 1, 'SDG_label': 2, 'target_label': 3, 'title': 4}
    
    for index, row in df.iterrows():
        feature_dict = {}
        for feature_name in selected_features:
            #components_index = feature_to_index.get(feature_name)
            feature_dict[feature_name] = row[feature_name]
        features.append(feature_dict)  
        gold.append(row['SDG_label'])
        
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
    
    #fitting the model to the features
    model.fit(features_vectorized, train_targets)

    return model, vec  

def print_confusion_matrix(predictions, goldlabels):
    '''
    Function that prints out a confusion matrix

    :param predictions: predicted labels
    :param goldlabels: gold standard labels
    :type predictions, goldlabels: list of strings

    :returns: confusion matrix
    '''

    # based on example from https://datatofish.com/confusion-matrix-python/
    data = {'Gold': goldlabels, 'Predicted': predictions}
    df = pd.DataFrame(data, columns=['Gold', 'Predicted'])

    confusion_matrix = pd.crosstab(df['Gold'], df['Predicted'], rownames=['Gold'], colnames=['Predicted'])
    print(confusion_matrix)
    return confusion_matrix


def print_precision_recall_fscore(predictions, goldlabels, selected_features):
    '''
    Function that prints out precision, recall and f-score in a complete report

    :param predictions: predicted output by classifier
    :param goldlabels: original gold labels
    :type predictions: list
    :type goldlabels: list
    
    '''
    report = classification_report(goldlabels,predictions,digits = 3)
    
    print('----> SVM with ' + ' and '.join(selected_features) + ' as features <----')
    print(report)
    
         
        
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
    
    train_features, train_gold = extract_features(train_set, selected_features)
    test_features, goldlabels = extract_features(test_set, selected_features)
    
    model, vec = create_classifier(train_features, train_gold)
    
    test_features = vec.transform(test_features)
    predictions = model.predict(test_features)
    
    print('CONFUSION MATRIX: ')
    print_confusion_matrix(predictions, goldlabels)
    
    print('METRICS: ')
    print_precision_recall_fscore(predictions, goldlabels, selected_features)
    
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
    filename = args.test_set.replace(".csv", "-predictions.csv")
    test.to_csv(filename, sep = ',', index = False)

if __name__ == '__main__':
    main()
