'''
authors: Dyon van der Ende, Eva den Uijl, Myrthe Buckens
a simple SDG classifier

Based on https://github.com/cltl/ma-ml4nlp-labs/blob/main/code/assignment1/basic_system.ipynb

'''
import sys
import argparse
import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

#constants
MODEL_NAME = 'xlm-roberta-base' #huggingface transformers model
MAX_LEN = 75                    #max token length of title
STEP_SIZE = 1                   #short title by STEP_SIZE until MAX_LEN tokens
                                #higher value is faster feature extraction

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
    #initializing tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    #creating lists for gold data and features
    gold = []
    features = []
     
    num_rows = df.shape[0]
    current_row = 1
    for index, row in df.iterrows():
        #print progress
        print(f'feature extraction: \t{str(current_row/num_rows*100)[:4]}%', end='\r')
        
        title = row['title']       

        #decrease title length until max number of tokens is reached
        while len(tokenizer.tokenize(title)) > MAX_LEN -2:
            title = title[:-STEP_SIZE]
        
        
        title_features = classifier(title)  #vectorize title by RoBERTa model
        title_len = len(title_features[0])
                                       
        title_vector = title_features[0]
        vector_length = len(title_vector[0])

        for i in range(title_len, MAX_LEN):
            title_vector.append(vector_length*[0])
        
        out = np.array(title_vector).flatten()

        assert len(title_vector) == MAX_LEN, "too much tokens"

        features.append(out)  
        gold.append(row['SDG_label'])
        current_row+=1
        
    print()
    return np.array(features), np.array(gold)

def create_classifier(train_features, train_targets):
    """
    Function to create a classifier. Variable 'model' denotes type of classifier.
    
    :param train_features: list with features extracted from training data
    :param train_targets: list with gold labels from training data 
    :type train_features: list
    :type train_targets: list 
    :return: trained model and vectors for features 
    
    """
    model = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3, verbose=1))

    print("fitting model")
       
    model.fit(train_features, train_targets)

    return model     
         
        
def run_classifier(train_set, test_set):
    """
    Function to run the classifier and get the predicted labels. 
    
    :param train_set: training data 
    :param test_set: test data
    :type train_set: pandas dataframe
    :type test_set: pandas dataframe
    :return: predictions (list with predicted labels)
    
    """
    
    classifier = pipeline('feature-extraction', model=MODEL_NAME)
    
    print("train data")
    train_features, train_gold = extract_features(train_set, classifier)
    
    
    model = create_classifier(train_features, train_gold)
    
    #free up memory
    train_features = None
    
    print("test data")
    test_features, goldlabels = extract_features(test_set, classifier)
    
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
    
    print("preparing")
    
    #checking paths for arguments 
    train_set = check_path(args.train_set)
    test_set = check_path(args.test_set)
    output_path = check_path(args.output_path)
    
    #reading in data 
    train_set = read_data(train_set)
    test_set = read_data(test_set)    
    
    #running classifier and generating statistics on performance 
    predictions = run_classifier(train_set, test_set)
    
    #writing the predictions to a new file
    test = pd.read_csv(args.test_set, encoding = 'utf-8', sep = ',')
    test['prediction'] = predictions
    filename = args.test_set.replace(".csv", "-predictions_title.csv")
    test.to_csv(filename, sep = ',', index = False)

if __name__ == '__main__':
    main()
