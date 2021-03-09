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
import torch
from transformers import BertModel, BertTokenizer
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
    title_list = []
    
    #mapping features to matching columns
    feature_to_index = {'retrieval_date': 0, 'id': 1, 'SDG_label': 2, 'target_label': 3, 'title': 4}
    
    for index, row in df.iterrows():
        feature_dict = {}
        for feature_name in selected_features:
            #components_index = feature_to_index.get(feature_name)
            feature_dict[feature_name] = row[feature_name]
        features.append(feature_dict)  
        title_list.append(row['title'])
        gold.append(row['SDG_label'])
        
    return features, gold, title_list  

def create_classifier(train_features, train_targets, title_list):
    """
    Function to create a classifier. Variable 'model' denotes type of classifier.
    
    :param train_features: list with features extracted from training data
    :param train_targets: list with gold labels from training data 
    :type train_features: list
    :type train_targets: list 
    :return: trained model and vectors for features 
    
    """
    #selected model and vectorizer
    MODEL_NAME = 'bert-base-uncased'
    model = BertModel.from_pretrained(MODEL_NAME)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    sentence_vectors = []
    all_tokens = []
    all_token_ids = []
    
    for title in title_list: 
        # Use the bert tokenizer
        tokens = [tokenizer.cls_token] + tokenizer.tokenize(title) + [tokenizer.sep_token]
        all_tokens.append(tokens)

        # Convert the tokens to token ids
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        all_token_ids.append(token_ids)
        tokens_tensor = torch.tensor(token_ids).unsqueeze(0)

        # Get the bert output
        model.eval()  # turn off dropout layers
        output = model(tokens_tensor)

        # The model provides a vector of 768 dimensions for each token
        # This vector corresponds to the last layer of hidden states of bert
        vector = output[0].detach().numpy()[0]
        sentence_vectors.append(vector)

    
    #fitting the model to the features
    model.fit(sentence_vectors, train_targets)

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
    
    train_features, train_gold, titles = extract_features(train_set, selected_features)
    test_features, goldlabels, test_titles = extract_features(test_set, selected_features)
    
    model, vec = create_classifier(train_features, train_gold, titles)
    
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
    filename = args.test_set.replace(".csv", "-predictions-bert_title.csv")
    test.to_csv(filename, sep = ',', index = False)

if __name__ == '__main__':
    main()
