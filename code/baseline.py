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
       

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_set',
                        help='file path to test data. Example path: "../data/SDG-test.csv"')
    parser.add_argument('output_path',
                        help='file path to output folder. Example path: "../data/"')

    args = parser.parse_args()
    
    #checking paths for arguments 
    test_set = check_path(args.test_set)
    output_path = check_path(args.output_path)
    
    #writing the predictions to a new file
    test = pd.read_csv(args.test_set, encoding = 'utf-8', sep = ',')
    test['prediction'] = 0
    filename = args.test_set.replace(".csv", "-predictions_baseline.csv")
    test.to_csv(filename, sep = ',', index = False)

if __name__ == '__main__':
    main()
