'''
authors: Dyon van der Ende, Eva den Uijl, Myrthe Buckens
a sscript voor evaluating the predicted data 

'''

import argparse
import sys
import os
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


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
    

def read_data(path):
    """
    Function that reads in the data and returns a dataframe. 
    
    :param path: path to data file
    :type path: string
    :return: pandas dataframe 
    """
    with open(path, encoding = 'utf-8') as infile:
        df = pd.read_csv(infile, delimiter=',')
        
        predictions = df['prediction']
        goldlabels = df['SDG_label']
        
    return predictions, goldlabels

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
    
    print('----> CONFUSION MATRIX <----')
    print(confusion_matrix)
    return confusion_matrix


def print_precision_recall_fscore(predictions, goldlabels):
    '''
    Function that prints out precision, recall and f-score in a complete report

    :param predictions: predicted output by classifier
    :param goldlabels: original gold labels
    :type predictions: list
    :type goldlabels: list
    
    '''
    report = classification_report(goldlabels,predictions,digits = 3)
    
    print('----> CLASSIFICATION REPORT <----')
    print(report)

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_predictions',
                        help='file path to training data. Example path: "../data/SDG-test-predictions.csv"')


    args = parser.parse_args()
    
    #checking paths for arguments 
    test_predictions = check_path(args.test_predictions)

    #reading in data 
    predictions, goldlabels = read_data(test_predictions)

    #getting evaluations 
    print_confusion_matrix(predictions, goldlabels)
    print_precision_recall_fscore(predictions, goldlabels)
    
    
if __name__ == '__main__':
    main()
