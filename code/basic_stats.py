import pandas as pd 
import numpy as np 
from collections import Counter
import argparse
from spacy.lang.en import English
import statistics
import matplotlib.pyplot as plt

nlp = English()

def print_statistics(input_file):
    """
    Function to print statistics on the inputfile to the terminal 
    :param input_file: path to file
    :type input_file: string
    """
    df = pd.read_csv(input_file, encoding = 'utf-8')
    columns = df.columns
    labels = df.iloc[:, 2]
    
    print('Data statistics:')
    print('These are the columns in your dataset: ', columns, '\n')
    print('This is the distribution of the labels in your dataset: ', Counter(labels), '\n')
    print('This is the size of your dataset: ', df.shape, '\n')
    
    tokenizer = nlp.tokenizer
    
    titles = df['title'].values
    title_lengths = []
    for title in titles:
        length = len(tokenizer(title))
        title_lengths.append(length)

        
    print("average number of tokens per title:\t"+str(statistics.mean(title_lengths)))
    
    plt.ylabel('Tokens' )
    plt.boxplot(title_lengths)
    plt.savefig('titleLength')
    
    abstracts = df['abstract'].values
    abstract_lengths = []
    for abstract in abstracts:
        length = len(tokenizer(abstract))
        abstract_lengths.append(length)
        
    print("average number of tokens per abstract:\t"+str(statistics.mean(abstract_lengths)))
    
    plt.boxplot(abstract_lengths)
    plt.savefig('abstractLength')
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file',
                        help='file path to the input data to compute statistics of. Example path: "../data/SDG-1-17-basic-excl-target13.0.csv"')
    args = parser.parse_args()
    input_file = args.input_file

    print_statistics(input_file)

if __name__ == '__main__':
    main()
