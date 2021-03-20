import pandas as pd 
import numpy as np 
from collections import Counter
import argparse
import statistics
import matplotlib.pyplot as plt
from pytorch_pretrained_bert import BertTokenizer


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
    
    return df
    
def create_plots(df):
    """
    Function to create boxplots based on the statistics of the input file
    :param df: dataframe with input data 
    :type df: pandas dataframe 
    """
    
    # initializing tokenizer from BERT multilingual model 
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    
    titles = df['title'].values
    title_lengths = []
    for title in titles:
        length = len(tokenizer.tokenize(title))
        title_lengths.append(length)

    # printing statistics on tokens per title     
    print("average number of tokens per title:\t"+str(statistics.mean(title_lengths)))
    
    # creating boxplot for token distribution in title
    plt.ylabel('Tokens' )
    plt.boxplot(title_lengths)
    plt.savefig('titleLength')
    
    abstracts = df['abstract'].values
    abstract_lengths = []
    for abstract in abstracts:
        length = len(tokenizer.tokenize(abstract))
        abstract_lengths.append(length)
    
    # printing statistics on tokens per abstract 
    print("average number of tokens per abstract:\t"+str(statistics.mean(abstract_lengths)))
    
    # creating boxplot for token distribution in abstract
    plt.boxplot(abstract_lengths)
    plt.savefig('abstractLength')
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file',
                        help='file path to the input data to compute statistics of. Example path: "../data/SDG-1-17-basic-excl-target13.0.csv"')
    args = parser.parse_args()
    input_file = args.input_file

    df = print_statistics(input_file)
    create_plots(df) 
    

if __name__ == '__main__':
    main()
