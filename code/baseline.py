'''
authors: Dyon van der Ende, Eva den Uijl, Myrthe Buckens
a simple SDG classifier
'''

import argparse
import sys
import os
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

SDGs = ["3", "14"]

new_column_names = {
                    "scopus_abstract_retrieval.coverDate": "retrieval_date",
                    "_id": "id",
                    "query_title": "SDG_label",
                    "query_id": "target_label",
                    "scopus_abstract_retrieval.title": "title",
                    "scopus_abstract_retrieval.abstract": "abstract",
                    "scopus_abstract_retrieval.idxterms": "terms",
                    "scopus_abstract_retrieval.authkeywords": "keywords",
                    "scopus_abstract_retrieval.doi": "doi",
                    "scopus_abstract_retrieval.url": "article_url",
                    "unpaywall_response.free_fulltext_url": "free_article_url"
                    }


def check_path(path):
    if os.path.isfile(path) == False and os.path.isdir(path) == False:
        print(f"{path} is not valid")
        sys.exit()
        
    if os.path.isdir(path) and path[:-1] != '/':
        path += '/'
    
    return path
    
def read_data(path):
    with open(path) as infile:
        df = pd.read_csv(infile, delimiter=',')
        
    df = df.rename(columns=new_column_names)
    df = df.drop(columns=['retrieval_date', 'free_article_url', 'article_url', 'doi'])
    df['SDG_label'] = df['SDG_label'].str[-2::]
    df = df.drop_duplicates(subset=["id"], keep=False)
    return df
    
def select_SDGs(df, SDGs):
    df = df[df['SDG_label'].isin(SDGs)] 
    return df
    
def create_negative_class(df, SDGs):
    df.loc[~df['SDG_label'].isin(SDGs), 'SDG_label'] = '0' 
    return df
    
def train_test_split(df, SDGs, ratio):    
    train_df_list = list()
    test_df_list = list()
    SDGs.append("0")
    
    for SDG in SDGs:
        temp_df = select_SDGs(df, [SDG])
        length = temp_df.shape[0]
        train_size = int(ratio*length)
        train_df_list.append(temp_df.iloc[:train_size-1,:])
        test_df_list.append(temp_df.iloc[train_size:,:])
    
    train_df = train_df_list[0]
    for i in range(1, len(train_df_list)):
        train_df = train_df.append(train_df_list[i], ignore_index=True)
      
    test_df = test_df_list[0]
    for i in range(1, len(test_df_list)):
        test_df = test_df.append(test_df_list[i], ignore_index=True)
        
    train_df = train_df.sample(frac=1).reset_index(drop=True, inplace=True)
    test_df = test_df.sample(frac=1).reset_index(drop=True, inplace=True)
        
    return train_df, test_df
           
         
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path',
                        help='file path to data')
    parser.add_argument('output_path',
                        help='file path to output folder')

    args = parser.parse_args()
    data_path = check_path(args.data_path)
    output_path = check_path(args.output_path)    
    
    data = read_data(data_path)
    data = create_negative_class(data, SDGs)
    train_set, test_set = train_test_split(data, SDGs, 0.8)
    

    
if __name__ == '__main__':
    main()
