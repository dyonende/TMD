'''
authors: Dyon van der Ende, Eva den Uijl, Myrthe Buckens
a preprocessing script that takes a csv file as input and creates a training and test dataset.

DISCLAIMER: the training and test data is selected randomly, every time this script is excecuted,
            different training and test data files are created. 
'''

import argparse
import pandas as pd

SDGs = ["03", "14"]

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


def load_data(input_file):
    """ 
    Preprocessing the input data and converting the column titles to more informative titles.
    
    :param path: path to data
    :type path: string 
    :return: loaded and converted pandas dataframe
    """
    
    with open(input_file, encoding = 'utf-8') as infile:
        df = pd.read_csv(infile, delimiter=',')
        
    df = df.rename(columns=new_column_names)
    df['SDG_label'] = df['SDG_label'].str[-2::]
    
    #dropping duplicates for training purposes 
    df = df.drop_duplicates(subset=["id"], keep=False)
    
    #replacing empty cells with empty string 
    df = df.fillna(value="")
    
    
    return df

def select_SDGs(df, SDGs):
    """
    Function to select the rows that contain a SDG in the SDGs list. 
    
    :param df: dataframe with the loaded data
    :param SDGs: list with selected SDGs 
    :return: pandas dataframe with selected SDGs data 
    """
    df = df[df['SDG_label'].isin(SDGs)] 
    
    return df

def create_negative_class(df, SDGs):
    """
    Function that creates a negative class in the data.
    All the SDG's that do not have a label that is in the selected SDGs list, 
    get the value 0 for negative class. 
    
    :param df: dataframe with the loaded data
    :param SDGs: list with selected SDGs
    :return: pandas dataframe with selected SDGs data and a negative class
    
    """
    df.loc[~df['SDG_label'].isin(SDGs), 'SDG_label'] = '0' 
    return df

def train_test_split(input_file, SDGs, ratio):  
    """
    Function to split the dataset into a training set and a test set.
    
    :param input_file: path to the datafile
    :param SDGs: list with the selected SDGs 
    :param ratio: the ratio for splitting the dataset into training and test set
    :type path: string
    :type SDGs: list
    :type ratio: float
    :return: pandas dataframe with test data, pandas dataframe with training data 
    """
    #checking if the ratio is below 1.0
    assert ratio < 1.0
    
    train_df_list = list()
    test_df_list = list()
    
    #appending the 0 category to SDGs list; needed for the negative class
    SDGs.append("0")
    
    #loading dataframe 
    df = load_data(input_file)
    df = create_negative_class(df, SDGs)
    
    for SDG in SDGs:
        temp_df = select_SDGs(df, [SDG])
        length = temp_df.shape[0]
        train_size = int(ratio*length)
        train_df_list.append(temp_df.iloc[:train_size-1,:])
        test_df_list.append(temp_df.iloc[train_size:,:])
    
    #creating training set
    train_df = train_df_list[0]
    for i in range(len(train_df_list)):
        train_df = train_df.append(train_df_list[i], ignore_index=True)
    
    #creating test set
    test_df = test_df_list[0]
    for i in range(len(test_df_list)):
        test_df = test_df.append(test_df_list[i], ignore_index=True)
    
    #resetting indices
    train_df = train_df.sample(frac=1)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.sample(frac=1)
    test_df = test_df.reset_index(drop=True)
        
    return train_df, test_df

def write_out(input_file):
    """
    Function that writes the preprocessed and split data to an output file.
    
    :param input_file: path to the data file
    :type input_file: string
    """
    #setting the ratio for splitting the data
    ratio = 0.8
    
    #loading the data 
    train_df, test_df = train_test_split(input_file, SDGs, ratio)
    
    #writing preprocessed data to file in folder 'data'
    train_df.to_csv('../data/SDG-training.csv')
    test_df.to_csv('../data/SDG-test.csv')
     

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file',
                        help='file path to the input data to preprocess. Example path: "../data/SDG-1-17-basic-excl-target13.0.csv"')
    args = parser.parse_args()

    write_out(args.input_file)

if __name__ == '__main__':
    main()
