'''
authors: Dyon van der Ende, Eva den Uijl, Myrthe Buckens
a BERT classifier
https://colab.research.google.com/drive/1ywsvwO6thOVOrfagjjfuxEf6xVRxbUNO#scrollTo=Cp9BPRd1tMIo
'''

import argparse
import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

#
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import io
import matplotlib.pyplot as plt

SDGs = ["03", "14"]
MAX_LEN = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    df['SDG_label'] = df['SDG_label'].str[-2::]
    df = df.drop_duplicates(subset=["id"], keep=False)
    df = df.fillna(value="")
    return df
    
def select_SDGs(df, SDGs):
    df = df[df['SDG_label'].isin(SDGs)] 
    return df
    
def create_negative_class(df, SDGs):
    df.loc[~df['SDG_label'].isin(SDGs), 'SDG_label'] = '0' 
    return df
    
def train_test_split(df, SDGs, ratio):  
    assert ratio < 1.0
    
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
    for i in range(len(train_df_list)):
        train_df = train_df.append(train_df_list[i], ignore_index=True)
      
    test_df = test_df_list[0]
    for i in range(len(test_df_list)):
        test_df = test_df.append(test_df_list[i], ignore_index=True)
        
    train_df = train_df.sample(frac=1)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.sample(frac=1)
    test_df = test_df.reset_index(drop=True)
        
    return train_df, test_df
    
def extract_abstracts_and_label(df):
    labels = []
    abstracts = []
    
    for index, row in df.iterrows():
        abstract_text = row['abstract'][:500]
        abstract_text = "[CLS] " +  abstract_text.replace(". ", ". [SEP] ")
        abstracts.append(abstract_text)
        labels.append(int(row['SDG_label']))
        
    return abstracts, labels  
    
def bert_preprocess(abstracts, labels):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    
    tokenized_abstracts = [tokenizer.tokenize(abstract) for abstract in abstracts]
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_abstracts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    
    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
      seq_mask = [float(i>0) for i in seq]
      attention_masks.append(seq_mask)   

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    labels = torch.tensor(labels)
    return input_ids, attention_masks, labels
    
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
      
        
def evaluate(predictions, gold):
    for i in range(len(gold)):
        print(predictions[i], gold[i])
        
    print(precision_recall_fscore_support(gold, predictions, average='macro'))
    print(confusion_matrix(gold, predictions))
         
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
    
    train, test = train_test_split(data, SDGs, 0.8)
    
    train_abstracts, train_labels = extract_abstracts_and_label(train)
    train_inputs, train_masks, train_labels =  bert_preprocess(train_abstracts, train_labels)

    test_abstracts, test_labels = extract_abstracts_and_label(test)
    test_inputs, test_masks, test_labels =  bert_preprocess(test_abstracts, test_labels)

    batch_size = 32
    
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    
    optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=2e-5,
                     warmup=.1)

    t = [] 

    # Store our loss and accuracy for plotting
    train_loss_set = []

    # Number of training epochs (authors recommend between 2 and 4)
    epochs = 4

    # trange is a tqdm wrapper around the normal python range
    for _ in trange(epochs, desc="Epoch"):
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
        # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            train_loss_set.append(loss.item())    
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()


            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

    print("Train loss: {}".format(tr_loss/nb_tr_steps))


    # Validation

    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    # Tracking variables 
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
          # Forward pass, calculate logit predictions
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))


if __name__ == '__main__':
    main()
