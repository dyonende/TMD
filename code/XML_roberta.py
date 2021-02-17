"""
Code from: https://huggingface.co/transformers/multilingual.html?highlight=tokenizer%20roberta#xlm-roberta
and Language as Data Lab 5.3

"""

import pandas as pd 
import numpy as np
import torch
from transformers import XLMTokenizer, XLMWithLMHeadModel


tokenizer = XLMTokenizer.from_pretrained("xlm-mlm-tlm-xnli15-1024")
model = XLMWithLMHeadModel.from_pretrained("xlm-mlm-tlm-xnli15-1024")


path = "../data/SDG-training.csv"



with open(path) as infile:
    df = pd.read_csv(infile, delimiter=',')
    sentences = df['abstract'] 
    
    sentence_vectors = []
    all_tokens = []
    all_token_ids = []
    
    
    for s in sentences:
        
        #The first 128 characters and the rest
        sentence = s[:127] + " " + s[383:]


        tokens = [tokenizer.cls_token] + tokenizer.tokenize(sentence) + [tokenizer.sep_token]
        all_tokens.append(tokens)
        #print(tokens)

        # Convert the tokens to token ids
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        all_token_ids.append(token_ids)
       # print(token_ids)
        tokens_tensor = torch.tensor(token_ids).unsqueeze(0)

        # Get the bert output output
        model.eval()  # turn off dropout layers
        output = model(tokens_tensor)

        # The model provides a vector of 768 dimensions for each token
        # This vector corresponds to the last layer of hidden states of bert
        vector = output[0].detach().numpy()[0]
        sentence_vectors.append(vector)

        #print(vector.shape)
        #print(vector)

