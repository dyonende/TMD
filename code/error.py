'''
authors: Dyon van der Ende, Eva den Uijl, Myrthe Buckens
a script for performing the error analysis

'''

import pandas as pd
import spacy
from collections import Counter
from spacy.lang.en.stop_words import STOP_WORDS

# English pipelines include a rule-based lemmatizer
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

error = pd.read_csv('../data/SDG-test-predictions-mBERT-abstract.csv', encoding = 'utf-8', sep = ',')

list14as0 = []
list3as0 = []
list0as3 = []
list0as14 = []
list14good = []
list3good = []


for gold, pred, abstract in zip(error['SDG_label'], error['prediction'], error['abstract']): 

    #creating lists for good predicted articles
    if gold == 3 and pred == 1:
        list3good.append(abstract)    
        
    if gold == 14 and pred == 2:
        list14good.append(abstract)        
    
    #creating lists for wrong predicted articles, per category
    if gold == 14 and pred == 0:
        list14as0.append(abstract)
        
    if gold == 3 and pred == 0:
        list3as0.append(abstract)
          
    if gold == 0 and pred == 1:
        list0as3.append(abstract)
        
    if gold == 0 and pred == 2:
        list0as14.append(abstract)

#creating strings for lemmatizing the abstracts
str14as0 = ""
str3as0 = ""
str0as3 = ""
str0as14 = ""
str14good = ""
str3good = ""

#concatenating all abstracts into one string per category
for abstracts in list14as0:
    str14as0 += abstracts + " "
    
for abstracts in list3as0:
    str3as0 += abstracts + " "
    
for abstracts in list0as3:
    str0as3 += abstracts + " "
    
for abstracts in list0as14:
    str0as14 += abstracts + " "
    
for abstracts in list3good:
    str3good += abstracts + " "
    
for abstracts in list14good:
    str14good += abstracts + " "

    
print('-----------------3 PREDICTED AS 0------------------')
lemmas3as0 = []
doc = nlp(str3as0)

#appending lemmas to cleaned list when lemma is not punctuation mark or on stopword list for English
lemmas3as0.append([token.lemma_ for token in doc if (token.is_punct == False and token.is_stop == False)])
for lemmalist in lemmas3as0:
    print(Counter(lemmalist).most_common(20))


print('-----------------14 PREDICTED AS 0------------------')
lemmas14as0 = []
doc = nlp(str14as0)

#appending lemmas to cleaned list when lemma is not punctuation mark or on stopword list for English
lemmas14as0.append([token.lemma_ for token in doc if (token.is_punct == False and token.is_stop == False)])
for lemmalist in lemmas14as0:
    print(Counter(lemmalist).most_common(20))
    

print('-----------------0 PREDICTED AS 3------------------')
lemmas0as3 = []
doc = nlp(str0as3)

#appending lemmas to cleaned list when lemma is not punctuation mark or on stopword list for English
lemmas0as3.append([token.lemma_ for token in doc if (token.is_punct == False and token.is_stop == False)])
for lemmalist in lemmas0as3:
    print(Counter(lemmalist).most_common(20))

print('-----------------0 PREDICTED AS 14------------------')
lemmas0as14 = []
doc = nlp(str0as14)

#appending lemmas to cleaned list when lemma is not punctuation mark or on stopword list for English
lemmas0as14.append([token.lemma_ for token in doc if (token.is_punct == False and token.is_stop == False)])
for lemmalist in lemmas0as14:
    print(Counter(lemmalist).most_common(20)))


print('-----------------3 PREDICTED GOOD------------------')
lemmas3good = []
doc = nlp(str3good)

#appending lemmas to cleaned list when lemma is not punctuation mark or on stopword list for English
lemmas3good.append([token.lemma_ for token in doc if (token.is_punct == False and token.is_stop == False)])
for lemmalist in lemmas3good:
    print(Counter(lemmalist).most_common(20))

print('-----------------14 PREDICTED GOOD------------------')
lemmas14good = []
doc = nlp(str14good)

#appending lemmas to cleaned list when lemma is not punctuation mark or on stopword list for English
lemmas14good.append([token.lemma_ for token in doc if (token.is_punct == False and token.is_stop == False)])
for lemmalist in lemmas14good:
    print(Counter(lemmalist).most_common(20))