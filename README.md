# SDG Classifier 
The github for the sustainability project for the course Text Mining Domains at Vrije Universiteit Amsterdam (2021).
Students: Myrthe Buckens, Dyon van der Ende and Eva den Uijl

### data 
In the data folder, you will find the following files: 
* original data file provided by Aurora Universities Alliance: `SDG-1-17-basic-excl-target13.0.csv`
* preprocessed training data with 2 SDG's and negative class: `SDG-training.csv`
* preprocessed test data with 2 SDG's and negative class: `SDG-test.csv`
* preprocessed training data with all SDG's: `SDG-training-all.csv`
* preprocessed test data with all SDG's: `SDG-test-all.csv`
* file with predictions for baseline: `SDG-predictions-baseline.csv`
* file with predictions for RoBERTa on title: `SDG-predictions-title.csv`
* file with predictions for RoBERTa on abstract: `SDG-predictions-abstract.csv`



### requirements 
The needed requirements can be found in `requirements.txt` and installed by running
```pip install requirements.txt``` from your terminal.

### code
In the code folder, you will find the following scripts: 
* baseline system: `baseline.py`
* preprocessing script: `preprocessing.py`
* basic statistics on the data: `basic_stats.py`
* RoBERTa classifier with abstracts: `RoBERTa_classifier_abstract.py`
* RoBERTa classifier with titles: `RoBERTa_classifier_title.py`
* evaluation script: `evaluation.py`

For running the mBERT models, we used google colaboration. It is possible to download this code on top of the page as .ipynb or .py, but for speed and possible hardware limitations, we advise you run the code online with the CPU from google. 
The code can be found by following the links below: 
* mBERT with titles: https://colab.research.google.com/drive/15GPBozxuIhkKkZumHa-fs8lYR98QiDm8?usp=sharing
* mBERT with abstracts: https://colab.research.google.com/drive/1WZa20GjK1KDPhbPMR945y7f6Hvp2Tx_z?usp=sharing
* mBERT with abstracts + titles: https://colab.research.google.com/drive/1hTiPH8Vh9kOAruSdgWYLlr64CS_fR1IW?usp=sharing
* mBERT with all SDG's: https://colab.research.google.com/drive/1lHlTAh8GKA7BBYXH8Xg92dNStbj7HaRJ?usp=sharing


