# Part-of-speech Taggers

## Introduction
This is a Homework at JHU CS 665 - Natural Language Processing Course. In this project, we built three part-of-speech taggers: HMM (Hidden Markov Model), CRF(Condition Random Field), and CRF with BiRNN (Bidirectional Random Field) word embedding.

## Setup
Please run 
```conda env create -f nlp-class.yml``` under /code directory. And run  ```conda activate nlp-class```

## Training
Please see ```code/tagger.py``` for training three different models.

## Additional Info
You would find the model setting for HMM in ```code/hmm.py``` and CRF as well as Improved CRF (CRF with BiRNN) in ```code/crf.py```.

