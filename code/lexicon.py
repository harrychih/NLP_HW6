#!/usr/bin/env python3

# CS465 at Johns Hopkins University.
# Module for constructing a lexicon of word attributes.

import logging
from pathlib import Path
from typing import Optional, Set

import torch
import math
from corpus import TaggedCorpus, BOS_WORD, EOS_WORD, OOV_WORD, Word

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

def build_lexicon(corpus: TaggedCorpus,
                  one_hot: bool =False,
                  embeddings_file: Optional[Path] = None,
                  log_counts: bool =False,
                  affixes: bool =False) -> torch.Tensor:
    """Returns a lexicon, implemented as a matrix Tensor
    where each row defines real-valued attributes for one of
    the words in corpus.vocab.  This is a wrapper method that
    horizontally concatenates 0 or more matrices that provide 
    different kinds of attributes."""

    matrices = [torch.empty(len(corpus.vocab), 0)]  # start with no features for each word

    if one_hot: 
        matrices.append(one_hot_lexicon(corpus))
    if embeddings_file is not None:
        matrices.append(embeddings_lexicon(corpus, embeddings_file))
    if log_counts:
        matrices.append(log_counts_lexicon(corpus))
    if affixes:
        matrices.append(affixes_lexicon(corpus))

    return torch.cat(matrices, dim=1)   # horizontally concatenate 

def one_hot_lexicon(corpus: TaggedCorpus) -> torch.Tensor:
    """Return a matrix with as many rows as corpus.vocab, where 
    each row specifies a one-hot embedding of the corresponding word.
    This allows us to learn features that are specific to the word."""

    return torch.eye(len(corpus.vocab))  # identity matrix

def embeddings_lexicon(corpus: TaggedCorpus, file: Path) -> torch.Tensor:
    """Return a matrix with as many rows as corpus.vocab, where 
    each row specifies a vector embedding of the corresponding word.
    
    The second argument is a lexicon file in the format of Homework 2 and 3, 
    which is used to look up the word embeddings.

    The lexicon entries BOS, EOS, OOV, and OOL will be treated appropriately
    if present.  In particular, any words that are not in the lexicon
    will get the embedding of OOL (or 0 if there is no such embedding).
    """

    vocab = corpus.vocab
    with open(file) as f:
        filerows, cols = [int(i) for i in next(f).split()]   # first line gives num of rows and cols
        matrix = torch.empty(len(vocab), cols)   # uninitialized matrix
        seen: Set[int] = set()                   # the words we've found embeddings for
        ool_vector = torch.zeros(cols)           # use this for other words if there is no OOL entry
        specials = {'BOS': BOS_WORD, 'EOS': EOS_WORD, 'OOV': OOV_WORD}

        # Run through the words in the lexicon, keeping those that are in the vocab.
        for line in f:
            first, *rest = line.strip().split("\t")
            word = Word(first)
            vector = torch.tensor([float(v) for v in rest])
            assert len(vector) == cols     # check that the file didn't lie about # of cols

            if word == 'OOL':
                assert word not in vocab   # make sure there's not an actual word "OOL"
                ool_vector = vector
            else:
                if word in specials:    # map the special word names that may appear in lexicon
                    word = specials[word]    
                w = vocab.index(word)   # vocab integer to use as row number
                if w is not None:
                    matrix[w] = vector  # fill the vector into that row
                    seen.add(w)

    # Fill in OOL for any other vocab entries that were not seen in the lexicon.
    for w in range(len(vocab)):
        if w not in seen:
            matrix[w] = ool_vector

    log.info(f"From {file.name}, got embeddings for {len(seen)} of {len(vocab)} word types")

    return matrix

def log_counts_lexicon(corpus: TaggedCorpus) -> torch.Tensor:
    """Return a feature matrix with as many rows as corpus.vocab, where each
    row represents a feature vector for the corresponding word w.
    There is one feature (column) for each tag in corpus.tagset.  The value of this
    feature is log(1+c) where c=count(t,w) is the number of times t emitted w in supervised
    training data.  Thus, if this feature has weight 1.0 and is the only feature,
    then p(w | t) will be proportional to 1+count(t,w), just as in add-1 smoothing."""

    from collections import defaultdict
    wordTagFreq = defaultdict(int)

    for sentence in corpus:
        senList = str(sentence).split()
        for tword in senList:
            try:
                word, tag = tword.split('/')
                word_idx = corpus.integerize_word(word)
                tag_idx = corpus.integerize_tag(tag)
                wordTagFreq[(word_idx, tag_idx)] += 1
            except:
                pass

    row = len(corpus.vocab)
    col = len(corpus.tagset)
    matrix = torch.zeros(row, col)

    for wt in wordTagFreq:
        c = wordTagFreq[wt]
        w_idx, t_idx = wt
        matrix[w_idx, t_idx] = math.log(1+c)
    
    log.info(f"From corpus, got feature matrix with {matrix.shape[0]} rows (number of vocabs) and {matrix.shape[1]} columns (number of tags)")

    return matrix
    

def affixes_lexicon(corpus: TaggedCorpus) -> torch.Tensor:
    """Return a feature matrix with as many rows as corpus.vocab, where each
    row represents a feature vector for the corresponding word w.
    Each row has binary features for common suffixes and affixes that the
    word has."""
    

    #row = len(corpus.vocab)
    ## list --> prefixes 
    ## list --> suffixes 
    ## corpus  words 
    ## for every word check if prefix or suffix to exists 
    ## set 1
   # V = len(corpus.vocab)
    common_prefixes = ['re', 'dis', 'over', 'un', 'mis', 'out', 'be', 'co', 'de', 'fore', 'inter', 'pre', 'sub', 'trans', 'under']
    common_suffixes = ['ise', 'ate', 'fy', 'en']
    V = len(corpus.vocab._objects)
    F = len(common_prefixes) + len(common_suffixes)
    matrix_ = torch.zeros(V, F)
    words_ = corpus.vocab._objects
    for i in range(len(words_)):
        for j in range(len(common_prefixes)):
            if common_prefixes[j] in words_[i]:
                matrix_[i][j] = 1 

        for k in range(len(common_suffixes)):
            if common_suffixes[k] in words_[i]:
                matrix_[i][k] = 1 
    
    return matrix_
            

   

def affixes_lexicon_2(corpus: TaggedCorpus, file_) -> torch.Tensor:
    
    prefix_l = 3 
    suffix_l = 3 
    prefix_dict = {}
    suffix_dict = {}
    f = open(file_,'r')
    lines = f.readlines()[1:]
    
    V = len(lines)
    
    for line in lines:
        line = line.split()
        curr_word = line[0]
        
        if len(curr_word) == 3: 
            prefix_word = curr_word[:3]
            suffix_word = curr_word[:3]
            if prefix_word not in prefix_dict.keys():
                prefix_dict[prefix_word] = 1 
            else:
                prefix_dict[prefix_word] +=1
            
            if suffix_word not in suffix_dict.keys():
                suffix_dict[suffix_word] = 1 
            else:
                suffix_dict[suffix_word] +=1 


        if len(curr_word) > 3: 
            prefix_word = curr_word[:3]
            suffix_word = curr_word[-3:]
            if prefix_word not in prefix_dict.keys():
                prefix_dict[prefix_word] = 1 
            else:
                prefix_dict[prefix_word] +=1
            
            if suffix_word not in suffix_dict.keys():
                suffix_dict[suffix_word] = 1 
            else:
                suffix_dict[suffix_word] +=1 
    
    for key, value in prefix_dict.copy().items():
        if value < 3:
            prefix_dict.pop(key)

    for key, value in suffix_dict.copy().items():
        if value < 3:
            suffix_dict.pop(key)
   
    all_prefix_suffix = []
    for key in prefix_dict.keys():
        all_prefix_suffix.append(key)
    
    for key in suffix_dict.keys():
        all_prefix_suffix.append(key)
    
    
    F = len(all_prefix_suffix)
   
    matrix_ = torch.zeros(V, F)
    
   
    for i in range(len(lines)):
        line = lines[i].split()
        curr_word = line[0] 
        prefix_word = curr_word[:3]
        suffix_word = curr_word[-3:]
       
        if prefix_word in all_prefix_suffix:
            j1 = all_prefix_suffix.index(prefix_word)
            matrix_[i][j1] = 1 
            
        if suffix_word in all_prefix_suffix:
            j2 = all_prefix_suffix.index(suffix_word)
            matrix_[i][j2] = 1 
   
    return matrix_
    



if __name__ == "__main__":
    trainPath = "../data/endev"
    lexiconPath = "../data/words-50.txt"
    train = TaggedCorpus(Path(trainPath))
    lexicon = build_lexicon(train, embeddings_file=Path(lexiconPath), log_counts=True, affixes=True)
    print(lexicon)
    # affix = affixes_lexicon(corpus=train)
    # print(affix)