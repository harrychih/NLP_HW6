# This file illustrates how you might experiment with the HMM interface at the prompt.
# You can also run it directly.

import logging
import math
import os
from pathlib import Path
from typing import Callable
from tqdm import tqdm
from corpus import TaggedCorpus
from eval import eval_tagging, model_cross_entropy, model_error_rate
from hmm import HiddenMarkovModel
from crf import CRFBiRNNModel
from lexicon import build_lexicon
import torch

# Set up logging
# log = logging.getLogger("test_en")       # For usage, see findsim.py in earlier assignment.
log = logging.getLogger("test_ic")       # For usage, see findsim.py in earlier assignment.
logging.basicConfig(format="%(levelname)s : %(message)s", level=logging.INFO)  # could change INFO to DEBUG
# torch.autograd.set_detect_anomaly(True)    # uncomment to improve error messages from .backward(), but slows down

# Get the corpora
os.chdir("../data")    # set working directory to the directory where the data live
entrain = TaggedCorpus(Path("ensup"), Path("enraw"))                               # all training
ictrain = TaggedCorpus(Path("icsup"), Path("icraw"))                               # all training
# ensup =   TaggedCorpus(Path("ensup"), tagset=entrain.tagset, vocab=entrain.vocab)  # supervised training
# ensup = TaggedCorpus(Path("ensup"))
# icsup = TaggedCorpus(Path("icsup"))
# endev =   TaggedCorpus(Path("endev"), tagset=entrain.tagset, vocab=entrain.vocab)

# log.info(f"Tagset: f{list(entrain.tagset)}")
# train = TaggedCorpus(*[Path('endev','enraw')])

known_vocab = TaggedCorpus(Path("ensup")).vocab    # words seen with supervised tags; used in evaluation


crf = CRFBiRNNModel.load("../model/ic_crf_birnn.pkl")
# hmm = HiddenMarkovModel.load('../model/en_hmm.pkl')

ensup =   TaggedCorpus(Path("ensup"), tagset=crf.tagset, vocab=crf.vocab)
icsup =   TaggedCorpus(Path("icsup"), tagset=crf.tagset, vocab=crf.vocab)
# lexicon = build_lexicon(ensup, embeddings_file=Path('../lexicon/words-50.txt'), log_counts=False)  # works better with more attributes!

endev =   TaggedCorpus(Path("endev"), tagset=crf.tagset, vocab=crf.vocab)  # evaluation
icdev =   TaggedCorpus(Path("icdev"), tagset=crf.tagset, vocab=crf.vocab)  # evaluation
# More detailed look at the first 10 sentences in the held-out corpus,
# including Viterbi tagging.
Tnum = 0
Tdenom = 0
for m, sentence in tqdm(enumerate(icdev)):
    if m >= 10: break
    # print(sentence)
    # assert False
    viterbi = crf.viterbi_tagging(sentence.desupervise(), icdev)
    counts = eval_tagging(predicted=viterbi, gold=sentence, 
                          known_vocab=known_vocab)
    num = counts['NUM', 'ALL']
    denom = counts['DENOM', 'ALL']
    Tnum += num
    Tdenom += denom
    log.info(f"Gold:    {sentence}")
    log.info(f"Viterbi: {viterbi}")
    log.info(f"Loss:    {denom - num}/{denom}")
    log.info(f"Prob:    {math.exp(crf.log_prob(sentence, icdev))}")
log.info(f"Average Accuracy on first 10 sentences:    {Tnum/Tdenom}")