# This file illustrates how you might experiment with the HMM interface at the prompt.
# You can also run it directly.

import logging
import math
import os
from pathlib import Path
from typing import Callable

from corpus import TaggedCorpus
from eval import eval_tagging, model_cross_entropy, model_error_rate
from hmm import HiddenMarkovModel
from lexicon import build_lexicon
import torch

# Set up logging
log = logging.getLogger("test_en")       # For usage, see findsim.py in earlier assignment.
logging.basicConfig(format="%(levelname)s : %(message)s", level=logging.INFO)  # could change INFO to DEBUG
# torch.autograd.set_detect_anomaly(True)    # uncomment to improve error messages from .backward(), but slows down

# Get the corpora
os.chdir("../data")    # set working directory to the directory where the data live
ictrain = TaggedCorpus(Path("icsup"), Path("icraw"))                               # all training
icsup =   TaggedCorpus(Path("icsup"), tagset=ictrain.tagset, vocab=ictrain.vocab)  # supervised training
icdev =   TaggedCorpus(Path("icdev"), tagset=ictrain.tagset, vocab=ictrain.vocab)  # evaluation
log.info(f"Tagset: f{list(ictrain.tagset)}")
known_vocab = TaggedCorpus(Path("icsup")).vocab    # words seen with supervised tags; used in evaluation

# Initialize an HMM
lexicon = build_lexicon(ictrain, embeddings_file=Path('words-50.txt'), log_counts=True)  # works better with more attributes!
hmm = HiddenMarkovModel(ictrain.tagset, ictrain.vocab, lexicon)

# Let's initialize with supervised training to approximately maximize the
# regularized log-likelihood.  If you want to speed this up, you can increase
# the tolerance of training (using the `tolerance` argument), since we don't 
# really have to train to convergence.
loss_sup = lambda model: model_cross_entropy(model, eval_corpus=icsup)
hmm.train(corpus=icsup, loss=loss_sup, minibatch_size=30, evalbatch_size=10000, lr=0.0001, reg=1) 

# Now let's throw in the unsupervised training data as well, and continue
# training to try to improve accuracy on held-out development data.
# We'll stop when accuracy is getting worse, so we can get away without regularization,
# but it would be better to search for the best `reg` and other hyperparameters in this call.
loss_dev = lambda model: model_error_rate(model, eval_corpus=icdev, known_vocab=known_vocab)
hmm.train(corpus=ictrain, loss=loss_dev, minibatch_size=30, evalbatch_size=10000, lr=0.0001, reg=0)

# hmm = HiddenMarkovModel.load("my_hmm.pkl")

# More detailed look at the first 10 sentences in the held-out corpus,
# including Viterbi tagging.
Tnum = 0
Tdenom = 0
for m, sentence in enumerate(icdev):
    # if m >= 10: break
    viterbi = hmm.viterbi_tagging(sentence.desupervise(), icdev)
    counts = eval_tagging(predicted=viterbi, gold=sentence, 
                          known_vocab=known_vocab)
    num = counts['NUM', 'ALL']
    denom = counts['DENOM', 'ALL']
    Tnum += num
    Tdenom += denom
    log.info(f"Gold:    {sentence}")
    log.info(f"Viterbi: {viterbi}")
    log.info(f"Loss:    {denom - num}/{denom}")
    log.info(f"Prob:    {math.exp(hmm.log_prob(sentence, icdev))}")
log.info(f"Average Accuracy on first 10 sentences:    {Tnum/Tdenom}")