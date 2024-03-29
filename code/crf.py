#!/usr/bin/env python3

# CS465 at Johns Hopkins University.
# Implementation of CRF-BiRNN.

from __future__ import annotations
import logging
from math import inf, log, exp, sqrt
from pathlib import Path
from typing import Callable, List, Optional, Tuple, cast, Any

import torch
from torch import Tensor as Tensor
from torch import tensor as tensor
from torch import optim as optim
from torch import nn as nn
from torch import cuda as cuda
from torch.nn import functional as F
from torch.nn.parameter import Parameter as Parameter
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from tqdm import tqdm # type: ignore
from logsumexp_safe import logaddexp_new, logsumexp_new

from corpus import (BOS_TAG, BOS_WORD, EOS_TAG, EOS_WORD, Sentence, Tag,
                    TaggedCorpus, Word)
from integerize import Integerizer
import sys

logger = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.
    # Note: We use the name "logger" this time rather than "log" since we
    # are already using "log" for the mathematical log!
logging.basicConfig(format="%(levelname)s : %(message)s", level=logging.INFO) 
# logger.setLevel(logging.DEBUG)

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available

patch_typeguard()   # makes @typechecked work with torchtyping

# Monkey patch: replace the old methods with our improved versions
if not hasattr(torch, 'logaddexp_old'):
    torch.logaddexp_old = torch.logaddexp  # save original def so we can call it above
    torch.logsumexp_old = torch.logsumexp  # save original def so we can call it above
torch.logaddexp = logaddexp_new
torch.Tensor.logaddexp = logaddexp_new
torch.logsumexp = logsumexp_new
torch.Tensor.logsumexp = logsumexp_new

###
# HMM tagger
###
class CRFBiRNNModel(nn.Module):
    """An implementation of an HMM, whose emission probabilities are
    parameterized using the word embeddings in the lexicon.
    
    We'll refer to the HMM states as "tags" and the HMM observations 
    as "words."
    """

    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 lexicon: Tensor,
                 unigram: bool = False,
                 withBirnn: bool = False):
        """Construct an HMM with initially random parameters, with the
        given tagset, vocabulary, and lexical features.
        
        Normally this is an ordinary first-order (bigram) HMM.  The unigram
        flag says to fall back to a zeroth-order HMM, in which the different
        positions are generated independently.  (The code could be extended
        to support higher-order HMMs: trigram HMMs used to be popular.)"""

        super().__init__() # type: ignore # pytorch nn.Module does not have type annotations

        # We'll use the variable names that we used in the reading handout, for
        # easy reference.  (It's typically good practice to use more descriptive names.)
        # As usual in Python, attributes starting with _ are intended as private;
        # in this case, they might go away if you changed the parametrization of the model.

        # We omit EOS_WORD and BOS_WORD from the vocabulary, as they can never be emitted.
        # See the reading handout section "Don't guess when you know."

        assert vocab[-2:] == [EOS_WORD, BOS_WORD]  # make sure these are the last two

        self.k = len(tagset)       # number of tag types
        self.V = len(vocab) - 2    # number of word types (not counting EOS_WORD and BOS_WORD)
        self.d = lexicon.size(1)   # dimensionality of a word's embedding in attribute space
        self.unigram = unigram     # do we fall back to a unigram model?
        self.hidden_size = 16      # hidden layer size
        self.withBirnn = withBirnn # if trained with Birnn
        

        self.tagset = tagset
        self.vocab = vocab
        self._E = lexicon[:-2]  # embedding matrix; omits rows for EOS_WORD and BOS_WORD

        # Useful constants that are invoked in the methods
        self.bos_t: Optional[int] = tagset.index(BOS_TAG)
        self.eos_t: Optional[int] = tagset.index(EOS_TAG)
        assert self.bos_t is not None    # we need this to exist
        assert self.eos_t is not None    # we need this to exist
        self.eye: Tensor = torch.eye(self.k)  # identity matrix, used as a collection of one-hot tag vectors

        self.init_params()     # create and initialize params


    @property
    def device(self) -> torch.device:
        """Get the GPU (or CPU) our code is running on."""
        # Why the hell isn't this already in PyTorch?
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return next(self.parameters()).device


    def _integerize_sentence(self, sentence: Sentence, corpus: TaggedCorpus) -> List[Tuple[int,Optional[int]]]:
        """Integerize the words and tags of the given sentence, which came from the given corpus."""

        # Make sure that the sentence comes from a corpus that this HMM knows
        # how to handle.
        if corpus.tagset != self.tagset or corpus.vocab != self.vocab:
            raise TypeError("The corpus that this sentence came from uses a different tagset or vocab")

        # If so, go ahead and integerize it.
        return corpus.integerize_sentence(sentence)

    def init_params(self) -> None:
        """Initialize params to small random values (which breaks ties in the fully unsupervised case).
        However, we initialize the BOS_TAG column of _WA to -inf, to ensure that
        we have 0 probability of transitioning to BOS_TAG (see "Don't guess when you know").
        See the "Parametrization" section of the reading handout."""

        # # See the reading handout section "Parametrization.""

        ThetaB = 0.01*torch.rand(self.k, self.d) 
        self._ThetaB = Parameter(ThetaB)    # params used to construct emission matrix

        WA = 0.01*torch.rand(1 if self.unigram # just one row if unigram model
                             else self.k,      # but one row per tag s if bigram model
                             self.k)          # one column per tag t
        # WA[:, self.bos_t] = -inf               # correct the BOS_TAG column
        self._WA = Parameter(WA)            # params used to construct transition matrix

        # Parameters for biRNN
        M = 0.01*torch.rand(self.hidden_size, 1+self.hidden_size+self._E.size(1))
        self.M = Parameter(M)
        MP = 0.01*torch.rand(self.hidden_size, 1+self.hidden_size+self._E.size(1))
        self.MP = Parameter(MP)

        # Tag Embedding Matrix: Need to learn
        # Might be better if we also learn the word embedding from lexicon
        T = 0.01*torch.rand(self.k, self._E.size(1))
        self._T = Parameter(T)

        # Parameters for computing 
        UA = 0.01*torch.rand(self.k,1+2*self.hidden_size+2*self._T.size(1))
        self.UA = Parameter(UA)
        UB = 0.01*torch.rand(self.k,1+2*self.hidden_size+self._E.size(1)+self._T.size(1))
        self.UB = Parameter(UB)

        

        # Parameters for potential functions
        self.Linear_PhiA = nn.Linear(self.k**2, self.k, bias=False)
        self.Linear_PhiB = nn.Linear(self.k**2, self.V, bias=False) 
        




    @typechecked
    def params_L2(self) -> TensorType[()]:
        """What's the L2 norm of the current parameter vector?
        We consider only the finite parameters."""
        l2 = tensor(0.0)
        for x in self.parameters():
            x_finite = x[x.isfinite()]
            l2 = l2 + x_finite @ x_finite   # add ||x_finite||^2
        return l2


    def updateAB(self) -> None:
        """Set the transition and emission matrices A and B, based on the current parameters.
        See the "Parametrization" section of the reading handout."""
        PhiA = self._WA
        
        #F.softmax(self._WA, dim=1)       # run softmax on params to get transition distributions
                                             # note that the BOS_TAG column will be 0, but each row will sum to 1
        if self.unigram:
            # A is a row vector giving unigram probabilities p(t).
            # We'll just set the bigram matrix to use these as p(t | s)
            # for every row s.  This lets us simply use the bigram
            # code for unigram experiments, although unfortunately that
            # preserves the O(nk^2) runtime instead of letting us speed 
            # up to O(nk).
            self.logPhiA = PhiA.repeat(self.k, 1)
            # self.A = A.repeat(self.k, 1)
        else:
            # A is already a full matrix giving p(t | s).
            self.logPhiA = PhiA
            # self.A = A

        WB = self._ThetaB @ self._E.t()  # inner products of tag weights and word embeddings
        logPhiB = WB
        # B = F.softmax(WB, dim=1)         # run softmax on those inner products to get emission distributions

        self.logPhiB = logPhiB 
        # self.logB[self.eos_t, :] = float('-inf')        
        # self.logB[self.bos_t, :] = float('-inf')     
        # self.B[self.eos_t, :] = 0        # but don't guess: EOS_TAG can't emit any column's word (only EOS_WORD)
        # self.B[self.bos_t, :] = 0        # same for BOS_TAG (although BOS_TAG will already be ruled out by other factors)


    # def printAB(self) -> None:
    #     """Print the A and B matrices in a more human-readable format (tab-separated)."""
    #     print("Transition matrix A:")
    #     col_headers = [""] + [str(self.tagset[t]) for t in range(self.A.size(1))]
    #     print("\t".join(col_headers))
    #     for s in range(self.A.size(0)):   # rows
    #         row = [str(self.tagset[s])] + [f"{self.A[s,t]:.3f}" for t in range(self.A.size(1))]
    #         print("\t".join(row))
    #     print("\nEmission matrix B:")        
    #     col_headers = [""] + [str(self.vocab[w]) for w in range(self.B.size(1))]
    #     print("\t".join(col_headers))
    #     for t in range(self.A.size(0)):   # rows
    #         row = [str(self.tagset[t])] + [f"{self.B[t,w]:.3f}" for w in range(self.B.size(1))]
    #         print("\t".join(row))
    #     print("\n")


    @typechecked
    def log_prob(self, sentence: Sentence, corpus: TaggedCorpus) -> TensorType[()]:
        """Compute the log probability of a single sentence under the current
        model parameters.  If the sentence is not fully tagged, the probability
        will marginalize over all possible tags.  

        When the logging level is set to DEBUG, the alpha and beta vectors and posterior counts
        are logged.  You can check this against the ice cream spreadsheet."""
        if self.withBirnn:
            h, hp = self.biRNN_forward(sentence, corpus)
            fa, fb = self.feature_extract(sentence, corpus, h, hp)
            return self.log_forward_biRNN(sentence, corpus, fa, fb)
        else:
            return self.log_forward(sentence, corpus) - self.log_forward(sentence.desupervise(), corpus)

    
    @typechecked
    def biRNN_forward(self, sentence: Sentence, corpus: TaggedCorpus) -> Tuple[list[TensorType[Any]], list[TensorType[Any]]]:
        '''Use Bidirectioanl RNN to construct vector embedding of the transition
        and emission in context for running CRF later'''
        sent = self._integerize_sentence(sentence, corpus)
        n = len(sentence) - 2

        # new params
        emptyh = torch.zeros(self.hidden_size)
        one = torch.ones(1)
        h = [torch.zeros(self.hidden_size) for _ in range(n+1)]
        hp = [torch.zeros(self.hidden_size) for _ in range(n+1)]
        emptyE = torch.zeros(self._E.size(1))
        # logger.info("Computing vector embeddings h and h' by using biRNN")
        # Forward
        for j in range(n+1):
            word_idx = sent[j+1][0]
            if j == 0:
                w_j = self._E[word_idx,:]
                vec =  torch.cat((one, emptyh, w_j), 0)
            elif j == n:
                vec = torch.cat((one, h[j-1], emptyE), 0)
            else:
                w_j = self._E[word_idx,:]
                vec = torch.cat((one, h[j-1], w_j), 0)
            h[j] = torch.sigmoid(self.M @ vec.unsqueeze(1)).t().squeeze()
        # backward
        for j in range(n, 0, -1):
            word_idx = sent[j+1][0]
            if j == n:
                vec = torch.cat((one, emptyE, emptyh), 0)
            else:
                w_j = self._E[word_idx,:]
                vec = torch.cat((one, w_j, hp[j+1]), 0)
            hp[j] = torch.sigmoid(self.MP @ vec.unsqueeze(1)).t().squeeze()
        
        return (h, hp)


    

    @typechecked
    def feature_extract(self, sentence: Sentence, corpus: TaggedCorpus, h: List[TensorType[Any]], hp: List[TensorType[Any]]) -> Tuple[List[TensorType[Any, Any]], List[TensorType[Any, Any]]]:
        """Extract feature matrix fa fb from h, h' we got in biRNN and run crf to get 
        conditional log likelihood objective later"""

        sent = self._integerize_sentence(sentence, corpus)
        n = len(sentence) - 2
        one = torch.ones(1)
        # h, hp = self.biRNN_forward(sentence, corpus)
        fA = [torch.empty(self.k, self.k**2) for _ in range(n+1)]
        fB = [torch.empty(self.k, self.k**2) for _ in range(n+1)]
        emptyh = torch.zeros(self.hidden_size)
        emptyE = torch.zeros(self._E.size(1))
        # print(len(sentence))
        # print(len(fA))
        for i in range(n+1):
            # _, prev_tag_idx = sent[i]
            curr_word_idx, _ = sent[i+1]
            vecA = []
            vecB = []
            for s in range(self.k):
                for t in range(self.k):
                    if i == 0:
                        vecA.append(torch.cat((one, h[i], self._T[s,:], self._T[t,:], emptyh.clone()), 0))
                        vecB.append(torch.cat((one, h[i], self._T[t,:], self._E[curr_word_idx,:], emptyh.clone()), 0))
                    elif i == 1:
                        vecA.append(torch.cat((one, h[i], self._T[s,:], self._T[t,:], emptyh.clone()), 0))
                        vecB.append(torch.cat((one, h[i], self._T[t,:], self._E[curr_word_idx,:], hp[i-1]), 0))
                    elif i == n:
                        vecA.append(torch.cat((one, h[i], self._T[s,:], self._T[t,:], hp[i-2]), 0))
                        vecB.append(torch.cat((one, h[i], self._T[t,:], emptyE.clone(), hp[i-1]), 0))
                    else:
                        vecA.append(torch.cat((one, h[i], self._T[s,:], self._T[t,:], hp[i-2]), 0))
                        vecB.append(torch.cat((one, h[i], self._T[t,:], self._E[curr_word_idx,:], hp[i-1]), 0))
            stEmbedd = torch.stack(vecA, dim=1)
            twEmbedd = torch.stack(vecB, dim=1)
            fA[i] = torch.sigmoid(self.UA @ stEmbedd)
            fB[i] = torch.sigmoid(self.UB @ twEmbedd)
            
            
        return (fA, fB)

    @typechecked
    def log_forward_biRNN(self, sentence: Sentence, corpus: TaggedCorpus, fA: List[TensorType[Any, Any]], fB: List[TensorType[Any, Any]]) -> TensorType[()]:
        """by using feature matrix fa fb, training with theta_a and theta_b to get the
        potension function, which is the conditional log likelihood objective"""

        sent = self._integerize_sentence(sentence, corpus)
        n = len(sentence) - 2
        logphiA = [torch.empty(self.k, self.k) for _ in range(n+1)]
        logphiB = [torch.empty(self.k, self.V) for _ in range(n+1)]
        alpha = [torch.tensor([float('-inf') for _ in range(self.k)]) for _ in sent]
        alpha_z = [torch.tensor([float('-inf') for _ in range(self.k)]) for _ in sent]
        # sent_desup = self._integerize_sentence(sentence.desupervise(), corpus)
        # cond_log_prob = torch.zeros(1)


        alpha[0][self.eos_t] = 0
        alpha_z[0][self.eos_t] = 0
 
        for i in range(n):
            curr_word, curr_tag = sent[i+1]
            logphiA[i] = self.Linear_PhiA(fA[i])
            logphiB[i] = self.Linear_PhiB(fB[i])
            if curr_tag is None:
                new_alpha = alpha[i].unsqueeze(1) + logphiA[i]
                alpha[i+1] = torch.logsumexp(new_alpha + logphiB[i][:,curr_word], dim=0, safe_inf=True)
            else:
                new_alpha = alpha[i] + logphiA[i][:,curr_tag]
                alpha[i+1][curr_tag] = torch.logsumexp(new_alpha + logphiB[i][curr_tag, curr_word], dim=0, safe_inf=True)

        logphiA[n] = self.Linear_PhiA(fA[n])
        logphiB[n] = self.Linear_PhiB(fB[n])
        alpha[n+1][self.eos_t] = torch.logsumexp(alpha[n] + logphiA[n][:, self.eos_t], dim=0, safe_inf=True)


        for i in range(n+1):
            curr_word, _ = sent[i+1]
            if i != n:
                new_alpha = alpha_z[i].unsqueeze(1) + logphiA[i]
                alpha_z[i+1] = torch.logsumexp(new_alpha + logphiB[i][:,curr_word], dim=0, safe_inf=True)
            else:
                alpha_z[i+1][self.eos_t] = torch.logsumexp(alpha_z[i] + logphiA[i][:, self.eos_t], dim=0, safe_inf=True)
        
            
        return alpha[n+1][self.eos_t] - alpha_z[n+1][self.eos_t]

    @typechecked
    def log_forward(self, sentence: Sentence, corpus: TaggedCorpus) -> TensorType[()]:
        '''The linear-chain CRF forward function, return the conditional log probability
        of the current sentence'''
        sent = self._integerize_sentence(sentence, corpus)

        n = len(sentence) - 2
        # path_cond_log_prob = torch.zeros(1)
        
        # Z = torch.zeros(1)
        # path_weight = [torch.empty(1) for _ in range(n+1)]
        # alpha = [torch.empty(self.k) for _ in range(n+1)]
        
        alpha = [torch.tensor([float('-inf') for _ in range(self.k)]) for _ in sent]  
        n = len(sentence) - 2
        # Supervised case
        if sent[1][1] is not None:
            prev_tag_idx = sent[0][1]
            alpha[0][prev_tag_idx] = 0
            for j in range(1,n+1):
                new_alpha = alpha[j].clone()
                
                curr_word_idx = sent[j][0]
                curr_tag_idx = sent[j][1]
                
                
                log_transition_weight = self.logPhiA[prev_tag_idx, curr_tag_idx].clone()
               
                log_emission_weight = self.logPhiB[curr_tag_idx, curr_word_idx].clone()

                log_prob_t = log_transition_weight + log_emission_weight

                new_alpha[curr_tag_idx] = alpha[j-1][prev_tag_idx] + log_prob_t
                alpha[j] = new_alpha
                
                prev_tag_idx = curr_tag_idx

            # EOS
            new_alpha = alpha[n+1].clone()
            
            log_transition_weight = self.logPhiA[prev_tag_idx, self.eos_t].clone()
            new_alpha[self.eos_t] = alpha[n][prev_tag_idx].clone() + log_transition_weight
            alpha[n+1] = new_alpha
            
            return alpha[n+1][self.eos_t]
            
        
        # Unsupervised case
        else:
            alpha[0][self.bos_t] = 0
            for j in range(n+1):
                curr_word_idx = sent[j+1][0]
                if j != n:
                    new_alpha = alpha[j].unsqueeze(1) + self.logPhiA
                    alpha[j+1] = torch.logsumexp(new_alpha + self.logPhiB[:, curr_word_idx].squeeze(), dim=0, safe_inf=True)
                else:
                    alpha[-1][self.eos_t] = torch.logsumexp(alpha[-2]+self.logPhiA[:,self.eos_t], dim=0, safe_inf=True)
            
            return alpha[-1][self.eos_t]



    def viterbi_tagging(self, sentence: Sentence, corpus: TaggedCorpus) -> Sentence:
        """Find the most probable tagging for the given sentence, according to the
        current model."""

        # Note: This code is mainly copied from the forward algorithm.
        # We just switch to using max, and follow backpointers.
        # I've continued to call the vector alpha rather than mu.

        # backpointer 
        backpointer = {}
        
        ## list of most probable tagging sequence 
        most_prob_tag_seq_list = []

        sent = self._integerize_sentence(sentence, corpus)
        n = len(sentence) - 2

        alpha = [torch.tensor([float("-inf") for _ in range(self.k)]) for _ in sent]

        if self.withBirnn:
            return self.viterbi_tagging_biRNN(sentence, corpus)
        
        else:
            alpha[0][self.bos_t] = 0
            for j in range(n+1):
                curr_word_idx = sent[j+1][0]
                if j != n:
                    new_alpha = alpha[j].unsqueeze(1) + self.logPhiA
                    weight = new_alpha + self.logPhiB[:, curr_word_idx].squeeze()
                    max_weight = torch.max(weight, 0)
                    alpha[j+1] = max_weight[0]
                    for curr, prev in enumerate(max_weight[1]):
                        backpointer[(j, curr)] = (j-1, prev.item())
                else:
                    max_weight = torch.max(alpha[-2].unsqueeze(1)+self.logPhiA, 0)
                    alpha[-1] = max_weight[0]
                    for curr, prev in enumerate(max_weight[1]):
                        backpointer[(j, curr)] = (j-1, prev.item())
            
            cur_pos = (n,self.eos_t)
            most_prob_tag_seq_list = [corpus.tagset[self.eos_t]]

            while cur_pos in backpointer:
                 most_prob_tag_seq_list.append(corpus.tagset[backpointer[cur_pos][1]])
                 cur_pos = backpointer[cur_pos]
            most_prob_tag_seq_list = most_prob_tag_seq_list[::-1]

            resList = [("_BOS_WORD_","_BOS_TAG_")]
            for i, tag in enumerate(most_prob_tag_seq_list[1:-1]):
                word = sentence[i+1][0]
                resList.append((word, tag))
            resList.append(("_EOS_WORD_","_EOS_TAG_"))
            # print(resList)
            res = Sentence(resList)

        return res
            
    def viterbi_tagging_biRNN(self, sentence: Sentence, corpus: TaggedCorpus) -> Sentence:
        """Find the most probable tagging for the given sentence, according to the
        current model."""
        # backpointer 
        backpointer = {}
        
        ## list of most probable tagging sequence 
        most_prob_tag_seq_list = []

        sent = self._integerize_sentence(sentence, corpus)
        n = len(sentence) - 2

        h, hp = self.biRNN_forward(sentence, corpus)
        fA, fB = self.feature_extract(sentence, corpus, h, hp)
        logphiA = [torch.empty(self.k, self.k) for _ in range(n+1)]
        logphiB = [torch.empty(self.k, self.V) for _ in range(n+1)]


        # alpha[0][prev_tag_idx] = 0
        alpha = [torch.tensor([float("-inf") for _ in range(self.k)]) for _ in range(n+1)]
        alpha[0][self.bos_t] = 0
        for j in range(n+1):
            curr_word_idx = sent[j+1][0]
            logphiA[j] = self.Linear_PhiA(fA[j])
            logphiB[j] = self.Linear_PhiB(fB[j])
            if j != n:
                new_alpha = alpha[j].unsqueeze(1) + logphiA[j]
                weight = new_alpha + logphiB[j][:, curr_word_idx].squeeze()
                max_weight = torch.max(weight, 0)
                alpha[j+1] = max_weight[0]
                for curr, prev in enumerate(max_weight[1]):
                    backpointer[(j, curr)] = (j-1, prev.item())
            else:
                max_weight = torch.max(alpha[-2].unsqueeze(1)+logphiA[j], 0)
                alpha[-1] = max_weight[0]
                for curr, prev in enumerate(max_weight[1]):
                    backpointer[(j, curr)] = (j-1, prev.item())


        eos_tag = sent[n+1][1]
        cur_pos = (n,eos_tag)
        most_prob_tag_seq_list = [corpus.tagset[eos_tag]]
        while cur_pos in backpointer:
            most_prob_tag_seq_list.append(corpus.tagset[backpointer[cur_pos][1]])
            cur_pos = backpointer[cur_pos]
        most_prob_tag_seq_list = most_prob_tag_seq_list[::-1]
        # print(most_prob_tag_seq_list)
        resList = [("_BOS_WORD_","_BOS_TAG_")]
        # resList = [()]
        for i, tag in enumerate(most_prob_tag_seq_list[1:-1]):
            word = sentence[i+1][0]
            resList.append((word, tag))
        resList.append(("_EOS_WORD_","_EOS_TAG_"))
        
        res = Sentence(resList)

        return res




    def train(self,
              corpus: TaggedCorpus,
              loss: Callable[[CRFBiRNNModel], float],
              tolerance: float =0.001,
              minibatch_size: int = 1,
              evalbatch_size: int = 500,
              lr: float = 1.0,
              reg: float = 0.0,
              max_iter: int = 50000,
              save_step: int = 5000,
              save_path: Path = Path("my_crf.pkl")) -> None:
        """Train the HMM on the given training corpus, starting at the current parameters.
        The minibatch size controls how often we do an update.
        (Recommended to be larger than 1 for speed; can be inf for the whole training corpus.)
        The evalbatch size controls how often we evaluate (e.g., on a development corpus).
        We will stop when evaluation loss is not better than the last evalbatch by at least the
        tolerance; in particular, we will stop if we evaluation loss is getting worse (overfitting).
        lr is the learning rate, and reg is an L2 batch regularization coefficient."""

        # This is relatively generic training code.  Notice however that the
        # updateAB step before each minibatch produces A, B matrices that are
        # then shared by all sentences in the minibatch.

        # All of the sentences in a minibatch could be treated in parallel,
        # since they use the same parameters.  The code below treats them
        # in series, but if you were using a GPU, you could get speedups
        # by writing the forward algorithm using higher-dimensional tensor 
        # operations that update alpha[j-1] to alpha[j] for all the sentences
        # in the minibatch at once, and then PyTorch could actually take
        # better advantage of hardware parallelism.


        assert minibatch_size > 0
        if minibatch_size > len(corpus):
            minibatch_size = len(corpus)  # no point in having a minibatch larger than the corpus
        assert reg >= 0

        old_dev_loss: Optional[float] = None    # we'll keep track of the dev loss here

        optimizer = optim.SGD(self.parameters(), lr=lr)  # optimizer knows what the params are
        if not self.withBirnn:
            self.updateAB()                                        # compute A and B matrices from current params
        log_likelihood = tensor(0.0, device=self.device)       # accumulator for minibatch log_likelihood
        for m, sentence in tqdm(enumerate(corpus.draw_sentences_forever())):
            # Before we process the new sentence, we'll take stock of the preceding
            # examples.  (It would feel more natural to do this at the end of each
            # iteration instead of the start of the next one.  However, we'd also like
            # to do it at the start of the first time through the loop, to print out
            # the dev loss on the initial parameters before the first example.)

            # m is the number of examples we've seen so far.
            # If we're at the end of a minibatch, do an update.
            if m % minibatch_size == 0 and m > 0:
                # with torch.autograd.detect_anomaly():
                # with torch.enable_grad():
                logger.debug(f"Training log-likelihood per example: {log_likelihood.item()/minibatch_size:.3f} nats")
                optimizer.zero_grad()          # backward pass will add to existing gradient, so zero it
                objective = -log_likelihood + (minibatch_size/corpus.num_tokens()) * reg * self.params_L2()
                objective.backward()           # type: ignore # compute gradient of regularized negative log-likelihod
                length = sqrt(sum((x.grad*x.grad).sum().item() for x in self.parameters()))
                logger.debug(f"Size of gradient vector: {length}")  # should approach 0 for large minibatch at local min
                optimizer.step()               # SGD step
                if not self.withBirnn:
                    self.updateAB()                # update A and B matrices from new params
                if m % evalbatch_size == 0:
                    logger.info(f"***running example {m} ------- log likelihood: {log_likelihood} ----------- dev_loss: {dev_loss}")
                log_likelihood = tensor(0.0, device=self.device)    # reset accumulator for next minibatch

            

            # If we're at the end of an eval batch, or at the start of training, evaluate.
            if m % evalbatch_size == 0:
                with torch.no_grad():       # type: ignore # don't retain gradients during evaluation
                    dev_loss = loss(self)   # this will print its own log messages
                # print(f"old_dev_loss: {old_dev_loss}, dev_loss: {dev_loss}")
                if old_dev_loss is not None and dev_loss >= old_dev_loss * (1-tolerance):
                    # we haven't gotten much better, so stop
                    logger.info("\t Early Stopping: ")
                    self.save(save_path)  # Store this model, in case we'd like to restore it later.
                    break
                if m > max_iter:
                    logger.info("\t stop training -- exceeds maximal iteration {max_iter}")
                    self.save(save_path)
                    break
                old_dev_loss = dev_loss            # remember for next eval batch
            # Save models every "save_setep" iteration
            if m % save_step == 0 and m != 0:
                logger.info(f"Saving the model every {save_step} iteration: currently {int(m / save_step)} times to save")
                self.save(save_path)
            # Finally, add likelihood of sentence m to the minibatch objective.
            log_likelihood = log_likelihood + self.log_prob(sentence, corpus)
            # print(log_likelihood)

                



    def save(self, destination: Path) -> None:
        import pickle
        with open(destination, mode="wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved model to {destination}")


    @classmethod
    def load(cls, source: Path) -> CRFBiRNNModel:
        import pickle  # for loading/saving Python objects
        logger.info(f"Loading model from {source}")
        with open(source, mode="rb") as f:
            result = pickle.load(f)
            if not isinstance(result, cls):
                raise ValueError(f"Type Error: expected object of type {cls} but got {type(result)} from pickled file.")
            logger.info(f"Loaded model from {source}")
            return result


if __name__ == "__main__":

    m = CRFBiRNNModel()
    # print(m.printAB())
    
