######################################################
# Use these package versions
#!pip install torchtext==0.6.0 torch==1.13.1
######################################################


import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] =':16:8' #This is a command to reduce non-deterministic behavior in CUDA
import warnings
warnings.simplefilter("ignore", UserWarning)
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data import get_tokenizer
import sys
import argparse
from LanguageModel import LanguageModel
from torch import softmax

import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main():
    chkpt = "got_language_model"

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device: {}'.format(dev))

    logging.info("Loading tokenizer and vocab from vocab.pkl")    
    text_field = pickle.load(open("vocab.pkl", "rb"))
    vocab_size = len(text_field.vocab.itos)

    logging.info("Loading checkpoint {}".format(chkpt))
    lm = LanguageModel(vocab_size).to(dev)
    # lm.load_state_dict(torch.load(chkpt, map_location=torch.device('cpu')))
    lm.load_state_dict(torch.load(chkpt, map_location=torch.device('cpu')))
    lm.eval()


    p = "the night is dark and full of terrors"
    
    # Torch is a bit frustrating at times and some things that ought to be deterministic are not. 
    # This is an attempt to resolve that, but it doesn't work 100% of the time
    torch.use_deterministic_algorithms(True)
    seed = 42
    mlen = 150

    # torch.manual_seed(seed); np.random.seed(seed)
    # print("\n----------- Vanilla Sampling -----------")
    # print(sample(lm, text_field, prompt=p, max_len=mlen))
    
    # torch.manual_seed(seed); np.random.seed(seed)
    # print("\n------- Temp-Scaled Sampling 0.0001 -------")
    # print(sample(lm, text_field, prompt=p, temp=0.0001, max_len=mlen))
    
    # torch.manual_seed(seed); np.random.seed(seed)
    # print("\n------- Temp-Scaled Sampling 100 --------")
    # print(sample(lm, text_field, prompt=p, temp=100, max_len=mlen))
    
    # torch.manual_seed(seed); np.random.seed(seed)
    # print("\n----------- Top-k Sampling 1 -----------")
    # print(sample(lm, text_field, prompt=p, k=1, max_len=mlen))
    
    # torch.manual_seed(seed); np.random.seed(seed)
    # print("\n----------- Top-k Sampling 20 -----------")
    # print(sample(lm, text_field, prompt=p, k=20, max_len=mlen))
    
    # torch.manual_seed(seed); np.random.seed(seed)
    # print("\n----------- Top-p Sampling 0.001 -----------")
    # print(sample(lm, text_field, prompt=p, p=0.001, max_len=mlen))

    # torch.manual_seed(seed); np.random.seed(seed)
    # print("\n----------- Top-p Sampling 0.75 -----------")
    # print(sample(lm, text_field, prompt=p, p=0.75, max_len=mlen))

    # torch.manual_seed(seed); np.random.seed(seed)
    # print("\n----------- Top-p Sampling 1 -----------")
    # print(sample(lm, text_field, prompt=p, p=1, max_len=mlen))
    
    
    torch.manual_seed(seed); np.random.seed(seed)
    print("\n----------- Beam Search B=1 -----------")
    print(beamsearch(lm, text_field, prompt=p, beams=1, max_len=mlen))

    torch.manual_seed(seed); np.random.seed(seed)
    print("\n----------- Beam Search B=10 -----------")
    print(beamsearch(lm, text_field, prompt=p, beams=10, max_len=mlen))

    torch.manual_seed(seed); np.random.seed(seed)
    print("\n----------- Beam Search B=50 -----------")
    print(beamsearch(lm, text_field, prompt=p, beams=50, max_len=mlen))

    print()



############################################################################################
# TASK 1.1
############################################################################################

def beamsearch(model, text_field, beams=5, prompt="", max_len=50):
    model.to(dev)
    model.eval()
    
    with torch.no_grad():
        prompt_tokens = text_field.process([text_field.tokenize(prompt.lower())]).to(dev)

        # Init
        h = torch.zeros([beams, model.rnn.num_layers, 1, model.rnn.hidden_size]).to(dev)
        c = torch.zeros([beams, model.rnn.num_layers, 1, model.rnn.hidden_size]).to(dev)
        all_top_word_values = torch.zeros([beams, beams])
        all_top_word_indices = torch.zeros([beams, beams], dtype=int)
        
        # Process the prefix
        out, h, c = model(prompt_tokens, h[0], c[0])
        
        log_probs = torch.log(softmax(torch.squeeze(out[-1]), 0))
        combine_top_words = torch.topk(log_probs, beams)
        
        h = h.repeat([beams, 1, 1, 1])
        c = c.repeat([beams, 1, 1, 1])
        
        w = combine_top_words.indices
        
        text = prompt_tokens.view([1,-1]).repeat([beams,1])
        text = text.type(torch.IntTensor)
        text = torch.cat((text, w.view([-1,1])), 1)

        sample_len = max_len - len(prompt_tokens) - 1
        
        # Init k prev log prob as 0 because log(1) = 0.
        prev_log_prob = combine_top_words.values
        
        for _ in range(0, sample_len):
            for i in range(0, beams):
                # out, h[i], c[i] = model(w[i], h[i], c[i])
                out, h[i], c[i] = model(w[i].view([1,-1]), h[i], c[i])
                
                # accumulate log prob (add)
                log_probs = torch.log(softmax(torch.squeeze(out), 0))
                top_words_per_beam = torch.topk(log_probs, beams)
                
                all_top_word_values[i] = prev_log_prob[i] + top_words_per_beam.values
                all_top_word_indices[i] = top_words_per_beam.indices
                
            # Pick the top-k log prob
            flatten_all_top_word_values = all_top_word_values.view([-1])
            flatten_all_top_word_indices = all_top_word_indices.view([-1])
            
            top_words = torch.topk(flatten_all_top_word_values, beams)
            
            prev_log_prob = top_words.values
            
            history_indices = top_words.indices / beams
            
            
            # Update top texts
            new_text = text[history_indices.type(torch.LongTensor)]
            w = flatten_all_top_word_indices[top_words.indices]
            new_text = torch.cat((new_text, w.view([-1,1])), 1)
            text = new_text
            
        # Most likely text
        _, best_word_idx = torch.topk(top_words.values, 1)
        best_text_idx = int(top_words.indices[best_word_idx] / beams)
        
    return reverseNumeralize(text[best_text_idx], text_field)

############################################################################################
# TASK 1.2
############################################################################################

def sample(model, text_field, prompt="", max_len=50, temp=1.0, k=0, p=1):
    assert (k==0 or p==1), "Cannot combine top-k and top-p sampling"
    model.to(dev)
    model.eval()
    
    with torch.no_grad():
        prompt_tokens = text_field.process([text_field.tokenize(prompt.lower())]).to(dev)

        w = prompt_tokens

        # Init
        h = torch.zeros([model.rnn.num_layers, 1, model.rnn.hidden_size]).to(dev)
        c = torch.zeros([model.rnn.num_layers, 1, model.rnn.hidden_size]).to(dev)

        sample_len = max_len - len(prompt_tokens)
        
        for _ in range(0, sample_len):
            out, h, c = model(w, h, c)
            out = torch.div(out[-1], temp)
            dist = torch.squeeze(softmax(out, 1))
            
            if k != 0:
                top_k = torch.topk(dist, k)
                sum_top_k = torch.sum(top_k.values)
                dist = torch.zeros_like(dist)
                dist[top_k.indices] = top_k.values
                dist = torch.div(dist, sum_top_k)
            elif p != 1:
                sorted_dist = torch.sort(dist, descending=True)
                prefix_sum = torch.cumsum(sorted_dist.values, 0)
                cut_off_idx = (prefix_sum >= p).nonzero(as_tuple=True)[0][0]
                
                # Zero out words not in the min-p set
                dist[sorted_dist.indices[cut_off_idx+1 :]] = 0
                
                sum_top_p = prefix_sum[cut_off_idx]
                
                dist = torch.div(dist, sum_top_p)
            
            w = torch.distributions.Categorical(dist).sample().resize(1,1)
            prompt_tokens = torch.cat((prompt_tokens, w), 0)
        
    return reverseNumeralize(prompt_tokens, text_field)

############################################################################################

def reverseNumeralize(numeralized_string, text_field):
    strings = [text_field.vocab.itos[i] for i in numeralized_string]
    return " ".join(strings)

if __name__ == "__main__":
    main()