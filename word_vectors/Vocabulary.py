from collections import Counter 
from re import sub, compile
import matplotlib.pyplot as plt
import numpy as np

class UnimplementedFunctionError(Exception):
    pass

class Vocabulary:

    def __init__(self, corpus):

        self.word2idx, self.idx2word, self.freq = self.build_vocab(corpus)
        self.size = len(self.word2idx)

    def __len__(self):
        return len(self.word2idx.keys())
    
    def most_common(self, k):
        freq = sorted(self.freq.items(), key=lambda x: x[1], reverse=True)
        return [t for t,f in freq[:k]]


    def text2idx(self, text):
        tokens = self.tokenize(text)
        return [self.word2idx[t] if t in self.word2idx.keys() else self.word2idx['UNK'] for t in tokens]

    def idx2text(self, idxs):
        return [self.idx2word[i] if i in self.idx2word.keys() else 'UNK' for i in idxs]


    ###########################
    ## TASK 1.1                ##
    ###########################
    def tokenize(self, text):
        """
        
        tokenize takes in a string of text, remove punctuations and returns an 
        array of strings splitting the text into discrete tokens.

        :params: 
        - text: a string, e.g. "The blue dog jumped, but not high."

        :returns:
        - tokens: a list of strings derived from the text, e.g. ["the", "blue", 
        "dog", "jumped", "but", "not", "high"] for word-level tokenization
        
        """ 
        text = sub(r'[^\w\s]', '', text)
        return text.split()



    ###########################
    ## TASK 1.2                 ##
    ###########################
    def build_vocab(self,corpus):
        """
        
        build_vocab takes in list of strings corresponding to a text corpus, tokenizes the strings, and builds a finite vocabulary

        :params:
        - corpus: a list string to build a vocabulary over

        :returns: 
        - word2idx: a dictionary mapping token strings to their numerical index in the dictionary e.g. { "dog": 0, "but":1, ..., "UNK":129}
        - idx2word: the inverse of word2idx mapping an index in the vocabulary to its word e.g. {0: "dog", 1:"but", ..., 129:"UNK"}
        - freq: a dictionary of words and frequency counts over the corpus (including words not in the dictionary), e.g. {"dog":102, "the": 18023, ...}

        """ 
        word2idx = {}
        idx2word = {}
        freq = {}
        
        # Count tokens
        for s in corpus:
            for token in self.tokenize(s):
                if token not in freq.keys():
                    freq[token] = 1
                else:
                    freq[token] += 1
                    
        # Cutoff the tail
        cutoff = 50
        sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        
        word2idx["UNK"] = 0
        idx2word[0] = "UNK"
        id = 1   # Preserve 0 for UNK
        for (token,cnt) in sorted_freq:
            if cnt > 50:
                word2idx[token] = id
                idx2word[id] = token
                id += 1
            else:
                break
            
        
        return word2idx, idx2word, freq
    
    ###########################
    ## TASK 1.3              ##
    ###########################
    def make_vocab_charts(self):
        """
        
        make_vocab_charts plots word frequency and cumulative coverage charts 
        for this vocabulary. See handout for more details

        
        """ 
        cutoff = 50
        
        ax1 = plt.figure(figsize=(16,6))
        ax1 = plt.subplot(1,2,1)
        
        sorted_freq = self.most_common(len(self.freq))
        most_freq = [self.freq[k] for k in sorted_freq]
        
        cutoff_idx = -1
        occ_sum = sum(most_freq)
        cfc = [0]
        for i in range(len(most_freq)):
            if cutoff_idx == -1 and most_freq[i] < cutoff:
                cutoff_idx = i - 1
                
            cfc.append((cfc[-1] + most_freq[i] / occ_sum))
        
        ax1.plot(list(range(len(most_freq))), most_freq)
        ax1.axhline(y=cutoff, color='r')
        ax1.text(0.8*len(most_freq), cutoff * 1.2,"freq = %d" % cutoff)
        ax1.set_yscale("log")
        ax1.set_xlabel("Token ID (sorted by frequency)")
        ax1.set_xlabel("Frequency")
        ax1.set_title("Token Frequency Distribution")

        ax2 = plt.subplot(1,2,2)
        ax2.plot(list(range(len(cfc[1:]))), cfc[1:])
        ax2.axvline(x=cutoff_idx, color='r', label=str(most_freq[cutoff_idx]))
        ax2.text(1.2*cutoff_idx, 0.95*cfc[cutoff_idx], "%.02f" % cfc[cutoff_idx])
        plt.title("Cumulative Fraction Covered")
        plt.xlabel("Token ID (sorted by frequency)")
        plt.ylabel("Fraction of Token Occurences Covered")
        
        plt.savefig("Vocabulary.jpg")
