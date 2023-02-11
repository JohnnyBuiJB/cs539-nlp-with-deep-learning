import torch
import torch.nn as nn
import torch.optim as optim

from torchtext import data
from torchtext import datasets

import spacy
import numpy as np

import time
import random
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


SEED = 224
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def visualizeSentenceWithTags(example):
    print("Token"+"".join([" "]*(15))+"POS Tag")
    print("---------------------------------")
    
    for w, t in zip ( example ['text'], example ['udtags']):
        print(w+"". join ([" " ]*(20 - len (w)))+t)
    

class BiLSTMPOSTagger(nn.Module):
    def __init__(self, 
                 input_dim, 
                 embed_dim, 
                 hidden_dim, 
                 output_dim, 
                 n_layers, 
                 bidirectional, 
                 dropout, 
                 pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, embed_dim, padding_idx = pad_idx)
        
        self.lstm = nn.LSTM(embed_dim, 
                            hidden_dim, 
                            num_layers = n_layers, 
                            bidirectional = bidirectional,
                            dropout = dropout)
        
        self.classifier = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):

        #text = [sent len, batch size]
        
        #pass text through embedding layer
        embedded = self.dropout(self.embedding(text))
        
        #embedded = [sent len, batch size, emb dim]
        
        #pass embeddings into LSTM
        outputs, (hidden, cell) = self.lstm(embedded)
        
        #outputs holds the backward and forward hidden states in the final layer
        #hidden and cell are the backward and forward hidden and cell states at the final time-step
        
        #output = [sent len, batch size, hid dim * n directions]
        #hidden/cell = [n layers * n directions, batch size, hid dim]
        
        #we use our outputs to make a prediction of what the tag should be
        predictions = self.classifier(self.dropout(outputs))
        
        #predictions = [sent len, batch size, output dim]
        
        return predictions

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean = 0, std = 0.1)
        

def per_word_accuracy(preds, y, tag_pad_idx):
    # Get the indices of the max probability
    max_preds = preds.argmax(dim = 1, keepdim = True)
    non_pad_elements = (y != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]]).to(dev)

def train(model, iterator, optimizer, criterion, tag_pad_idx):
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        text = batch.text
        tags = batch.udtags
        
        # Zero out grads
        optimizer.zero_grad()
        
        preds = model(text)
        
        preds = preds.view(-1, preds.shape[-1])
        tags = tags.view(-1)
        loss = criterion(preds, tags)
                
        acc = per_word_accuracy(preds, tags, tag_pad_idx)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, tag_pad_idx):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            text = batch.text
            tags = batch.udtags
            
            predictions = model(text)
            
            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)
            
            loss = criterion(predictions, tags)
            
            acc = per_word_accuracy(predictions, tags, tag_pad_idx)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def tag_sentence(model, dev, sentence, text_field, tag_field):
    # Enter evaluation mode
    model.eval()
    with torch.no_grad():
        if isinstance(sentence, str):
            nlp = spacy.load("en_core_web_sm")
            words = [word.text for word in nlp(sentence)]
            print(words)
        else:
            words = [word for word in sentence]

        if text_field.lower:
            words = [w.lower() for w in words]

        tokens = [text_field.vocab.stoi[t] for t in words]
        unk_idx = text_field.vocab.stoi[text_field.unk_token]
        unks = [t for t, n in zip(words, tokens) if n == unk_idx]
        token_tensor = torch.LongTensor(tokens)
        token_tensor = token_tensor.unsqueeze(-1).to(dev)

        preds = model(token_tensor)
        top_preds = preds.argmax(-1)

        pred_tags = [tag_field.vocab.itos[t.item()] for t in top_preds]
    
    return words, pred_tags, unks

def main():
    logging.info("TASK 3.1: LOAD DATA")
    TEXT = data.Field(lower = True)
    UD_TAGS = data.Field(unk_token = None)
    fields = (("text", TEXT), ("udtags", UD_TAGS))
    train_data, valid_data, test_data = datasets.UDPOS.splits(fields)
    
    
    batch_size = 128
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size = batch_size,
        device = dev)
    
    logging.info("TASK 3.1: TOKENIZATION")
    min_freq = 2
    TEXT.build_vocab(train_data, 
                    min_freq = min_freq,
                    vectors = "glove.6B.100d",
                    unk_init = torch.Tensor.normal_)

    UD_TAGS.build_vocab(train_data)
    
    input_dim = len(TEXT.vocab)
    embed_dim = 100
    hidden_dim = 128
    output_dim = len(UD_TAGS.vocab)
    n_layers = 2
    is_bidirectional = True
    drop_out = 0.25
    pad_idx = TEXT.vocab.stoi[TEXT.pad_token]

    logging.info("TASK 3.2: TRAINING")
    model = BiLSTMPOSTagger(input_dim, 
                            embed_dim, 
                            hidden_dim, 
                            output_dim, 
                            n_layers, 
                            is_bidirectional, 
                            drop_out, 
                            pad_idx)
    
    model.apply(init_weights)
    
    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)
    model.embedding.weight.data[pad_idx] = torch.zeros(embed_dim)
    
    optimizer = optim.Adam(model.parameters())
    tag_pad_idx = UD_TAGS.vocab.stoi[UD_TAGS.pad_token]

    criterion = nn.CrossEntropyLoss(ignore_index = tag_pad_idx)
    model = model.to(dev)
    criterion = criterion.to(dev)
    
    n_epochs = 20
    best_valid_loss = float('inf')
    for epoch in range(n_epochs):
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, tag_pad_idx)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, tag_pad_idx)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best-model.pt')
        
        logging.info("EPOCH #%d" % epoch)
        logging.info('\tTrain Loss: %.2f | Train Acc: {train_acc*100:.2f}%' % train_loss)
        logging.info('\t Val. Loss: %.2f |  Val. Acc: {valid_acc*100:.2f}%')
    
    
    test_loss, test_acc = evaluate(model, test_iterator, criterion, tag_pad_idx)
    logging.info(f'\Test Loss: {test_loss:.2f} | Test Acc: {test_acc*100:.2f}%')
    
    logging.info("TASK 3.3: Tag sample sentences")
    for sentence in ["The old man the boat",
                "The man who hunts ducks out on weekends.",
                "The complex houses married and single soldiers and their families."]:

        words, pred_tags, unks = tag_sentence(model, 
                                            dev, 
                                            sentence,
                                            TEXT, 
                                            UD_TAGS)

        print("%10s%10s\n" % ('word','pred_tag'))

        for word, pred_tag in zip(words, pred_tags):
            print("%10s%10s" % (word,pred_tag))

if __name__ != "main":
    main()
    