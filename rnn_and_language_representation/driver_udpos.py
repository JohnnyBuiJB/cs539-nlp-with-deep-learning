import torch
import torch.nn as nn
import torch.optim as optim

from torchtext import data
from torchtext import datasets

import spacy
import numpy as np

import time
import random

# TODO: Try other seed
SEED = 1234
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
                 embedding_dim, 
                 hidden_dim, 
                 output_dim, 
                 n_layers, 
                 bidirectional, 
                 dropout, 
                 pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx = pad_idx)
        
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers = n_layers, 
                            bidirectional = bidirectional,
                            dropout = dropout if n_layers > 1 else 0)
        
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
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
        predictions = self.fc(self.dropout(outputs))
        
        #predictions = [sent len, batch size, output dim]
        
        return predictions

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean = 0, std = 0.1)
        

def categorical_accuracy(preds, y, tag_pad_idx):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    non_pad_elements = (y != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]]).to(dev)

# TODO: Can use Stefan's train program instead
def train(model, iterator, optimizer, criterion, tag_pad_idx):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        text = batch.text
        tags = batch.udtags
        
        optimizer.zero_grad()
        
        #text = [sent len, batch size]
        
        predictions = model(text)
        
        #predictions = [sent len, batch size, output dim]
        #tags = [sent len, batch size]
        
        predictions = predictions.view(-1, predictions.shape[-1])
        tags = tags.view(-1)
        
        #predictions = [sent len * batch size, output dim]
        #tags = [sent len * batch size]
        
        loss = criterion(predictions, tags)
                
        acc = categorical_accuracy(predictions, tags, tag_pad_idx)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# TODO: Can use Stefan's evaluation instead
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
            
            acc = categorical_accuracy(predictions, tags, tag_pad_idx)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def tag_sentence(model, device, sentence, text_field, tag_field):
    model.eval()
    
    if isinstance(sentence, str):
        nlp = spacy.load("en_core_web_sm")
        tokens = [token.text for token in nlp(sentence)]
    else:
        tokens = [token for token in sentence]

    if text_field.lower:
        tokens = [t.lower() for t in tokens]
        
    numericalized_tokens = [text_field.vocab.stoi[t] for t in tokens]

    unk_idx = text_field.vocab.stoi[text_field.unk_token]
    
    unks = [t for t, n in zip(tokens, numericalized_tokens) if n == unk_idx]
    
    token_tensor = torch.LongTensor(numericalized_tokens)
    
    token_tensor = token_tensor.unsqueeze(-1).to(device)
         
    predictions = model(token_tensor)
    
    top_predictions = predictions.argmax(-1)
    
    predicted_tags = [tag_field.vocab.itos[t.item()] for t in top_predictions]
    
    return tokens, predicted_tags, unks

def main():
    TEXT = data.Field(lower = True)
    UD_TAGS = data.Field(unk_token = None)
    fields = (("text", TEXT), ("udtags", UD_TAGS))
    train_data, valid_data, test_data = datasets.UDPOS.splits(fields)
    
    
    BATCH_SIZE = 128

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size = BATCH_SIZE,
        device = dev)
    
    # Tokenization
    min_freq = 2
    TEXT.build_vocab(train_data, 
                    min_freq = min_freq,
                    vectors = "glove.6B.100d",
                    unk_init = torch.Tensor.normal_)

    UD_TAGS.build_vocab(train_data)
    
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128
    OUTPUT_DIM = len(UD_TAGS.vocab)
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.25
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    model = BiLSTMPOSTagger(INPUT_DIM, 
                            EMBEDDING_DIM, 
                            HIDDEN_DIM, 
                            OUTPUT_DIM, 
                            N_LAYERS, 
                            BIDIRECTIONAL, 
                            DROPOUT, 
                            PAD_IDX)
    
    # TODO: Move this to the models
    model.apply(init_weights)
    
    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
    
    optimizer = optim.Adam(model.parameters())
    TAG_PAD_IDX = UD_TAGS.vocab.stoi[UD_TAGS.pad_token]

    criterion = nn.CrossEntropyLoss(ignore_index = TAG_PAD_IDX)
    model = model.to(dev)
    criterion = criterion.to(dev)
    
    ### TRAIN ##################################################################
    N_EPOCHS = 3

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()
        
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, TAG_PAD_IDX)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, TAG_PAD_IDX)
        
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut1-model.pt')
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    
    
    example_index = 1
    sentence = vars(train_data.examples[example_index])['text']
    actual_tags = vars(train_data.examples[example_index])['udtags']    

    tokens, pred_tags, unks = tag_sentence(model, 
                                       dev, 
                                       "The old man the boat", 
                                       TEXT, 
                                       UD_TAGS)

    print(unks)
    
    print("Pred. Tag\tActual Tag\tCorrect?\tToken\n")

    for token, pred_tag, actual_tag in zip(tokens, pred_tags, actual_tags):
        correct = '✔' if pred_tag == actual_tag else '✘'
        print(f"{pred_tag}\t\t{actual_tag}\t\t{correct}\t\t{token}")

if __name__ != "main":
    main()
    