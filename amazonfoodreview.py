"""*****************************************************************************************


                    Sentiment Analysis of Amazon Food Review Using BERT

                                
*****************************************************************************************"""
#%% Importing Libraries

# General
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Bert
from transformers import BertTokenizer
from transformers import BertModel

#%% Tokenizer

tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
print('Length of Tokenizer Vocab is {}'.format(len(tokenizer.vocab)))

init_token=tokenizer.cls_token
eos_token=tokenizer.sep_token
pad_token=tokenizer.pad_token
unk_token=tokenizer.unk_token

init_token_idx=tokenizer.convert_tokens_to_ids(init_token)
eos_token_idx=tokenizer.convert_tokens_to_ids(eos_token)
pad_token_idx=tokenizer.convert_tokens_to_ids(pad_token)
unk_token_idx=tokenizer.convert_tokens_to_ids(unk_token)

max_input_length=tokenizer.max_model_input_sizes['bert-base-uncased']

#%% Reading Dataset

def sentence_tokenizer(sentence):
    tokens=tokenizer.tokenize(sentence)
    tokens=tokens[:max_input_length-2]
    return tokens


TEXT=data.Field(batch_first=True,
                use_vocab=False,
                tokenize=sentence_tokenizer,
                preprocessing=tokenizer.convert_tokens_to_ids,
                init_token=init_token_idx,
                eos_token=eos_token_idx,
                pad_token=pad_token_idx,
                unk_token=unk_token_idx)


LABEL=data.LabelField(dtype=torch.float)


datafields=[("Id",None),
            ("ProductId",None),
            ("UserId",None),
            ("ProfileName",None),
            ("HelpfulnessNumerator",None),
            ("HelpfulnessDenominator",None),
            ("Score",LABEL),
            ("Time",None),
            ("Summary",None),
            ("Text",TEXT)]


dataset = data.TabularDataset(
           path="./Reviews.csv", # the file path
           format='csv',
           skip_header=True, # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
           fields=datafields)

print(dataset.examples[0].__dict__)

train_data, test_data = dataset.split()
train_data, valid_data = train_data.split()

LABEL.build_vocab(train_data)

print(LABEL.vocab.stoi)

#%% Batch Loading

BATCH_SIZE = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE, 
    sort_within_batch = True,
    sort_key = lambda x: len(x.Text),
    device = device)

for batch in train_iterator:
    print(batch.Text)
    print(batch.Score)
    break

#%% Constructiong the Model

bert=BertModel.from_pretrained('bert-base-uncased')

class BERTGRUSentiment(nn.Module):
  def __init__(self,bert,hidden_dim, output_dim, n_layers, bidirectional, dropout):
      super().__init__()
      self.bert=bert
      embedding_dim=bert.config.to_dict()['hidden_size']
      self.rnn=nn.GRU(embedding_dim,
                      hidden_dim,
                      num_layers=n_layers,
                      bidirectional=bidirectional,
                      batch_first=True,
                      dropout=0 if n_layers<2 else dropout)
      self.dropout=nn.Dropout(dropout)
      self.out_layer=nn.Linear(2*hidden_dim if bidirectional else hidden_dim, output_dim)
  def forward(self,text):
      # [batch_size, sentence_length]
      with torch.no_grad():
           embeddings=self.bert(text)[0]
      # [batch_size, sentence_length, embedding_dim]
      _,hiddens=self.rnn(embeddings)
      # [num_layers*num_directions, batch_size, hidden_dim]
      if self.rnn.bidirectional:
           hidden=self.dropout(torch.cat((hiddens[-2,:,:],hiddens[-1,:,:]),dim=1))
      else:
           hidden=self.dropout(hiddens[-1,:,:])
      # [batch_size, 2*hidden_dim if bidirectional else hidden_dim]
      output=self.out_layer(hidden)
      # [batch_size, output_dim]
      return output


hidden_dim = 256
output_dim = 5
n_layers = 2
bidirectional = True
dropout = 0.25

model = BERTGRUSentiment(bert,hidden_dim,output_dim,n_layers,bidirectional,dropout)

model = model.to(device)

#%% Freezing the parameters of the bert

print("Before Freezing: The Model has {} trainable parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

for name, param in model.named_parameters():                
    if name.startswith('bert'):
        param.requires_grad = False

print("After Freezing: The Model has {} trainable parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

for name, param in model.named_parameters():                
    if param.requires_grad:
        print(name)

#%% Training and Evaluation

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(reduction='sum')
criterion = criterion.to(device)
softmaxx = nn.Softmax(dim=1)
def train(model, iterator, optimizer, criterion):    
    num_samp = 0
    epoch_loss = 0
    epoch_corr_pred = 0   
    model.train()    
    for batch in iterator:        
        optimizer.zero_grad()        
        predictions = model(batch.Text)        
        loss = criterion(predictions, batch.Score.long())        
        corr_pred = (torch.argmax(softmaxx(predictions),1)==batch.Score).float().sum() 
        loss.backward()        
        optimizer.step()        
        num_samp+=batch.Text.shape[0]
        epoch_loss += loss.item()
        epoch_corr_pred += corr_pred.item()        
    return epoch_loss, epoch_corr_pred / num_samp

def evaluate(model, iterator, criterion):    
    num_samp = 0
    epoch_loss = 0
    epoch_corr_pred = 0    
    model.eval()    
    with torch.no_grad():    
        for batch in iterator:
            predictions = model(batch.Text)         
            loss = criterion(predictions, batch.Score.long())        
            corr_pred = (torch.argmax(softmaxx(predictions),1)==batch.Score).float().sum()
            num_samp+=batch.Text.shape[0]
            epoch_loss += loss.item()
            epoch_corr_pred += corr_pred.item()       
    return epoch_loss, epoch_corr_pred / num_samp

num_epochs = 5
for epoch in range(num_epochs): 
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

#%% Testing

test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

#%% Inference

def predict_sentiment(model, tokenizer, sentence):
    model.eval()
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
    input_samp = torch.LongTensor(indexed).to(device)
    input_samp = input_samp.unsqueeze(0)
    prediction = torch.argmax(softmaxx(model(input_samp)),1)
    return prediction.item()

print(predict_sentiment(model,tokenizer,"This is a great book"))

print(predict_sentiment(model, tokenizer, "This is a terrible product"))