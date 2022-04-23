import enum
import torch 
import transformers
from torch.utils.data import DataLoader as DL
from transformers import AutoTokenizer,AdamW,get_linear_schedule_with_warmup,AutoModelForSequenceClassification

import argparse
from engine import Engine
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import *

from create_torch_dataset import CreateTweetsDataset
import optuna 
from optuna.trial import TrialState

def parse():
    parser = argparse.ArgumentParser(description="Train and test the model for training toxicity classifier")
    parser.add_argument('--train_batch_size',help='Set the training batch size',type=int,default=4)
    parser.add_argument('--test_batch_size',help='Set the testing batch size',type=int,default=16)
    parser.add_argument('--model',help='Select the transformer model',default='bert-base-multilingual-cased')
    parser.add_argument('--lr',help = 'Learning rate set',type=float,default=2e-5)
    parser.add_argument('--train_data',help='Training Dataset Location',type = str,required = True)
    parser.add_argument('--checkpoint',help='Set the location for checkpoint saving',type = str,default='checkpoints')
    parser.add_argument('--starting_epoch',help='Starting epoch for training ',type=int,default=0)
    parser.add_argument('--max_epochs',help='Set the maximum epochs to train the model on ',type = int,default=4)
    global args 
    args = parser.parse_args()

def read_data(train_data,test_data):
    train_data = pd.read_csv(train_data)
    test_data = pd.read_csv(test_data)
    train_data.tweet = train_data.tweet.apply(text_preprocessing)
    train_data.label = train_data.label.apply(label2idx)
    train_data = train_data[['tweet','label']]
    train_data,val_data = train_test_split(train_data,test_size=0.15,stratify=train_data['label'])
    return train_data,val_data


def process_for_transformer(datasets):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    processed_data = []
    for dataset in datasets:
        batch = tokenizer(list(dataset['tweet']), padding=True, truncation=True, return_tensors="pt")
        data = {key:value.clone() for key,value in batch.items()}
        data['label'] = list(dataset['label'])
        processed_data.append(data)
    return processed_data
    

if __name__=='__main__':
    parse()

    train_data,val_data = read_data()
    train_data,val_data = process_for_transformer([train_data,val_data])
    train_data = CreateTweetsDataset(train_data)
    val_data = CreateTweetsDataset(val_data)
    
    global train_loader,val_loader

    train_loader = DL(train_data,batch_size=args.train_batch_size,shuffle = True)
    val_loader = DL(val_data,batch_size=args.test_batch_size,shuffle = True)

    state = {'max_epochs':args.max_epochs,'start_epoch':args.starting_epoch}
    model = AutoModelForSequenceClassification.from_pretrained(args.model,output_attentions = True)
    optimizer = AdamW(model.parameters(),
                  lr = args.lr,
                  eps = 1e-8 
                )
    total_steps = len(train_loader) * (args.max_epochs-args.starting_epoch)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

    engine = Engine(state)

    engine.learn(train_loader,val_loader,model,optimizer,scheduler)




    

