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
# import optuna 
# from optuna.trial import TrialState

from model import Transformer
import json

def parse():
    parser = argparse.ArgumentParser(description="Train and test the model for training toxicity classifier")
    parser.add_argument('--train_batch_size',help='Set the training batch size',type=int,default=32)
    parser.add_argument('--test_batch_size',help='Set the testing batch size',type=int,default=32)
    parser.add_argument('--model',help='Select the transformer model',default='roberta-base')
    parser.add_argument('--lr',help = 'Learning rate set',type=float,default=2.2e-5)
    parser.add_argument('--train_data',help='Training Dataset Location',type = str,default="/speech/sreyan/IEMOCAP/iemocap.csv",required = False)
    parser.add_argument('--test_data',help='Test Dataset Location',type = str,required=False)
    parser.add_argument('--checkpoint',help='Set the location for checkpoint saving',type = str,default='checkpoints')
    parser.add_argument('--starting_epoch',help='Starting epoch for training ',type=int,default=0)
    parser.add_argument('--max_epochs',help='Set the maximum epochs to train the model on ',type = int,default=50)
    global args 
    args = parser.parse_args()

def read_data(train_data, test_data):
    # train_data = pd.read_csv(train_data)
    # test_data = pd.read_csv(test_data)
    # train_data.dropna(axis='columns', inplace=True)
    # test_data.dropna(axis='columns',inplace = True)
    train_data.Sentences = train_data.Sentences.apply(text_preprocessing)
    test_data.Sentences = test_data.Sentences.apply(text_preprocessing)
    train_data.Label = train_data.Label.apply(label2idx)
    test_data.Label = test_data.Label.apply(label2idx)
    train_data = train_data[['Sentences','Label']]
    test_data = test_data[['Sentences','Label']]
    return train_data,test_data,test_data


def process_for_transformer(datasets):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    processed_data = []
    for dataset in datasets:
        batch = tokenizer(list(dataset['Sentences']), padding=True, truncation=True, return_tensors="pt")
        data = {key:value.clone() for key,value in batch.items()}
        data['label'] = list(dataset['Label'])
        processed_data.append(data)
    return processed_data
    

if __name__=='__main__':
    parse()

    df_emotion = pd.read_csv(args.train_data)
    
    for i in range(1,6):

        stats_file = open("/speech/sreyan/IEMOCAP/working_model"  + "/" + str(i) + '_' + 'stats.txt', 'a', buffering=1)

        valid_session = "Ses0" + str(i)
        valid_data_csv = df_emotion[df_emotion["FileName"].str.match(valid_session)]
        train_data_csv = pd.DataFrame(df_emotion, index = set(df_emotion.index).difference(set(valid_data_csv.index))).reset_index(drop= True)
        valid_data_csv.reset_index(drop= True, inplace= True)
        train_data,val_data,test_data = read_data(train_data_csv,valid_data_csv)
        train_data,val_data,test_data = process_for_transformer([train_data,val_data,test_data])
    
        train_data = CreateTweetsDataset(train_data)
        val_data = CreateTweetsDataset(val_data)
        test_data = CreateTweetsDataset(test_data)
    
        global train_loader,val_loader,test_loader

        train_loader = DL(train_data,batch_size=args.train_batch_size,shuffle = True)
        val_loader = DL(val_data,batch_size=args.test_batch_size,shuffle = True)
        test_loader = DL(test_data,batch_size=args.test_batch_size,shuffle = True)

        state = {'max_epochs':args.max_epochs,'start_epoch':args.starting_epoch}
        #model = AutoModelForSequenceClassification.from_pretrained(args.model,output_attentions = True)
        model = Transformer(args.model)
        optimizer = AdamW(model.parameters(),
                    lr = args.lr,
                    eps = 1e-8 
                    )
        total_steps = len(train_loader) * (args.max_epochs-args.starting_epoch)
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)

        engine = Engine(state)

        metrics_dict = engine.learn(train_loader,val_loader,model,optimizer,scheduler,test_loader)

        stats = dict(f1 = float(metrics_dict['f1_score'].detach().cpu().numpy()), accuracy = float(metrics_dict['accuracy'].detach().cpu().numpy()), best_score = float(metrics_dict['best_score'].detach().cpu().numpy()))
        print(json.dumps(stats), file=stats_file)
        
        print("Test f1 score :: ",metrics_dict['f1_score'])
        print("Test accuracy ::",metrics_dict['accuracy'])




    

