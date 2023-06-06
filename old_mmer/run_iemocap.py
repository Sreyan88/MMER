import re
import os
import time
import sys
import json
# from sqlalchemy import true
import yaml
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import librosa
import pandas as pd
from functools import reduce
import random

import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, BertConfig

from utils import create_processor, prepare_example, text_preprocessing
from mmi_module import MMI_Model, CAI_SOTA, UNI_BASELINE

import warnings
warnings.filterwarnings("ignore")


tokenizer = AutoTokenizer.from_pretrained("roberta-base")
audio_processor = create_processor("facebook/wav2vec2-base")

vocabulary_chars_str = "".join(t for t in audio_processor.tokenizer.get_vocab().keys() if len(t) == 1)
vocabulary_text_cleaner = re.compile(  # remove characters not in vocabulary
        f"[^\s{re.escape(vocabulary_chars_str)}]",  # allow space in addition to chars in vocabulary
        flags=re.IGNORECASE if audio_processor.tokenizer.do_lower_case else 0,
    )



def evaluate_metrics(pred_label, true_label):
    pred_label = np.array(pred_label)
    true_label = np.array(true_label)
    ua = np.mean(pred_label.astype(int) == true_label.astype(int))
    pred_onehot = np.eye(4)[pred_label.astype(int)]
    true_onehot = np.eye(4)[true_label.astype(int)]
    wa = np.mean(np.sum((pred_onehot==true_onehot)*true_onehot,axis=0)/np.sum(true_onehot,axis=0))
    key_metric, report_metric = 0.9*wa+0.1*ua, {'wa':wa,'ua':ua}
    return wa, report_metric #take wa as key metric

def label2idx(label):
    label2idx = {
        "hap":0,
        "ang":1,
        "neu":2,
        "sad":3,
        "exc":0}

    return label2idx[label]


class IEMOCAPDataset(object):
    def __init__(self, config, data_list):
        self.data_list = data_list
        self.unit_length = int(8 * 16000)
        self.audio_length = config['acoustic']['audio_length']
        self.feature_name = config['acoustic']['feature_name']
        self.feature_dim = config['acoustic']['embedding_dim']

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        audio_path, bert_path, bert_text, label, asr_text = self.data_list[index]
        audio_name = os.path.basename(audio_path)

        #------------- extract the audio features -------------#
        wave,sr = librosa.core.load(audio_path + ".wav", sr=None)
        audio_length = len(wave)
        # if audio_length > self.unit_length:
        #     wave = wave[0:self.unit_length]

        #------------- extract the text contexts -------------#
        tokenized_word = np.load(bert_path + ".npy")
        tokenized_word = torch.from_numpy(tokenized_word).squeeze(0)
        text_length = tokenized_word.shape[0]
        
        bert_text = text_preprocessing(bert_text)

        #------------- clean asr target text -------------#
        asr_text = prepare_example(asr_text,vocabulary_text_cleaner)

        #------------- labels -------------#
        label = label2idx(label)

        #------------- wrap up all the output info the dict format -------------#
        return {'audio_input':wave,'text_input':bert_text,'audio_length':audio_length,
                'text_length':text_length,'label':label,'audio_name':audio_name,'asr_target':asr_text, 'bert_output':tokenized_word}


def collate(sample_list):
    
    batch_audio = [x['audio_input'] for x in sample_list]
    batch_bert_text = [x['text_input'] for x in sample_list]
    batch_asr_text = [x['asr_target'] for x in sample_list]

    #----------------tokenize and pad the audio----------------------#

    batch_audio = audio_processor(batch_audio, sampling_rate=16000).input_values

    batch_audio = [{"input_values": audio} for audio in batch_audio]
    batch_audio = audio_processor.pad(
            batch_audio,
            padding=True,
            return_tensors="pt",
        )

    with audio_processor.as_target_processor():
        label_features = audio_processor(batch_asr_text).input_ids

    label_features = [{"input_ids": labels} for labels in label_features]

    with audio_processor.as_target_processor():
        labels_batch = audio_processor.pad(
                label_features,
                padding=True,
                return_tensors="pt",
            )

    ctc_labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

    #----------------tokenize and pad the text----------------------#
    batch_text = tokenizer(batch_bert_text, padding=True, truncation=True, return_tensors="pt")
    batch_text_inputids = batch_text['input_ids']
    batch_text_attention = batch_text['attention_mask']

    bert_output = [x['bert_output'] for x in sample_list]
    bert_output = pad_sequence(bert_output,batch_first = True)

    #----------------tokenize and pad the extras----------------------#
    audio_length = torch.LongTensor([x['audio_length'] for x in sample_list])
    text_length = torch.LongTensor([x['text_length'] for x in sample_list])

    batch_label = torch.tensor([x['label'] for x in sample_list], dtype=torch.long)
    batch_name = [x['audio_name'] for x in sample_list]

    return (batch_text_inputids,batch_text_attention,text_length,bert_output),(batch_audio,audio_length),(ctc_labels,batch_label)


def run(args, config, train_data, valid_data, session):

    ############################ PARAMETER SETTING ##########################
    num_workers = 1
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate #5e-5 for cai et.al re-iplementation #0.00001 or 1e-5 for all other
    accum_iter = args.accum_grad
    stats_file = open(os.path.join(args.final_save_path, session) + '_' + 'stats.txt', 'a', buffering=1)

    ############################## PREPARE DATASET ##########################
    train_dataset = IEMOCAPDataset(config, train_data)
    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset, batch_size = batch_size, collate_fn=lambda x: collate(x),
        shuffle = True, num_workers = num_workers
    )
    valid_dataset = IEMOCAPDataset(config, valid_data)
    valid_loader = torch.utils.data.DataLoader(
        dataset = valid_dataset, batch_size = batch_size, collate_fn=lambda x: collate(x),
        shuffle = False, num_workers = num_workers
    )
    # valid_loader = train_loader
    # train_loader = []
    ########################### CREATE MODEL #################################
    config_mmi = BertConfig('configs/config.json')

    if args.run == "baseline":
        model = MMI_Model(config_mmi,len(audio_processor.tokenizer),4,alpha = args.alpha)
    elif args.run == "cai_sota":
        model = CAI_SOTA(config_mmi,len(audio_processor.tokenizer),4,alpha = args.alpha)
    elif args.run == "mmer":
        model = UNI_BASELINE(config_mmi,len(audio_processor.tokenizer),4,alpha = args.alpha)
    else:
        print("Model run not supported")
        sys.exit(1)

    model.cuda()
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name) #to check if all params are trainable
    
    ########################### TRAINING #####################################
    count, best_metric, save_metric, best_epoch = 0, -np.inf, None, 0

    for epoch in range(epochs):
        epoch_train_loss = []
        epoch_train_cls_loss = []
        epoch_train_ctc_loss = []
        model.train()
        start_time = time.time()
        batch_idx = 0
        time.sleep(2) # avoid the deadlock during the switch between the different dataloaders
        progress = tqdm(train_loader, desc='Epoch {:0>3d}'.format(epoch))
        for bert_input, audio_input, label_input in progress:
            
            input_ids, attention_mask, text_length, bert_output =  bert_input[0].cuda(),bert_input[1].cuda(),bert_input[2].cuda(),bert_input[3].cuda()
            acoustic_input, acoustic_length = audio_input[0]['input_values'].cuda(),audio_input[1].cuda()
            ctc_labels, emotion_labels = label_input[0].cuda(),label_input[1].cuda()
            
            #model.zero_grad()
            loss, _, cls_loss, ctc_loss = model(input_ids, attention_mask, 0, text_length, acoustic_input, acoustic_length, ctc_labels, emotion_labels, bert_output)

            loss = loss/accum_iter
            loss.backward()

            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
                epoch_train_loss.append(loss)
                epoch_train_cls_loss.append(cls_loss)
                epoch_train_ctc_loss.append(ctc_loss)
                optimizer.step()
                optimizer.zero_grad()

            batch_idx += 1
            count += 1
            acc_train_loss = torch.mean(torch.tensor(epoch_train_loss)).cpu().detach().numpy()
            cls_loss = torch.mean(torch.tensor(epoch_train_cls_loss)).cpu().detach().numpy()
            ctc_loss = torch.mean(torch.tensor(epoch_train_ctc_loss)).cpu().detach().numpy()
            progress.set_description("Epoch {:0>3d} - Loss {:.4f} - CLS_Loss {:.4f} - CTC_Loss {:.4f}".format(epoch, acc_train_loss, cls_loss, ctc_loss))



        model.eval()
        pred_y, true_y = [], []
        with torch.no_grad():
            time.sleep(2) # avoid the deadlock during the switch between the different dataloaders
            for bert_input, audio_input, label_input in tqdm(valid_loader):
                input_ids, attention_mask, text_length, bert_output =  bert_input[0].cuda(),bert_input[1].cuda(),bert_input[2].cuda(),bert_input[3].cuda()
                acoustic_input, acoustic_length = audio_input[0]['input_values'].cuda(),audio_input[1].cuda()
                ctc_labels, emotion_labels = label_input[0].cuda(),label_input[1].cuda()

                true_y.extend(list(emotion_labels.cpu().numpy()))

                _, logits, cls_loss, ctc_loss = model(input_ids, attention_mask, 0, text_length, acoustic_input, acoustic_length, ctc_labels, emotion_labels, bert_output)
                
                prediction = torch.argmax(logits, axis=1)
                label_outputs = prediction.cpu().detach().numpy().astype(int)
                
                pred_y.extend(list(label_outputs))

        key_metric, report_metric = evaluate_metrics(pred_y, true_y)

        epoch_train_loss = torch.mean(torch.tensor(epoch_train_loss)).cpu().detach().numpy()

        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
                time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print('Valid Metric: {} - Train Loss: {:.3f}'.format(
            ' - '.join(['{}: {:.3f}'.format(key, value) for key, value in report_metric.items()]),
            epoch_train_loss))
        stats = dict(epoch=epoch, key_accuracy = key_metric, report_accuracy = report_metric)
        print(json.dumps(stats), file=stats_file)

        if key_metric > best_metric:
            torch.save({'state_dict': model.state_dict()}, os.path.join(args.final_save_path, session) + '_' + "model.pt")
            best_metric, best_epoch = key_metric, epoch
            print('Better Metric found on dev, calculate performance on Test')
            pred_y, true_y = [], []
            with torch.no_grad():
                time.sleep(2) # avoid the deadlock during the switch between the different dataloaders
                for bert_input, audio_input, label_input in valid_loader:
                    input_ids, attention_mask, text_length, bert_output =  bert_input[0].cuda(),bert_input[1].cuda(),bert_input[2].cuda(),bert_input[3].cuda()
                    acoustic_input, acoustic_length = audio_input[0]['input_values'].cuda(),audio_input[1].cuda()
                    ctc_labels, emotion_labels = label_input[0].cuda(),label_input[1].cuda()

                    true_y.extend(list(emotion_labels.cpu().numpy()))

                    _, logits, _, _ = model(input_ids, attention_mask, 0, text_length, acoustic_input, acoustic_length, ctc_labels, emotion_labels, bert_output)
                

                    prediction = torch.argmax(logits, axis=1)
                    label_outputs = prediction.cpu().detach().numpy().astype(int)

                    pred_y.extend(list(label_outputs))        
            
            _, save_metric = evaluate_metrics(pred_y, true_y)
            print("Test Metric: {}".format(
                ' - '.join(['{}: {:.3f}'.format(key, value) for key, value in save_metric.items()])
            ))

    print("End. Best epoch {:03d}: {}".format(best_epoch, ' - '.join(['{}: {:.3f}'.format(key, value) for key, value in save_metric.items()])))
    return save_metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help='configuration file path')
    parser.add_argument("--epochs", type=int, default=100, help="training epoches")
    parser.add_argument("--csv_path", type=str, required=True, help="path of csv")
    parser.add_argument("--save_path", type=str, default="./", help="report or ckpt save path")
    parser.add_argument("--data_path_audio", type=str, required=True, help="path to raw audio wav files")
    parser.add_argument("--data_path_roberta", type=str, required=True, help="path to roberta embeddings for text")
    parser.add_argument("--run", type=str, required=True, help="type of model you want to run, options are cai_sota, unimodal_baseline and mmer")
    parser.add_argument("--learning_rate", type=float, default=0.00001, help="learning rate for the specific run")
    parser.add_argument("--alpha", type=float, default=0.1, help="value of alpha for CTC weight, only applicable when running cai_sota and mmer")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--accum_grad", type=int, default=4, help="gradient accumulation steps")

    args = parser.parse_args()

    args.final_save_path = os.path.join(args.save_path,args.run)

    os.makedirs(args.final_save_path, exist_ok=True)

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    report_result = []

    df_emotion = pd.read_csv(args.csv_path)
    
    for i in range(1,6):

        valid_session = "Ses0" + str(i)
        valid_data_csv = df_emotion[df_emotion["FileName"].str.match(valid_session)]
        train_data_csv = pd.DataFrame(df_emotion, index = set(df_emotion.index).difference(set(valid_data_csv.index))).reset_index(drop= True)
        valid_data_csv.reset_index(drop= True, inplace= True)
        
        train_data = []
        valid_data = []

        for row in train_data_csv.itertuples():
            file_name = args.data_path_audio + row.FileName
            bert_path = args.data_path_roberta + row.FileName
            train_data.append((file_name,bert_path,row.Sentences,row.Label,row.text))

        for row in valid_data_csv.itertuples():
            file_name = args.data_path_audio + row.FileName
            bert_path = args.data_path_roberta + row.FileName
            valid_data.append((file_name,bert_path,row.Sentences,row.Label,row.text))


        report_metric = run(args, config, train_data, valid_data, str(i))

        report_result.append(report_metric)

    
    #os.makedirs(args.save_path, exist_ok=True)
    pickle.dump(report_result, open(os.path.join(args.final_save_path, 'metric_report.pkl'),'wb'))
