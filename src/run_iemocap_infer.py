import argparse
import os
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--session', type=int, help='Session index')
parser.add_argument('--config_path', type=str, help='Path to config')
parser.add_argument('--csv_path', type=str, help='Path to csv file')
parser.add_argument('--data_path_audio', type=str, help='Path to audio data')
parser.add_argument('--data_path_roberta', type=str, help='Path to text data')
parser.add_argument('--checkpoint_path', type=str, help='Path to checkpoint')
parser.add_argument('--gpu','-g', default=0, type=int, help='gpu number')
parser.add_argument('--batch_size','-bs', type=int, default=16, help='batch size')
args = parser.parse_args()


from cgitb import text
import re
import os
import time
import sys
import json
from tkinter import NONE
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
import copy
import math

import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, BertConfig, AutoConfig
from transformers import  Wav2Vec2Model, RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaEncoder
from infonce_loss import InfoNCE, SupConLoss
from mmi_module import MMI_Model


# loss =  InfoNCE()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

from utils import create_processor, prepare_example, text_preprocessing
# from mmi_module import MMI_Model, CAI_SOTA, UNI_BASELINE

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
        # self.unit_length = int(8 * 16000)
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
        if len(wave)>210000:
            wave = wave[:210000]
        audio_length = len(wave)

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
                'text_length':text_length,'label':label,'audio_name':audio_name,'asr_target':asr_text, 'bert_output':tokenized_word,
                }


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

    #-----------------pad the pre-generated bert embeddings----------#
    bert_output = [x['bert_output'] for x in sample_list]
    bert_output = pad_sequence(bert_output,batch_first = True)

    batch_label = torch.tensor([x['label'] for x in sample_list], dtype=torch.long)

    #----------------tokenize and pad the extras----------------------#
    audio_length = torch.LongTensor([x['audio_length'] for x in sample_list])
    text_length = torch.LongTensor([x['text_length'] for x in sample_list])


    return (batch_text_inputids,batch_text_attention,text_length,bert_output),(batch_audio,audio_length),(ctc_labels,batch_label)

class ActivateFun(nn.Module):
    def __init__(self, activate_fun):
        super(ActivateFun, self).__init__()
        self.activate_fun = activate_fun

    def _gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x):
        if self.activate_fun == 'relu':
            return torch.relu(x)
        elif self.activate_fun == 'gelu':
            return self._gelu(x)


class FuseModel(nn.Module):

    def __init__(self, text_config):

        super().__init__()

        tran_dim = 768

        self.config_mmi = BertConfig('config.json')
        self.model_mmi = MMI_Model(self.config_mmi,len(audio_processor.tokenizer),4)

        self.temperature = 0.07

        self.orgin_linear_change = nn.Sequential(
            nn.Linear(tran_dim*2, tran_dim),
            ActivateFun('gelu'),
            nn.Linear(tran_dim, tran_dim)
        )

        self.augment_linear_change = nn.Sequential(
            nn.Linear(tran_dim*2, tran_dim),
            ActivateFun('gelu'),
            nn.Linear(tran_dim, tran_dim)
        )

    def forward_encoder(self, text_output, attention_mask, audio_inputs, audio_length, ctc_labels, emotion_labels, augmentation = False):

        #bert_attention_mask, audio_input, audio_length, ctc_labels, emotion_labels, text_output, augmentation = False
        emotion_logits, logits, loss_cls, loss_ctc = self.model_mmi(text_output, attention_mask, audio_inputs, audio_length, ctc_labels, emotion_labels, augmentation = augmentation)

        return emotion_logits, logits, loss_cls, loss_ctc

    def forward(self, text_output, attention_mask, audio_inputs, audio_length, ctc_labels, emotion_labels):

        emotion_logits, logits, loss_cls, loss_ctc = self.forward_encoder(text_output, attention_mask, audio_inputs, audio_length, ctc_labels, emotion_labels)

        return emotion_logits



def run_infer(config, train_data, valid_data, checkpoint_path, session):

    ############################ PARAMETER SETTING ##########################
    num_workers = 4
    batch_size = args.batch_size

    ############################## PREPARE DATASET ##########################
    train_dataset = IEMOCAPDataset(config, train_data)
    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset, batch_size = batch_size, collate_fn=collate,
        shuffle = True, num_workers = num_workers
    )
    valid_dataset = IEMOCAPDataset(config, valid_data)
    valid_loader = torch.utils.data.DataLoader(
        dataset = valid_dataset, batch_size = batch_size, collate_fn=collate,
        shuffle = False, num_workers = num_workers
    )

    ############################## CREATE MODEL ##########################

    print("*"*40)
    print("Create model")
    print("*"*40)

    config_mmi = BertConfig('config.json')
    # roberta_model = RobertaModel.from_pretrained("roberta-base")
    # text_config = copy.deepcopy(roberta_model.config)

    model = FuseModel(config_mmi)

    del config_mmi

    checkpoint = torch.load(checkpoint_path)
    state_dict = {key.replace('module.', ''):value for key, value in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)

    model.cuda()

 

    # ########################### INFERENCE #####################################
    print("*"*40)
    print("Inference started")
    print("*"*40)


    start_time = time.time()
    pred_y, true_y = [], []
    with torch.no_grad():
        time.sleep(2) # avoid the deadlock during the switch between the different dataloaders
        for bert_input, audio_input, label_input in tqdm(valid_loader):
            torch.cuda.empty_cache()
            attention_mask, text_length, bert_output =  bert_input[1].cuda(),bert_input[2].cuda(),bert_input[3].cuda()
            acoustic_input, acoustic_length = audio_input[0]['input_values'].cuda(),audio_input[1].cuda()
            ctc_labels, emotion_labels = label_input[0].cuda(),label_input[1].cuda()

            true_y.extend(list(emotion_labels.cpu().numpy()))

            logits = model(bert_output, attention_mask, acoustic_input, acoustic_length, ctc_labels, emotion_labels)

            prediction = torch.argmax(logits, axis=1)
            label_outputs = prediction.cpu().detach().numpy().astype(int)

            pred_y.extend(list(label_outputs))
            # del valid_loader

        key_metric, report_metric = evaluate_metrics(pred_y, true_y)

        elapsed_time = time.time() - start_time
        print("The time elapse is: " +
                time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print('Valid Metric: {} '.format(
            ' - '.join(['{}: {:.3f}'.format(key, value) for key, value in report_metric.items()])
            ))


if __name__ == "__main__":

    config_path = args.config_path
    csv_path = args.csv_path
    data_path_audio =args.data_path_audio
    data_path_roberta = args.data_path_roberta
    checkpoint_path = args.checkpoint_path

    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    report_result = []

    df_emotion = pd.read_csv(csv_path)



    valid_session = "Ses0" + str(args.session)
    valid_data_csv = df_emotion[df_emotion["FileName"].str.match(valid_session)]
    train_data_csv = pd.DataFrame(df_emotion, index = list(set(df_emotion.index).difference(set(valid_data_csv.index)))).reset_index(drop= True)
    valid_data_csv.reset_index(drop= True, inplace= True)

    train_data = []
    valid_data = []

    for row in train_data_csv.itertuples():
        file_name = os.path.join(data_path_audio + row.FileName)
        bert_path = data_path_roberta + row.FileName
        train_data.append((file_name,bert_path,row.Sentences,row.Label,row.text))

    for row in valid_data_csv.itertuples():
        file_name = os.path.join(data_path_audio + row.FileName)
        bert_path = data_path_roberta + row.FileName
        valid_data.append((file_name,bert_path,row.Sentences,row.Label,row.text))
        


    report_metric = run_infer(config, train_data, valid_data, checkpoint_path, str(args.session))
