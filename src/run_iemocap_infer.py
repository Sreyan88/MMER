import argparse
import os
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--gpu','-g', default=0, type=int, help='gpu number')
parser.add_argument('--batch_size','-bs', type=int, default=16, help='batch size')
parser.add_argument('--ga','-ga', type=int, default=1, help='gradient accumulation')
args = parser.parse_args()

unqiue_path = f'./model_store_{args.batch_size}_{args.gpu}_{args.ga}_all'

if not os.path.exists(unqiue_path):
    os.makedirs(unqiue_path, exist_ok=True)


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
from mmi_module_ner_slt_fbank_5 import MMI_Model


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

        self.augmented_audio_path = "/fs/nexus-projects/audio-visual_dereverberation/clmlf/out/"
        self.augmented_audio = os.listdir("/fs/nexus-projects/audio-visual_dereverberation/clmlf/out")

        self.augmented_audio_dictionary = {}

        for item in self.augmented_audio:
            gt_audio = "Ses" + item.split("Ses")[-1]
            if gt_audio in self.augmented_audio_dictionary:
                self.augmented_audio_dictionary[gt_audio].append(self.augmented_audio_path + item)
            else:
                self.augmented_audio_dictionary[gt_audio] = [self.augmented_audio_path + item]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        audio_path, bert_path, bert_text, label, asr_text, augmented_text, bert_path_augment = self.data_list[index]
        audio_name = os.path.basename(audio_path)

        #------------- extract the audio features -------------#
        wave,sr = librosa.core.load(audio_path + ".wav", sr=None)
        if len(wave)>210000:
            wave = wave[:210000]
        audio_length = len(wave)

        # figure out augmnted audio
        original_audio = audio_path.split("/")[-1] + ".wav" #.wav
        augmented_wav_index = random.randint(0,len(self.augmented_audio_dictionary[original_audio])-1)
        augmented_wav_path = self.augmented_audio_dictionary[original_audio][augmented_wav_index]
        augmented_wav, sr = librosa.core.load(augmented_wav_path, sr=None)
        if len(augmented_wav)>210000:
            augmented_wav = augmented_wav[:210000]
        augmented_audio_length = len(augmented_wav)
        # if audio_length > self.unit_length:
        #     wave = wave[0:self.unit_length]

        #------------- extract the text contexts -------------#
        tokenized_word = np.load(bert_path + ".npy")
        tokenized_word = torch.from_numpy(tokenized_word).squeeze(0)
        text_length = tokenized_word.shape[0]

        tokenized_word_augment = np.load(bert_path_augment + ".npy")
        tokenized_word_augment = torch.from_numpy(tokenized_word_augment).squeeze(0)

        bert_text = text_preprocessing(bert_text)
        augmented_text = text_preprocessing(augmented_text)

        #------------- clean asr target text -------------#
        asr_text = prepare_example(asr_text,vocabulary_text_cleaner)

        #------------- labels -------------#
        label = label2idx(label)

        #------------- wrap up all the output info the dict format -------------#
        return {'audio_input':wave,'text_input':bert_text,'audio_length':audio_length,
                'text_length':text_length,'label':label,'audio_name':audio_name,'asr_target':asr_text, 'bert_output':tokenized_word,
                'augmented_text_input': augmented_text, 'augmented_audio_input': augmented_wav, 'augmented_audio_length': augmented_audio_length,
                'bert_output_augment': tokenized_word_augment}


def collate(sample_list):

    batch_audio = [x['audio_input'] for x in sample_list]
    batch_augmented_audio = [x['augmented_audio_input'] for x in sample_list]
    batch_bert_text = [x['text_input'] for x in sample_list]
    batch_augmented_text = [x['augmented_text_input'] for x in sample_list]
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

    #-------------tokenize and pad augmented audio--------------------#

    batch_augmented_audio = audio_processor(batch_augmented_audio, sampling_rate=16000).input_values

    batch_augmented_audio = [{"input_values": audio} for audio in batch_augmented_audio]
    batch_augmented_audio = audio_processor.pad(
            batch_augmented_audio,
            padding=True,
            return_tensors="pt",
        )

    #----------------tokenize and pad the text----------------------#
    batch_text = tokenizer(batch_bert_text, padding=True, truncation=True, return_tensors="pt")
    batch_text_inputids = batch_text['input_ids']
    batch_text_attention = batch_text['attention_mask']

    #----------------tokenize and pad the augmented text-------------#
    batch_augmented_text = tokenizer(batch_augmented_text, padding=True, truncation=True, return_tensors="pt")
    batch_augmented_text_inputids = batch_augmented_text['input_ids']
    batch_augmented_text_attention = batch_augmented_text['attention_mask']

    #-----------------pad the pre-generated bert embeddings----------#
    bert_output = [x['bert_output'] for x in sample_list]
    bert_output = pad_sequence(bert_output,batch_first = True)

    #-----------------pad the pre-generated augmented bert embeddings----------#
    bert_output_augment = [x['bert_output_augment'] for x in sample_list]
    bert_output_augment = pad_sequence(bert_output_augment,batch_first = True)

    #----------------tokenize and pad the extras----------------------#
    audio_length = torch.LongTensor([x['audio_length'] for x in sample_list])
    text_length = torch.LongTensor([x['text_length'] for x in sample_list])

    augmented_audio_length = torch.LongTensor([x['augmented_audio_length'] for x in sample_list])
    # augmented_text_length = torch.LongTensor([x['text_length'] for x in sample_list])

    batch_label = torch.tensor([x['label'] for x in sample_list], dtype=torch.long)
    batch_name = [x['audio_name'] for x in sample_list]

    target_labels = []

    for label_idx in range(4):
        temp_labels = []
        for idx, _label in enumerate(batch_label):
            if _label == label_idx:
                temp_labels.append(idx)

        target_labels.append(torch.LongTensor(temp_labels[:]))

    return (batch_text_inputids,batch_text_attention,text_length,bert_output),(batch_audio,audio_length),(ctc_labels,batch_label, target_labels), (bert_output_augment,batch_augmented_text_attention),\
        (batch_augmented_audio,augmented_audio_length)

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

    def forward(self, text_output, attention_mask, audio_inputs, audio_length, ctc_labels, emotion_labels, target_labels, augmented_text_output, augmented_attention_mask, augmented_audio_inputs, augmented_audio_length, mode = "train"):

        emotion_logits, logits, loss_cls, loss_ctc = self.forward_encoder(text_output, attention_mask, audio_inputs, audio_length, ctc_labels, emotion_labels)

        # Self Contrastive loss
        orgin_res_change = self.orgin_linear_change(logits)

        l_pos_neg_self = torch.einsum('nc,ck->nk', [orgin_res_change, orgin_res_change.T])
        l_pos_neg_self = torch.log_softmax(l_pos_neg_self, dim=-1)
        l_pos_neg_self = l_pos_neg_self.view(-1)

        cl_self_labels = target_labels[emotion_labels[0]]

        for index in range(1, logits.size(0)):
            cl_self_labels = torch.cat((cl_self_labels, target_labels[emotion_labels[index]] + index*emotion_labels.size(0)), 0)

        l_pos_neg_self = l_pos_neg_self / self.temperature
        cl_self_loss = torch.gather(l_pos_neg_self, dim=0, index=cl_self_labels)
        cl_self_loss = - cl_self_loss.sum() / cl_self_labels.size(0)

        # cl_self_loss = self.cl_loss(text_audio_ftrs, text_audio_ftrs)
        # cl_self_loss = self.cl_loss(text_audio_ftrs, text_audio_ftrs) #contrastive loss

        if ((augmented_text_output is not None) and (augmented_audio_inputs is not None) and (mode == "train")):

            augmented_emotion_logits, augmented_logits, augmented_loss_cls, augmented_loss_ctc = self.forward_encoder(augmented_text_output, augmented_attention_mask, augmented_audio_inputs, augmented_audio_length, ctc_labels, emotion_labels, augmentation=True)

            # orgin_res_change = self.orgin_linear_change(text_audio_ftrs)
            augment_res_change = self.augment_linear_change(augmented_logits)

            l_pos_neg = torch.einsum('nc,ck->nk', [orgin_res_change, augment_res_change.T])
            cl_lables = torch.arange(l_pos_neg.size(0))

            # if self.set_cuda:
            cl_lables = cl_lables.cuda()
            l_pos_neg /= self.temperature

            cl_loss = self._cls_loss(l_pos_neg, cl_lables)

        if mode == "train":
            return emotion_logits, cl_loss, cl_self_loss, loss_ctc, loss_cls
        else:
            return emotion_logits, None, None, None, None

    def _ctc_loss(self, logits, labels, input_lengths, attention_mask=None):

        loss = None
        if labels is not None:

            # # retrieve loss input_lengths from attention_mask
            # attention_mask = (
            #     attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            # )
            # input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1))

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = F.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=0,
                    reduction="sum",
                    zero_infinity=False,
                    )

        return loss

    def _cls_loss(self, logits, cls_labels): # sum hidden_states over dim 1 (the sequence length), then feed into self.cls
        loss = None
        if cls_labels is not None:
            loss = F.cross_entropy(logits, cls_labels.to(logits.device))
        return loss




def run(config, train_data, valid_data, session):

    ############################ PARAMETER SETTING ##########################
    num_workers = 4
    batch_size = args.batch_size
    epochs = 100
    learning_rate = 5e-5 #5e-5 for cai et.al re-iplementation #0.00001 or 1e-5 for all other
    accum_iter = args.ga
    final_save_path = unqiue_path
    # stats_file = open(os.path.join(final_save_path, session) + '_' + 'stats.txt', 'a', buffering=1)

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

    print("*"*40)
    print("Model params")
    print(sum(p.numel() for p in model.parameters()))
    print("*"*40)

    print("*"*40)
    print("Load to CUDA")
    print("*"*40)

    model.cuda()

    print("*"*40)
    print("Loaded to CUDA ...")
    print("*"*40)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name) #to check if all params are trainable


    ########################### TRAINING #####################################
    print("*"*40)
    print("Training started")
    print("*"*40)

    count, best_metric, save_metric, best_epoch = 0, -np.inf, None, 0

    for epoch in range(epochs):
        # epoch_train_loss = []
        # epoch_train_cls_loss = []
        # epoch_train_ctc_loss = []
        # epoch_train_cl_loss = []
        # epoch_train_cl_self_loss = []
        # model.train()
        # start_time = time.time()
        # batch_idx = 0
        # time.sleep(2) # avoid the deadlock during the switch between the different dataloaders
        # progress = tqdm(train_loader, desc='Epoch {:0>3d}'.format(epoch))
        # for bert_input, audio_input, label_input, bert_augmented_input, audio_augmented_input in progress:
        #     # torch.cuda.empty_cache()
        #     attention_mask, text_length, bert_output =  bert_input[1].cuda(),bert_input[2].cuda(),bert_input[3].cuda()
        #     acoustic_input, acoustic_length = audio_input[0]['input_values'].cuda(),audio_input[1].cuda()
        #     ctc_labels, emotion_labels = label_input[0].cuda(),label_input[1].cuda()
        #     augmented_bert_output, augmented_attention_mask = bert_augmented_input[0].cuda(),bert_augmented_input[1].cuda()
        #     augmented_acoustic_input, augmented_acoustic_length = audio_augmented_input[0]['input_values'].cuda(),audio_augmented_input[1].cuda()
        #     target_labels = [_target.cuda() for _target in label_input[2]]

        #     #model.zero_grad()                                input_ids, attention_mask, audio_inputs, ctc_labels, emotion_labels, target_labels, augmented_input_ids, augmented_attention_mask, augmented_audio
        #     logits, cl_loss, cl_self_loss, ctc_loss, cls_loss = model(bert_output, attention_mask, acoustic_input, acoustic_length, ctc_labels, emotion_labels, target_labels, augmented_bert_output, augmented_attention_mask, augmented_acoustic_input, augmented_acoustic_length)
        #     # print(l_pos_neg)
        #     # print(cl_self_loss)
        #     # print(loss_cls)
        #     loss = 0.1*cl_loss + 0.1*cl_self_loss + cls_loss + 0.1*ctc_loss
        #     loss = loss/accum_iter
        #     loss.backward()

        #     if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
        #         epoch_train_loss.append(loss)
        #         epoch_train_cl_loss.append(cl_loss)
        #         epoch_train_cls_loss.append(cls_loss)
        #         epoch_train_ctc_loss.append(ctc_loss)
        #         epoch_train_cl_self_loss.append(cl_self_loss)
        #         optimizer.step()
        #         optimizer.zero_grad()

        #     batch_idx += 1
        #     count += 1
        #     acc_train_loss = torch.mean(torch.tensor(epoch_train_loss)).cpu().detach().numpy()
        #     cls_loss = torch.mean(torch.tensor(epoch_train_cls_loss)).cpu().detach().numpy()
        #     ctc_loss = torch.mean(torch.tensor(epoch_train_ctc_loss)).cpu().detach().numpy()
        #     cl_self_loss = torch.mean(torch.tensor(epoch_train_cl_self_loss)).cpu().detach().numpy()
        #     cl_loss = torch.mean(torch.tensor(epoch_train_cl_loss)).cpu().detach().numpy()

        #     # cl_self_loss = 0
        #     progress.set_description("Epoch {:0>3d} - Loss {:.4f} - CLS_Loss {:.4f} - CTC_Loss {:.4f} - CL Loss {:.4f}".format(epoch, acc_train_loss, cls_loss, ctc_loss, cl_self_loss))
        # del progress
        model.load_state_dict(torch.load('/fs/nexus-projects/audio-visual_dereverberation/clmlf/SCLMLF/model_store_8_1_2/2_model.pt')['state_dict'])
        model.eval()
        pred_y, true_y = [], []
        with torch.no_grad():
            time.sleep(2) # avoid the deadlock during the switch between the different dataloaders
            for bert_input, audio_input, label_input, bert_augmented_input, audio_augmented_input in tqdm(valid_loader):
                torch.cuda.empty_cache()
                attention_mask, text_length, bert_output =  bert_input[1].cuda(),bert_input[2].cuda(),bert_input[3].cuda()
                acoustic_input, acoustic_length = audio_input[0]['input_values'].cuda(),audio_input[1].cuda()
                ctc_labels, emotion_labels = label_input[0].cuda(),label_input[1].cuda()
                augmented_bert_output, augmented_attention_mask = bert_augmented_input[0].cuda(),bert_augmented_input[1].cuda()
                augmented_acoustic_input, augmented_acoustic_length = audio_augmented_input[0]['input_values'].cuda(),audio_augmented_input[1].cuda()
                target_labels = [_target.cuda() for _target in label_input[2]]

                true_y.extend(list(emotion_labels.cpu().numpy()))

                logits, cl_loss, cl_self_loss, ctc_loss, cls_loss = model(bert_output, attention_mask, acoustic_input, acoustic_length, ctc_labels, emotion_labels, target_labels, augmented_bert_output, augmented_attention_mask, augmented_acoustic_input, augmented_acoustic_length, mode = "valid")

                prediction = torch.argmax(logits, axis=1)
                label_outputs = prediction.cpu().detach().numpy().astype(int)

                pred_y.extend(list(label_outputs))
            # del valid_loader

        key_metric, report_metric = evaluate_metrics(pred_y, true_y)
        pickle.dump(pred_y, open(os.path.join(final_save_path, 'pred_y_83.pkl'),'wb'))
        pickle.dump(true_y, open(os.path.join(final_save_path, 'true_y_83.pkl'),'wb'))

        epoch_train_loss = torch.mean(torch.tensor(epoch_train_loss)).cpu().detach().numpy()

        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " +
                time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print('Valid Metric: {} - Train Loss: {:.3f}'.format(
            ' - '.join(['{}: {:.3f}'.format(key, value) for key, value in report_metric.items()]),
            epoch_train_loss))
        stats = dict(epoch=epoch, key_accuracy = key_metric, report_accuracy = report_metric)
        # print(json.dumps(stats), file=stats_file)


        if key_metric > best_metric:
            # torch.save({'state_dict': model.state_dict()}, os.path.join(final_save_path, session) + '_' + "model.pt")
            best_metric, best_epoch = key_metric, epoch
            print('Better Metric found on dev, calculate performance on Test')
            # pred_y, true_y = [], []
            # with torch.no_grad():
            #     time.sleep(2) # avoid the deadlock during the switch between the different dataloaders
            #     for bert_input, audio_input, label_input, bert_augmented_input, audio_augmented_input in valid_loader:
            #         torch.cuda.empty_cache()
            #         input_ids, attention_mask, text_length, bert_output =  bert_input[0].cuda(),bert_input[1].cuda(),bert_input[2].cuda(),bert_input[3].cuda()
            #         acoustic_input, acoustic_length = audio_input[0]['input_values'].cuda(),audio_input[1].cuda()
            #         ctc_labels, emotion_labels = label_input[0].cuda(),label_input[1].cuda()
            #         augmented_input_ids, augmented_attention_mask = bert_augmented_input[0].cuda(),bert_augmented_input[1].cuda()
            #         augmented_acoustic_input, augmented_acoustic_length = audio_augmented_input[0]['input_values'].cuda(),audio_augmented_input[1].cuda()
            #         target_labels = [_target.cuda() for _target in label_input[2]]

            #         true_y.extend(list(emotion_labels.cpu().numpy()))

            #         logits, cl_loss, cl_self_loss, ctc_loss, cls_loss = model(input_ids, attention_mask, acoustic_input, ctc_labels, emotion_labels, target_labels, augmented_input_ids, augmented_attention_mask, augmented_acoustic_input)


            #         prediction = torch.argmax(logits, axis=1)
            #         label_outputs = prediction.cpu().detach().numpy().astype(int)

            #         pred_y.extend(list(label_outputs))

            _, save_metric = evaluate_metrics(pred_y, true_y)
            print("Test Metric: {}".format(
                ' - '.join(['{}: {:.3f}'.format(key, value) for key, value in save_metric.items()])
            ))

    print("End. Best epoch {:03d}: {}".format(best_epoch, ' - '.join(['{}: {:.3f}'.format(key, value) for key, value in save_metric.items()])))
    return save_metric


if __name__ == "__main__":

    config_path = 'conf/config.yaml'
    csv_path = '/fs/nexus-projects/audio-visual_dereverberation/clmlf/SCLMLF/iemocap.csv'
    augmented_csv_path = '/fs/nexus-projects/audio-visual_dereverberation/clmlf/SCLMLF/dedup_iemocap_v2.2_final.csv'
    data_path_audio = '/fs/nexus-projects/audio-visual_dereverberation/clmlf/iemocap_files/'
    data_path_roberta = '/fs/nexus-projects/audio-visual_dereverberation/clmlf/numpy_roberta/'
    data_path_roberta_augmented = '/fs/nexus-projects/audio-visual_dereverberation/clmlf/numpy_roberta_augment/'


    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    report_result = []

    df_emotion = pd.read_csv(augmented_csv_path)

    for i in range(2,3):

        valid_session = "Ses0" + str(i)
        valid_data_csv = df_emotion[df_emotion["FileName"].str.match(valid_session)]
        train_data_csv = pd.DataFrame(df_emotion, index = list(set(df_emotion.index).difference(set(valid_data_csv.index)))).reset_index(drop= True)
        valid_data_csv.reset_index(drop= True, inplace= True)

        train_data = []
        valid_data = []

        for row in train_data_csv.itertuples():
            file_name = os.path.join(data_path_audio + row.FileName)
            bert_path = data_path_roberta + row.FileName
            bert_path_augmented = data_path_roberta_augmented + row.FileName
            train_data.append((file_name,bert_path,row.Sentences,row.Label,row.text,row.AugmentedText,bert_path_augmented))

        for row in valid_data_csv.itertuples():
            file_name = os.path.join(data_path_audio + row.FileName)
            bert_path = data_path_roberta + row.FileName
            bert_path_augmented = data_path_roberta_augmented + row.FileName
            valid_data.append((file_name,bert_path,row.Sentences,row.Label,row.text,row.AugmentedText,bert_path_augmented))

        report_metric = run(config, train_data, valid_data, str(i))
        report_result.append(report_metric)

        final_save_path = unqiue_path

        # pickle.dump(report_result, open(os.path.join(final_save_path, 'metric_report.pkl'),'wb'))