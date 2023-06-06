#!/usr/bin/env python
# coding: utf-8

# # Demo zero-shot TTS with YourTTS

# ##TTS Model setup

# ### Download and install Coqui TTS
# 

# In[5]:


# !git clone https://github.com/Edresson/Coqui-TTS -b multilingual-torchaudio-SE TTS
# !python -m pip install -q -e TTS/
# !python -m pip install -q torchaudio==0.9.0


# ###Download TTS Checkpoint

# In[1]:


# TTS checkpoints

# # download config  
# ! gdown --id 1-PfXD66l1ZpsZmJiC-vhL055CDSugLyP
# # download language json 
# ! gdown --id 1_Vb2_XHqcC0OcvRF82F883MTxfTRmerg
# # download speakers json
# ! gdown --id 1SZ9GE0CBM-xGstiXH2-O2QWdmSXsBKdC -O speakers.json
# # download checkpoint
# ! gdown --id 1sgEjHt0lbPSEw9-FSbC_mBoOPwNi87YR -O best_model.pth.tar  


# ### Imports

# In[1]:


import sys
TTS_PATH = "TTS/"

# add libraries into environment
sys.path.append(TTS_PATH) # set this if TTS is not installed globally

import os
import string
import time
import argparse
import json

import numpy as np
import IPython
from IPython.display import Audio


import torch

from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text.symbols import make_symbols, phonemes, symbols
try:
  from TTS.utils.audio import AudioProcessor
except:
  from TTS.utils.audio import AudioProcessor


from TTS.tts.models import setup_model
from TTS.config import load_config
from TTS.tts.models.vits import *


# ### Paths definition

# In[2]:


OUT_PATH = 'out/'

# create output path
os.makedirs(OUT_PATH, exist_ok=True)

# model vars 
MODEL_PATH = 'best_model.pth.tar'
CONFIG_PATH = 'config.json'
TTS_LANGUAGES = "language_ids.json"
TTS_SPEAKERS = "speakers.json"
USE_CUDA = torch.cuda.is_available()


# ### Restore model

# In[3]:


# load the config
C = load_config(CONFIG_PATH)


# load the audio processor
ap = AudioProcessor(**C.audio)

speaker_embedding = None

C.model_args['d_vector_file'] = TTS_SPEAKERS
C.model_args['use_speaker_encoder_as_loss'] = False

model = setup_model(C)
model.language_manager.set_language_ids_from_file(TTS_LANGUAGES)
# print(model.language_manager.num_languages, model.embedded_language_dim)
# print(model.emb_l)
cp = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
# remove speaker encoder
model_weights = cp['model'].copy()
for key in list(model_weights.keys()):
  if "speaker_encoder" in key:
    del model_weights[key]

model.load_state_dict(model_weights)


model.eval()

if USE_CUDA:
    model = model.cuda()

# synthesize voice
use_griffin_lim = False


# ##Speaker encoder setup

# ### Install helper libraries

# In[5]:


# ! pip install -q pydub ffmpeg-normalize==1.21.0


# ### Paths definition

# In[4]:


CONFIG_SE_PATH = "config_se.json"
CHECKPOINT_SE_PATH = "SE_checkpoint.pth.tar"

# download config 
# ! gdown --id  19cDrhZZ0PfKf2Zhr_ebB-QASRw844Tn1 -O $CONFIG_SE_PATH
# download checkpoint  
# ! gdown --id   17JsW6h6TIh7-LkU2EvB_gnNrPcdBxt7X -O $CHECKPOINT_SE_PATH


# ###Imports

# In[5]:


from TTS.tts.utils.speakers import SpeakerManager
from pydub import AudioSegment
# from google.colab import files
import librosa


# ###Load the Speaker encoder

# In[6]:


SE_speaker_manager = SpeakerManager(encoder_model_path=CHECKPOINT_SE_PATH, encoder_config_path=CONFIG_SE_PATH, use_cuda=USE_CUDA)


# ###Define helper function

# In[7]:


def compute_spec(ref_file):
  y, sr = librosa.load(ref_file, sr=ap.sample_rate)
  spec = ap.spectrogram(y)
  spec = torch.FloatTensor(spec).unsqueeze(0)
  return spec


# ## TTS

# ###Upload, normalize and resample your reference wav files

# Please upload wav files

# In[ ]:


# print("Select speaker reference audios files:")
# reference_files = files.upload()
# reference_files = list(reference_files.keys())
# for sample in reference_files:
#     !ffmpeg-normalize $sample -nt rms -t=-27 -o $sample -ar 16000 -f


# ###Compute embedding

# In[8]:


all_files = os.listdir("/speech/sreyan/IEMOCAP/iemocap_files")


# In[9]:


all_speakers = []

for file in all_files:
    
    speaker = file.split(".")[0][:-3]
    
    all_speakers.append(speaker)
    
print(len(set(all_speakers)))

print(len(all_speakers))

unique_speakers = list(set(all_speakers))


# In[18]:


# IPython.display.display(Audio(reference_files[10], rate=ap.sample_rate))


# In[10]:


# reference_files = ["/speech/sreyan/IEMOCAP/iemocap_files/" + item for item in all_files if item.startswith(unique_speakers[0])]


# In[11]:


# reference_emb = SE_speaker_manager.compute_d_vector_from_clip(reference_files)


# ###Define inference variables

# In[26]:


model.length_scale = 1.5  # scaler for the duration predictor. The larger it is, the slower the speech.
model.inference_noise_scale = 0.0 # defines the noise variance applied to the random z vector at inference.
model.inference_noise_scale_dp = 0.0 # defines the noise variance applied to the duration predictor z vector at inference.
text = "It took me quite a long time to develop a voice and now that I have it I am not going to be silent."


# ###Chose language id

# In[13]:


model.language_manager.language_id_mapping


# In[14]:


language_id = 0


# ### Sythesis

# In[27]:


# print(" > text: {}".format(text))
# wav, alignment, _, _ = synthesis(
#                     model,
#                     text,
#                     C,
#                     "cuda" in str(next(model.parameters()).device),
#                     ap,
#                     speaker_id=None,
#                     d_vector=reference_emb,
#                     style_wav=None,
#                     language_id=language_id,
#                     enable_eos_bos_chars=C.enable_eos_bos_chars,
#                     use_griffin_lim=True,
#                     do_trim_silence=False,
#                 ).values()
# print("Generated Audio")
# IPython.display.display(Audio(wav, rate=ap.sample_rate))
# file_name = text.replace(" ", "_")
# file_name = file_name.translate(str.maketrans('', '', string.punctuation.replace('_', ''))) + '.wav'
# out_path = os.path.join(OUT_PATH, file_name)
# print(" > Saving output to {}".format(out_path))
# ap.save_wav(wav, out_path)


# In[ ]:


from tqdm import tqdm as tqdm

import pandas as pd

df = pd.read_csv("iemocap_v1.csv")

for i in tqdm(range(len(unique_speakers))):

    reference_files = ["/speech/sreyan/IEMOCAP/iemocap_files/" + item for item in all_files if item.startswith(unique_speakers[i])]
    reference_emb = SE_speaker_manager.compute_d_vector_from_clip(reference_files)
    
    for i,row in tqdm(df.iterrows()):
        text = row["Sentence_v1"]
        wav, alignment, _, _ = synthesis(
                        model,
                        text,
                        C,
                        "cuda" in str(next(model.parameters()).device),
                        ap,
                        speaker_id=None,
                        d_vector=reference_emb,
                        style_wav=None,
                        language_id=language_id,
                        enable_eos_bos_chars=C.enable_eos_bos_chars,
                        use_griffin_lim=True,
                        do_trim_silence=False,
                    ).values()
        
        
        file_name = str(unique_speakers[i]) + "_" + row["FileName"] + ".wav"
        
        out_path = os.path.join(OUT_PATH, file_name)
        
        ap.save_wav(wav, out_path)

