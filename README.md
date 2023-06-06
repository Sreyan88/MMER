# MMER

This repository contains code for the InterSpeech 2023 paper [MMER: Multimodal Multi-task Learning for Speech Emotion Recognition](https://arxiv.org/abs/2203.16794)  

### Proposed MMER Architecture:  

![Proposed Architecture :](./assets/MMER.pdf)  

Tu run our model, first download roberta embeddings using gdown with this [link](https://drive.google.com/file/d/1xCpOWpwuw8eLyjm1fxcyJF8U_qtQDSCc/view?usp=sharing) in the data folder. Then prepare and extract IEMOCAP audio files in data/audio using instructions in data_prep folder.

To run our sota implementation of MMER in the paper, please run:  
```
sh best_run.sh path_to_audio_files \  
path_to_roberta_embeddings \  
path_to_iemocap_csv \  
path_to_save_directory
```
To run other variants, please change the arguments accordingly. Some main arguments are listed below:    
```
--run : you have 3 model variants you can run, cai_sota (implementation of the paper (https://www.isca-speech.org/archive/pdfs/interspeech_2021/cai21b_interspeech.pdf), unimodal_baseline (wav2vec-2.0 baseline) and mmer (our paper). 

--alpha : weight for CTC loss in the final loss  
```

If you find this work useful, please do cite our paper:  
```
@inproceedings{ghosh22b_interspeech,
  author={Sreyan Ghosh and Utkarsh Tyagi and S Ramaneswaran and Harshvardhan Srivastava and Dinesh Manocha},
  title={{MMER: Multimodal Multi-task Learning for Speech Emotion Recognition}},
  year=2023,
  booktitle={Proc. Interspeech 2023},
}
```
