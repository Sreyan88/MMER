#!/usr/bin/env bash

root="$(pwd)"
# Pre-defined variables
config="$root/configs/iemocap-ours.yaml"
bert_config="$root/configs/config.json"
data_path_csv="$root/data/iemocap.csv"
data_path_iemocap="$root/data/iemocap/"
data_path_iemocap_augmented="$root/data/iemocap_augmented"
data_path_roberta="$root/data/roberta"
data_path_roberta_augmented="$root/data/roberta_augmented"
output_path="$root/output/"

python src/run_iemocap.py \
--data_path_audio $data_path_iemocap \
--data_path_roberta $data_path_roberta \
--csv_path $data_path_csv \
--save_path $output_path \
--config $config \
--bert_config $bert_config \
--data_path_audio_augmented $data_path_iemocap_augmented \
--data_path_roberta_augmented $data_path_roberta_augmented