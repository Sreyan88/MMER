#!/usr/bin/env bash

python src/run_iemocap.py --run mmer --data_path_audio $1 --data_path_roberta $2 --csv_path $3 --save_path $4 \
--config configs/iemocap-ours.yaml