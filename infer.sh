session=$1
config=$2
csv_path=$3
data_path_audio=$4
data_path_roberta=$5
checkpoint_path=$6

python src/run_iemocap_infer.py \
--session $session \
--data_path_audio $data_path_audio \
--data_path_roberta $data_path_roberta \
--csv_path $csv_path \
--config_path $config \
--checkpoint_path $checkpoint_path \