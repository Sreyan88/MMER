

python src/run_iemocap_infer.py \
--data_path_audio $data_path_iemocap \
--data_path_roberta $data_path_roberta \
--csv_path $data_path_csv \
--save_path $output_path \
--config $config \
--bert_config $bert_config \
--data_path_audio_augmented $data_path_iemocap_augmented \
--data_path_roberta_augmented $data_path_roberta_augmented