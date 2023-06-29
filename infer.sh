config='/fs/nexus-projects/audio-visual_dereverberation/clmlf/MMER/configs/iemocap-ours.yaml'
csv_path='/fs/nexus-projects/audio-visual_dereverberation/clmlf/SCLMLF/iemocap.csv'
data_path_audio='/fs/nexus-projects/audio-visual_dereverberation/clmlf/iemocap_files/'
data_path_roberta='/fs/nexus-projects/audio-visual_dereverberation/clmlf/numpy_roberta/'
checkpoint_path='/fs/nexus-projects/audio-visual_dereverberation/clmlf/SCLMLF/1_model.pt'

python src/run_iemocap_infer.py \
--data_path_audio $data_path_audio \
--data_path_roberta $data_path_roberta \
--csv_path $csv_path \
--config_path $config \
--checkpoint_path $checkpoint_path \