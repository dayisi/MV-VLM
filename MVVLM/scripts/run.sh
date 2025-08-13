#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Xinlong Hou and collaborators

dataset="screen_proj"

maxLen=30
minNewTokens=10

folder_base="<path_to_your_base_folder>"  # Replace with your actual base folder path

annotation="${folder_base}/example_data/annotation_file/example_caption.json"
base_dir_img="${folder_base}/example_data"
base_dir_text="${folder_base}/example_data/caption/example_data_textemb"

savepath="${folder_base}/MVVLM/results/test"
delta_file="${folder_base}/MVVLM/saved_ckpt/checkpoint_epoch5_step437.pth"

if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi
# Function to get the current time in seconds since epoch
current_time_in_seconds() {
  date +%s
}
# Function to format time from seconds since epoch
format_time() {
  date -d @$1 +"%Y-%m-%d %H:%M:%S"
}
# Capture the start time in seconds
start_time=$(current_time_in_seconds)
formatted_start_time=$(format_time $start_time)
echo "Start Time: $formatted_start_time"

python -u MVVLM_train_textEmb.py \
    --test \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir_img} \
    --base_dir_text ${base_dir_text} \
    --delta_file ${delta_file} \
    --batch_size 6 \
    --val_batch_size 5 \
    --freeze_vm True \
    --freeze_alignment True \
    --vis_use_lora False \
    --llm_use_lora False \
    --savedmodel_path ${savepath} \
    --max_length ${maxLen} \
    --min_new_tokens ${minNewTokens} \
    --max_new_tokens ${maxLen} \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 12 \
    --devices 1 \
    --max_epochs 1 \
    --limit_val_batches 1 \
    --val_check_interval 0.25 \
    --num_sanity_val_steps 2 \
    --TRAINMODE len4 \
    --llama_mode "<path_to_your_Llama_folder>" \
    2>&1 |tee -a ${savepath}/log.txt


# Capture the end time in seconds
end_time=$(current_time_in_seconds)
formatted_end_time=$(format_time $end_time)
echo "End Time: $formatted_end_time"

# Calculate the duration
duration=$((end_time - start_time))
echo "Duration: $duration seconds"

# Optionally, you can format the duration in HH:MM:SS
formatted_duration=$(printf '%02d:%02d:%02d' $((duration/3600)) $(( (duration%3600)/60 )) $((duration%60)))
echo "Formatted Duration: $formatted_duration"
