# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Xinlong Hou and collaborators

import os, json

def validate_file_path(file_path):
    # Get the directory name from the file path
    parent_dir = os.path.dirname(file_path)
    # Check if the directory exists
    if not os.path.exists(parent_dir):
        # Create the directory (including intermediate directories)
        os.makedirs(parent_dir)
        
screens = ["TV", "tv", "smartphone", "television", "smart phone", "cell phone", "mobile phone", "iPhone", "Portable PC", "computer monitor", "personal computer", "PC", "MacBook", "Notebook", "computer", "monitor", "tablet", "laptop", "telephone", "phone"]

screen_token = "[SCREEN_TYPE]"
# screen_token = "<screen_type>"

def mask_screen(content):
    masked_content = content.lower()
    for screen in screens:
        if screen.lower() in masked_content:
            masked_content = masked_content.replace(screen.lower(), screen_token)
    return masked_content

test_version = "textTest"
folder_base="<your_path>/git_publish/MV-VLM" # Please set your base folder here
input_json_file = os.path.join(folder_base, "example_data/annotation_file/example_caption.json")

output_group_caption_folder = os.path.join(folder_base, "example_data/caption")
output_data = {}
with open(input_json_file, "r") as input_f:
    ref_data = json.load(input_f)
    for split in ref_data:
        output_data[split] = []
        for group_obj in ref_data[split]:
            group_caption = ("; ").join(group_obj["caption"])
            screen_type = group_obj["label"]
            subject_id = group_obj["subject_id"]
            group_caption_filename = group_obj["study_id"] + ".txt"
            group_caption_filepath = os.path.join(output_group_caption_folder, subject_id, "images", screen_type, group_caption_filename)
            validate_file_path(group_caption_filepath)
            with open(group_caption_filepath, "w", newline='') as output_f:
                output_f.write(group_caption)
                print(f"{group_caption_filename} written")