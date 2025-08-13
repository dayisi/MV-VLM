# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Xinlong Hou and collaborators

import os
import requests
import numpy as np
import re

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

model_id = "sentence-transformers/all-MiniLM-L6-v2"
hf_token = "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"

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', cache_dir="/media/yuganlab/blackstone/xinlong/cache/miniLM_cache")
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', cache_dir="/media/yuganlab/blackstone/xinlong/cache/miniLM_cache")
device="cuda:0"


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

model = model.to(device)
def extract_features_and_save(input_dir, output_dir, clear_gpu_every=100):
    counter = 0
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    text = f.read()

                encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
                encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
                
                with torch.no_grad():
                    model_output = model(**encoded_input)

                sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
                
                # Move to CPU immediately before saving
                sentence_embeddings_cpu = sentence_embeddings.cpu()
                
                save_path = file_path.replace(input_dir, output_dir).replace(".txt", ".npy")
                print(f"[{counter+1}] Saving: {save_path}")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.save(save_path, sentence_embeddings_cpu)
                
                counter += 1

                # Optional: clear GPU cache every N files
                if counter % clear_gpu_every == 0:
                    print(f"Clearing GPU cache at file {counter}...")
                    torch.cuda.empty_cache()
    print(f"Done. Processed {counter} files.")

folder_base="<your_base_directory_here>" # Please set your base folder here
input_directory = os.path.join(folder_base, "example_data/caption/example_data")
output_directory = input_directory + "_textemb"
extract_features_and_save(input_directory, output_directory)
