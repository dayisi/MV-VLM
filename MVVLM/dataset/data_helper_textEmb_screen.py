# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Xinlong Hou and collaborators
import os
import json
import re
import numpy as np
from PIL import Image
import torch.utils.data as data
from transformers import BertTokenizer, AutoImageProcessor


class FieldParser:
    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args
        self.dataset = args.dataset
        self.vit_feature_extractor = AutoImageProcessor.from_pretrained(args.vision_model)


    def _parse_image(self, img):
        pixel_values = self.vit_feature_extractor(img, return_tensors="pt").pixel_values
        return pixel_values[0] 

    # from https://github.com/cuhksz-nlp/R2Gen/blob/main/modules/tokenizers.py
    def clean_report(self, report):
        if self.dataset == "screen_proj":
            return report
        else:
            report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
                .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
                .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
                .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
                .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ').replace(':', ' :') \
                .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+()\[\]{}]', '', t.replace('"', '').replace('/', '')
                                .replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .' 
        return report


    def parse(self, features):
        #to_return = {'id': features['id']}
        to_return = {'study_id': features['study_id']}
        report = features.get("report", "")
        report = self.clean_report(report)
        to_return['input_text'] = report
        to_return['img_len'] = len(features['image_path'])
        
        if self.args.dataset == 'screen_proj':
            image_path = features['image_path'][0]
            text_emb_path = os.path.dirname(image_path)
            text_emb_path = os.path.join(text_emb_path, features["study_id"])
            text_emb_path += ".npy"
            text_emb_path = os.path.join(self.args.base_dir_text , text_emb_path)
        else:
            image_path = features['image_path'][0]
            text_emb_path = image_path.split('/')[0]
            text_emb_path = text_emb_path.split('_')[0]
            text_emb_path = os.path.join(self.args.base_dir_text , text_emb_path  , text_emb_path + '.npy')

        text_emb = np.load(text_emb_path)
        text_emb = np.max(text_emb, axis=0)
        to_return['addition_text'] = text_emb
        
        images = []
        for image_path in features['image_path']:
            with Image.open(os.path.join(self.args.base_dir, image_path)) as pil:
                array = np.array(pil, dtype=np.uint8)
                if array.shape[-1] != 3 or len(array.shape) != 3:
                    array = np.array(pil.convert("RGB"), dtype=np.uint8)
                image = self._parse_image(array)
                images.append(image)

        TRAINMODE = self.args.TRAINMODE
        if TRAINMODE == "len4":
            if len(images) == 1:
                res_img = [images[0], images[0], images[0], images[0]]
            elif len(images) == 2:
                res_img = [images[0], images[1], images[0], images[1]]
            elif len(images) == 3:
                res_img = [images[0], images[1], images[2], (images[0] + images[1] + images[2]) / 3]
            else:
                res_img = [images[0], images[1], images[2], images[3]]
        else:
            if len(images) == 1:
                res_img = [images[0], images[0]]
            else:
                res_img = [images[0], images[1]]
                
        to_return["image"] = res_img
        return to_return


    def transform_with_parse(self, inputs):
        return self.parse(inputs)


class ParseDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        self.meta = json.load(open(args.annotation, 'r'))
        self.meta = self.meta[split]
        self.parser = FieldParser(args)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        return self.parser.transform_with_parse(self.meta[index])


def create_datasets(args):
    train_dataset = ParseDataset(args, 'train')
    dev_dataset = ParseDataset(args, 'val')
    test_dataset = ParseDataset(args, 'test')
    return train_dataset, dev_dataset, test_dataset


