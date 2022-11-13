from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor
import json
import torch
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
class VQADataSet(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, json_path,root_image_path, tokenizer, source_len=64, target_len=64,mode="train",model_params=None
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.target_text = []
        self.source_text = []
        self.image_ids = []
        self.ids = []
        self.langs = []
        if mode=="train":
            images_id2name = json.load(open("./datasets/evjvqa_train.json", encoding="utf-8"))["images"]
            obj_ = json.load(open("./datasets/train_obj.json", encoding="utf-8"))
        elif mode=="val":
            images_id2name = json.load(open("./datasets/evjvqa_public_test.json", encoding="utf-8"))["images"]
            obj_ = json.load(open("./datasets/public_test_obj.json", encoding="utf-8"))
        else:
            images_id2name = json.load(open("./datasets/prepared_evjvqa_private_test.json", encoding="utf-8"))["images"]
            obj_ = json.load(open("./datasets/private_test_obj.json", encoding="utf-8"))
        self.id2img_name = {}
        for e in images_id2name:
            self.id2img_name[e["id"]] = e["filename"]
        with open(json_path,"r",encoding="utf-8") as fr:
            data = json.load(fr)
        for e in data["annotations"]:
            self.source_text.append( f"object: {', '.join(obj_[self.id2img_name[e['image_id']]])} ; answer in {e['lang']}: "+e["question"])
            self.target_text.append(e["answer"])
            self.image_ids.append(e["image_id"])
            self.ids.append(e["id"])
            self.langs.append(e["lang"])
        self.source_len = source_len
        self.summ_len = target_len
        self.root_image_path = root_image_path
        self.mode = mode
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_params["IMG_MODEL"])

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()
        image_path = os.path.join(self.root_image_path,self.id2img_name[self.image_ids[index]])
        image = Image.open(image_path)
        pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.squeeze()
        qid = self.ids[index]
        qid = torch.tensor([qid])
        if self.mode=="test":
            return {
                "input_ids": source_ids.to(dtype=torch.long),
                "attention_mask": source_mask.to(dtype=torch.long),
                "labels": target_ids.to(dtype=torch.long),
                "pixel_values": pixel_values,
                "qid":  qid
            }

        return {
            "input_ids": source_ids.to(dtype=torch.long),
            "attention_mask": source_mask.to(dtype=torch.long),
            "labels": target_ids.to(dtype=torch.long),
            "pixel_values": pixel_values
        }