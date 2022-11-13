import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import os
import json
from torch import cuda
from data_loader import VQADataSet
from transformers import T5Tokenizer
from modeling_mt5 import MT5ForConditionalGeneration
from transformers import (Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)
from torch import cuda

import numpy as np
device = 'cuda' if cuda.is_available() else 'cpu'
# let's define model parameters specific to T5
model_params = {
    "MODEL": "google/mt5-large",  # model_type: t5-base/t5-large
    "IMG_MODEL": "google/vit-large-patch16-224-in21k",
    "MAX_SOURCE_TEXT_LENGTH" : 64,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 64,  # max length of target text
}

# Creation of Dataset and Dataloader
# Defining the train size. So 80% of the data will be used for training and the rest for validation.
train_json_path = "./datasets/vqa_train.json"
root_img_train = "./datasets/train-images"

test_json_path = "./datasets/vqa_public_test.json"
root_img_test = "./datasets/public-test-images"

# tokenzier for encoding the text
tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])
# Creating the Training and Validation dataset for further creation of Dataloader
training_set = VQADataSet(
        train_json_path,
        root_img_train,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        mode="train",
        model_params=model_params
    )
val_set = VQADataSet(
        test_json_path,
        root_img_test,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        mode="val",
        model_params=model_params
    )

batch_size = 4
model_name = "vit-large-mt5-large"
model_dir = f"outputs/{model_name}"


args = Seq2SeqTrainingArguments(
    model_dir,
    evaluation_strategy="epoch",
    eval_steps=2000,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="epoch",
    save_steps=2000,
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=None,
    num_train_epochs=50,
    gradient_accumulation_steps=4,
    predict_with_generate=True,
    gradient_checkpointing=False,
    fp16=False,
    load_best_model_at_end=True,
    report_to="tensorboard"
)
model = MT5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
model = model.to(device)
# model.config.use_cache = False

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=training_set,
    eval_dataset=val_set,
    tokenizer=tokenizer
)


trainer.train()

