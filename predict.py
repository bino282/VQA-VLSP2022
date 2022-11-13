import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from modeling_mt5 import MT5ForConditionalGeneration
from torch import cuda
from data_loader import VQADataSet
os.environ["TOKENIZERS_PARALLELISM"]="true"
device = 'cuda' if cuda.is_available() else 'cpu'

model_params = {
    "MODEL": "google/mt5-large",  # model_type: t5-base/t5-large
    "IMG_MODEL": "google/vit-large-patch16-224-in21k",
    "MAX_SOURCE_TEXT_LENGTH": 64,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 64,  # max length of target text
}

model_name = "./outputs/vit-large-mt5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)
model = model.to(device)
batch_size = 8
test_json_path = "./datasets/vqa_private_test.json"
root_img_test = "./datasets/private-test-images"

test_params = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": 0,
    }
test_set = VQADataSet(
        test_json_path,
        root_img_test,
        tokenizer,
        64,
        64,
        mode="test",
        model_params=model_params
    )

test_loader = DataLoader(test_set, **test_params)

model.eval()
predictions = []
actuals = []
qid2pred = {}
with torch.no_grad():
    for batch in tqdm(test_loader):
        qid = batch["qid"].squeeze().cpu().tolist()
        labels = batch["labels"]
        del batch['qid']
        del batch['labels']
        for k in batch:
            batch[k] = batch[k].to(device)
        generated_ids = model.generate(**batch,num_beams=1, max_length=64)
        preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
        target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in labels]
        assert len(preds) == len(qid)
        predictions.extend(preds)
        actuals.extend(target)
        for k in range(len(qid)):
            id_ = qid[k]
            qid2pred[id_]  = preds[k].replace("<extra_id_0>","").strip().lower()
output_dir ="outputs/"
final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals})
final_df.to_csv(os.path.join(output_dir, "predictions.csv"))
with open(os.path.join(output_dir,"results.json"),"w",encoding="utf-8") as fw:
    json.dump(qid2pred,fw, indent=4, ensure_ascii=False)