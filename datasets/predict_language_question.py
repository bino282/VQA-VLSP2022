import json
from langdetect import detect

in_file = "prepared_evjvqa_private_test.json"
out_file = "vqa_private_test.json"

with open(in_file,"r",encoding="utf-8") as fr:
    data= json.load(fr)
raw_annotations = data["annotations"]
annotations = []
for e in raw_annotations:
    lang = detect(e["question"])
    if lang not in ["vi","en","ja"]:
        lang = "en"
    e["lang"] = lang
    annotations.append(e)

with open(out_file,"w",encoding="utf-8") as fw:
    json.dump({"annotations":annotations},fw, indent=4, ensure_ascii=False)