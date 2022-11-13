
from PIL import Image
import torch
from tqdm import tqdm
import torchvision.transforms as T
torch.set_grad_enabled(False)
# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
model.eval()
def get_obj_list(img_path):
    im = Image.open(img_path)
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)
    # propagate through the model
    outputs = model(img)
    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    probas_score = probas.max(-1).values
    keep = probas_score > 0.5
    # convert boxes from [0; 1] to image scales
    results = []
    for p in probas[keep]:
        cl = p.argmax()
        object_ = CLASSES[cl]
        results.append(object_)
    return list(set(results))

import os, json

image_dir = "./private-test-images"
output_file = "private_test_obj.json"
train_img_list = os.listdir(image_dir)
train_obj = {}
for idx,img_name  in tqdm(enumerate(train_img_list)):
    objs  = get_obj_list(os.path.join(image_dir,img_name))
    train_obj[img_name] = objs
with open(output_file,"w",encoding="utf-8") as fw:
    json.dump(train_obj,fw,indent=4, ensure_ascii=False)