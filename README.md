# Vietnamese Visual Question Answering at VLSP 2022

We use VIT model to encode image, and concate it with text and forward it to mt5 encode decode model. We can also prefix for text input. Use DETR detection to get object list from image and use as prefix for text input.


## Prepare dataset

Download and unzip at here:
    datasets/private-test-images/
    datasets/public-test-images/
    datasets/train-images/

### If you want recreate vqa_train.json,vqa_public_test.json,vqa_private_test.json(question language detection)  you can :

Edit input file path and output file path in datasets/predict_language_question.py

cd ./datasets

python predict_language_question.py

### If you want recreate *_obj.json(list object from detr dection for each image). You can:

cd ./datasets

python predict_language_question.py


## Inference
Download model and push it into outputs. Link download at: ./outputs/download.txt

To inference on private test with  file and model is trained. you can run:

python predict.py

Results at ./outputs/results.json

## Traning

To training you run:

python train.py

if GPU is limits, you can set gradient_checkpoiting=True and model.config.use_cache=False in mt5_trainer.py

you can change size of model from large to base in model_params in mt5_trainer.py
