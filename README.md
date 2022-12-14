# Vietnamese Visual Question Answering at VLSP 2022

We use VIT model to encode image, and concate it with text embedding and forward it to mt5 encode decode model. Use DETR detection to get object list from image and use as prefix for text input. We use also question language detection to make prefix for text input. Example text in put for mt5 encode decode : 

```
object: object1 object2 ... ; answer in ja: question

```


## Prepare dataset

Download and unzip at here:

```
datasets
    private-test-images/
    public-test-images/
    train-images/
    evjvqa_train.json
    evjvqa_public_test.json
    prepared_evjvqa_private_test.json
```

### If you want recreate vqa_train.json,vqa_public_test.json,vqa_private_test.json(question language detection)  you can :

Edit input file path and output file path in datasets/predict_language_question.py

```
cd ./datasets
python predict_language_question.py

```
### If you want recreate *_obj.json(list object from detr dection for each image). You can:

```
cd ./datasets
python detr_predict_obj.py
```

## Setup ENV
```
docker build -t nhanv-vqa .
```


## Inference
Download model and push it into outputs. Link download at: ./outputs/download.txt

```
outputs/
    vit-large-mt5-large/
    vit-large-patch16-224-in21k/
```
To inference on private test with  file and model is trained. you can run:

```
./interactive.sh
python3 predict.py
```

Results at ./outputs/results.json

## Traning

To training you run:
```
./interactive.sh
python3 train.py
```
if GPU is limited, you can set gradient_checkpoiting=True and model.config.use_cache=False in mt5_trainer.py

you can change size of model from large to base in model_params in mt5_trainer.py
