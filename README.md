# Natural Language Understanding for Explanational Information Tagging Competition


Competition Link: [TBrain AI 實戰吧](https://tbrain.trendmicro.com.tw/Competitions/Details/26)
Model List: [Huggingface Pre-trained Model List](https://huggingface.co/transformers/v3.3.1/pretrained_models.html)

---
# My Method - Token Classification

對 q 跟 r 分別作標記，有出現在答案的標 1，沒出現得標 0

## Training Steps
0. 把訓練數據載進 `data` 裡
1. `preprocess.py`: Data preprocessing for training data.
2. `train.py`: Finetuning the pretrained model on your data.
3. `evaluate.py`: (optional) Evaluation on training data to see the results.
4. `try.py`: (optional) See the lcs results on training data.

## Submission
0. Download submission data in folder `data`
1. `preprocess-sub.py`: Data preprocessing for submission data. (Same tokenizer as training model)
2. `predict-sub.py`: Generate model prediction on submission dataset.
3. `try-sub.py`: Make the submission csv.
> 1. First Part: Make answer for submission data.
> 2. Second Part: Tidy csv for submissiion. 

---
Training resource: 1 GPU (GeForce RTX 2080 Ti)