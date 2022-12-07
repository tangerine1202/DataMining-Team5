import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

from datasets import Dataset

from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification

data_path = "data/Batch_answers - train_data (no-blank).csv"
goal_path = "data/tc-bert-base-dataset-"
model_name = "bert-base-uncased"

"""
"""
tokenizer = AutoTokenizer.from_pretrained(model_name)
df = pd.read_csv(data_path)
# drop unnecessary columns
df = df.drop(['Unnamed: 6', 'total no.: 7987'], axis=1)
# remove quotes
df['q'] = df['q'].str.strip('"')
df['r'] = df['r'].str.strip('"')
df["q'"] = df["q'"].str.strip('"')
df["r'"] = df["r'"].str.strip('"')
# add original index to df
df = df.reset_index()
# drop duplicated rows
df = df.drop_duplicates()
# some information about the dataset
print(df['s'].value_counts())
print('# of distinct data:\t', len(df['id'].unique()))
print('# of data:\t', len(df))

def get_tokenized_text(data, category):
  input_ids = []
  attention_mask = []

  for ind in data.index:
    tokenized_text = tokenizer(data.loc[ind][category])
    input_ids.append(tokenized_text["input_ids"])
    attention_mask.append(tokenized_text["attention_mask"])

  return input_ids, attention_mask

df["q_input_ids"], df["q_attention_mask"] = get_tokenized_text(df, "q")
df["r_input_ids"], df["r_attention_mask"] = get_tokenized_text(df, "r")
df["q'_input_ids"], df["q'_attention_mask"] = get_tokenized_text(df, "q'")
df["r'_input_ids"], df["r'_attention_mask"] = get_tokenized_text(df, "r'")

print("\nTokenized Q and R")
print(df.shape)

no_one_id = []

def make_labels_sep(tokens, tags, ind):
    tokens = tokens[1:-1]
    tags = tags[1:-1]
    ans = [-100]  # [CLS]
    tid = 0
    tlen, alen = len(tokens), len(tags)

    while tid < tlen:
        if tid + alen - 1 >= tlen:
            ans.append(0)
        elif tokens[tid] == tags[0] and tokens[tid + alen - 1] == tags[-1]:
            aid = 0
            stop = False
            while aid < alen:
                if tokens[tid + aid] != tags[aid]:
                    stop = True
                    break
                aid += 1
            if not stop:
                ans.extend([1 for _ in range(alen)])
                tid += (alen - 1)
            else:
                ans.append(0)
        else:
            ans.append(0)
        tid += 1
    ans.append(-100)
    if 1 not in ans:
        no_one_id.append(ind)
    return ans

df["q_labels"] = df[["q_input_ids", "q'_input_ids", "index"]].apply(lambda x : make_labels_sep(x["q_input_ids"], x["q'_input_ids"], x["index"]), axis=1)
df["r_labels"] = df[["r_input_ids", "r'_input_ids", "index"]].apply(lambda x : make_labels_sep(x["r_input_ids"], x["r'_input_ids"], x["index"]), axis=1)

df = df.drop(no_one_id)
print("\nMake Labels and Clean")
print(df.shape)

# combine token and label
def combine_input(q_input_ids, r_input_ids, s):
  s = tokenizer.convert_tokens_to_ids(s.lower())
  endpoint = 1012

  input_ids = q_input_ids[:-1]
  input_ids.extend([s, endpoint])
  input_ids.extend(r_input_ids[1:])

  # truncation
  if len(input_ids) > 512:
    input_ids = input_ids[:511]
    input_ids.append(102)
  return input_ids

def combine_label(q_label, r_label):
  label_list = q_label[:-1]
  label_list.extend([0, 0])
  label_list.extend(r_label[1:])

  # truncation
  if len(label_list) > 512:
    label_list = label_list[:511]
    label_list.append(-100)
  return label_list

def make_attention_mask(input_ids):
  attention_mask = [1 for _ in range(len(input_ids))]
  return attention_mask

df["input_ids"] = df[["q_input_ids", "r_input_ids", "s"]].apply(lambda x: combine_input(x["q_input_ids"], x["r_input_ids"], x["s"]), axis=1)
df["labels"] = df[["q_labels", "r_labels"]].apply(lambda x: combine_label(x["q_labels"], x["r_labels"]), axis=1)
df["attention_mask"] = df["input_ids"].apply(lambda x : make_attention_mask(x))

print("\nCombine Q and R as Input")
print(df.shape)

# Ouput to dataset
total = df.shape[0]
df1 = df.iloc[:int(total * 0.8), :]
df2 = df.iloc[int(total * 0.8):, :]

# prepare training dataset
train_ds = Dataset.from_pandas(df1)
train_ds.save_to_disk(goal_path + "train")
print(train_ds)
valid_ds = Dataset.from_pandas(df2)
valid_ds.save_to_disk(goal_path + "valid")
print(valid_ds)

# prepare submission dataset
# test_ds = Dataset.from_pandas(test_data)
# test_ds.save_to_disk(dir + "test_dataset")
# print(test_ds)


