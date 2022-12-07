import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

from datasets import Dataset

from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification

data_path = "data/Batch_answers - test_data(no_label).csv"
goal_path = "data/tc-bert-base-submission-dataset"
model_name = "tc-bert-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
df = pd.read_csv(data_path)
# remove quotes
df['q'] = df['q'].str.strip('"')
df['r'] = df['r'].str.strip('"')
# some information about the dataset
print(df['s'].value_counts())
print('# of distinct data:\t', len(df['id'].unique()))
print('# of data:\t', len(df))
print(df.head(3))

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
df["attention_mask"] = df["input_ids"].apply(lambda x : make_attention_mask(x))

print("\nCombine Q and R as Input")
print(df.shape)

ds = Dataset.from_pandas(df)
ds.save_to_disk(goal_path)


