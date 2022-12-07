import numpy as np
import pandas as pd
import math
import nltk
import pickle
from nltk.tokenize import word_tokenize

from datasets import Dataset, load_from_disk

from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("tc-bert-base")
print("tokenizer")
data_path = "data/Batch_answers - test_data(no_label).csv"
df = pd.read_csv(data_path)
print("read df")
sub_ds = load_from_disk("data/tc-bert-base-submission-dataset")
print(sub_ds)
prediction = pickle.load(open("submission.pkl", "rb"))
prediction = np.argmax(prediction[0], axis=-1)
print("read prediction")
special_characters = {
    "ä": "a",
    "â": "a",
    "ã": "a",
}

def get_ans_id(id):
    q = tokenizer.convert_ids_to_tokens(sub_ds["q_input_ids"][id])
    r = tokenizer.convert_ids_to_tokens(sub_ds["r_input_ids"][id])
    q_len = len(q) - 1
    r_start = len(q) + 2
    r_len = len(r)
    if q_len > 512:
        q_len = 512

    ans_q = []
    # print("Q: ")
    for i in range(1, q_len):
        if prediction[id][i] == 1:
            # print(q[i], end = " ")
            ans_q.append(i - 1)
    # print()
    
    ans_r = []
    # print("R: ")
    if q_len < 512:
        for i in range(r_start, r_start + r_len):
            if i >= 512:
                break
            if prediction[id][i] == 1:
                # print(r[i - r_start + 1], end=" ")
                ans_r.append(i - r_start)
    return ans_q, ans_r

def get_dict(split_words, token_words):
    # print("split_words: ", split_words)
    # print("===")
    # print("token_words: ", token_words)
    i, j = 0, 0
    ii, jj = 0, 0
    word_to_token = {}
    split_word = split_words[0].lower()
    while i < len(split_words) and j < len(token_words):
        # print(f"{i} {j}: {split_word} {token_words[j]}")
        while ii < len(split_word) and jj < len(token_words[j]):
            if ascii(split_word[ii]) == ascii('\xad'):
                ii += 1
                continue
            # print(f"{ii}, {jj}: {ascii(split_word[ii])}, {split_word[ii]}, {token_words[j][jj]}")
            if token_words[j][jj] == "#" and token_words[j] != "#":
                jj += 1
                continue
            if split_word[ii] in special_characters:
                my_list = list(split_word)
                my_list[ii] = special_characters[split_word[ii]]
                split_word = ''.join(my_list)
            if split_word[ii] == token_words[j][jj]:
                ii += 1
                jj += 1
            else:
                break
        if jj == len(token_words[j]):
            word_to_token[j] = i
            # print(f"{j}: {i},  {token_words[j]} -> {split_words[i]}")
            jj = 0
            j += 1
        if ii == len(split_word):
            ii = 0
            i += 1
            if i < len(split_words):
                split_word = split_words[i].lower()
    return word_to_token

def get_ans(id):
    ans_q, ans_r = get_ans_id(id)
    # for q
    # print("Q: ", sub_ds["q"][id])
    split_words = sub_ds["q"][id].split(" ")
    token_words = tokenizer.convert_ids_to_tokens(sub_ds["q_input_ids"][id])[1:-1]
    q_word_to_token = get_dict(split_words, token_words)
    # print()
    appear_words = []
    info_q = ""
    for word_id in ans_q:
        if split_words[q_word_to_token[word_id]] in appear_words:
            continue
        appear_words.append(split_words[q_word_to_token[word_id]])
        info_q += (split_words[q_word_to_token[word_id]] + " ")
    info_q = info_q[:-1]
    # for word in appear_words:
        # print(word, end=" ")
    # print("\n=========")

    # for r
    # print("R: ", sub_ds["r"][id])
    split_words = sub_ds["r"][id].split(" ")
    token_words = tokenizer.convert_ids_to_tokens(sub_ds["r_input_ids"][id])[1:-1]
    r_word_to_token = get_dict(split_words, token_words)
    # print()
    appear_words = []
    info_r = ""
    for word_id in ans_r:
        if word_id >= len(r_word_to_token):
            break
        if split_words[r_word_to_token[word_id]] in appear_words:
            continue
        appear_words.append(split_words[r_word_to_token[word_id]])
        info_r += (split_words[r_word_to_token[word_id]] + " ")
    info_r = info_r[:-1]
    # for word in appear_words:
    #     print(word, end=" ")
    # print()
    return info_q, info_r


"""First Part - Make answer of submission data"""

# infos_q, infos_r = [], []
# for sentence_id in range(df.shape[0]):
#     print(f"{sentence_id}: ")
#     print(df.loc[sentence_id]["id"])
#     # print()
#     info_q, info_r = get_ans(sentence_id)
#     infos_q.append(info_q)
#     infos_r.append(info_r)
#     # print("\n", sub_ds["s"][sentence_id])
#     # print("--------------------------------------------------------")

# df["ans_q"] = infos_q
# df["ans_r"] = infos_r

# print(df.head(20))

# df.to_csv("submission1.csv", index=False)

"""Second Part - Tidy the csv """

def post_process(x):
    if x == "":
        return "\"\""
    return "\"" + str(x) + "\""


df = pd.read_csv("submission1.csv")
print(df.shape)
print(df.head(3))

df["ans_q"] = df["ans_q"].apply(lambda x: post_process(x))
df["ans_r"] = df["ans_r"].apply(lambda x: post_process(x))

df["q"] = df["ans_q"]
df["r"] = df["ans_r"]
df.drop(["ans_q", "ans_r", "s"], axis=1, inplace=True)
df = df.sort_values(by=["id"])
print(df.shape)
print(df.head(3))
df.to_csv("submission2_comp.csv", index=False)
