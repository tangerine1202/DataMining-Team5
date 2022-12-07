import pickle
import string
import numpy as np

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

a = pickle.load(open("predictions.pkl", "rb"))
# y_pred = np.array(a[1])
y_pred = np.argmax(a[0], axis=-1)
test_ds = load_from_disk("data/tc_valid_dataset")
q_input_ids = test_ds["q_input_ids"]
r_input_ids = test_ds["r_input_ids"]
tokenizer = AutoTokenizer.from_pretrained("tc-distilbert")

def calculate_acc(y_pred, y_true):
    correct_list = []
    for i in range(y_pred.shape[0]):
        l = len(y_true[i])
        correct = 0
        for (pred, label) in zip(y_pred[i][:l], y_true[i]):
            if pred == label:
                correct += 1
        # print("=======")
        # print(i)
        # print(correct)
        # print(correct / l)
        correct_list.append(correct / l)

    print(sum(correct_list) / y_pred.shape[0])

def compute_lcs(X, Y):
    # find the length of the strings
    # X_token = word_tokenize(X)
    # Y_token = word_tokenize(Y)
    X_token = X
    Y_token = Y

    X = [x for x in X_token if x not in string.punctuation]	
    Y = [x for x in Y_token if x not in string.punctuation]	

    m = len(X)
    n = len(Y)

    total_length = m + n

    if total_length == 0:
        return -1
 
    # declaring the array for storing the dp values
    L = [[None]*(n + 1) for i in range(m + 1)]
 
    """Following steps build L[m + 1][n + 1] in bottom up fashion
    Note: L[i][j] contains length of LCS of X[0..i-1]
    and Y[0..j-1]"""
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0 :
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1]+1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
 
    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    LSC = L[m][n]
    score = (LSC/(total_length - LSC))
    return score

def calculate_lcs(y_pred, input_ids, Q_ans, R_ans, S):
    data_num = y_pred.shape[0]
    lcs_avg_sum = 0
    count = 0
    for entry_id in range(y_pred.shape[0]):
        sentence = tokenizer.convert_ids_to_tokens(input_ids[entry_id])
        s = S[entry_id].lower()

        # get Q and R from prediction
        r_start = 0
        for i, word in enumerate(sentence):
            if word == s:
                r_start = i + 2
                break
        if r_start != 0:
            Q_s, R_s = sentence[1:r_start - 2], sentence[r_start:-1]
            Q_id, R_id = y_pred[entry_id][1:r_start - 2], y_pred[entry_id][r_start:len(sentence)-1]
        else:
            Q_s, R_s = sentence[1:-1], []
            Q_id, R_id = y_pred[entry_id][1:-1], []
        q_pred = []
        for i in range(1, r_start - 2):
            if y_pred[entry_id][i] == 1:
                q_pred.append(Q_s[i - 1])

        r_pred = []
        if (r_start != 0):
            for i in range(r_start, len(sentence) - 1):
                if y_pred[entry_id][i] == 1:
                    r_pred.append(R_s[i - r_start])
        
        # get Q and R from ans
        q_true = tokenizer.convert_ids_to_tokens(Q_ans[entry_id][1:-1])
        r_true = tokenizer.convert_ids_to_tokens(R_ans[entry_id][1:-1])
        # print("q_pred: ", q_pred)
        # print("r_pred: ", r_pred)
        # print("q_true: ", q_true)
        # print("r_true: ", r_true)
        # print()
        # print("Q: ", compute_lcs(q_pred, q_true))
        # print("R: ", compute_lcs(r_pred, r_true))
        # print("=========")
        q_score = compute_lcs(q_pred, q_true)
        if q_score > 0:
            lcs_avg_sum += q_score
            count += 1

        if r_pred != []:
            r_score = compute_lcs(r_pred, r_true)
            if r_score > 0:
                lcs_avg_sum += r_score
            count += 1
    return lcs_avg_sum / count

avg_lcs = calculate_lcs(y_pred, test_ds["input_ids"], test_ds["q'_input_ids"], test_ds["r'_input_ids"], test_ds["s"])
print(avg_lcs)


