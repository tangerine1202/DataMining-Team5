import pandas
from datasets import Dataset, load_from_disk

from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification


train_ds = load_from_disk("data/tc-bert-base-dataset-train")
valid_ds = load_from_disk("data/tc-bert-base-dataset-valid")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=3)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    total = predictions.shape[0]
    case_correct = 0
    for i in range(total):
        word_correct = 0
        for j in range(512):
            if labels[i][j] == -100 and predictions[i][j] == 0:
                word_correct += 1
            if predictions[i][j] == labels[i][j]:
                word_correct += 1
        case_correct += word_correct / 512
    return {'accuracy': case_correct / total}

training_args = TrainingArguments(
    output_dir="data/results/",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model("data/models/tc-bert-base")

pred = trainer.predict(train_ds)
print(pred.metrics)
pred = trainer.predict(valid_ds)
print(pred.metrics)
with open("predictions.pkl", "wb") as f:
    pickle.dump(pred, f)