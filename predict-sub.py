import pandas
import pickle
from datasets import Dataset, load_from_disk

from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification

train_ds = load_from_disk("data/tc-bert-base-dataset-train")
valid_ds = load_from_disk("data/tc-bert-base-dataset-valid")
sub_ds = load_from_disk("data/tc-bert-base-submission-dataset")

tokenizer = AutoTokenizer.from_pretrained("tc-bert-base")
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
model = AutoModelForTokenClassification.from_pretrained("tc-bert-base", num_labels=3)

training_args = TrainingArguments(
    output_dir="data/results/",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

pred = trainer.predict(sub_ds)
with open("submission.pkl", "wb") as f:
    pickle.dump(pred, f)
