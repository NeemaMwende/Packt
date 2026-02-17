# import numpy as np
# import evaluate
# from datasets import load_dataset
# from transformers import (
#     AutoTokenizer,
#     AutoModelForSequenceClassification,
#     Trainer,
#     TrainingArguments,
# )

# # 1️⃣ Load Dataset
# dataset = load_dataset("imdb")

# # Optional: use small shard for faster testing
# train_data = dataset["train"].shard(num_shards=4, index=0)
# test_data = dataset["test"].shard(num_shards=4, index=0)

# # 2️⃣ Load Model + Tokenizer
# model_name = "bert-base-uncased"

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(
#     model_name,
#     num_labels=2
# )

# # 3️⃣ Tokenize Function
# def tokenize_function(examples):
#     return tokenizer(
#         examples["text"],
#         padding="max_length",
#         truncation=True,
#         max_length=64
#     )

# # 4️⃣ Apply Tokenization
# tokenized_train = train_data.map(tokenize_function, batched=True)
# tokenized_test = test_data.map(tokenize_function, batched=True)

# # 5️⃣ Set Format for PyTorch
# tokenized_train.set_format(
#     type="torch",
#     columns=["input_ids", "attention_mask", "label"]
# )

# tokenized_test.set_format(
#     type="torch",
#     columns=["input_ids", "attention_mask", "label"]
# )

# # 6️⃣ Load Metric
# accuracy = evaluate.load("accuracy")

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return accuracy.compute(predictions=predictions, references=labels)

# # 7️⃣ Training Arguments
# training_args = TrainingArguments(
#     output_dir="./results",
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     logging_dir="./logs",
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=1,
#     weight_decay=0.01,
#     logging_steps=50,
#     load_best_model_at_end=True
# )

# # 8️⃣ Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_train,
#     eval_dataset=tokenized_test,
#     compute_metrics=compute_metrics,
# )

# # 9️⃣ Train
# trainer.train()

# # 🔟 Evaluate
# results = trainer.evaluate()
# print(results)


from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

train_data = load_dataset("imdb", split="train")
train_data = train_data.shard(num_shards=4, index=0)

test_data = load_dataset("imdb", split="test")
test_data = test_data.shard(num_shards=4, index=0)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# ✅ tokenize function added
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=64
    )

# ✅ use map instead of manual tokenizer call
tokenized_training_data = train_data.map(tokenize_function, batched=True)
tokenized_test_data = test_data.map(tokenize_function, batched=True)

print(tokenized_training_data) 
