import os
import shutil
import random
import datetime
import pandas as pd

import torch
import numpy as np

import datasets
from datasets import load_dataset, load_metric

import speechbrain as sb

import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer


def show_random_elements(dataset, num_examples=5):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    print(df)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    if optimization_metric == "bleu":
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}
    
    elif optimization_metric == "wer":
        for ids in range(len(decoded_preds)):
            metric.append([ids], [decoded_preds[ids]], [decoded_labels[ids][0]])
        result = {"wer": metric.summarize("error_rate")}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def preprocess_function(dataset):
    try:
        inputs = [ex for ex in dataset["src"]]
        targets = [ex for ex in dataset["trg"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
    except:
        try:
            for idx in range(len(inputs)):
                text = inputs[idx]
                target = targets[idx]
                tokenizer(text, max_length=max_input_length, truncation=True)

        except:
            print(text)
            print(target)
            raise Exception("Bad input format.")
        
    return model_inputs

# optimization metric
optimization_metric = "bleu" # bleu | wer

if optimization_metric == "bleu":
    metric = load_metric("sacrebleu")
    greater_is_better = True
elif optimization_metric == "wer":
    metric = sb.utils.metric_stats.ErrorRateStats()
    greater_is_better = False

# Timestamp
date = datetime.datetime.now()
timestamp = str(date).replace(" ", "_").replace("-", "").replace(":", "").split(".")[0]

# Constants
max_input_length = 200
max_target_length = 200
source_lang = "wrong-sp"
target_lang = "sp"

# Training stage
batch_size = 60
epochs = 20
lr = 1e-4
patience = 5

# Checkpoint
#model_checkpoint = "Helsinki-NLP/opus-mt-es-es" # spanish to spanish
model_checkpoint = "Helsinki-NLP/opus-mt-ca-es" # catalán to spanish
#model_checkpoint = "corrector" #local

# Load train data
raw_datasets = load_dataset('csv', data_files={
    'train': 'data/csv/train.csv',
    'valid': 'data/csv/valid.csv',
    'test': 'data/csv/test.csv'})

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

#print("Showing random samples: \n", show_random_elements(raw_datasets["train"]))

print("Tokenizing dataset...")
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

print("Build model")
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

model_name = model_checkpoint.split("/")[-1]

# Scheduler
"""
scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer
)"""

args = Seq2SeqTrainingArguments(
    f"models/{model_name}-finetuned-corrector",
    evaluation_strategy = "epoch",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=epochs,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
    lr_scheduler_type="cosine_with_restarts",
    logging_dir="logs/" + optimization_metric  + '/' + timestamp,
    metric_for_best_model=optimization_metric,
    greater_is_better=greater_is_better,
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="tensorboard"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model) 

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks = [transformers.EarlyStoppingCallback(early_stopping_patience=patience)]
)

trainer.train()

# copy info json
shutil.copyfile("data/dataset.json", "logs/" + optimization_metric  + '/' + timestamp + "/dataset.json")

trainer.save_model("corrector") # save generic
trainer.save_model("logs/" + optimization_metric  + '/' + timestamp + "/model") # save in folder

# Final check
from transformers import pipeline

#translator = pipeline("translation", model="/home/fernandol/transformer-translator-pytorch/opus-mt-es-es-finetuned-wrong-sp-to-sp/checkpoint-4000")
translator = pipeline("translation", model="corrector")
print(translator("PERO YO ANTES ERA DE NA MANERA".lower()))