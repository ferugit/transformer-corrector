import torch
import numpy as np

import random
import pandas as pd

import datasets
from datasets import load_dataset, load_metric

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

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def preprocess_function(dataset):
    inputs = [ex for ex in dataset["src"]]
    targets = [ex for ex in dataset["trg"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs



# Constants
max_input_length = 200
max_target_length = 200
source_lang = "wrong-sp"
target_lang = "sp"

# Training stage
batch_size = 60
epochs = 200
lr = 2e-5

# CHeckpoint
#model_checkpoint = "Helsinki-NLP/opus-mt-es-es" #remote

model_checkpoint = "Helsinki-NLP/opus-mt-ca-es"
#model_checkpoint = "corrector" #local

# Load train data
raw_datasets = load_dataset('csv', data_files={
    'train': '/home/fernandol/transformer-translator-pytorch/data/tsv/train.csv',
    'valid': '/home/fernandol/transformer-translator-pytorch/data/tsv/valid.csv',
    'test': '/home/fernandol/transformer-translator-pytorch/data/tsv/test.csv'})

# Metric
metric = load_metric("sacrebleu")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

#print("Showing random samples: \n", show_random_elements(raw_datasets["train"]))

print("Tokenizing dataset...")
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

print("Build model")
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

model_name = model_checkpoint.split("/")[-1]

args = Seq2SeqTrainingArguments(
    f"models/{model_name}-finetuned-corrector",
    evaluation_strategy = "epoch",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=epochs,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model) 

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model("corrector")


# Final check
from transformers import pipeline

#translator = pipeline("translation", model="/home/fernandol/transformer-translator-pytorch/opus-mt-es-es-finetuned-wrong-sp-to-sp/checkpoint-4000")
translator = pipeline("translation", model="corrector")
print(translator("PERO YO ANTES ERA DE NA MANERA"))