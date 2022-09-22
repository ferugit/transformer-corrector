import sentencepiece as spm

import speechbrain as sb

import pandas as pd

from transformers import pipeline


# Metrics
wer_metric = sb.utils.metric_stats.ErrorRateStats()
cer_metric = sb.utils.metric_stats.ErrorRateStats(split_tokens=True)

# data
test_csv_path = "data/csv/test.csv"
test_df = pd.read_csv(test_csv_path, header=0)

asr_hypothesis_list = test_df["src"].tolist()
reference_list = test_df["trg"].tolist()

for ids in range(len(asr_hypothesis_list)):
    wer_metric.append([ids], [asr_hypothesis_list[ids]], [reference_list[ids]])
    cer_metric.append([ids], [asr_hypothesis_list[ids]], [reference_list[ids]])

print("Origin metrics")
cer = round(cer_metric.summarize("error_rate"), 2)
wer = round(wer_metric.summarize("error_rate"), 2)
print(f"CER= {cer}%")
print(f"WER= {wer}%")

# Load corrector
#model_checkpoint = 'corrector' # last checkpoint
model_checkpoint = 'models/opus-mt-ca-es-finetuned-corrector/checkpoint-16686' # last checkpoint
translator = pipeline("translation", model=model_checkpoint)

for ids in range(len(reference_list)):
    corrected_sentence = translator(asr_hypothesis_list[ids])[0]['translation_text']
    wer_metric.append([ids], [corrected_sentence], [reference_list[ids]])
    cer_metric.append([ids], [corrected_sentence], [reference_list[ids]])

print("Corrected metrics")
cer = round(cer_metric.summarize("error_rate"), 2)
wer = round(wer_metric.summarize("error_rate"), 2)
print(f"CER= {cer}%")
print(f"WER= {wer}%")
