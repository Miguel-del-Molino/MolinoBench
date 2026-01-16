import os
import json
import evaluate
import torch
from datasets import Dataset, load_metric
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# CARGA GLOBAL DE MÉTRICAS PARA EVITAR ERRORES DE TOKEN
metric_bleu = load_metric("sacrebleu")
metric_rouge = load_metric("rouge")
metric_chrf = load_metric("chrf") # Añadida chrF

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # SacreBLEU y chrF esperan una lista de listas para las referencias
    bleu_refs = [[ref] for ref in decoded_labels]
    
    bleu_res = metric_bleu.compute(predictions=decoded_preds, references=bleu_refs)
    rouge_res = metric_rouge.compute(predictions=decoded_preds, references=decoded_labels)
    chrf_res = metric_chrf.compute(predictions=decoded_preds, references=bleu_refs) # Cálculo chrF

    return {
        "bleu": bleu_res["score"],
        "rougeL": rouge_res["rougeL"].mid.fmeasure * 100,
        "chrf": chrf_res["score"] # Retorno chrF
    }

def load_parallel_files_as_dataset(src_path, tgt_path):
    with open(src_path, "r", encoding="utf-8") as f_src:
        src_lines = [line.strip() for line in f_src]
    with open(tgt_path, "r", encoding="utf-8") as f_tgt:
        tgt_lines = [line.strip() for line in f_tgt]
    return Dataset.from_dict({"source": src_lines, "target": tgt_lines})

def preprocess_dataset(dataset, tokenizer, src_lang, tgt_lang, max_length=128):
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    def preprocess_function(examples):
        inputs = examples["source"]
        targets = examples["target"]
        model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)
        labels = tokenizer(text_target=targets, max_length=max_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return dataset.map(preprocess_function, batched=True)

def main():
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    src_lang, tgt_lang = "la_XX", "es_XX"

    # 1. Load Datasets
    train_dataset = load_parallel_files_as_dataset("train.lt", "train.es")
    eval_dataset = load_parallel_files_as_dataset("valid.lt", "valid.es")
    test_dataset = load_parallel_files_as_dataset("test.lt", "test.es")

    # 2. Load Model & Tokenizer
    global tokenizer
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)

    # 3. Preprocess
    train_ds = preprocess_dataset(train_dataset, tokenizer, src_lang, tgt_lang)
    eval_ds = preprocess_dataset(eval_dataset, tokenizer, src_lang, tgt_lang)
    test_ds = preprocess_dataset(test_dataset, tokenizer, src_lang, tgt_lang)

    # 4. Training Args
    training_args = Seq2SeqTrainingArguments(
        output_dir="./mbart_latin_spanish",
        eval_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=5,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(), # Auto-enable GPU acceleration
        logging_steps=10,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=compute_metrics,
    )

    # 5. Train
    print("--- Starting Training ---")
    trainer.train()

    # 6. Test / Evaluation + RUNTIME
    print("--- Starting Testing on Test Set ---")
    test_results = trainer.predict(test_ds)
    
    # Extraer métricas de tiempo del objeto test_results
    metrics = test_results.metrics
    runtime = metrics.get("test_runtime", 0)
    samples_per_sec = metrics.get("test_samples_per_second", 0)
    
    print(f"Test Metrics: {metrics}")
    print(f"Runtime: {runtime:.2f} seconds")
    print(f"Throughput: {samples_per_sec:.2f} samples/s")

    # 7. Save Predictions for Manual Review
    predictions = tokenizer.batch_decode(test_results.predictions, skip_special_tokens=True)
    
    with open("test_results_mbart_finetune.txt", "w", encoding="utf-8") as f:
        # Añadimos el tiempo al inicio del archivo
        f.write(f"TEST RUNTIME: {runtime:.2f} seconds | THROUGHPUT: {samples_per_sec:.2f} samples/s\n")
        f.write("-" * 50 + "\n")
        f.write("LATIN SOURCE\t| MODEL PREDICTION\t| GROUND TRUTH\n")
        for i in range(len(predictions)):
            src = test_dataset[i]["source"]
            ref = test_dataset[i]["target"]
            pred = predictions[i]
            f.write(f"{src}\t| {pred}\t| {ref}\n")

    print("Predictions saved to test_results_mbart_finetune.txt")

if __name__ == "__main__":
    main()
