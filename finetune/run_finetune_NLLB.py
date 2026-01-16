import os
import torch
from datasets import Dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load metrics globally for stability
metric_rouge = load_metric("rouge")
metric_bleu = load_metric("sacrebleu")
metric_chrf = load_metric("chrf")

# -----------------------------
# 1. Función para cargar archivos de texto plano
# -----------------------------
def load_parallel_files_as_dataset(src_path, tgt_path):
    print(f"Cargando {src_path} y {tgt_path}...")
    with open(src_path, "r", encoding="utf-8") as f_src:
        src_lines = [line.strip() for line in f_src]
    with open(tgt_path, "r", encoding="utf-8") as f_tgt:
        tgt_lines = [line.strip() for line in f_tgt]
    return Dataset.from_dict({"source": src_lines, "target": tgt_lines})

# -----------------------------
# 2. Preprocesamiento (No changes made)
# -----------------------------
def preprocess_dataset(dataset, tokenizer, src_lang, tgt_lang, max_length=256):
    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["source"],
            max_length=max_length,
            padding="max_length",
            truncation=True
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["target"],
                max_length=max_length,
                padding="max_length",
                truncation=True
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(preprocess_function, batched=True)

# -----------------------------
# 3. Métricas (ROUGE + SacreBLEU + chrF)
# -----------------------------
def compute_metrics(eval_pred):
    preds, labels = eval_pred

    if isinstance(preds, tuple):
        preds = preds[0]

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = [
        [token if token != -100 else tokenizer.pad_token_id for token in label]
        for label in labels
    ]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Format for SacreBLEU and chrF (list of lists for references)
    bleu_refs = [[ref] for ref in decoded_labels]

    # Compute ROUGE
    rouge_results = metric_rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    
    # Compute SacreBLEU
    bleu_results = metric_bleu.compute(
        predictions=decoded_preds, 
        references=bleu_refs
    )

    # Compute chrF
    chrf_results = metric_chrf.compute(
        predictions=decoded_preds, 
        references=bleu_refs
    )

    # Extract results
    return {
        "bleu": round(bleu_results["score"], 4),
        "chrf": round(chrf_results["score"], 4),
        "rouge1": round(rouge_results["rouge1"].mid.fmeasure * 100, 4),
        "rouge2": round(rouge_results["rouge2"].mid.fmeasure * 100, 4),
        "rougeL": round(rouge_results["rougeL"].mid.fmeasure * 100, 4),
    }

# -----------------------------
# MAIN
# -----------------------------
def main():
    model_name = "facebook/nllb-200-distilled-600M"
    src_lang = "lat_Latn"
    tgt_lang = "spa_Latn"

    # A. Cargar archivos específicos
    train_dataset = load_parallel_files_as_dataset("train.lt", "train.es")
    eval_dataset = load_parallel_files_as_dataset("valid.lt", "valid.es")
    test_dataset = load_parallel_files_as_dataset("test.lt", "test.es")

    print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)} | Test: {len(test_dataset)}")

    # B. Tokenizador y Modelo
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # C. Procesar Datasets
    train_ds = preprocess_dataset(train_dataset, tokenizer, src_lang, tgt_lang)
    eval_ds = preprocess_dataset(eval_dataset, tokenizer, src_lang, tgt_lang)
    test_ds = preprocess_dataset(test_dataset, tokenizer, src_lang, tgt_lang)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # D. Argumentos
    training_args = Seq2SeqTrainingArguments(
        output_dir="./nllb_lat_es_results",
        evaluation_strategy="epoch",  
        save_strategy="epoch",        
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        weight_decay=0.01,
        warmup_steps=500,
        logging_dir="./logs",
        logging_steps=50,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # E. Entrenar
    print("\n--- Iniciando Fine-tuning con NLLB ---")
    trainer.train()

    # F. Predicción sobre el Test Set y Guardado de Resultados
    print("\n--- Generando predicciones finales sobre test.lt ---")
    test_results = trainer.predict(test_ds)
    
    # Decodificar predicciones
    predictions = tokenizer.batch_decode(test_results.predictions, skip_special_tokens=True)
    
    # Extraer métricas de tiempo
    m = test_results.metrics
    runtime = m.get("test_runtime", 0)
    samples_per_sec = m.get("test_samples_per_second", 0)
    
    output_file = "test_results_nllb_finetune.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"TEST RUNTIME: {runtime:.2f}s | THROUGHPUT: {samples_per_sec:.2f} samples/s\n")
        f.write("-" * 80 + "\n")
        f.write("LATIN SOURCE\t| NLLB PREDICTION\t| GROUND TRUTH (test.es)\n")
        for i in range(len(predictions)):
            src = test_dataset[i]["source"]
            ref = test_dataset[i]["target"]
            pred = predictions[i]
            f.write(f"{src}\t| {pred}\t| {ref}\n")

    print(f"Predicciones guardadas en: {output_file}")
    print(f"Métricas finales en test: {test_results.metrics}")

    # G. Guardar modelo final
    trainer.save_model("nllb-latin-spanish-trained")
    tokenizer.save_pretrained("nllb-latin-spanish-trained")

if __name__ == "__main__":
    main()
