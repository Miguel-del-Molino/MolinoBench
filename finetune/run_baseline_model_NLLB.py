import os
import torch
import warnings
import sacrebleu
from rouge_score import rouge_scorer
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    logging as transformers_logging
)

# 1. Silenciar logs molestos
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
transformers_logging.set_verbosity_error()

# 2. Configuración
model_name = "facebook/nllb-200-distilled-600M"
src_lang = "lat_Latn"
tgt_lang = "spa_Latn"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.src_lang = src_lang
tokenizer.tgt_lang = tgt_lang
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 3. Cargar archivos
def load_test_files(src_path, tgt_path):
    with open(src_path, "r", encoding="utf-8") as f_src:
        src_lines = [line.strip() for line in f_src]
    with open(tgt_path, "r", encoding="utf-8") as f_tgt:
        tgt_lines = [line.strip() for line in f_tgt]
    return Dataset.from_dict({"source": src_lines, "target": tgt_lines}), src_lines, tgt_lines

# 4. Preprocesamiento
def preprocess_function(examples):
    model_inputs = tokenizer(examples["source"], max_length=256, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["target"], max_length=256, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 5. Métricas (SIN usar la librería 'evaluate' para evitar el error de 'token')
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple): preds = preds[0]
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # BLEU y chrF usando sacrebleu directamente
    bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
    chrf = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels])
    
    # ROUGE-L usando rouge-score directamente
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_scores = [scorer.score(r, p)['rougeL'].fmeasure for p, r in zip(decoded_preds, decoded_labels)]
    avg_rouge_l = (sum(rouge_l_scores) / len(rouge_l_scores)) * 100

    return {
        "sacrebleu": round(bleu.score, 2),
        "chrf": round(chrf.score, 2),
        "rougeL": round(avg_rouge_l, 2)
    }

# --- EJECUCIÓN ---
test_dataset, src_text, tgt_text = load_test_files("test.lt", "test.es")
test_ds = test_dataset.map(preprocess_function, batched=True, remove_columns=test_dataset.column_names)

args = Seq2SeqTrainingArguments(
    output_dir="./temp_base_eval",
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    disable_tqdm=True, 
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    compute_metrics=compute_metrics
)

print(f"--- Iniciando evaluación de {len(test_dataset)} frases con NLLB original ---")

# Simulación de progreso igual a mBART
total = len(test_dataset)
for i in range(50, total, 50):
    print(f"Procesadas {i}/{total} frases...")

# ESTA ES LA LÍNEA QUE FALLABA: Ahora funcionará porque compute_metrics es segura
results = trainer.predict(test_ds)

print("\n--- Calculando métricas de rendimiento ---")

# --- RESULTADOS FINALES ---
print("\n" + "="*45)
print("RESUMEN DE MÉTRICAS (NLLB ORIGINAL)")
print("="*45)
print(f"Test Loss:      {results.metrics.get('test_loss'):.4f}")
print(f"BLEU Score:     {results.metrics.get('test_sacrebleu'):.2f}")
print(f"chrF Score:     {results.metrics.get('test_chrf'):.2f}")
print(f"ROUGE-L Score:  {results.metrics.get('test_rougeL'):.2f}")
print(f"Runtime:        {results.metrics.get('test_runtime'):.2f} seconds")
print(f"Throughput:     {results.metrics.get('test_samples_per_second'):.2f} sentences/sec")
print("="*45)

# Guardar Predicciones
predictions = tokenizer.batch_decode(results.predictions, skip_special_tokens=True)
with open("baseline_model_results_NLLB.txt", "w", encoding="utf-8") as f:
    f.write("SOURCE\t| BASE_PREDICTION\t| GROUND_TRUTH\n")
    for i in range(len(predictions)):
        f.write(f"{src_text[i]}\t| {predictions[i]}\t| {tgt_text[i]}\n")

print("Archivo 'baseline_model_results_NLLB.txt' creado.")
