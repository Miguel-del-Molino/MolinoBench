import torch
import sacrebleu
import time
import warnings
from rouge_score import rouge_scorer
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Ocultar advertencias de deprecación para mantener logs limpios
warnings.filterwarnings("ignore", category=UserWarning)

def translate_original(model_name, src_sentences, reference_sentences, src_lang="la_XX", tgt_lang="es_XX"):
    # 1. Cargar modelo y tokenizador originales
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    
    # Configurar idiomas en el tokenizador (ESTO CORRIGE EL ERROR)
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    translations = []
    total_loss = 0.0

    print(f"--- Iniciando traducción de {len(src_sentences)} frases con mBART original ---")
    
    model.eval()
    start_time = time.time()
    
    # ID del idioma para la generación
    forced_bos_id = tokenizer.lang_code_to_id[tgt_lang]
    
    with torch.no_grad():
        for i, (text, ref) in enumerate(zip(src_sentences, reference_sentences)):
            # Preparar entrada
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
            
            # Preparar etiquetas para LOSS
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(ref, return_tensors="pt", truncation=True, max_length=128).to(device)
            
            # Calcular Loss
            outputs = model(**inputs, labels=labels["input_ids"])
            total_loss += outputs.loss.item()

            # Generar traducción
            generated_tokens = model.generate(
                **inputs, 
                forced_bos_token_id=forced_bos_id
            )
            
            result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            translations.append(result)
            
            if (i + 1) % 50 == 0:
                print(f"Procesadas {i + 1}/{len(src_sentences)} frases...")
    
    total_runtime = time.time() - start_time
    avg_loss = total_loss / len(src_sentences)
                
    return translations, total_runtime, avg_loss

def main():
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    
    try:
        with open("test.lt", "r", encoding="utf-8") as f:
            latin_sentences = [line.strip() for line in f]
        with open("test.es", "r", encoding="utf-8") as f:
            reference_sentences = [line.strip() for line in f]
    except FileNotFoundError as e:
        print(f"Error: No se encontró el archivo {e.filename}")
        return

    # Ejecución
    original_preds, runtime, test_loss = translate_original(model_name, latin_sentences, reference_sentences)
    samples_per_sec = len(latin_sentences) / runtime

    # Guardar resultados en archivo
    with open("baseline_model_results_mBart50.txt", "w", encoding="utf-8") as f:
        f.write(f"LOSS: {test_loss:.4f} | RUNTIME: {runtime:.2f}s | THROUGHPUT: {samples_per_sec:.2f} samples/s\n")
        f.write("-" * 80 + "\n")
        f.write("LATIN SOURCE | ORIGINAL mBART PREDICTION | GROUND TRUTH\n")
        for src, pred, ref in zip(latin_sentences, original_preds, reference_sentences):
            f.write(f"{src} | {pred} | {ref}\n")

    # Métricas
    print("\n--- Calculando métricas de rendimiento ---")
    bleu = sacrebleu.corpus_bleu(original_preds, [reference_sentences])
    chrf = sacrebleu.corpus_chrf(original_preds, [reference_sentences])
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_scores = [scorer.score(r, p)['rougeL'].fmeasure for p, r in zip(original_preds, reference_sentences)]
    avg_rouge_l = (sum(rouge_l_scores) / len(rouge_l_scores)) * 100

    # Resumen Final
    print("\n" + "="*40)
    print("RESUMEN DE MÉTRICAS (MODELO ORIGINAL)")
    print("="*40)
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"BLEU Score:    {bleu.score:.2f}")
    print(f"chrF Score:    {chrf.score:.2f}")
    print(f"ROUGE-L Score: {avg_rouge_l:.2f}")
    print(f"Runtime:       {runtime:.2f} seconds")
    print(f"Throughput:    {samples_per_sec:.2f} sentences/sec")
    print("="*40)

if __name__ == "__main__":
    main()
