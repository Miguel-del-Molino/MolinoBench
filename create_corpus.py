import pandas as pd
import json

def generate_parallel_corpus_with_log(input_excel, output_jsonl, log_csv,
                                      id_col='orden_lt',
                                      src_col='s_lt',
                                      tgt_col='s_es',
                                      file_col='foro_lt'):
    # Read full Excel
    df = pd.read_excel(input_excel)

    # Identify rows with any missing or empty key columns (source, target, id)
    mask_valid = df[id_col].notna() & df[src_col].notna() & df[tgt_col].notna() & df[file_col].notna()
    mask_nonempty = df[src_col].str.strip().astype(bool) & df[tgt_col].str.strip().astype(bool)
    mask = mask_valid & mask_nonempty

    # Rows to keep and rows to log as incomplete
    df_keep = df[mask]
    df_log = df[~mask]

    # Write filtered parallel corpus
    with open(output_jsonl, 'w', encoding='utf-8') as fw:
        for _, row in df_keep.iterrows():
            example = {
                "id": int(row[id_col]),
                "source": row[src_col].strip(),
                "target": row[tgt_col].strip(),
                "file": row[file_col].strip()
            }
            json.dump(example, fw, ensure_ascii=False)
            fw.write("\n")

    # Save log of removed rows separately, for inspection/debugging
    if not df_log.empty:
        df_log.to_csv(log_csv, index=False, encoding='utf-8')
        print(f"[LOG] Saved {len(df_log)} removed/incomplete rows to {log_csv}")
        print("\n[LOG] Here are the removed rows:")
        print(df_log[[id_col, src_col, tgt_col, file_col]])
    else:
        print("[LOG] No rows removed; no log file created.")

    print(f"[DONE] Parallel corpus saved to {output_jsonl}")

# Example usage:
generate_parallel_corpus_with_log(
    input_excel='corpusParalelo_aligned.xlsx',
    output_jsonl='parallel_corpus.jsonl',
    log_csv='removed_rows_log.csv'
)

