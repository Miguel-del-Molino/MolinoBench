import pandas as pd
import re
import os

# -----------------------------
# Helper: Parse "Part" sections from a text block
# -----------------------------
def parse_parts_sentences(block):
    """
    Parse a text block into parts (Part1., Part2., etc.)
    Each part is split into sentences.
    Sentences are separated by at least one blank line,
    possibly containing spaces or tabs.
    """
    parts_found = re.findall(r'(Part\d+\.)', block)
    parts_split = re.split(r'Part\d+\.', block)
    part_dict = {}

    for i, part_tag in enumerate(parts_found):
        part_num = int(re.findall(r'\d+', part_tag)[0])
        text_part = parts_split[i + 1]

        # Split sentences by at least one empty line (possibly with spaces)
        sentences = [s.strip() for s in re.split(r'\n\s*\n', text_part.strip()) if s.strip()]
        part_dict[part_num] = sentences

    return part_dict


# -----------------------------
# Helper: Extract Lithuanian + Spanish parts from text file
# -----------------------------
def extract_parallel_parts(txt_path):
    """
    Extract Lithuanian (lt) and Spanish (es) blocks from a text file.
    Each text file is expected to contain:
      f.<n>.
      <Lithuanian parts>
      f.<n+1>.
      <Spanish parts>
    """
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split blocks by f.<number>.
    blocks = re.split(r'\nf\.\d+\.\n', '\n' + content)

    lt_block = blocks[1] if len(blocks) > 1 else ''
    es_block = blocks[2] if len(blocks) > 2 else ''

    lt_parts = parse_parts_sentences(lt_block)
    es_parts = parse_parts_sentences(es_block)

    return lt_parts, es_parts


# -----------------------------
# Main function: Expand corpus by parts (preserving Excel order)
# -----------------------------
def expand_by_parts_preserve_order(input_xlsx, txt_dir, output_xlsx):
    """
    Reads an Excel file with columns 'orden_lt' and 'foro_lt',
    extracts bilingual text parts from corresponding .txt files,
    and outputs an expanded sentence-aligned Excel.

    ✅ Keeps the same row order as in the input Excel.
    ✅ Outputs one row per sentence without 'sentence_idx_in_part'.
    """
    df = pd.read_excel(input_xlsx)

    # Validate required columns
    if 'orden_lt' not in df.columns:
        raise ValueError("❌ Missing required column 'orden_lt' in Excel file.")
    if 'foro_lt' not in df.columns:
        raise ValueError("❌ Missing required column 'foro_lt' in Excel file.")

    results = []

    # Group by foro_lt (text file identifier)
    grouped = df.groupby('foro_lt', sort=False)  # keep original order

    for foro_lt, group_df in grouped:
        clean_forolt = foro_lt.lstrip('f.').rstrip('.')
        txt_file = os.path.join(txt_dir, f'f.{clean_forolt}.txt')

        if not os.path.exists(txt_file):
            print(f'[ERROR] Missing {txt_file} for foro_lt {foro_lt}, skipping group')
            continue

        lt_parts, es_parts = extract_parallel_parts(txt_file)

        # Each Excel row corresponds to one PartX (based on its position)
        for idx_in_group, (_, row) in enumerate(group_df.iterrows()):
            part_num = idx_in_group + 1  # 1-based index for Part number

            part_lt_sents = lt_parts.get(part_num, [])
            part_es_sents = es_parts.get(part_num, [])

            max_sents = max(len(part_lt_sents), len(part_es_sents))

            for j in range(max_sents):
                r = row.to_dict()
                r['s_lt'] = part_lt_sents[j] if j < len(part_lt_sents) else ''
                r['s_es'] = part_es_sents[j] if j < len(part_es_sents) else ''
                results.append(r)

    # Build final DataFrame and save
    df_out = pd.DataFrame(results)
    df_out.to_excel(output_xlsx, index=False)
    print(f'[DONE] Saved expanded aligned corpus to {output_xlsx}')
    return df_out


# -----------------------------
# Example usage
# -----------------------------
expand_by_parts_preserve_order('corpusParalelo.xlsx', 'files', 'corpusParalelo_aligned.xlsx')

