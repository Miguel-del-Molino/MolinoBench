import json
import random

def read_data(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. Organize into categories
    categories = {"cat_f": [], "cat_P": [], "cat_other": []}

    for item in data:
        entry = item.get("Entry") or ""
        # Store as a dictionary to keep the pair linked during shuffling
        pair = {
            "lt": str(item.get("s_lt", "")).strip(),
            "es": str(item.get("s_es", "")).strip()
        }
        
        if entry.startswith("f."):
            categories["cat_f"].append(pair)
        elif entry.startswith("P."):
            categories["cat_P"].append(pair)
        else:
            categories["cat_other"].append(pair)

    train_list, valid_list, test_list = [], [], []

    # 2. Stratified Split
    for items in categories.values():
        random.shuffle(items)
        n = len(items)
        train_end = int(n * 0.8)
        valid_end = int(n * 0.9)

        train_list.extend(items[0:train_end])
        valid_list.extend(items[train_end:valid_end])
        test_list.extend(items[valid_end:])

    # 3. Final Shuffles
    random.shuffle(train_list)
    random.shuffle(valid_list)
    random.shuffle(test_list)

    # 4. Save into Parallel Files (The NLLB Standard)
    save_parallel(train_list, "train")
    save_parallel(valid_list, "valid")
    save_parallel(test_list, "test")

    print(f"Files created successfully. Train: {len(train_list)}, Valid: {len(valid_list)}, Test: {len(test_list)}")

def save_parallel(data_list, prefix):
    # Save Latin side
    with open(f"{prefix}.lt", 'w', encoding='utf-8') as f_lt:
        # Save Spanish side
        with open(f"{prefix}.es", 'w', encoding='utf-8') as f_es:
            for pair in data_list:
                f_lt.write(pair['lt'] + "\n")
                f_es.write(pair['es'] + "\n")

if __name__ == "__main__":
    read_data('MolinoBench.json')
