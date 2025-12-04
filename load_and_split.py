import json
from sklearn.model_selection import train_test_split

def load_and_split_jsonl(jsonl_file, test_ratio=0.1):
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            data.append(entry)

    train_data, test_data = train_test_split(data, test_size=test_ratio, random_state=42)
    print(f"Total samples: {len(data)}, Train: {len(train_data)}, Test: {len(test_data)}")
    return train_data, test_data

if __name__ == "__main__":
    train, test = load_and_split_jsonl('parallel_corpus.jsonl')
    # Optionally save train/test splits to separate jsonl files or pass downstream
