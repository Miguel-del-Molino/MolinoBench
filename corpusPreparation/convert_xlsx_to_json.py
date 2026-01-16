import pandas as pd

# ---- CONFIG ----
input_file = "MolinoBench.xlsx"     # Path to your original Excel file
output_file = "MolinoBench.json"   # Path for the output JSON file
sheet_name = 0                # Sheet index or sheet name
# ----------------

# Read Excel file, preserving all text
df = pd.read_excel(input_file, sheet_name=sheet_name, engine="openpyxl")

# Select the columns we want
df_corpus = df[[
    "Source",
    "Entry",
    "order_lt",
    "s_lt",
    "s_es"
]].copy()

# Create a sequential Id column starting from 1
df_corpus.insert(2, "Id", range(1, len(df_corpus) + 1))

# Save as JSON, preserving all text exactly
df_corpus.to_json(output_file, orient="records", force_ascii=False, indent=4)

print("JSON corpus successfully created!")
