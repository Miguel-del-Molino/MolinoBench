# MolinoBench


MolinoBench is a parallel corpus prepared for medieval Latin to Spanish translation. 

The corpus is provided in its original aligned form as a single JSON file (MolinoBench.json), which contains the full dataset before any division. This file preserves the complete sentence-level parallel structure and serves as the reference version of the corpus.

For machine learning usage, the corpus is additionally released in plain text format and split into three standard subsets:

- Training set: train.lt, train.es

- Validation set: valid.lt, valid.es

- Testing set: test.lt, test.es

Each pair of files represents a sentence-aligned parallel corpus. All text files are also regrouped in a compressed archive (data.zip) to facilitate download and reuse.

As part of the validation of the corpus, we additionally release pretrained neural machine translation models based on mBART and NLLB, which were fine-tuned using the MolinoBench training data. The resulting model predictions and evaluation outputs are provided in a dedicated experiments folder. These resources allow users to reproduce the reported experiments, assess the quality of the corpus, and use the provided models as baselines for further research.

MolinoBench is released as part of the Miguel del Molino project (https://migueldelmolino.es/)
