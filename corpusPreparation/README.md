# Preparation of corpus: 

This directory contains list of the main scripts used in the process of preparing and converting the corpus.



###  Update the original xlsx file :

The script reads bilingual text from the file correspondant to text of fueros and creates aligned lists of sentences for Latin (lt) and Spanish (es). Each file contains multiple parts (Part1., Part2., etc.), and each part is split into sentences by blank lines.
With the name of the file we identify the name of fueros.

```bash
python  update_xlxs.py

```



###  Convert a XLSX file to JSON:

```bash
python convert_xlsx_to_json.py

```
