# Translare Legem 


To update the xml file with the splitted text to parts, you can run the following command : 


´

python update_xlxs.py

´


python create_corpus.py : to create the corpus for training , the name of the output is "parallel_corpus.jsonl"


When I will train the model I can use :

python load_and_split.py to load the train , test data , with splitting test_ratio =0.1

