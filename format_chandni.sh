mkdir -p bert_input_data
sed 's/\([^,]*\),\(.*\)/\2/' data/lang_id_eval.csv > bert_input_data/eval.csv
sed 1d bert_input_data/eval.csv > bert_input_data/eval.txt
sed 's/\([^,]*\),\(.*\)/\2/' data/lang_id_train.csv > bert_input_data/train.csv
sed 1d bert_input_data/train.csv > bert_input_data/train.txt
sed 's/\([^,]*\),\(.*\)/\2/' data/lang_id_test.csv > bert_input_data/test.csv
sed 1d bert_input_data/test.csv > bert_input_data/test.txt
