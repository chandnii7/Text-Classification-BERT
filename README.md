# Using pre-trained BERT vectors for text classification

Program performs text classification of following 10 classes using BERT vector:
1. Arabic
2. Cantonese
3. Japanese
4. Korean
5. Mandarin
6. Polish
7. Russian
8. Spanish
9. Thai
10. Vietnamese

Program was implemented using Python and BERT. Refer the report for further implementation details, and instructions to run the code:
<a href="https://github.com/chandnii7/UsingBERT/blob/main/doc/bert_report_chandni.pdf">View Report</a>
<br/><br/>

### Results:
1. Logistic Regression Model Predictions: Among all languages, highest precision, recall, and f1-score is for Thai, whereas lowest is for Mandarin. Misclassification is highest for Mandarin and Cantonese, whereas lowest for Thai.
<img src="https://github.com/chandnii7/UsingBERT/blob/main/data/img_lr.jpg" height="200" width="350"/>
<br/>

2. Neural Network Model Predictions: Using MLP Classifier, highest precision, recall, and f1-score is for Thai, whereas lowest is for Mandarin. Misclassification is highest for Mandarin, whereas lowest for Thai.
<img src="https://github.com/chandnii7/UsingBERT/blob/main/data/img_nn.jpg" height="200" width="350"/>
<br/>
