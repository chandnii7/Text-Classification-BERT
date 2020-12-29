
# coding: utf-8

# # Text categorization model using the features derived from BERT

# In[ ]:


import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report


# In[2]:


ORIGINAL_DATA_DIR = os.path.join("data")
BERT_FEATURE_DIR = "bert_output_data"


# In[3]:


train_df = pd.read_csv(os.path.join(ORIGINAL_DATA_DIR, "lang_id_train.csv"))
print(train_df.shape)

bert_vectors_train = []
with open(os.path.join(BERT_FEATURE_DIR, "train.jsonlines"), "rt") as infile:
    for line in infile:
        bert_data = json.loads(line)
        for t in bert_data["features"]:
            # Only extract the [CLS] vector used for classification
            if t["token"] == "[CLS]":
                # We only use the representation at the final layer of the network
                bert_vectors_train.append(t["layers"][0]["values"])
                break
print(len(bert_vectors_train))

X_train = np.array(bert_vectors_train)
y_train = train_df["native_language"].values


# In[4]:


eval_df = pd.read_csv(os.path.join(ORIGINAL_DATA_DIR, "lang_id_eval.csv"))
print(eval_df.shape)

bert_vectors_eval = []
with open(os.path.join(BERT_FEATURE_DIR, "eval.jsonlines"), "rt") as infile:
    for line in infile:
        bert_data = json.loads(line)
        for t in bert_data["features"]:
            # Only extract the [CLS] vector used for classification
            if t["token"] == "[CLS]":
                # We only use the representation at the final layer of the network
                bert_vectors_eval.append(t["layers"][0]["values"])
                break
print(len(bert_vectors_eval))

X_eval = np.array(bert_vectors_eval)
y_eval = eval_df["native_language"].values


# In[5]:


test_df = pd.read_csv(os.path.join(ORIGINAL_DATA_DIR, "lang_id_test.csv"))
print(test_df.shape)

bert_vectors_test = []
with open(os.path.join(BERT_FEATURE_DIR, "test.jsonlines"), "rt") as infile:
    for line in infile:
        bert_data = json.loads(line)
        for t in bert_data["features"]:
            # Only extract the [CLS] vector used for classification
            if t["token"] == "[CLS]":
                # We only use the representation at the final layer of the network
                bert_vectors_test.append(t["layers"][0]["values"])
                break
print(len(bert_vectors_test))

X_test = np.array(bert_vectors_test)
y_test = test_df["native_language"].values


# # Logistic Regression

# In[6]:


lr_model = LogisticRegression(penalty="l2", C=1.0)
lr_model.fit(X_train, y_train)

print("Training Accuarcy: ", lr_model.score(X_train, y_train))


# In[15]:


# Adding predicted value in the dataframe
test_df['predicted1'] = lr_model.predict(X_test)
# Class list
list_of_languages = sorted(test_df['native_language'].unique())
# Precision, recall, f-measure and support for each class
print("Evaluation for each class")
print(classification_report(y_test,test_df['predicted1'].values,target_names=list_of_languages))
print()
print("**********************************************************************************************")
print()


# Confusion matrix
matrix = confusion_matrix(test_df['native_language'], test_df['predicted1'])
plt.figure(figsize = (10,5))
ax = sns.heatmap(matrix, annot=True, xticklabels=list_of_languages, yticklabels=list_of_languages)
plt.show()
print("**********************************************************************************************")
print()


# Calculate misclassification
test_predicted = test_df.groupby('predicted1').count()['native_language']
test_misclassifications = []
for i in range(len(list_of_languages)):
    misclassification = ((200 - matrix[i][i] + (test_predicted[i] - matrix[i][i])) / 2000) * 100
    test_misclassifications.append(misclassification)

# Misclassification for each class into one dataframe
evaluation_by_class = pd.DataFrame(columns=['Language', 'Misclassification'])
for i in range(len(list_of_languages)):
    evaluation_by_class = evaluation_by_class.append(pd.DataFrame([[list_of_languages[i], test_misclassifications[i]]], 
                                        columns=['Language', 'Misclassification']))
print("Misclassification for each class")
print(evaluation_by_class.to_string())
print()
print("**********************************************************************************************")
print()


# Evaluate misclassification between all classes
evaluation_between_classes = pd.DataFrame(columns=['Language', 'Predicted', 'Misclassification'])
for i in list_of_languages:
    for j in list_of_languages:
        if(i != j):
            evaluation_between_classes = evaluation_between_classes.append(pd.DataFrame([[i, j, 
                                                              matrix[list_of_languages.index(i)][list_of_languages.index(j)]]], 
                                                              columns=['Language', 'Predicted', 'Misclassification']))
print("Misclassification between each pair of classes")
print(evaluation_between_classes.sort_values(by=['Misclassification']).to_string())
print()
print("**********************************************************************************************")
print()


print("Summary")
print("Total records:", test_df.shape[0])
print("Incorrect predictions:", evaluation_between_classes['Misclassification'].sum())
print("Correct predictions:", (test_df.shape[0] - evaluation_between_classes['Misclassification'].sum()))


# #  Neural Network

# In[8]:


nn_model = MLPClassifier(solver='lbfgs')
nn_model.fit(X_train, y_train) 

print("Training Accuarcy: ", nn_model.score(X_train, y_train))


# In[16]:


# Adding predicted value in the dataframe
test_df['predicted2'] = nn_model.predict(X_test)
# Class list
list_of_languages = sorted(test_df['native_language'].unique())
# Precision, recall, f-measure and support for each class
print("Evaluation for each class")
print(classification_report(y_test,test_df['predicted2'].values,target_names=list_of_languages))
print()
print("**********************************************************************************************")
print()


# Confusion matrix
matrix = confusion_matrix(test_df['native_language'], test_df['predicted2'])
plt.figure(figsize = (10,5))
ax = sns.heatmap(matrix, annot=True, xticklabels=list_of_languages, yticklabels=list_of_languages)
plt.show()
print("**********************************************************************************************")
print()


# Calculate misclassification
test_predicted = test_df.groupby('predicted2').count()['native_language']
test_misclassifications = []
for i in range(len(list_of_languages)):
    misclassification = ((200 - matrix[i][i] + (test_predicted[i] - matrix[i][i])) / 2000) * 100
    test_misclassifications.append(misclassification)

# Misclassification for each class into one dataframe
evaluation_by_class = pd.DataFrame(columns=['Language', 'Misclassification'])
for i in range(len(list_of_languages)):
    evaluation_by_class = evaluation_by_class.append(pd.DataFrame([[list_of_languages[i], test_misclassifications[i]]], 
                                        columns=['Language', 'Misclassification']))
print("Misclassification for each class")
print(evaluation_by_class.to_string())
print()
print("**********************************************************************************************")
print()


# Evaluate misclassification between all classes
evaluation_between_classes = pd.DataFrame(columns=['Language', 'Predicted', 'Misclassification'])
for i in list_of_languages:
    for j in list_of_languages:
        if(i != j):
            evaluation_between_classes = evaluation_between_classes.append(pd.DataFrame([[i, j, 
                                                              matrix[list_of_languages.index(i)][list_of_languages.index(j)]]], 
                                                              columns=['Language', 'Predicted', 'Misclassification']))
print("Misclassification between each pair of classes")
print(evaluation_between_classes.sort_values(by=['Misclassification']).to_string())
print()
print("**********************************************************************************************")
print()


print("Summary")
print("Total records:", test_df.shape[0])
print("Incorrect predictions:", evaluation_between_classes['Misclassification'].sum())
print("Correct predictions:", (test_df.shape[0] - evaluation_between_classes['Misclassification'].sum()))

