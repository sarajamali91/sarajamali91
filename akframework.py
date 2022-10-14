import matplotlib.pyplot as plt
import pandas as pd
import time
import datetime
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
#import tensorflow_datasets as tfds
import openml
from openml.datasets import get_dataset
#import os.path


metrics = []
Dataset = input('Please enter Dataset ID:')  # provide dataset id
dataset = openml.datasets.get_dataset(Dataset)

# Print a summary
print(
    f"This is dataset '{dataset.name}', the target feature is "
    f"'{dataset.default_target_attribute}'" 
     )
print(f"URL: {dataset.url}")
DATASET_NAME={dataset.name}

print(dataset.description)

# LabelEncoded Target
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="array", target=dataset.default_target_attribute)

dataset = pd.DataFrame(X, columns=attribute_names)
dataset["target"] = y

# training and test data split
data = dataset.sample(frac=0.80, random_state=1234)

data_unseen = dataset.drop(data.index)
data.reset_index(inplace=True, drop=True)
data_unseen.reset_index(inplace=True, drop=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))
#----------------------------------------------------------------------------------#
print("--------------------------------------------------------------------")

import autokeras as ak

MODEL_NAME='AUTOKERAS'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
y_train = y_train.astype('str')
y_test  = y_test.astype('str')

start_experiment = time.time()

print()
# It tries 5 different models by defualt.
clf = ak.StructuredDataClassifier(overwrite=True, max_trials=5)
#clf = ak.StructuredDataRegressor(overwrite=True, max_trials=10)

# Feed the structured data classifier with training data.
clf.fit(X_train, y_train, validation_split=0.15, epochs=10, batch_size=32, verbose=1)

#print(); print("Model training is complete ... ... ...")
end_expriment  = time.time()
print()
#print("Local current time :", localtime)
clf
# Evaluate the best model with testing data.
print(); print();
print(clf.evaluate(X_test, y_test, verbose=1))
            
# For Classification
import scikitplot as skplt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import pandas as pd

# Predict with the best model.
predicted_y = clf.predict(X_test, verbose=0)

# Evaluate the skill of the Trained model
acc                 = accuracy_score(y_test, predicted_y)
classReport         = classification_report(y_test, predicted_y)
confMatrix          = confusion_matrix(y_test, predicted_y)

cls_report_ak= pd.DataFrame(classification_report(y_test, predicted_y, output_dict=True)).transpose()
cls_report_ak.to_csv(f'{DATASET_NAME}_AK_Classification.csv', index= True)
metrics.append({
'Accuracy': round(accuracy_score(y_test, predicted_y),4),
'Time_sec': round(end_expriment - start_experiment),
'Time': datetime.datetime.now(),
  })
pd.DataFrame(metrics).to_csv(f'{DATASET_NAME}_{MODEL_NAME}_metrics.csv', index=False,)
print(); print('Testing Results of the trained model: ')
print(); print('Accuracy : ', acc)
print(); print('Confusion Matrix :\n', confMatrix)
print(); print('Classification Report :\n',classReport)

# Confusion matrix
skplt.metrics.plot_confusion_matrix(y_test, predicted_y,figsize=(9, 9)); plt.show()
            
print(); print("Model is complete ... ... ...")
